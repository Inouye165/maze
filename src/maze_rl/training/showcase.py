"""Checkpoint showcase helpers for sequential playback and summaries."""

from __future__ import annotations

from copy import deepcopy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from maze_rl.config import MazeConfig, maze_config_from_dict
from maze_rl.envs.maze_env import MazeEnv
from maze_rl.envs.entities import Position
from maze_rl.policies.model_factory import (
    CheckpointCompatibilityError,
    action_probabilities,
    load_model_from_checkpoint,
    predict_action,
)
from maze_rl.policies.action_helpers import (
    DIRECTION_DELTAS,
    WAIT_ACTION,
    WAIT_DIRECTION,
    HeuristicMoveChoice,
    choose_heuristic_action,
    describe_move_choice,
    is_wait_action,
    policy_confidence as _policy_confidence,
    policy_decision_label as _policy_decision_label,
    project_action_target as _project_action_target,
    rank_legal_moves,
    should_override_policy,
    wait_action_for_env,
)
from maze_rl.training.checkpointing import load_checkpoint_metadata, resolve_checkpoint_path


@dataclass(frozen=True)
class ShowcaseResult:
    """One showcase row for a checkpoint episode."""

    checkpoint: str
    status: str
    outcome: str
    escape_rate: float
    coverage: float
    steps: int
    revisits: int
    oscillations: int
    dead_ends: int
    start_monster_distance: float | None
    time_to_capture: float | None
    frontier_rate: float
    peak_no_progress_streak: int
    final_player_position: tuple[int, int] | None = None
    final_monster_position: tuple[int, int] | None = None
    final_distance: int | None = None
    capture_rule: str | None = None
    final_state: dict[str, Any] | None = None
    checkpoint_path: str | None = None
    seed: int | None = None
    policy_override_count: int = 0
    last_override_reason: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the showcase row."""

        payload = asdict(self)
        final_state = payload.get("final_state")
        if isinstance(final_state, dict):
            final_state.pop("player", None)
            final_state.pop("monster", None)
            final_state.pop("exit", None)
        return payload


@dataclass(frozen=True)
class RecordedRun:
    """Recorded micro-step frames and final result for exact replay."""

    frames: list[dict[str, Any]]
    result: ShowcaseResult


@dataclass
class RecordedPlaybackSession:
    """Replay a previously recorded micro-step frame sequence."""

    recorded_run: RecordedRun

    def __post_init__(self) -> None:
        self.frames = [deepcopy(frame) for frame in self.recorded_run.frames]
        self.index = 0
        self.latest_state = deepcopy(self.frames[0]) if self.frames else {}
        self.done = False
        self.result: ShowcaseResult | None = None

    def advance(self) -> tuple[dict[str, Any], ShowcaseResult | None]:
        """Advance one recorded micro-step frame."""

        if self.done:
            return self.latest_state, self.result
        if not self.frames:
            self.result = self.recorded_run.result
            self.done = True
            return self.latest_state, self.result
        if self.index >= len(self.frames) - 1:
            self.result = self.recorded_run.result
            self.done = True
            return self.latest_state, self.result
        self.index += 1
        self.latest_state = deepcopy(self.frames[self.index])
        if self.index >= len(self.frames) - 1:
            self.result = self.recorded_run.result
            self.done = True
            return self.latest_state, self.result
        return self.latest_state, None


def _decorate_committed_state(
    checkpoint_label: str,
    snapshot: dict[str, Any],
    outcome: str,
    turn_step: int,
    replay_turn: dict[str, Any] | None,
    capture_diagnostics: dict[str, Any],
    frontier_rate: float,
    start_monster_distance: float | None,
    time_to_capture: float | None,
    action_index: int | None,
    action_direction: int | None,
    action_speed: int | None,
    peak_no_progress_streak: int,
    policy_kind: str = "trained",
    policy_override_enabled: bool = False,
    policy_override_count: int = 0,
    policy_override_reason: str | None = None,
) -> dict[str, Any]:
    """Build the authoritative committed state for one completed turn."""

    state = dict(snapshot)
    player_position = tuple(state.get("player_position", (0, 0)))
    monster_position = tuple(state.get("monster_position", (0, 0)))
    state.update(
        {
            "checkpoint_label": checkpoint_label,
            "outcome": outcome,
            "peak_no_progress_streak": peak_no_progress_streak,
            "start_monster_distance": start_monster_distance,
            "time_to_capture": time_to_capture,
            "frontier_rate": frontier_rate,
            "action_index": action_index,
            "action_direction": action_direction,
            "action_speed": action_speed,
            "capture_diagnostics": capture_diagnostics,
            "replay_turn": replay_turn,
            "turn_step": turn_step,
            "policy_kind": policy_kind,
            "policy_override_enabled": policy_override_enabled,
            "policy_override_count": policy_override_count,
            "policy_override_reason": policy_override_reason,
            "policy_decision_label": _policy_decision_label(
                policy_kind,
                policy_override_enabled,
                policy_override_reason,
            ),
            "current_micro_step": 0,
            "micro_step_count": 0,
            "micro_actor": None,
            "micro_phase": "idle",
            "rendered_player_position": player_position,
            "rendered_monster_position": monster_position,
            "committed_player_position": player_position,
            "committed_monster_position": monster_position,
            "player_position": player_position,
            "monster_position": monster_position,
            "player": Position(*player_position),
            "monster": Position(*monster_position),
        }
    )
    return state


def _build_microstep_frames(committed_state: dict[str, Any], replay_turn: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Build rendered micro-step frames from one structured replay event."""

    if not replay_turn:
        frame = deepcopy(committed_state)
        frame["micro_phase"] = "committed"
        return [frame]

    rendered_player = tuple(replay_turn["player_start_position"])
    rendered_monster = tuple(replay_turn["monster_start_position"])
    micro_steps = replay_turn.get("micro_steps", [])
    total_micro_steps = len(micro_steps)
    if total_micro_steps == 0:
        frame = deepcopy(committed_state)
        frame.update({"micro_phase": "committed", "micro_step_count": 0, "current_micro_step": 0})
        return [frame]

    frames: list[dict[str, Any]] = []
    for index, micro_step in enumerate(micro_steps, start=1):
        if micro_step["actor"] == "player":
            rendered_player = tuple(micro_step["position"])
        else:
            rendered_monster = tuple(micro_step["position"])
        frame = deepcopy(committed_state)
        frame.update(
            {
                "player": Position(*rendered_player),
                "monster": Position(*rendered_monster),
                "player_position": rendered_player,
                "monster_position": rendered_monster,
                "player_monster_distance": abs(rendered_player[0] - rendered_monster[0]) + abs(rendered_player[1] - rendered_monster[1]),
                "rendered_player_position": rendered_player,
                "rendered_monster_position": rendered_monster,
                "current_micro_step": index,
                "micro_step_count": total_micro_steps,
                "micro_actor": micro_step["actor"],
                "micro_phase": micro_step["phase"],
                "capture_event": replay_turn.get("capture_event") if micro_step.get("capture") else None,
            }
        )
        if index == total_micro_steps:
            frame.update(
                {
                    "player": Position(*committed_state["committed_player_position"]),
                    "monster": Position(*committed_state["committed_monster_position"]),
                    "player_position": committed_state["committed_player_position"],
                    "monster_position": committed_state["committed_monster_position"],
                    "rendered_player_position": committed_state["committed_player_position"],
                    "rendered_monster_position": committed_state["committed_monster_position"],
                    "player_monster_distance": committed_state["player_monster_distance"],
                    "micro_phase": "committed",
                    "capture_event": replay_turn.get("capture_event"),
                }
            )
        frames.append(frame)
    return frames


@dataclass
class PlaybackSession:
    """Incremental deterministic playback session for app and viewer controls."""

    checkpoint_path: str | Path
    checkpoint_label: str
    seed: int
    max_no_progress_streak: int = 25
    wall_time_timeout_s: float = 30.0
    debug_trace: bool = False
    allow_policy_override: bool = False

    def __post_init__(self) -> None:
        self.model, self.env, _ = load_checkpoint_for_playback(self.checkpoint_path)
        self.observation, _ = self.env.reset(seed=self.seed, options={"maze_seed": self.seed})
        self.recurrent_state = None
        self.episode_start = np.ones((1,), dtype=bool)
        self.started_at = time.monotonic()
        self.no_progress_streak = 0
        self.last_frontier_count = 0
        self.done = False
        self.result: ShowcaseResult | None = None
        self.pending_states: list[dict[str, Any]] = []
        self.pending_result: ShowcaseResult | None = None
        self.last_override_reason: str | None = None
        self.policy_override_count = 0
        self.latest_state = _decorate_committed_state(
            checkpoint_label=self.checkpoint_label,
            snapshot=self.env.get_render_state(),
            outcome="running",
            turn_step=0,
            replay_turn=None,
            capture_diagnostics={},
            frontier_rate=0.0,
            start_monster_distance=self.env.start_monster_distance,
            time_to_capture=None,
            action_index=None,
            action_direction=None,
            action_speed=None,
            peak_no_progress_streak=0,
            policy_kind="trained",
            policy_override_enabled=self.allow_policy_override,
            policy_override_count=0,
            policy_override_reason=None,
        )
        self.recorded_frames: list[dict[str, Any]] = [deepcopy(self.latest_state)]

    def advance(self) -> tuple[dict[str, Any], ShowcaseResult | None]:
        """Advance one micro-step frame and return the live state and optional final result."""

        if self.done:
            return self.latest_state, self.result
        if self.pending_states:
            self.latest_state = deepcopy(self.pending_states.pop(0))
            self.recorded_frames.append(deepcopy(self.latest_state))
            if not self.pending_states and self.pending_result is not None:
                self.result = self.pending_result
                self.pending_result = None
                self.done = True
                return self.latest_state, self.result
            return self.latest_state, None
        action_masks = np.asarray(self.env.action_masks(), dtype=bool)
        action, self.recurrent_state = predict_action(
            model=self.model,
            observation=self.observation,
            deterministic=True,
            recurrent_state=self.recurrent_state,
            episode_start=self.episode_start,
            action_masks=action_masks,
        )
        probabilities = action_probabilities(self.model, self.observation, action_masks=action_masks)
        chosen_confidence, confidence_gap = _policy_confidence(probabilities, int(action))
        ranked_moves = rank_legal_moves(self.env)
        chosen_move = describe_move_choice(self.env, int(action))
        best_move = ranked_moves[0] if ranked_moves else None
        self.last_override_reason = None
        should_override = should_override_policy(chosen_move, best_move, chosen_confidence, confidence_gap)
        override_applied = False
        if self.allow_policy_override and should_override and best_move is not None:
            action = best_move.action
            override_applied = True
            self.policy_override_count += 1
            self.last_override_reason = "heuristic-override"
            if chosen_move is not None and best_move is not None:
                if chosen_move.visits > 0 and best_move.visits == 0:
                    self.last_override_reason = "prefer-unvisited"
                elif chosen_move.immediate_reverse and not best_move.immediate_reverse:
                    self.last_override_reason = "avoid-reverse"
                elif chosen_move.short_loop_risk > best_move.short_loop_risk:
                    self.last_override_reason = "break-loop"
                else:
                    self.last_override_reason = "low-confidence"
        if is_wait_action(self.env, int(action)):
            direction = None
            speed = 0
            self.observation, _, terminated, truncated, info = self.env.step_wait()
        else:
            direction, speed = self.env.decode_action(int(action))
            self.observation, _, terminated, truncated, info = self.env.step(int(action))
        self.episode_start = np.array([terminated or truncated], dtype=bool)

        frontier_count = int(info.get("frontier_cells_visited", 0))
        if frontier_count > self.last_frontier_count:
            self.no_progress_streak = 0
            self.last_frontier_count = frontier_count
        else:
            self.no_progress_streak += 1

        replay_turn = info.get("replay_turn")
        committed_state = _decorate_committed_state(
            checkpoint_label=self.checkpoint_label,
            snapshot=info["state_snapshot"],
            outcome=info.get("outcome", "running"),
            turn_step=int(info["state_snapshot"].get("steps", 0)),
            replay_turn=replay_turn,
            capture_diagnostics=info.get("capture_diagnostics", {}),
            frontier_rate=1.0 if info.get("reached_new_frontier") else 0.0,
            start_monster_distance=info.get("start_monster_distance"),
            time_to_capture=info.get("time_to_capture"),
            action_index=int(action),
            action_direction=direction,
            action_speed=speed,
            peak_no_progress_streak=max(self.env.peak_no_progress_steps, self.no_progress_streak),
            policy_kind="heuristic-override" if override_applied else "trained",
            policy_override_enabled=self.allow_policy_override,
            policy_override_count=self.policy_override_count,
            policy_override_reason=self.last_override_reason,
        )
        self.pending_states = _build_microstep_frames(committed_state=committed_state, replay_turn=replay_turn)

        if self.debug_trace:
            print(
                f"step={committed_state['steps']:03d} ckpt={self.checkpoint_label} action={int(action):02d} dir={direction} speed={speed} "
                f"player={committed_state['player_position']} monster={committed_state['monster_position']} "
                f"distance={committed_state['player_monster_distance']} outcome={committed_state['outcome']} override={self.last_override_reason}"
            )

        if terminated or truncated:
            metrics = info["episode_metrics"]
            self.pending_result = ShowcaseResult(
                checkpoint=self.checkpoint_label,
                status="ok",
                outcome=metrics.outcome,
                escape_rate=1.0 if metrics.outcome == "escaped" else 0.0,
                coverage=metrics.coverage,
                steps=metrics.steps,
                revisits=metrics.revisits,
                oscillations=metrics.oscillations,
                dead_ends=metrics.dead_end_entries,
                start_monster_distance=float(metrics.start_monster_distance),
                time_to_capture=float(metrics.time_to_capture) if metrics.time_to_capture is not None else None,
                frontier_rate=1.0 if metrics.reached_new_frontier else 0.0,
                peak_no_progress_streak=metrics.peak_no_progress_steps,
                final_player_position=metrics.final_player_position,
                final_monster_position=metrics.final_monster_position,
                final_distance=metrics.final_player_monster_distance,
                capture_rule=metrics.capture_rule,
                final_state=deepcopy(committed_state),
                checkpoint_path=str(self.checkpoint_path),
                seed=self.seed,
                policy_override_count=self.policy_override_count,
                last_override_reason=self.last_override_reason,
            )
        if not self.pending_states:
            self.pending_states = [deepcopy(committed_state)]
        return self.advance()

    def build_recorded_run(self) -> RecordedRun | None:
        """Return the recorded frame sequence for exact replay."""

        if self.result is None:
            return None
        return RecordedRun(frames=[deepcopy(frame) for frame in self.recorded_frames], result=self.result)

@dataclass
class BaselinePlaybackSession:
    """Incremental playback session for a deterministic non-learning legal mover."""

    maze_config: MazeConfig
    checkpoint_label: str
    seed: int
    max_no_progress_streak: int = 25
    wall_time_timeout_s: float = 30.0
    debug_trace: bool = False

    def __post_init__(self) -> None:
        self.env = MazeEnv(self.maze_config, training_mode=False)
        self.observation, _ = self.env.reset(seed=self.seed, options={"maze_seed": self.seed})
        self.started_at = time.monotonic()
        self.no_progress_streak = 0
        self.last_frontier_count = 0
        self.done = False
        self.result: ShowcaseResult | None = None
        self.pending_states: list[dict[str, Any]] = []
        self.pending_result: ShowcaseResult | None = None
        self.latest_state = _decorate_committed_state(
            checkpoint_label=self.checkpoint_label,
            snapshot=self.env.get_render_state(),
            outcome="running",
            turn_step=0,
            replay_turn=None,
            capture_diagnostics={},
            frontier_rate=0.0,
            start_monster_distance=self.env.start_monster_distance,
            time_to_capture=None,
            action_index=None,
            action_direction=None,
            action_speed=None,
            peak_no_progress_streak=0,
            policy_kind="innate",
            policy_override_enabled=False,
            policy_override_count=0,
            policy_override_reason=None,
        )
        self.recorded_frames: list[dict[str, Any]] = [deepcopy(self.latest_state)]

    def advance(self) -> tuple[dict[str, Any], ShowcaseResult | None]:
        """Advance one micro-step frame and return the live state and optional final result."""

        if self.done:
            return self.latest_state, self.result
        if self.pending_states:
            self.latest_state = deepcopy(self.pending_states.pop(0))
            self.recorded_frames.append(deepcopy(self.latest_state))
            if not self.pending_states and self.pending_result is not None:
                self.result = self.pending_result
                self.pending_result = None
                self.done = True
                return self.latest_state, self.result
            return self.latest_state, None

        action = self._choose_legal_action()
        if is_wait_action(self.env, int(action)):
            direction = None
            speed = 0
            self.observation, _, terminated, truncated, info = self.env.step_wait()
        else:
            direction, speed = self.env.decode_action(int(action))
            self.observation, _, terminated, truncated, info = self.env.step(int(action))

        frontier_count = int(info.get("frontier_cells_visited", 0))
        if frontier_count > self.last_frontier_count:
            self.no_progress_streak = 0
            self.last_frontier_count = frontier_count
        else:
            self.no_progress_streak += 1

        replay_turn = info.get("replay_turn")
        committed_state = _decorate_committed_state(
            checkpoint_label=self.checkpoint_label,
            snapshot=info["state_snapshot"],
            outcome=info.get("outcome", "running"),
            turn_step=int(info["state_snapshot"].get("steps", 0)),
            replay_turn=replay_turn,
            capture_diagnostics=info.get("capture_diagnostics", {}),
            frontier_rate=1.0 if info.get("reached_new_frontier") else 0.0,
            start_monster_distance=info.get("start_monster_distance"),
            time_to_capture=info.get("time_to_capture"),
            action_index=int(action),
            action_direction=direction,
            action_speed=speed,
            peak_no_progress_streak=max(self.env.peak_no_progress_steps, self.no_progress_streak),
            policy_kind="innate",
            policy_override_enabled=False,
            policy_override_count=0,
            policy_override_reason=None,
        )
        self.pending_states = _build_microstep_frames(committed_state=committed_state, replay_turn=replay_turn)

        if self.debug_trace:
            print(
                f"step={committed_state['steps']:03d} baseline action={int(action):02d} dir={direction} speed={speed} "
                f"player={committed_state['player_position']} monster={committed_state['monster_position']} outcome={committed_state['outcome']}"
            )

        if terminated or truncated:
            metrics = info["episode_metrics"]
            self.pending_result = ShowcaseResult(
                checkpoint=self.checkpoint_label,
                status="ok",
                outcome=metrics.outcome,
                escape_rate=1.0 if metrics.outcome == "escaped" else 0.0,
                coverage=metrics.coverage,
                steps=metrics.steps,
                revisits=metrics.revisits,
                oscillations=metrics.oscillations,
                dead_ends=metrics.dead_end_entries,
                start_monster_distance=float(metrics.start_monster_distance),
                time_to_capture=float(metrics.time_to_capture) if metrics.time_to_capture is not None else None,
                frontier_rate=1.0 if metrics.reached_new_frontier else 0.0,
                peak_no_progress_streak=metrics.peak_no_progress_steps,
                final_player_position=metrics.final_player_position,
                final_monster_position=metrics.final_monster_position,
                final_distance=metrics.final_player_monster_distance,
                capture_rule=metrics.capture_rule,
                final_state=deepcopy(committed_state),
                checkpoint_path=None,
                seed=self.seed,
                notes="innate play",
            )
        if not self.pending_states:
            self.pending_states = [deepcopy(committed_state)]
        return self.advance()

    def build_recorded_run(self) -> RecordedRun | None:
        """Return the recorded frame sequence for exact replay."""

        if self.result is None:
            return None
        return RecordedRun(frames=[deepcopy(frame) for frame in self.recorded_frames], result=self.result)

    def _choose_legal_action(self) -> int:
        """Choose a reproducible non-learning move that favors exploration."""

        return choose_heuristic_action(self.env)


def load_checkpoint_for_playback(checkpoint_path: str | Path) -> tuple[Any, MazeEnv, dict[str, Any]]:
    """Load a checkpoint model and environment for deterministic playback."""

    metadata = load_checkpoint_metadata(checkpoint_path)
    saved = metadata["maze_config"]
    maze_config = maze_config_from_dict(saved)
    # Always use the current code's step limit so old checkpoints get the
    # updated budget without needing to retrain.
    from maze_rl.training.train import maze_config_for_training_mode  # noqa: E402 — local to avoid circular import
    current_cfg = maze_config_for_training_mode(MazeConfig(), "maze-only")
    maze_config = MazeConfig(
        **{**maze_config.__dict__, "max_episode_steps": current_cfg.max_episode_steps}
    )
    env = MazeEnv(maze_config, training_mode=False)
    model = load_model_from_checkpoint(checkpoint_path, env)
    return model, env, metadata


def run_checkpoint_showcase_episode(
    checkpoint_path: str | Path,
    checkpoint_label: str,
    seed: int,
    max_no_progress_streak: int = 25,
    wall_time_timeout_s: float = 30.0,
    on_step: Callable[[dict[str, Any]], bool] | None = None,
    debug_trace: bool = False,
    allow_policy_override: bool = False,
) -> ShowcaseResult:
    """Run one deterministic checkpoint episode with guardrails."""

    try:
        session = PlaybackSession(
            checkpoint_path=checkpoint_path,
            checkpoint_label=checkpoint_label,
            seed=seed,
            max_no_progress_streak=max_no_progress_streak,
            wall_time_timeout_s=wall_time_timeout_s,
            debug_trace=debug_trace,
            allow_policy_override=allow_policy_override,
        )
    except CheckpointCompatibilityError as error:
        return build_incompatible_result(
            checkpoint_label=checkpoint_label,
            checkpoint_path=checkpoint_path,
            seed=seed,
            reason=str(error),
        )
    while True:
        state, result = session.advance()
        if on_step is not None and not on_step(state):
            return ShowcaseResult(
                checkpoint=checkpoint_label,
                status="aborted",
                outcome="aborted",
                escape_rate=0.0,
                coverage=session.env.coverage,
                steps=session.env.step_count,
                revisits=session.env.revisits,
                oscillations=session.env.oscillations,
                dead_ends=session.env.dead_end_entries,
                start_monster_distance=float(session.env.start_monster_distance),
                time_to_capture=None,
                frontier_rate=1.0 if session.env.frontier_cells_visited > 0 else 0.0,
                peak_no_progress_streak=max(session.env.peak_no_progress_steps, session.no_progress_streak),
                final_player_position=session.env.player.as_tuple() if session.env.player is not None else None,
                final_monster_position=session.env.monster.as_tuple() if session.env.monster is not None else None,
                final_distance=state.get("player_monster_distance"),
                capture_rule=session.env.last_capture_rule,
                final_state=state,
                checkpoint_path=str(checkpoint_path),
                seed=seed,
                policy_override_count=session.policy_override_count,
                last_override_reason=session.last_override_reason,
                notes="playback aborted by viewer",
            )
        if result is not None:
            return result


def build_missing_result(checkpoint_episode: int, checkpoint_path: str | Path, seed: int) -> ShowcaseResult:
    """Build a skipped row for a missing checkpoint."""

    return ShowcaseResult(
        checkpoint=f"ckpt {checkpoint_episode:04d}",
        status="missing",
        outcome="skipped",
        escape_rate=0.0,
        coverage=0.0,
        steps=0,
        revisits=0,
        oscillations=0,
        dead_ends=0,
        start_monster_distance=None,
        time_to_capture=None,
        frontier_rate=0.0,
        peak_no_progress_streak=0,
        final_player_position=None,
        final_monster_position=None,
        final_distance=None,
        capture_rule=None,
        final_state=None,
        checkpoint_path=str(checkpoint_path),
        seed=seed,
        notes="checkpoint missing",
    )


def build_incompatible_result(
    checkpoint_label: str,
    checkpoint_path: str | Path,
    seed: int,
    reason: str,
) -> ShowcaseResult:
    """Build a skipped row for a checkpoint that no longer matches the env shape."""

    return ShowcaseResult(
        checkpoint=checkpoint_label,
        status="incompatible",
        outcome="incompatible",
        escape_rate=0.0,
        coverage=0.0,
        steps=0,
        revisits=0,
        oscillations=0,
        dead_ends=0,
        start_monster_distance=None,
        time_to_capture=None,
        frontier_rate=0.0,
        peak_no_progress_streak=0,
        final_player_position=None,
        final_monster_position=None,
        final_distance=None,
        capture_rule=None,
        final_state=None,
        checkpoint_path=str(checkpoint_path),
        seed=seed,
        notes=reason,
    )


def run_showcase_headless(
    checkpoint_dir: str | Path,
    checkpoints: list[int],
    seed: int,
    max_no_progress_streak: int = 25,
    wall_time_timeout_s: float = 30.0,
    debug_trace: bool = False,
    allow_policy_override: bool = False,
) -> list[ShowcaseResult]:
    """Run a sequential headless showcase over checkpoint episodes."""

    results: list[ShowcaseResult] = []
    for checkpoint_episode in checkpoints:
        checkpoint_path = resolve_checkpoint_path(checkpoint_dir, checkpoint_episode)
        if not Path(checkpoint_path).exists():
            results.append(build_missing_result(checkpoint_episode, checkpoint_path, seed))
            continue
        results.append(
            run_checkpoint_showcase_episode(
                checkpoint_path=checkpoint_path,
                checkpoint_label=f"ckpt {checkpoint_episode:04d}",
                seed=seed,
                max_no_progress_streak=max_no_progress_streak,
                wall_time_timeout_s=wall_time_timeout_s,
                debug_trace=debug_trace,
                allow_policy_override=allow_policy_override,
            )
        )
    return results


def save_showcase_summary(
    results: list[ShowcaseResult],
    seed: int,
    output_path: str | Path | None = None,
) -> Path:
    """Write showcase results to a JSON file."""

    output = (
        Path(output_path)
        if output_path is not None
        else Path("replays") / f"showcase_seed_{seed}_{int(time.time())}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "generated_at": int(time.time()),
        "results": [item.to_dict() for item in results],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def format_showcase_table(results: list[ShowcaseResult]) -> list[str]:
    """Build a readable row-per-checkpoint summary table."""

    header = (
        "checkpoint | status | outcome | escape_rate | coverage | steps | revisits | "
        "oscillations | dead_ends | start_monster_distance | time_to_capture | "
        "frontier_rate | peak_no_progress_streak"
    )
    lines = [header]
    for item in results:
        lines.append(
            f"{item.checkpoint} | {item.status} | {item.outcome} | {item.escape_rate:.2f} | "
            f"{item.coverage:.2f} | {item.steps} | {item.revisits} | {item.oscillations} | "
            f"{item.dead_ends} | "
            f"{item.start_monster_distance if item.start_monster_distance is not None else 'n/a'} | "
            f"{item.time_to_capture if item.time_to_capture is not None else 'n/a'} | "
            f"{item.frontier_rate:.2f} | "
            f"{item.peak_no_progress_streak}"
        )
    return lines
