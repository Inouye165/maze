"""Local pygame control app for the maze RL lab."""

# pylint: disable=too-many-lines,line-too-long

from __future__ import annotations

import inspect
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pygame

from maze_rl.config import MazeConfig, maze_config_from_dict
from maze_rl.policies.model_factory import CheckpointCompatibilityError
from maze_rl.render.view_state import (
    viewer_cell_color,
    viewer_dead_end_cells,
    viewer_explored_cells,
    viewer_exit_position,
    viewer_grid,
    viewer_monster_position,
    viewer_policy_badge,
    viewer_player_position,
    viewer_traveled_cells,
    viewer_visible_cells,
)
from maze_rl.training.checkpointing import checkpoint_is_complete, latest_checkpoint, list_checkpoints, load_checkpoint_metadata, resolve_checkpoint_path
from maze_rl.training.showcase import (
    BaselinePlaybackSession,
    PlaybackSession,
    RecordedPlaybackSession,
    RecordedRun,
    ShowcaseResult,
    build_incompatible_result,
    build_missing_result,
)
from maze_rl.training.train import continue_training_from_latest, format_training_progress, maze_config_for_training_mode

QUIT = getattr(pygame, "QUIT", 256)
MOUSEBUTTONDOWN = getattr(pygame, "MOUSEBUTTONDOWN", 1025)
KEYDOWN = getattr(pygame, "KEYDOWN", 768)
K_BACKSPACE = getattr(pygame, "K_BACKSPACE", 8)
K_RETURN = getattr(pygame, "K_RETURN", 13)

MILESTONE_EPISODES = (0, 50, 100, 200, 500, 1000)


@dataclass
class Button:
    """Clickable button definition."""

    label: str
    rect: pygame.Rect
    action: Callable[[], None]
    enabled: bool = True
    kind: str = "default"
    active: bool = False


@dataclass(frozen=True)
class TrainingStatCard:
    """Compact training performance block shown in the app dashboard."""

    label: str
    cycles: int
    wins: int
    losses: int
    percentage: float


@dataclass(frozen=True)
class CompareRunEntry:
    """One queued compare run entry."""

    checkpoint_episode: int
    checkpoint_path: Path


class LabAppController:
    """State and actions for the app-first local control app."""

    def __init__(self, checkpoint_dir: str | Path = "checkpoints") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.seed_text = "12345"
        self.debug_trace = False
        self.speed_options = [("Slow", 4), ("Normal", 8), ("Fast", 16)]
        self.speed_index = 1
        self.available_checkpoints: list[tuple[int, Path]] = []
        self.selected_index = 0
        self.session: PlaybackSession | BaselinePlaybackSession | RecordedPlaybackSession | None = None
        self.last_recorded_run: RecordedRun | None = None
        self.last_result: ShowcaseResult | None = None
        self.last_state: dict[str, Any] | None = None
        self.paused = False
        self.current_mode = "idle"
        self.selected_mode = "baseline-legal-mover"
        self.training_mode = "maze-only"
        self.training_thread: threading.Thread | None = None
        self.training_stop_event: threading.Event | None = None
        self.training_status = "idle"
        self.training_message = "ready"
        self.training_error: str | None = None
        self.training_progress: dict[str, Any] | None = None
        self.cycle_input_text = "1"
        self.last_training_increment = 100
        self.compare_queue: list[CompareRunEntry] = []
        self.compare_results: list[ShowcaseResult] = []
        self.compare_pause_until = 0.0
        self.compare_pause_s = 0.9
        self.refresh_checkpoints()
        if self.available_checkpoints:
            self.selected_mode = "current-learned-ai"
            self.selected_index = len(self.available_checkpoints) - 1

    @property
    def selected_checkpoint(self) -> tuple[int, Path] | None:
        """Return the currently selected checkpoint entry, if any."""

        if not self.available_checkpoints:
            return None
        self.selected_index = max(0, min(self.selected_index, len(self.available_checkpoints) - 1))
        return self.available_checkpoints[self.selected_index]

    @property
    def fps(self) -> int:
        """Return the active playback frame rate."""

        return self.speed_options[self.speed_index][1]

    @property
    def active_checkpoint_dir(self) -> Path:
        """Return the checkpoint directory for the active training stage."""

        return self.checkpoint_dir / ("maze_only" if self.training_mode == "maze-only" else "full_monster")

    @property
    def can_start_run(self) -> bool:
        """Return whether the app can start a new playback session."""

        return self.session is None and not self.is_training_active

    @property
    def is_training_active(self) -> bool:
        """Return whether background training is still running."""

        return self.training_thread is not None and self.training_thread.is_alive()

    def refresh_checkpoints(self) -> None:
        """Reload available checkpoints from disk."""

        previous_episode = self.selected_checkpoint[0] if self.selected_checkpoint else None
        self.available_checkpoints = list_checkpoints(self.active_checkpoint_dir)
        if not self.available_checkpoints:
            self.selected_index = 0
            return
        if previous_episode is None:
            self.selected_index = len(self.available_checkpoints) - 1
            return
        for index, (episode, _) in enumerate(self.available_checkpoints):
            if episode == previous_episode:
                self.selected_index = index
                return
        self.selected_index = len(self.available_checkpoints) - 1

    def parse_seed(self) -> int:
        """Return the current numeric seed."""

        try:
            return max(1, int(self.seed_text))
        except ValueError:
            return 12345

    def parse_cycle_count(self) -> int:
        """Return the requested training cycle count."""

        try:
            return max(1, int(self.cycle_input_text))
        except ValueError:
            return 1

    def set_speed_index(self, index: int) -> None:
        """Set playback speed by index."""

        self.speed_index = max(0, min(index, len(self.speed_options) - 1))

    def cycle_checkpoint(self, direction: int) -> None:
        """Move the checkpoint selection backward or forward."""

        if not self.available_checkpoints or not self.can_start_run:
            return
        self.selected_index = (self.selected_index + direction) % len(self.available_checkpoints)

    def use_latest_checkpoint(self) -> None:
        """Select the latest available checkpoint."""

        if not self.available_checkpoints or not self.can_start_run:
            return
        self.selected_index = len(self.available_checkpoints) - 1
        self.training_message = f"selected latest checkpoint ckpt {self.available_checkpoints[-1][0]:04d}"

    def set_mode(self, mode: str) -> None:
        """Remember the currently selected app mode."""

        self.selected_mode = mode

    def start_selected_mode(self) -> None:
        """Run the currently selected app mode."""

        if self.selected_mode == "baseline-legal-mover":
            self.start_baseline_legal_mover()
            return
        if self.selected_mode == "current-learned-ai":
            self.start_current_ai_run()
            return
        if self.selected_mode == "compare-milestones":
            self.start_compare_milestones()
            return
        if self.selected_mode == "replay":
            self.replay_last_run()
            return

    def start_baseline_legal_mover(self) -> None:
        """Run the deterministic non-learning legal-move baseline."""

        if not self.can_start_run:
            self.training_message = "finish training or the current run first"
            return
        seed = self.parse_seed()
        maze_config = self._maze_config_for_active_run()
        self._clear_compare_state()
        self.session = BaselinePlaybackSession(
            maze_config=maze_config,
            checkpoint_label="innate",
            seed=seed,
            debug_trace=self.debug_trace,
        )
        self.current_mode = "baseline-legal-mover"
        self.selected_mode = "baseline-legal-mover"
        self.last_result = None
        self.last_state = self.session.latest_state
        self.paused = False
        self.training_message = f"running innate play on seed {seed}"

    def start_current_ai_run(self) -> None:
        """Run one frozen episode using the selected checkpoint."""

        if not self.can_start_run:
            self.training_message = "finish training or the current run first"
            return
        selected = self.selected_checkpoint
        if selected is None:
            self.training_message = "no checkpoint available for current AI"
            return
        episode, checkpoint_path = selected
        seed = self.parse_seed()
        self._clear_compare_state()
        try:
            self.session = PlaybackSession(
                checkpoint_path=checkpoint_path,
                checkpoint_label=f"ckpt {episode:04d}",
                seed=seed,
                debug_trace=self.debug_trace,
                allow_policy_override=True,
            )
        except CheckpointCompatibilityError:
            self.session = None
            self.last_result = None
            self.last_state = None
            self.training_message = (
                f"ckpt {episode:04d} is incompatible with the current observation shape; retrain on this branch"
            )
            return
        self.current_mode = "current-learned-ai"
        self.selected_mode = "current-learned-ai"
        self.last_result = None
        self.last_state = self.session.latest_state
        self.paused = False
        self.training_message = f"running trained play from ckpt {episode:04d} on seed {seed}"

    def start_play(self) -> None:
        """Run the active play mode: innate without checkpoints, trained otherwise."""

        if self.selected_checkpoint is None:
            self.start_baseline_legal_mover()
            return
        self.start_current_ai_run()

    def start_marks_play(self) -> None:
        """Backward-compatible alias for the plain-English Play action."""

        self.start_play()

    def replay_last_run(self) -> None:
        """Replay the previous recorded event stream exactly."""

        if not self.can_start_run:
            self.training_message = "finish training or the current run first"
            return
        if self.last_recorded_run is None:
            self.training_message = "nothing recorded to replay"
            return
        self._clear_compare_state()
        self.session = RecordedPlaybackSession(recorded_run=self.last_recorded_run)
        self.current_mode = "replay"
        self.selected_mode = "replay"
        self.last_result = None
        self.last_state = self.session.latest_state
        self.paused = False
        self.training_message = "replaying last recorded run"

    def start_compare_milestones(self) -> None:
        """Queue milestone checkpoints for sequential in-app comparison."""

        if not self.can_start_run:
            self.training_message = "finish training or the current run first"
            return
        seed = self.parse_seed()
        self._clear_compare_state()
        self.compare_results = []
        self.compare_queue = []
        for episode in MILESTONE_EPISODES:
            checkpoint_path = resolve_checkpoint_path(self.active_checkpoint_dir, episode)
            self.compare_queue.append(CompareRunEntry(checkpoint_episode=episode, checkpoint_path=checkpoint_path))
        self.current_mode = "compare-milestones"
        self.selected_mode = "compare-milestones"
        self.last_result = None
        self.paused = False
        self.compare_pause_until = 0.0
        if not self.compare_queue:
            self.training_message = "compare finished: no milestone checkpoints found"
            return
        self.training_message = f"comparing milestones on seed {seed}"
        self._start_next_compare_run()

    def pause(self) -> None:
        """Pause the active replay."""

        self.paused = True

    def resume(self) -> None:
        """Resume the active replay."""

        if self.session is not None:
            self.paused = False

    def reset(self) -> None:
        """Clear current playback and compare state."""

        self.session = None
        self.last_state = None
        self.last_result = None
        self.paused = False
        self.current_mode = "idle"
        self._clear_compare_state()
        self.training_message = "reset"

    def step_once(self) -> None:
        """Advance one playback micro-step."""

        if self.session is None:
            if self.is_training_active:
                return
            self.start_selected_mode()
            self.paused = True
        if self.session is None:
            return
        self._advance_session()
        self.paused = True

    def start_training(self, additional_episodes: int) -> None:
        """Start background training from the latest checkpoint in the selected training mode."""

        if self.is_training_active or self.session is not None:
            self.training_message = "stop the current run before training"
            return
        self.last_training_increment = additional_episodes
        self.training_stop_event = threading.Event()
        self.training_status = "running"
        self.training_error = None
        self.training_progress = None
        self.current_mode = "training"
        self.training_message = f"training +{additional_episodes} episodes in {self.training_mode}"

        def _report_progress(progress: dict[str, Any]) -> None:
            self.training_progress = progress
            self.training_message = format_training_progress(progress)

        training_kwargs = {
            "additional_episodes": additional_episodes,
            "checkpoint_dir": self.active_checkpoint_dir,
            "training_mode": self.training_mode,
            "stop_event": self.training_stop_event,
        }
        if "progress_callback" in inspect.signature(continue_training_from_latest).parameters:
            training_kwargs["progress_callback"] = _report_progress

        def _worker() -> None:
            try:
                continue_training_from_latest(**training_kwargs)
            except (OSError, RuntimeError, ValueError) as error:  # pragma: no cover - surfaced in app status
                self.training_error = str(error)
                self.training_message = f"training error: {error}"

        self.training_thread = threading.Thread(target=_worker, daemon=True)
        self.training_thread.start()

    def continue_training(self) -> None:
        """Continue training with the last requested episode increment."""

        self.start_training(self.last_training_increment)

    def reset_training(self) -> None:
        """Clear learned checkpoints so the app returns to an untrained state."""

        if self.is_training_active:
            self.training_message = "stop training before resetting learned knowledge"
            return
        if self.session is not None:
            self.training_message = "finish or reset the current run before clearing learned knowledge"
            return

        removed_any = False
        if self.checkpoint_dir.exists():
            for path in self.checkpoint_dir.rglob("*"):
                if path.is_file() and path.name.startswith("ckpt_") and path.suffix in {".zip", ".json"}:
                    path.unlink()
                    removed_any = True

        for stage_name in ("maze_only", "full_monster"):
            stage_dir = self.checkpoint_dir / stage_name
            if stage_dir.exists() and not any(stage_dir.iterdir()):
                shutil.rmtree(stage_dir)

        self.selected_index = 0
        self.last_recorded_run = None
        self.last_result = None
        self.last_state = None
        self.training_error = None
        self.training_status = "idle"
        self.training_progress = None
        self.current_mode = "idle"
        self.selected_mode = "baseline-legal-mover"
        self.paused = False
        self._clear_compare_state()
        self.refresh_checkpoints()
        self.training_message = "cleared all learned checkpoints" if removed_any else "no learned checkpoints to clear"

    def stop_training(self) -> None:
        """Request graceful stop for background training."""

        if self.training_stop_event is None or not self.is_training_active:
            return
        self.training_stop_event.set()
        self.training_status = "stopping"
        self.training_message = "stopping training after the current episode"

    def toggle_training_mode(self) -> None:
        """Switch between simplified maze-only and full-monster training."""

        if self.is_training_active:
            return
        self.training_mode = "full-monster" if self.training_mode == "maze-only" else "maze-only"
        self.refresh_checkpoints()
        self.training_message = f"training mode set to {self.training_mode}"

    def update(self) -> None:
        """Advance playback, compare sequencing, and training state."""

        if self.training_thread is not None and not self.training_thread.is_alive():
            self.training_thread = None
            self.training_stop_event = None
            self.refresh_checkpoints()
            if self.training_error is None:
                self.training_status = "idle"
                self.training_message = "training finished"
                if self.available_checkpoints:
                    self.selected_mode = "current-learned-ai"
                    self.selected_index = len(self.available_checkpoints) - 1
            else:
                self.training_status = "error"

        if self.session is not None and not self.paused:
            self._advance_session()

        if self.current_mode == "compare-milestones" and self.session is None and self.compare_queue:
            if time.monotonic() >= self.compare_pause_until:
                self._start_next_compare_run()

    def _advance_session(self) -> None:
        if self.session is None:
            return
        active_session = self.session
        state, result = active_session.advance()
        self.last_state = state
        if result is None:
            return
        self.last_result = result
        if isinstance(active_session, (PlaybackSession, BaselinePlaybackSession)):
            self.last_recorded_run = active_session.build_recorded_run()
        self.session = None
        self.paused = False
        if self.current_mode == "compare-milestones":
            self.compare_results.append(result)
            if self.compare_queue:
                self.compare_pause_until = time.monotonic() + self.compare_pause_s
                self.training_message = f"compare pause after {result.checkpoint}"
            else:
                self.training_message = "compare milestones finished"
        else:
            self.training_message = f"run finished: {result.outcome}"

    def _start_next_compare_run(self) -> None:
        while self.compare_queue:
            entry = self.compare_queue.pop(0)
            if not checkpoint_is_complete(entry.checkpoint_path):
                self.compare_results.append(build_missing_result(entry.checkpoint_episode, entry.checkpoint_path, self.parse_seed()))
                continue
            try:
                self.session = PlaybackSession(
                    checkpoint_path=entry.checkpoint_path,
                    checkpoint_label=f"ckpt {entry.checkpoint_episode:04d}",
                    seed=self.parse_seed(),
                    debug_trace=self.debug_trace,
                    allow_policy_override=True,
                )
            except CheckpointCompatibilityError as error:
                self.compare_results.append(
                    build_incompatible_result(
                        checkpoint_label=f"ckpt {entry.checkpoint_episode:04d}",
                        checkpoint_path=entry.checkpoint_path,
                        seed=self.parse_seed(),
                        reason=str(error),
                    )
                )
                continue
            self.last_state = self.session.latest_state
            self.last_result = None
            self.training_message = f"compare running ckpt {entry.checkpoint_episode:04d}"
            return
        self.training_message = "compare milestones finished"

    def _clear_compare_state(self) -> None:
        self.compare_queue = []
        self.compare_results = []
        self.compare_pause_until = 0.0

    def _maze_config_for_active_run(self) -> MazeConfig:
        """Resolve maze configuration for baseline playback."""

        selected = self.selected_checkpoint
        if selected is not None:
            metadata = load_checkpoint_metadata(selected[1])
            return maze_config_from_dict(metadata["maze_config"])
        latest = latest_checkpoint(self.active_checkpoint_dir)
        if latest is not None:
            metadata = load_checkpoint_metadata(latest[1])
            return maze_config_from_dict(metadata["maze_config"])
        return maze_config_for_training_mode(MazeConfig(), self.training_mode)

    def mode_label(self) -> str:
        labels = {
            "idle": "Idle",
            "baseline-legal-mover": "Innate",
            "current-learned-ai": "Trained",
            "training": "Training",
            "replay": "Replay",
            "compare-milestones": "Compare Milestones",
        }
        return labels.get(self.current_mode, self.current_mode)

    def training_mode_label(self) -> str:
        return "Maze-only learning" if self.training_mode == "maze-only" else "Full monster mode"

    def selected_mode_label(self) -> str:
        labels = {
            "baseline-legal-mover": "Play Innate",
            "current-learned-ai": "Play Trained",
            "replay": "Replay Last Run",
            "compare-milestones": "Auto Compare Milestones",
        }
        return labels.get(self.selected_mode, self.selected_mode)

    def monster_visibility_label(self) -> str:
        """Describe whether monster pressure is active in the current stage."""

        if self.training_mode == "maze-only":
            return "Monster: slow chase (moves every 3rd step)"
        return "Monster pressure ON in full monster mode"

    def compare_progress_label(self) -> str:
        total = len(MILESTONE_EPISODES)
        completed = len(self.compare_results)
        running = 1 if self.current_mode == "compare-milestones" and self.session is not None else 0
        return f"{completed + running}/{total}"

    def has_marks_policy(self) -> bool:
        """Return whether a trained checkpoint is available for Play."""

        return self.selected_checkpoint is not None

    def play_mode_status(self) -> str:
        """Return the small plain-English mode label shown near Play."""

        selected = self.selected_checkpoint
        if selected is None:
            return "Mode: Innate"
        return f"Mode: Trained (ckpt {selected[0]:04d})"

    def all_time_training_card(self) -> TrainingStatCard:
        """Return the compact all-time training summary for the Basic tab."""

        return self._build_training_stat_card("All Time", self.latest_training_summary(), None)

    def recent_10_outcomes(self) -> list[str]:
        """Return the latest ten outcomes for the compact recent row."""

        summary = self.latest_training_summary()
        if summary is None:
            return []
        return self._summary_outcomes(summary, 10)

    def training_stat_cards(self) -> list[TrainingStatCard]:
        """Return training outcome cards for last 10, last 50, and all-time."""

        summary = self.latest_training_summary()
        return [
            self._build_training_stat_card("Last 10", summary, 10),
            self._build_training_stat_card("Last 50", summary, 50),
            self._build_training_stat_card("All Time", summary, None),
        ]

    def primary_status_lines(self) -> list[str]:
        """Return concise status lines for the basic tab."""

        latest_summary = self.latest_training_summary()
        last_outcome = self.last_result.outcome if self.last_result is not None else "none yet"
        cycles_seen = 0 if latest_summary is None else int(latest_summary.get("episodes_seen", 0))
        active_seed = self.active_training_seed()
        seed_suffix = "" if active_seed is None else f" | Live train seed: {active_seed}"
        return [
            f"Seed: {self.parse_seed()} | Train cycles: {self.parse_cycle_count()}",
            f"Play mode: {self.play_mode_status()} | Last outcome: {last_outcome}",
            f"Training mode: {self.training_mode_label()} | Current view: {self.mode_label()}",
            f"Training status: {self.training_status} | Cycles completed: {cycles_seen}",
            f"Status: {self.training_message}{seed_suffix}",
        ]

    def training_progress_summary(self) -> str:
        """Return one compact line for the current training progress."""

        progress = self.training_progress
        if not progress:
            return "Training progress: idle"

        completed = int(progress.get("completed_episodes", 0))
        target = max(1, int(progress.get("target_episodes", 1)))
        percent = 100.0 * completed / target
        active_cycle = int(progress.get("active_cycle", completed))
        episode_steps = int(progress.get("episode_steps", 0))
        parts = [f"Training progress: cycle {active_cycle} | move {episode_steps} | {completed}/{target} ({percent:.1f}%)"]

        maze_seed = progress.get("maze_seed")
        if maze_seed is not None:
            parts.append(f"seed {int(maze_seed)}")

        current_seconds = progress.get("current_episode_elapsed_seconds")
        if isinstance(current_seconds, (int, float)):
            parts.append(f"cycle {current_seconds:.0f}s")

        average_seconds = progress.get("average_seconds_per_cycle")
        if isinstance(average_seconds, (int, float)):
            parts.append(f"avg {average_seconds:.1f}s/cycle")

        eta_seconds = progress.get("estimated_remaining_seconds")
        if isinstance(eta_seconds, (int, float)):
            parts.append(f"eta {eta_seconds / 60.0:.1f}m")

        no_progress_steps = int(progress.get("no_progress_steps", 0))
        if no_progress_steps > 0:
            parts.append(f"no-progress {no_progress_steps}")

        return " | ".join(parts)

    def active_training_seed(self) -> int | None:
        """Return the current live training seed when a training cycle is active."""

        progress = self.training_progress
        if not progress:
            return None
        maze_seed = progress.get("maze_seed")
        return int(maze_seed) if maze_seed is not None else None

    def training_preview_state(self) -> dict[str, Any] | None:
        """Return the live training snapshot to render while training is active."""

        if not self.is_training_active:
            return None
        progress = self.training_progress
        if not progress:
            return None
        snapshot = progress.get("state_snapshot")
        if not isinstance(snapshot, dict):
            return None
        preview = dict(snapshot)
        preview["checkpoint_label"] = f"training {self.training_mode}"
        preview["policy_decision_label"] = "live training maze"
        preview["policy_kind"] = "innate"
        return preview

    def render_state(self) -> dict[str, Any] | None:
        """Return the state that should currently drive the main maze panel."""

        training_preview = self.training_preview_state()
        if training_preview is not None:
            return training_preview
        return self.last_state

    def training_progress_ratio(self) -> float:
        """Return a 0..1 completion ratio for the active training run."""

        progress = self.training_progress
        if not progress:
            return 0.0
        completed = int(progress.get("completed_episodes", 0))
        target = max(1, int(progress.get("target_episodes", 1)))
        return max(0.0, min(1.0, completed / target))

    def summary_lines(self) -> list[str]:
        """Return compact app summary lines for the side panel."""

        lines: list[str] = []
        latest_summary = self.latest_training_summary()
        if latest_summary is not None:
            lines.extend(
                [
                    f"Exit rate: {latest_summary.get('win_rate', 0.0):.2f} | Coverage: {latest_summary.get('recent_average_coverage', 0.0):.2f}",
                    f"Discovered: {latest_summary.get('recent_average_discovered_cells', 0.0):.1f} | Frontier count: {latest_summary.get('recent_frontier_expansion_count', 0.0):.1f}",
                    f"Frontier rate: {latest_summary.get('recent_frontier_reached_rate', 0.0):.2f} | Illegal moves: {latest_summary.get('recent_average_illegal_moves', 0.0):.1f}",
                    f"Revisits: {latest_summary.get('recent_average_revisits', 0.0):.1f} | Oscillations: {latest_summary.get('recent_average_oscillations', 0.0):.1f}",
                    f"Timeouts: {latest_summary.get('timeout_count', 0)} | Stalls: {latest_summary.get('stall_count', 0)}",
                    f"Avoidable captures: {latest_summary.get('avoidable_capture_count', 0)} | Recent avoidable rate: {latest_summary.get('recent_avoidable_capture_rate', 0.0):.2f}",
                ]
            )
        if self.last_result is not None:
            lines.extend(
                [
                    f"Last outcome: {self.last_result.outcome}",
                    f"Coverage: {self.last_result.coverage:.2f} | Steps: {self.last_result.steps}",
                    f"Discovered cells: {self.last_result.final_state.get('discovered_cells', 0) if self.last_result.final_state else 0} | Frontier: {self.last_result.frontier_rate:.2f}",
                    f"Oscillations: {self.last_result.oscillations} | Revisits: {self.last_result.revisits} | Repeat streak: {self.last_state.get('repeat_move_streak', 0) if self.last_state else 0}",
                    f"Illegal/blocked: {self.last_state.get('blocked_moves', 0) if self.last_state else 0}",
                    f"Exit/caught: {self.last_result.outcome == 'escaped'} / {self.last_result.outcome == 'caught'}",
                ]
            )
        if self.compare_results:
            lines.append("Compare milestones:")
            for item in self.compare_results[-6:]:
                lines.append(f"{item.checkpoint}: {item.status}/{item.outcome} cov={item.coverage:.2f} steps={item.steps}")
        return lines or ["No run summary yet."]

    def review_lines(self) -> list[str]:
        """Return compact review information for replay and compare tabs."""

        lines: list[str] = []
        if self.last_result is not None:
            lines.extend(
                [
                    f"Last run: {self.last_result.outcome} via {self.last_result.checkpoint}",
                    f"Coverage {self.last_result.coverage:.2f} | Steps {self.last_result.steps} | Revisits {self.last_result.revisits}",
                    f"Oscillations {self.last_result.oscillations} | Frontier rate {self.last_result.frontier_rate:.2f}",
                ]
            )
        if self.compare_results:
            lines.append(f"Compare progress: {self.compare_progress_label()}")
            for item in self.compare_results[-4:]:
                lines.append(f"{item.checkpoint}: {item.status} / {item.outcome} / cov {item.coverage:.2f}")
        if not lines:
            return ["Run a game, replay it, or compare milestones from the Review tab."]
        return lines

    def latest_training_summary(self) -> dict[str, Any] | None:
        """Load the latest checkpoint training summary for the active mode."""

        latest = latest_checkpoint(self.active_checkpoint_dir)
        if latest is None:
            return None
        try:
            metadata = load_checkpoint_metadata(latest[1])
        except FileNotFoundError:
            return None
        summary = metadata.get("training_summary")
        return summary if isinstance(summary, dict) else None

    def _active_policy_label(self) -> str:
        """Describe the current play mode in plain English."""

        return self.play_mode_status()

    def _build_training_stat_card(
        self,
        label: str,
        summary: dict[str, Any] | None,
        window_size: int | None,
    ) -> TrainingStatCard:
        if summary is None:
            return TrainingStatCard(label=label, cycles=0, wins=0, losses=0, percentage=0.0)
        if window_size is None:
            cycles = int(summary.get("episodes_seen", 0))
            wins = int(summary.get("wins", 0))
        else:
            outcomes = self._summary_outcomes(summary, window_size)
            cycles = len(outcomes)
            wins = sum(1 for item in outcomes if item == "escaped")
        losses = max(0, cycles - wins)
        percentage = wins / cycles if cycles else 0.0
        return TrainingStatCard(label=label, cycles=cycles, wins=wins, losses=losses, percentage=percentage)

    @staticmethod
    def _summary_outcomes(summary: dict[str, Any], window_size: int) -> list[str]:
        if window_size == 10:
            outcomes = summary.get("recent_10_outcomes")
            if isinstance(outcomes, list):
                return [str(item) for item in outcomes]
        if window_size == 50:
            outcomes = summary.get("recent_50_outcomes", summary.get("recent_outcomes"))
            if isinstance(outcomes, list):
                return [str(item) for item in outcomes][-window_size:]
        outcomes = summary.get("recent_outcomes")
        if isinstance(outcomes, list):
            return [str(item) for item in outcomes][-window_size:]
        return []


class LabControlApp:
    """Pygame control panel app for training and watching the human AI."""

    def __init__(self, checkpoint_dir: str | Path = "checkpoints", auto_quit_ms: int | None = None) -> None:
        self.controller = LabAppController(checkpoint_dir=checkpoint_dir)
        self.auto_quit_ms = auto_quit_ms
        self.window_size = (1480, 920)
        self.game_area = pygame.Rect(24, 24, 900, 872)
        self.panel_area = pygame.Rect(944, 24, 512, 872)
        self.seed_input_rect = pygame.Rect(0, 0, 0, 0)
        self.cycle_input_rect = pygame.Rect(0, 0, 0, 0)
        self.active_input: str | None = None
        self.active_tab = "basic"
        self.buttons: list[Button] = []
        self._running = False
        self.font: Any = None
        self.small_font: Any = None
        self.heading_font: Any = None
        self.title_font: Any = None

    def run(self) -> None:
        """Open the control app loop."""

        pygame.display.init()
        pygame.font.init()
        screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Maze RL Lab App")
        self.font = pygame.font.SysFont("segoeui", 18)
        self.small_font = pygame.font.SysFont("segoeui", 15)
        self.heading_font = pygame.font.SysFont("segoeui", 23, bold=True)
        self.title_font = pygame.font.SysFont("trebuchetms", 30, bold=True)
        clock = pygame.time.Clock()
        self._running = True
        started_at = time.monotonic()

        while self._running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self._running = False
                elif event.type == MOUSEBUTTONDOWN:
                    self._handle_click(event.pos)
                elif event.type == KEYDOWN:
                    self._handle_key(event)

            if self.auto_quit_ms is not None and (time.monotonic() - started_at) * 1000 >= self.auto_quit_ms:
                self._running = False

            self.controller.update()
            self.buttons = self.build_buttons()
            self._draw(screen)
            pygame.display.flip()
            clock.tick(self.controller.fps if self.controller.session is not None else 30)

        pygame.display.quit()
        pygame.font.quit()

    def build_buttons(self) -> list[Button]:
        """Return the currently visible buttons for the active tab."""

        return self._build_buttons()

    def visible_button_labels(self) -> list[str]:
        """Return the currently visible button labels for lightweight UI tests."""

        return [button.label for button in self.build_buttons()]

    def _handle_click(self, position: tuple[int, int]) -> None:
        if self.seed_input_rect.collidepoint(position):
            self.active_input = "seed"
            return
        if self.cycle_input_rect.collidepoint(position):
            self.active_input = "cycles"
            return
        self.active_input = None
        for button in self.buttons:
            if button.enabled and button.rect.collidepoint(position):
                button.action()
                return

    def _handle_key(self, event: pygame.event.Event) -> None:
        if self.active_input is None:
            return
        if event.key == K_BACKSPACE:
            if self.active_input == "seed":
                self.controller.seed_text = self.controller.seed_text[:-1]
            else:
                self.controller.cycle_input_text = self.controller.cycle_input_text[:-1]
        elif event.key == K_RETURN:
            self.active_input = None
        elif event.unicode.isdigit():
            if self.active_input == "seed" and len(self.controller.seed_text) < 10:
                self.controller.seed_text += event.unicode
            if self.active_input == "cycles" and len(self.controller.cycle_input_text) < 6:
                self.controller.cycle_input_text += event.unicode

    def _panel_layout(self) -> dict[str, pygame.Rect]:
        panel_x = self.panel_area.x + 18
        panel_width = self.panel_area.width - 36
        return {
            "tabs": pygame.Rect(panel_x, self.panel_area.y + 76, panel_width, 42),
            "content": pygame.Rect(panel_x, self.panel_area.y + 134, panel_width, self.panel_area.height - 154),
        }

    def _build_buttons(self) -> list[Button]:
        layout = self._panel_layout()
        buttons: list[Button] = []
        row_height = 34
        gap = 10
        tabs = layout["tabs"]
        tab_width = (tabs.width - 2 * gap) // 3
        tab_names = [("basic", "Basic"), ("review", "Review"), ("advanced", "Advanced")]
        for index, (tab_key, tab_label) in enumerate(tab_names):
            buttons.append(
                Button(
                    tab_label,
                    pygame.Rect(tabs.x + index * (tab_width + gap), tabs.y, tab_width, tabs.height),
                    lambda key=tab_key: self._set_tab(key),
                    kind="tab",
                    active=self.active_tab == tab_key,
                )
            )

        content = layout["content"]
        if self.active_tab == "basic":
            actions_y = content.y + 126
            play_width = 216
            train_width = 116
            reset_width = 136
            buttons.extend(
                [
                    Button(
                        "Play",
                        pygame.Rect(content.x + 12, actions_y, play_width, 54),
                        self.controller.start_play,
                        enabled=self.controller.can_start_run,
                        kind="primary",
                    ),
                    Button(
                        "Train",
                        pygame.Rect(content.x + 12 + play_width + 12, actions_y + 5, train_width, 44),
                        lambda: self.controller.start_training(self.controller.parse_cycle_count()),
                        enabled=not self.controller.is_training_active and self.controller.session is None,
                        kind="default",
                    ),
                    Button(
                        "Reset Training",
                        pygame.Rect(content.x + 12 + play_width + 12 + train_width + 12, actions_y + 5, reset_width, 44),
                        self.controller.reset_training,
                        enabled=not self.controller.is_training_active and self.controller.session is None,
                        kind="danger",
                    ),
                ]
            )
        elif self.active_tab == "review":
            buttons.extend(
                [
                    Button(
                        "Pause",
                        pygame.Rect(content.x + 12, content.y + 72, 102, row_height),
                        self.controller.pause,
                        enabled=self.controller.session is not None and not self.controller.paused,
                    ),
                    Button(
                        "Resume",
                        pygame.Rect(content.x + 126, content.y + 72, 102, row_height),
                        self.controller.resume,
                        enabled=self.controller.session is not None and self.controller.paused,
                    ),
                    Button(
                        "Step",
                        pygame.Rect(content.x + 240, content.y + 72, 102, row_height),
                        self.controller.step_once,
                        enabled=not self.controller.is_training_active,
                    ),
                    Button(
                        "Reset Run",
                        pygame.Rect(content.x + 354, content.y + 72, 110, row_height),
                        self.controller.reset,
                        enabled=self.controller.session is not None or self.controller.last_state is not None,
                    ),
                    Button(
                        "Replay Last Run",
                        pygame.Rect(content.x + 12, content.y + 122, 220, 40),
                        self.controller.replay_last_run,
                        enabled=self.controller.can_start_run and self.controller.last_recorded_run is not None,
                        kind="primary",
                    ),
                    Button(
                        "Compare Milestones",
                        pygame.Rect(content.x + 244, content.y + 122, 220, 40),
                        self.controller.start_compare_milestones,
                        enabled=self.controller.can_start_run,
                        kind="accent",
                    ),
                    Button(
                        "Previous Ckpt",
                        pygame.Rect(content.x + 12, content.y + 178, 140, row_height),
                        lambda: self.controller.cycle_checkpoint(-1),
                        enabled=bool(self.controller.available_checkpoints) and self.controller.can_start_run,
                    ),
                    Button(
                        "Next Ckpt",
                        pygame.Rect(content.x + 164, content.y + 178, 140, row_height),
                        lambda: self.controller.cycle_checkpoint(1),
                        enabled=bool(self.controller.available_checkpoints) and self.controller.can_start_run,
                    ),
                    Button(
                        "Latest",
                        pygame.Rect(content.x + 316, content.y + 178, 148, row_height),
                        self.controller.use_latest_checkpoint,
                        enabled=bool(self.controller.available_checkpoints) and self.controller.can_start_run,
                    ),
                ]
            )
        else:
            buttons.extend(
                [
                    Button(
                        "Continue Training",
                        pygame.Rect(content.x + 12, content.y + 72, 172, 40),
                        self.controller.continue_training,
                        enabled=not self.controller.is_training_active and self.controller.session is None,
                        kind="accent",
                    ),
                    Button(
                        "Stop Training",
                        pygame.Rect(content.x + 196, content.y + 72, 132, 40),
                        self.controller.stop_training,
                        enabled=self.controller.is_training_active,
                    ),
                    Button(
                        "Toggle Train Mode",
                        pygame.Rect(content.x + 340, content.y + 72, 124, 40),
                        self.controller.toggle_training_mode,
                        enabled=not self.controller.is_training_active,
                    ),
                    Button(
                        "Watch Innate",
                        pygame.Rect(content.x + 12, content.y + 128, 140, row_height),
                        self.controller.start_baseline_legal_mover,
                        enabled=self.controller.can_start_run,
                    ),
                    Button(
                        "Play Trained",
                        pygame.Rect(content.x + 164, content.y + 128, 140, row_height),
                        self.controller.start_current_ai_run,
                        enabled=self.controller.can_start_run and self.controller.has_marks_policy(),
                    ),
                    Button(
                        "Step Once",
                        pygame.Rect(content.x + 316, content.y + 128, 148, row_height),
                        self.controller.step_once,
                        enabled=not self.controller.is_training_active,
                    ),
                ]
            )
        return buttons

    def _draw(self, screen: Any) -> None:
        screen.fill((241, 236, 226))
        pygame.draw.circle(screen, (232, 202, 168), (112, 120), 180)
        pygame.draw.circle(screen, (199, 223, 209), (1380, 160), 150)
        pygame.draw.circle(screen, (214, 224, 242), (1280, 760), 190)
        pygame.draw.rect(screen, (251, 249, 244), self.game_area, border_radius=28)
        pygame.draw.rect(screen, (248, 244, 236), self.panel_area, border_radius=28)
        pygame.draw.rect(screen, (215, 205, 190), self.game_area, width=1, border_radius=28)
        pygame.draw.rect(screen, (215, 205, 190), self.panel_area, width=1, border_radius=28)
        self._draw_game_area(screen)
        self._draw_panel(screen)

    def _draw_game_area(self, screen: Any) -> None:
        inner = self.game_area.inflate(-18, -18)
        pygame.draw.rect(screen, (255, 252, 247), inner, border_radius=24)
        state = self.controller.render_state()
        if state is None or not state.get("grid"):
            title = self.title_font.render("Watch the maze before training", True, (46, 57, 71))
            screen.blit(title, (self.game_area.x + 30, self.game_area.y + 30))
            subtitle = self.font.render("Press Play to use the active model, then train by cycles and review the result.", True, (106, 116, 128))
            screen.blit(subtitle, (self.game_area.x + 30, self.game_area.y + 72))
            tip_lines = [
                "The basic tab keeps only the core flow visible.",
                "Replay, compare, and checkpoint controls live in Review and Advanced.",
            ]
            self._draw_wrapped_text(
                screen,
                self.heading_font,
                tip_lines,
                (63, 86, 104),
                pygame.Rect(self.game_area.x + 30, self.game_area.y + 146, self.game_area.width - 60, 120),
                line_gap=8,
            )
            return

        grid = viewer_grid(state)
        visible_cells = viewer_visible_cells(state)
        explored_cells = viewer_explored_cells(state)
        traveled_cells = viewer_traveled_cells(state)
        dead_end_cells = viewer_dead_end_cells(state)
        rows = len(grid)
        cols = len(grid[0])
        cell_size = min((self.game_area.width - 64) // cols, (self.game_area.height - 224) // rows)
        offset_x = self.game_area.x + 30
        offset_y = self.game_area.y + 30
        board_rect = pygame.Rect(offset_x, offset_y, cols * cell_size, rows * cell_size)

        for row_index, row in enumerate(grid):
            for col_index, cell in enumerate(row):
                rect = pygame.Rect(offset_x + col_index * cell_size, offset_y + row_index * cell_size, cell_size - 1, cell_size - 1)
                position = (row_index, col_index)
                color = viewer_cell_color(
                    cell,
                    position in visible_cells,
                    is_explored=position in explored_cells,
                    is_traveled=position in traveled_cells,
                    is_dead_end=position in dead_end_cells,
                )
                pygame.draw.rect(screen, color, rect)

        player_visible = bool(state.get("player_visible", True))
        rendered_player = viewer_player_position(state)
        rendered_monster = viewer_monster_position(state)
        exit_position = viewer_exit_position(state)
        npc_monster_visible = bool(state.get("monster_visible", True))
        if player_visible and rendered_player is not None and rendered_monster is not None and rendered_player == rendered_monster:
            self._draw_overlap_entities(screen, board_rect, cell_size, rendered_player)
        else:
            if player_visible and rendered_player is not None:
                self._draw_entity(screen, board_rect, cell_size, rendered_player, (60, 110, 220), "H", "Human AI")
            if rendered_monster is not None:
                self._draw_entity(screen, board_rect, cell_size, rendered_monster, (205, 60, 60), "M", "Monster")
        if exit_position is not None:
            self._draw_entity(screen, board_rect, cell_size, exit_position, (77, 145, 95), "E", "Exit")

        overlay_rect = pygame.Rect(self.game_area.x + 24, board_rect.bottom + 18, self.game_area.width - 48, self.game_area.bottom - board_rect.bottom - 32)
        pygame.draw.rect(screen, (244, 239, 229), overlay_rect, border_radius=18)
        pygame.draw.rect(screen, (214, 204, 188), overlay_rect, width=1, border_radius=18)
        badge_label, badge_color, badge_text_color = viewer_policy_badge(state)
        badge_rect = pygame.Rect(overlay_rect.x + 18, overlay_rect.y + 14, min(380, max(190, len(badge_label) * 10)), 30)
        pygame.draw.rect(screen, badge_color, badge_rect, border_radius=10)
        badge_text = self.font.render(badge_label, True, badge_text_color)
        screen.blit(badge_text, (badge_rect.x + 10, badge_rect.y + 5))
        training_preview = self.controller.training_preview_state() is not None
        overlay_lines = [
            f"Mode: {'Training Preview' if training_preview else self.controller.mode_label()} | Seed: {state.get('seed')} | Checkpoint: {state.get('checkpoint_label')}",
            f"Turn: {state.get('turn_step', state.get('steps'))} | Outcome: {state.get('outcome')} | Coverage: {state.get('coverage', 0.0):.2f}",
            f"Human: {rendered_player if player_visible else 'hidden'} | Monster: {rendered_monster} | Exit: {exit_position}",
            f"Seen cells: {state.get('discovered_cells', 0)} | Vision-shaded cells: {len(visible_cells)} | NPC sees monster: {npc_monster_visible} | Exit seen: {state.get('exit_seen', False)}",
            f"Distance: {state.get('player_monster_distance')} | Repeat streak: {state.get('repeat_move_streak', 0)} | Micro-step: {state.get('current_micro_step', 0)}/{state.get('micro_step_count', 0)} | Speed: {self.controller.speed_options[self.controller.speed_index][0]}",
            (
                f"Policy: {state.get('policy_kind', 'trained')} | Override enabled: {state.get('policy_override_enabled', False)} | "
                f"Override count: {state.get('policy_override_count', 0)} | Reason: {state.get('policy_override_reason', 'n/a')}"
            ),
        ]
        if training_preview:
            overlay_lines.append("This is the live maze being used for the current training cycle. It updates as the cycle and seed change.")
        if self.controller.last_result is not None:
            overlay_lines.append(
                f"Last finished run: {self.controller.last_result.outcome} | Steps {self.controller.last_result.steps} | Revisits {self.controller.last_result.revisits} | Oscillations {self.controller.last_result.oscillations} | Peak repeat {state.get('peak_repeat_move_streak', 0)}"
            )
        if self.controller.debug_trace:
            overlay_lines.extend(
                [
                    f"Debug rendered H={state.get('rendered_player_position')} | rendered M={state.get('rendered_monster_position')}",
                    f"Debug committed H={state.get('committed_player_position')} | committed M={state.get('committed_monster_position')}",
                    f"Debug actor={state.get('micro_actor')} | phase={state.get('micro_phase')} | action={state.get('action_index')} dir={state.get('action_direction')} speed={state.get('action_speed')}",
                ]
            )
        self._draw_wrapped_text(screen, self.font, overlay_lines, (57, 63, 75), pygame.Rect(overlay_rect.x + 18, overlay_rect.y + 50, overlay_rect.width - 36, overlay_rect.height - 64), line_gap=4)

    def _draw_panel(self, screen: Any) -> None:
        layout = self._panel_layout()
        clip_rect = self.panel_area.inflate(-10, -10)
        previous_clip = screen.get_clip()
        screen.set_clip(clip_rect)

        title = self.title_font.render("Maze Lab Control", True, (44, 54, 67))
        screen.blit(title, (self.panel_area.x + 18, self.panel_area.y + 18))
        subtitle = self.small_font.render("A simpler flow: Play, Train, then review runs when you need detail.", True, (108, 118, 128))
        screen.blit(subtitle, (self.panel_area.x + 18, self.panel_area.y + 52))

        if self.active_tab == "basic":
            self._draw_basic_tab(screen, layout["content"])
        elif self.active_tab == "review":
            self._draw_review_tab(screen, layout["content"])
        else:
            self._draw_advanced_tab(screen, layout["content"])

        for button in self.buttons:
            self._draw_button(screen, button)
        screen.set_clip(previous_clip)

    def _draw_basic_tab(self, screen: Any, rect: pygame.Rect) -> None:
        hero_rect = pygame.Rect(rect.x, rect.y, rect.width, 178)
        stats_rect = pygame.Rect(rect.x, hero_rect.bottom + 18, rect.width, 156)
        progress_rect = pygame.Rect(rect.x, stats_rect.bottom + 18, rect.width, 96)
        status_rect = pygame.Rect(rect.x, progress_rect.bottom + 18, rect.width, 92)

        self._draw_card(screen, hero_rect, "Play The Maze")
        helper_lines = [
            "Use one seed to make behavior comparisons obvious.",
            "A cycle is one full game. Start with 1 to see immediate cause and effect.",
        ]
        self._draw_wrapped_text(screen, self.small_font, helper_lines, (101, 109, 120), pygame.Rect(hero_rect.x + 14, hero_rect.y + 38, hero_rect.width - 28, 34), line_gap=2)
        self.seed_input_rect = pygame.Rect(hero_rect.x + 14, hero_rect.y + 92, 120, 34)
        self.cycle_input_rect = pygame.Rect(hero_rect.x + 160, hero_rect.y + 92, 96, 34)
        self._draw_input(screen, self.seed_input_rect, "Seed", self.controller.seed_text, self.active_input == "seed")
        self._draw_input(screen, self.cycle_input_rect, "Cycles", self.controller.cycle_input_text, self.active_input == "cycles")

        mode_surface = self.heading_font.render(self.controller.play_mode_status(), True, (57, 97, 151))
        screen.blit(mode_surface, (hero_rect.x + 14, hero_rect.y + 138))
        helper_surface = self.small_font.render("Play always uses the active model for this training mode.", True, (101, 109, 120))
        screen.blit(helper_surface, (hero_rect.x + 14, hero_rect.y + 162))

        self._draw_card(screen, stats_rect, "Training Summary")
        self._draw_basic_training_summary(screen, stats_rect)

        self._draw_card(screen, progress_rect, "Training Progress")
        self._draw_training_progress(screen, progress_rect)

        self._draw_card(screen, status_rect, "Status")
        self._draw_wrapped_text(screen, self.font, self.controller.primary_status_lines(), (56, 64, 76), pygame.Rect(status_rect.x + 14, status_rect.y + 38, status_rect.width - 28, status_rect.height - 50), line_gap=4)

    def _draw_review_tab(self, screen: Any, rect: pygame.Rect) -> None:
        review_rect = pygame.Rect(rect.x, rect.y, rect.width, 238)
        summary_rect = pygame.Rect(rect.x, review_rect.bottom + 14, rect.width, 238)
        detail_rect = pygame.Rect(rect.x, summary_rect.bottom + 14, rect.width, 232)

        self._draw_card(screen, review_rect, "Review")
        lines = [
            "Review puts the extra tools behind a tab so the default screen stays focused.",
            "Pause, step, replay, and compare checkpoints on the same seed.",
        ]
        self._draw_wrapped_text(screen, self.small_font, lines, (101, 109, 120), pygame.Rect(review_rect.x + 14, review_rect.y + 38, review_rect.width - 28, 32), line_gap=2)

        self._draw_card(screen, summary_rect, "Review Summary")
        self._draw_wrapped_text(screen, self.font, self.controller.review_lines(), (56, 64, 76), pygame.Rect(summary_rect.x + 14, summary_rect.y + 38, summary_rect.width - 28, summary_rect.height - 50), line_gap=4)

        self._draw_card(screen, detail_rect, "Run Details")
        self._draw_wrapped_text(screen, self.small_font, self.controller.summary_lines(), (73, 81, 93), pygame.Rect(detail_rect.x + 14, detail_rect.y + 38, detail_rect.width - 28, detail_rect.height - 50), line_gap=4)

    def _draw_advanced_tab(self, screen: Any, rect: pygame.Rect) -> None:
        control_rect = pygame.Rect(rect.x, rect.y, rect.width, 186)
        status_rect = pygame.Rect(rect.x, control_rect.bottom + 14, rect.width, 214)
        summary_rect = pygame.Rect(rect.x, status_rect.bottom + 14, rect.width, 256)

        self._draw_card(screen, control_rect, "Advanced Controls")
        lines = [
            f"Training mode: {self.controller.training_mode_label()}",
            self.controller.monster_visibility_label(),
            f"Checkpoint root: {self.controller.active_checkpoint_dir}",
            f"Playback speed: {self.controller.speed_options[self.controller.speed_index][0]} ({self.controller.fps} fps)",
        ]
        self._draw_wrapped_text(screen, self.small_font, lines, (101, 109, 120), pygame.Rect(control_rect.x + 14, control_rect.y + 38, control_rect.width - 28, 56), line_gap=4)

        self._draw_card(screen, status_rect, "Checkpoint State")
        selected = self.controller.selected_checkpoint
        latest_entry: tuple[int, Path] | None = self.controller.available_checkpoints[-1] if self.controller.available_checkpoints else None
        selected_label = "none" if selected is None else f"ckpt {selected[0]:04d}"
        latest_label = "none"
        if latest_entry is not None:
            latest_episode, _ = latest_entry
            latest_label = f"ckpt {latest_episode:04d}"
        status_lines = [
            f"Selected checkpoint: {selected_label}",
            f"Latest checkpoint: {latest_label}",
            f"Training status: {self.controller.training_status}",
            f"Compare progress: {self.controller.compare_progress_label()}",
            f"Message: {self.controller.training_message}",
        ]
        self._draw_wrapped_text(screen, self.font, status_lines, (56, 64, 76), pygame.Rect(status_rect.x + 14, status_rect.y + 38, status_rect.width - 28, status_rect.height - 50), line_gap=4)

        self._draw_card(screen, summary_rect, "Telemetry")
        self._draw_wrapped_text(screen, self.small_font, self.controller.summary_lines(), (73, 81, 93), pygame.Rect(summary_rect.x + 14, summary_rect.y + 38, summary_rect.width - 28, summary_rect.height - 50), line_gap=4)

    def _draw_training_cards(self, screen: Any, rect: pygame.Rect) -> None:
        cards = self.controller.training_stat_cards()
        card_width = (rect.width - 40) // 3
        top = rect.y + 52
        for index, card in enumerate(cards):
            card_rect = pygame.Rect(rect.x + 14 + index * (card_width + 6), top, card_width, 94)
            pygame.draw.rect(screen, (255, 250, 242), card_rect, border_radius=16)
            pygame.draw.rect(screen, (218, 207, 188), card_rect, width=1, border_radius=16)
            title = self.small_font.render(card.label, True, (91, 99, 110))
            screen.blit(title, (card_rect.x + 12, card_rect.y + 10))
            stats = [
                f"Cycles {card.cycles}",
                f"Wins {card.wins}",
                f"Losses {card.losses}",
                f"Win % {card.percentage * 100:.0f}",
            ]
            self._draw_wrapped_text(screen, self.small_font, stats, (50, 58, 69), pygame.Rect(card_rect.x + 12, card_rect.y + 30, card_rect.width - 24, card_rect.height - 36), line_gap=2)

    def _draw_basic_training_summary(self, screen: Any, rect: pygame.Rect) -> None:
        """Draw the compact all-time summary and last-ten row for the Basic tab."""

        card = self.controller.all_time_training_card()
        label_color = (91, 99, 110)
        value_color = (50, 58, 69)
        labels = ["All-time cycles", "Wins", "Losses", "Win %"]
        values = [str(card.cycles), str(card.wins), str(card.losses), f"{card.percentage * 100:.0f}%"]
        column_width = (rect.width - 28) // 4
        for index, (label, value) in enumerate(zip(labels, values, strict=True)):
            item_x = rect.x + 14 + index * column_width
            label_surface = self.small_font.render(label, True, label_color)
            value_surface = self.heading_font.render(value, True, value_color)
            screen.blit(label_surface, (item_x, rect.y + 44))
            screen.blit(value_surface, (item_x, rect.y + 68))

        recent = self.controller.recent_10_outcomes()
        recent_label = self.small_font.render("Last 10", True, label_color)
        screen.blit(recent_label, (rect.x + 14, rect.y + 112))
        if recent:
            symbols = " ".join("W" if item == "escaped" else "L" for item in recent)
            recent_surface = self.small_font.render(symbols, True, value_color)
        else:
            recent_surface = self.small_font.render("No completed cycles yet", True, value_color)
        screen.blit(recent_surface, (rect.x + 74, rect.y + 112))

    def _draw_training_progress(self, screen: Any, rect: pygame.Rect) -> None:
        """Draw the live training progress bar and timing hint."""

        progress = self.controller.training_progress
        ratio = self.controller.training_progress_ratio()
        label_surface = self.small_font.render(self.controller.training_progress_summary(), True, (56, 64, 76))
        screen.blit(label_surface, (rect.x + 14, rect.y + 18))

        bar_rect = pygame.Rect(rect.x + 14, rect.y + 44, rect.width - 28, 18)
        pygame.draw.rect(screen, (233, 227, 216), bar_rect, border_radius=9)
        fill_width = int(bar_rect.width * ratio)
        if fill_width > 0:
            fill_color = (57, 97, 151)
            if progress is not None and progress.get("status") == "no-progress":
                fill_color = (188, 119, 58)
            fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, max(8, fill_width), bar_rect.height)
            pygame.draw.rect(screen, fill_color, fill_rect, border_radius=9)
        pygame.draw.rect(screen, (200, 191, 177), bar_rect, width=1, border_radius=9)

        if progress is None:
            hint = "Start training to watch the bar fill and the cycle age update."
        else:
            active_cycle = int(progress.get("active_cycle", progress.get("completed_episodes", 0)))
            move_number = int(progress.get("episode_steps", 0))
            current_seconds = progress.get("current_episode_elapsed_seconds")
            current_label = f"cycle {active_cycle}, move {move_number}"
            eta_seconds = progress.get("estimated_remaining_seconds")
            eta_label = f"eta {eta_seconds / 60.0:.1f}m" if isinstance(eta_seconds, (int, float)) else "eta n/a"
            if isinstance(current_seconds, (int, float)):
                hint = f"{current_label} | age {current_seconds:.0f}s | {eta_label} | status {progress.get('status', 'running')}"
            else:
                hint = f"{current_label} | {eta_label} | status {progress.get('status', 'running')}"
        hint_surface = self.small_font.render(hint, True, (101, 109, 120))
        screen.blit(hint_surface, (rect.x + 14, rect.y + 66))

    def _draw_input(self, screen: Any, rect: pygame.Rect, label: str, value: str, active: bool) -> None:
        label_surface = self.small_font.render(label, True, (95, 103, 113))
        screen.blit(label_surface, (rect.x, rect.y - 18))
        pygame.draw.rect(screen, (255, 251, 246), rect, border_radius=10)
        pygame.draw.rect(screen, (87, 120, 148) if active else (206, 196, 182), rect, width=2 if active else 1, border_radius=10)
        value_surface = self.font.render(value or "", True, (43, 52, 64))
        screen.blit(value_surface, (rect.x + 10, rect.y + 5))

    def _draw_button(self, screen: Any, button: Button) -> None:
        if button.kind == "tab":
            fill = (67, 93, 116) if button.active else (233, 227, 216)
            text_color = (248, 248, 244) if button.active else (61, 69, 80)
        elif not button.enabled:
            fill = (207, 201, 193)
            text_color = (125, 128, 133)
        elif button.kind == "primary":
            fill = (57, 97, 151)
            text_color = (250, 251, 252)
        elif button.kind == "danger":
            fill = (155, 76, 66)
            text_color = (251, 248, 242)
        elif button.kind == "accent":
            fill = (205, 122, 76)
            text_color = (251, 248, 242)
        elif button.active:
            fill = (96, 125, 97)
            text_color = (248, 249, 246)
        else:
            fill = (234, 228, 218)
            text_color = (56, 63, 74)
        pygame.draw.rect(screen, fill, button.rect, border_radius=10)
        pygame.draw.rect(screen, (200, 191, 177), button.rect, width=1, border_radius=10)
        text = self.small_font.render(button.label, True, text_color)
        screen.blit(text, (button.rect.centerx - text.get_width() / 2, button.rect.centery - text.get_height() / 2))

    def _draw_card(self, screen: Any, rect: pygame.Rect, title: str) -> None:
        pygame.draw.rect(screen, (244, 239, 231), rect, border_radius=18)
        pygame.draw.rect(screen, (216, 206, 190), rect, width=1, border_radius=18)
        heading = self.heading_font.render(title, True, (49, 58, 71))
        screen.blit(heading, (rect.x + 14, rect.y + 10))

    def _set_tab(self, tab_key: str) -> None:
        self.active_tab = tab_key
        self.active_input = None

    def _draw_entity(
        self,
        screen: Any,
        board_rect: pygame.Rect,
        cell_size: int,
        position: tuple[int, int],
        color: tuple[int, int, int],
        glyph: str,
        label: str,
    ) -> None:
        rect = pygame.Rect(
            board_rect.x + position[1] * cell_size + 3,
            board_rect.y + position[0] * cell_size + 3,
            cell_size - 6,
            cell_size - 6,
        )
        pygame.draw.rect(screen, color, rect, border_radius=8)
        glyph_text = self.heading_font.render(glyph, True, (255, 255, 255))
        screen.blit(glyph_text, (rect.centerx - glyph_text.get_width() / 2, rect.centery - glyph_text.get_height() / 2))
        label_surface = self.small_font.render(label, True, (248, 249, 250))
        pill_width = label_surface.get_width() + 12
        pill = pygame.Rect(rect.x, max(board_rect.y - 2, rect.y - 22), pill_width, 18)
        if pill.right > board_rect.right:
            pill.x = board_rect.right - pill.width
        pygame.draw.rect(screen, (10, 12, 18), pill, border_radius=8)
        pygame.draw.rect(screen, color, pill, width=1, border_radius=8)
        screen.blit(label_surface, (pill.x + 6, pill.y + 2))

    def _draw_overlap_entities(self, screen: Any, board_rect: pygame.Rect, cell_size: int, position: tuple[int, int]) -> None:
        rect = pygame.Rect(
            board_rect.x + position[1] * cell_size + 3,
            board_rect.y + position[0] * cell_size + 3,
            cell_size - 6,
            cell_size - 6,
        )
        left_half = pygame.Rect(rect.x, rect.y, rect.width // 2, rect.height)
        right_half = pygame.Rect(rect.x + rect.width // 2, rect.y, rect.width - rect.width // 2, rect.height)
        pygame.draw.rect(screen, (60, 110, 220), left_half, border_top_left_radius=8, border_bottom_left_radius=8)
        pygame.draw.rect(screen, (205, 60, 60), right_half, border_top_right_radius=8, border_bottom_right_radius=8)
        left_text = self.heading_font.render("H", True, (255, 255, 255))
        right_text = self.heading_font.render("M", True, (255, 255, 255))
        screen.blit(left_text, (left_half.centerx - left_text.get_width() / 2, rect.centery - left_text.get_height() / 2))
        screen.blit(right_text, (right_half.centerx - right_text.get_width() / 2, rect.centery - right_text.get_height() / 2))
        label_surface = self.small_font.render("Human AI + Monster", True, (248, 249, 250))
        pill_width = label_surface.get_width() + 12
        pill = pygame.Rect(rect.x, max(board_rect.y - 2, rect.y - 22), pill_width, 18)
        if pill.right > board_rect.right:
            pill.x = board_rect.right - pill.width
        pygame.draw.rect(screen, (10, 12, 18), pill, border_radius=8)
        pygame.draw.rect(screen, (205, 60, 60), pill, width=1, border_radius=8)
        screen.blit(label_surface, (pill.x + 6, pill.y + 2))

    def _draw_wrapped_text(
        self,
        screen: Any,
        font: Any,
        lines: list[str],
        color: tuple[int, int, int],
        rect: pygame.Rect,
        line_gap: int = 2,
    ) -> None:
        y = rect.y
        for line in lines:
            for wrapped in self._wrap_line(font, line, rect.width):
                if y + font.get_height() > rect.bottom:
                    return
                surface = font.render(wrapped, True, color)
                screen.blit(surface, (rect.x, y))
                y += font.get_height() + line_gap

    @staticmethod
    def _wrap_line(font: Any, text: str, width: int) -> list[str]:
        if not text:
            return [""]
        words = text.split()
        if not words:
            return [text]
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if font.size(candidate)[0] <= width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines


def run_app(checkpoint_dir: str | Path = "checkpoints", auto_quit_ms: int | None = None) -> None:
    """Launch the local control app."""

    LabControlApp(checkpoint_dir=checkpoint_dir, auto_quit_ms=auto_quit_ms).run()
