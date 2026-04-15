"""Frozen evaluation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from maze_rl.config import MazeConfig, maze_config_from_dict
from maze_rl.envs.maze_env import MazeEnv
from maze_rl.policies.model_factory import load_model_from_checkpoint, predict_action
from maze_rl.training.checkpointing import load_checkpoint_metadata
from maze_rl.training.metrics import EpisodeMetrics


@dataclass(frozen=True)
class EvaluationSummary:
    """Summary of one or more frozen evaluation episodes."""

    checkpoint: str
    seed: int | None
    seeds: list[int]
    episodes: int
    outcomes: dict[str, int]
    average_steps: float
    average_coverage: float
    average_revisits: float
    average_oscillations: float
    average_dead_end_entries: float
    average_blocked_moves: float
    average_start_monster_distance: float
    average_time_to_capture: float | None
    frontier_reached_rate: float
    average_peak_no_progress_steps: float
    escape_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Convert the summary to a dictionary."""

        return asdict(self)


def _format_trace_line(snapshot: dict[str, Any], action: int, direction: int, speed: int, outcome: str) -> str:
    """Format one concise step trace line."""

    return (
        f"step={snapshot['steps']:03d} action={action:02d} dir={direction} speed={speed} "
        f"player={snapshot['player_position']} monster={snapshot['monster_position']} "
        f"distance={snapshot['player_monster_distance']} outcome={outcome}"
    )


def run_frozen_episode(
    model: Any,
    env: MazeEnv,
    seed: int,
    deterministic: bool = True,
    debug_trace: bool = False,
) -> EpisodeMetrics:
    """Run one deterministic episode without mutating training artifacts."""

    observation, _ = env.reset(seed=seed, options={"maze_seed": seed})
    recurrent_state = None
    episode_start = np.ones((1,), dtype=bool)
    while True:
        action, recurrent_state = predict_action(
            model=model,
            observation=observation,
            deterministic=deterministic,
            recurrent_state=recurrent_state,
            episode_start=episode_start,
            action_masks=np.asarray(env.action_masks(), dtype=bool),
        )
        is_wait = int(action) == env.wait_action_index
        if is_wait:
            direction, speed = None, 0
        else:
            direction, speed = env.decode_action(int(action))
        observation, _, terminated, truncated, info = env.step(int(action))
        episode_start = np.array([terminated or truncated], dtype=bool)
        if debug_trace:
            print(_format_trace_line(info["state_snapshot"], int(action), direction, speed, info.get("outcome", "running")))
            if terminated or truncated:
                print(f"capture={info.get('capture_diagnostics', {})}")
        if terminated or truncated:
            return info["episode_metrics"]


def evaluate_model(
    model: Any,
    maze_config: MazeConfig,
    seed: int | None = None,
    episodes: int = 1,
    seeds: list[int] | None = None,
    debug_trace: bool = False,
) -> EvaluationSummary:
    """Evaluate a loaded model on a fixed seed or seed range."""

    evaluation_seeds = seeds if seeds is not None else [seed + offset for offset in range(episodes)]
    if not evaluation_seeds:
        raise ValueError("At least one evaluation seed is required.")
    results: list[EpisodeMetrics] = []
    for evaluation_seed in evaluation_seeds:
        env = MazeEnv(maze_config, training_mode=False)
        results.append(run_frozen_episode(model, env, seed=evaluation_seed, deterministic=True, debug_trace=debug_trace))

    outcomes: dict[str, int] = {}
    for item in results:
        outcomes[item.outcome] = outcomes.get(item.outcome, 0) + 1

    return EvaluationSummary(
        checkpoint="<loaded-model>",
        seed=seed,
        seeds=evaluation_seeds,
        episodes=len(evaluation_seeds),
        outcomes=outcomes,
        average_steps=float(np.mean([item.steps for item in results])),
        average_coverage=float(np.mean([item.coverage for item in results])),
        average_revisits=float(np.mean([item.revisits for item in results])),
        average_oscillations=float(np.mean([item.oscillations for item in results])),
        average_dead_end_entries=float(np.mean([item.dead_end_entries for item in results])),
        average_blocked_moves=float(np.mean([item.blocked_moves for item in results])),
        average_start_monster_distance=float(np.mean([item.start_monster_distance for item in results])),
        average_time_to_capture=float(np.mean([item.time_to_capture for item in results if item.time_to_capture is not None]))
        if any(item.time_to_capture is not None for item in results)
        else None,
        frontier_reached_rate=float(np.mean([item.reached_new_frontier for item in results])),
        average_peak_no_progress_steps=float(np.mean([item.peak_no_progress_steps for item in results])),
        escape_rate=float(np.mean([item.outcome == "escaped" for item in results])),
    )


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    seed: int | None = None,
    episodes: int = 1,
    seeds: list[int] | None = None,
    debug_trace: bool = False,
) -> EvaluationSummary:
    """Load a checkpoint and evaluate it on frozen seeds."""

    metadata = load_checkpoint_metadata(checkpoint_path)
    maze_config = maze_config_from_dict(metadata["maze_config"])
    env = MazeEnv(maze_config, training_mode=False)
    model = load_model_from_checkpoint(checkpoint_path, env)
    summary = evaluate_model(model=model, maze_config=maze_config, seed=seed, episodes=episodes, seeds=seeds, debug_trace=debug_trace)
    summary_dict = summary.to_dict()
    summary_dict["checkpoint"] = str(checkpoint_path)
    return EvaluationSummary(**summary_dict)