"""Training entry point and checkpoint callback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any, Callable

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from maze_rl.config import (
    MazeConfig,
    RewardConfig,
    TrainingConfig,
    maze_config_from_dict,
    training_config_from_dict,
)
from maze_rl.envs.maze_env import MazeEnv
from maze_rl.policies.model_factory import CheckpointCompatibilityError, create_model, load_model_from_checkpoint
from maze_rl.training.checkpointing import (
    CheckpointManager,
    latest_checkpoint,
    load_checkpoint_metadata,
)
from maze_rl.training.evaluate import evaluate_model
from maze_rl.training.metrics import EpisodeMetrics, RollingTrainingSummary


@dataclass(frozen=True)
class TrainingArtifacts:
    """Paths and metadata produced by training."""

    checkpoint_dir: Path
    final_episode_count: int
    total_timesteps: int


class ImmutableCheckpointCallback(BaseCallback):
    """Save immutable checkpoints on episode boundaries."""

    def __init__(
        self,
        manager: CheckpointManager,
        training_config: TrainingConfig,
        maze_config: MazeConfig,
        initial_training_summary: dict[str, Any] | None = None,
        start_episode: int = 0,
        target_episode: int | None = None,
        save_initial_checkpoint: bool = True,
        stop_event: threading.Event | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        report_every_timesteps: int = 8,
        report_every_seconds: float = 0.2,
        report_no_progress_steps: int = 10,
    ) -> None:
        super().__init__()
        self.manager = manager
        self.training_config = training_config
        self.maze_config = maze_config
        self.summary = RollingTrainingSummary()
        self.summary.load_snapshot(initial_training_summary)
        self.completed_episodes = start_episode
        self.target_episode = (
            target_episode if target_episode is not None else training_config.episodes
        )
        self.save_initial_checkpoint = save_initial_checkpoint
        self.stop_event = stop_event
        self.progress_callback = progress_callback
        self.report_every_timesteps = report_every_timesteps
        self.report_every_seconds = report_every_seconds
        self.report_no_progress_steps = report_no_progress_steps
        self.last_saved_episode = start_episode if not save_initial_checkpoint else -1
        self._session_start_episode = start_episode
        self._started_at = time.monotonic()
        self._current_episode_started_at = self._started_at
        self._last_progress_timesteps = 0
        self._last_progress_emit_at = self._started_at
        self._last_progress_no_progress_steps = -1

    def _on_training_start(self) -> None:
        if not self.save_initial_checkpoint:
            return
        self._save_checkpoint(episode=0, timesteps=0)

    def _save_checkpoint(self, episode: int, timesteps: int) -> None:
        """Save one immutable checkpoint with a fresh held-out evaluation."""

        evaluation = evaluate_model(
            model=self.model,
            maze_config=self.maze_config,
            seed=self.maze_config.held_out_seed,
            episodes=self.training_config.held_out_eval_episodes,
        ).to_dict()
        self.manager.save(
            model=self.model,
            episode=episode,
            timesteps=timesteps,
            training_summary=self.summary.snapshot(),
            evaluation_summary=evaluation,
        )
        self.last_saved_episode = episode

    def _emit_progress(self, info: dict[str, Any] | None, status: str) -> None:
        if self.progress_callback is None:
            return

        now = time.monotonic()
        snapshot = info.get("state_snapshot", {}) if info is not None else {}
        snapshot_summary = self.summary.snapshot()
        completed = self.completed_episodes
        target = max(1, self.target_episode)
        active_cycle = completed + 1 if status in {"running", "no-progress"} else completed
        elapsed_seconds = now - self._started_at
        average_seconds_per_cycle = elapsed_seconds / completed if completed > 0 else None
        estimated_remaining_seconds = (
            average_seconds_per_cycle * (target - completed)
            if average_seconds_per_cycle is not None
            else None
        )
        payload = {
            "status": status,
            "completed_episodes": completed,
            "target_episodes": target,
            "training_summary_snapshot": snapshot_summary,
            "session_start_episode": self._session_start_episode,
            "session_completed_episodes": completed - self._session_start_episode,
            "session_target_episodes": target - self._session_start_episode,
            "active_cycle": active_cycle,
            "progress_ratio": completed / target,
            "maze_seed": int(info.get("maze_seed", 0)) if info is not None else None,
            "state_snapshot": snapshot if info is not None else None,
            "timesteps": self.num_timesteps,
            "episode_steps": int(snapshot.get("steps", 0)),
            "coverage": float(info.get("coverage", 0.0)) if info is not None else 0.0,
            "outcome": info.get("outcome", "running") if info is not None else "running",
            "no_progress_steps": (
                int(info.get("no_progress_steps", 0)) if info is not None else 0
            ),
            "peak_no_progress_steps": (
                int(info.get("peak_no_progress_steps", 0)) if info is not None else 0
            ),
            "repeat_move_streak": (
                int(info.get("repeat_move_streak", 0)) if info is not None else 0
            ),
            "repeat_loop_detected": (
                bool(info.get("repeat_loop_detected", False)) if info is not None else False
            ),
            "avoidable_capture": (
                bool(info.get("avoidable_capture", False)) if info is not None else False
            ),
            "avoidable_capture_reason": (
                info.get("avoidable_capture_reason") if info is not None else None
            ),
            "recent_win_rate": snapshot_summary["recent_win_rate"],
            "recent_stall_rate": snapshot_summary["recent_stall_rate"],
            "recent_timeout_rate": snapshot_summary["recent_timeout_rate"],
            "recent_avoidable_capture_rate": (
                snapshot_summary["recent_avoidable_capture_rate"]
            ),
            "recent_average_coverage": snapshot_summary["recent_average_coverage"],
            "elapsed_seconds": elapsed_seconds,
            "current_episode_elapsed_seconds": now - self._current_episode_started_at,
            "average_seconds_per_cycle": average_seconds_per_cycle,
            "estimated_remaining_seconds": estimated_remaining_seconds,
        }
        self._last_progress_emit_at = now
        self.progress_callback(payload)

    def _maybe_emit_step_progress(self, info: dict[str, Any]) -> None:
        if self.progress_callback is None:
            return

        no_progress_steps = int(info.get("no_progress_steps", 0))
        if self.num_timesteps - self._last_progress_timesteps >= self.report_every_timesteps:
            self._last_progress_timesteps = self.num_timesteps
            self._last_progress_no_progress_steps = no_progress_steps
            self._emit_progress(info, status="running")
            return

        if time.monotonic() - self._last_progress_emit_at >= self.report_every_seconds:
            self._emit_progress(info, status="running")
            return

        if (
            no_progress_steps >= self.report_no_progress_steps
            and no_progress_steps != self._last_progress_no_progress_steps
        ):
            self._last_progress_no_progress_steps = no_progress_steps
            self._emit_progress(info, status="no-progress")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            self._maybe_emit_step_progress(info)
            metrics = info.get("episode_metrics")
            if isinstance(metrics, EpisodeMetrics):
                self.completed_episodes += 1
                self.summary.add(metrics)
                self._emit_progress(info, status=metrics.outcome)
                self._current_episode_started_at = time.monotonic()
                if self.manager.should_save(self.completed_episodes):
                    self._save_checkpoint(
                        episode=self.completed_episodes,
                        timesteps=self.num_timesteps,
                    )
                if self.completed_episodes >= self.target_episode:
                    return False
        if self.stop_event is not None and self.stop_event.is_set():
            return False
        return True

    def _on_training_end(self) -> None:
        if self.completed_episodes > self.last_saved_episode:
            self._save_checkpoint(episode=self.completed_episodes, timesteps=self.num_timesteps)


def build_training_env(maze_config: MazeConfig) -> DummyVecEnv:
    """Build the single-environment vec wrapper used for SB3 training."""

    return DummyVecEnv([lambda: Monitor(MazeEnv(maze_config, training_mode=True))])


def train_from_scratch(
    training_config: TrainingConfig | None = None,
    maze_config: MazeConfig | None = None,
    stop_event: threading.Event | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> TrainingArtifacts:
    """Train a fresh model from episode 0."""

    training_config = training_config or TrainingConfig()
    maze_config = maze_config or MazeConfig()
    training_config.log_dir.mkdir(parents=True, exist_ok=True)
    training_config.replay_dir.mkdir(parents=True, exist_ok=True)
    manager = CheckpointManager(training_config=training_config, maze_config=maze_config)
    env = build_training_env(maze_config)
    model = create_model(training_config=training_config, env=env)
    callback = ImmutableCheckpointCallback(
        manager=manager,
        training_config=training_config,
        maze_config=maze_config,
        stop_event=stop_event,
        progress_callback=progress_callback,
    )
    model.learn(total_timesteps=10_000_000, callback=callback, progress_bar=False)
    return TrainingArtifacts(
        checkpoint_dir=training_config.checkpoint_dir,
        final_episode_count=callback.completed_episodes,
        total_timesteps=model.num_timesteps,
    )


def continue_training_from_latest(
    additional_episodes: int,
    checkpoint_dir: str | Path = "checkpoints",
    training_mode: str = "maze-only",
    fixed_maze_seed: int | None = None,
    stop_event: threading.Event | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> TrainingArtifacts:
    """Continue training from the latest available checkpoint, or start fresh if none exists."""

    def _with_fixed_seed(maze_config: MazeConfig) -> MazeConfig:
        if fixed_maze_seed is None:
            return maze_config
        return MazeConfig(
            **{
                **maze_config.__dict__,
                "fixed_maze_seed": int(fixed_maze_seed),
                "train_seed_base": int(fixed_maze_seed),
            }
        )

    latest = latest_checkpoint(checkpoint_dir)
    if latest is None:
        base_config = _with_fixed_seed(
            maze_config_for_training_mode(MazeConfig(), training_mode)
        )
        return train_from_scratch(
            TrainingConfig(
                episodes=additional_episodes,
                checkpoint_dir=Path(checkpoint_dir),
                algorithm="maskable_ppo",
                n_steps=64 if training_mode == "maze-only" else 128,
                learning_rate=3e-4 if training_mode == "maze-only" else 2.5e-4,
                ent_coef=0.05 if training_mode == "maze-only" else 0.02,
            ),
            maze_config=base_config,
            stop_event=stop_event,
            progress_callback=progress_callback,
        )

    latest_episode, checkpoint_path = latest
    metadata = load_checkpoint_metadata(checkpoint_path)
    previous_training_summary = metadata.get("training_summary")
    maze_config = _with_fixed_seed(
        maze_config_for_training_mode(
            maze_config_from_dict(metadata["maze_config"]),
            training_mode,
        )
    )
    previous_training_config = training_config_from_dict(metadata["training_config"])
    if previous_training_config.algorithm != "maskable_ppo":
        if latest_episode <= 0:
            return train_from_scratch(
                TrainingConfig(
                    episodes=additional_episodes,
                    checkpoint_dir=Path(checkpoint_dir),
                    algorithm="maskable_ppo",
                    seed=previous_training_config.seed,
                    n_steps=64 if training_mode == "maze-only" else 128,
                    learning_rate=(
                        3e-4 if training_mode == "maze-only" else 2.5e-4
                    ),
                    ent_coef=0.05 if training_mode == "maze-only" else 0.02,
                ),
                maze_config=maze_config,
                stop_event=stop_event,
                progress_callback=progress_callback,
            )
        raise ValueError(
            "Existing checkpoint uses plain PPO, so innate safety masks cannot be "
            "enforced during training. "
            "Reset this training branch and retrain with maskable_ppo to preserve "
            "flee behavior."
        )
    resumed_seed = previous_training_config.seed + latest_episode
    training_config = TrainingConfig(
        **{
            **previous_training_config.__dict__,
            "checkpoint_dir": Path(checkpoint_dir),
            "episodes": latest_episode + additional_episodes,
            "seed": resumed_seed,
            "algorithm": "maskable_ppo",
            "n_steps": (
                64 if training_mode == "maze-only" else previous_training_config.n_steps
            ),
            "learning_rate": (
                3e-4 if training_mode == "maze-only" else previous_training_config.learning_rate
            ),
            "ent_coef": 0.05 if training_mode == "maze-only" else previous_training_config.ent_coef,
        }
    )
    training_config.log_dir.mkdir(parents=True, exist_ok=True)
    training_config.replay_dir.mkdir(parents=True, exist_ok=True)
    manager = CheckpointManager(training_config=training_config, maze_config=maze_config)
    env = build_training_env(maze_config)
    try:
        model = load_model_from_checkpoint(checkpoint_path, env)
    except CheckpointCompatibilityError:
        model = create_model(training_config=training_config, env=env)
    model.set_random_seed(training_config.seed)
    callback = ImmutableCheckpointCallback(
        manager=manager,
        training_config=training_config,
        maze_config=maze_config,
        initial_training_summary=previous_training_summary if isinstance(previous_training_summary, dict) else None,
        start_episode=latest_episode,
        target_episode=latest_episode + additional_episodes,
        save_initial_checkpoint=False,
        stop_event=stop_event,
        progress_callback=progress_callback,
    )
    model.learn(
        total_timesteps=10_000_000,
        callback=callback,
        progress_bar=False,
        reset_num_timesteps=False,
    )
    return TrainingArtifacts(
        checkpoint_dir=training_config.checkpoint_dir,
        final_episode_count=callback.completed_episodes,
        total_timesteps=model.num_timesteps,
    )


def maze_config_for_training_mode(maze_config: MazeConfig, training_mode: str) -> MazeConfig:
    """Adjust maze settings for the requested app training mode."""

    if training_mode == "full-monster":
        return maze_config
    if training_mode != "maze-only":
        raise ValueError(f"Unsupported training mode: {training_mode}")

    reward = RewardConfig(
        survival_reward=-0.08,
        exploration_reward=2.5,
        frontier_reward=1.0,
        revisit_penalty=-0.6,
        revisit_depth_penalty=-0.2,
        oscillation_penalty=-1.5,
        dead_end_penalty=-0.45,
        # Keep this below exploration and exit progress so PPO still has to trade off,
        # but make avoidable visible dead ends costly enough to learn before contact.
        avoidable_visible_dead_end_penalty=-0.8,
        trap_threat_penalty=-12.0,
        blocked_move_penalty=-2.0,
        exit_progress_reward=1.6,
        safety_gain_reward=1.8,
        safety_loss_penalty=-3.2,
        win_reward=100.0,
        caught_penalty=-50.0,
        timeout_penalty=-10.0,
        stall_penalty=-10.0,
    )
    return MazeConfig(
        **{
            **maze_config.__dict__,
            "rows": 19,
            "cols": 19,
            "max_player_speed": 1,
            "monster_speed": 1,
            "monster_activation_delay": 0,
            "monster_move_interval": 3,
            "max_episode_steps": 560,
            "stall_threshold": 90,
            "curriculum_enabled": False,
            "reward": reward,
        }
    )


def format_training_progress(progress: dict[str, Any]) -> str:
    """Format a compact progress line for CLI and app status surfaces."""

    completed = int(progress.get("completed_episodes", 0))
    target = max(1, int(progress.get("target_episodes", 1)))
    progress_pct = 100.0 * completed / target
    active_cycle = int(progress.get("active_cycle", completed))
    episode_steps = int(progress.get("episode_steps", 0))
    no_progress_steps = int(progress.get("no_progress_steps", 0))
    peak_no_progress_steps = int(progress.get("peak_no_progress_steps", 0))
    coverage = float(progress.get("coverage", 0.0))
    recent_win_rate = float(progress.get("recent_win_rate", 0.0))
    recent_stall_rate = float(progress.get("recent_stall_rate", 0.0))
    recent_timeout_rate = float(progress.get("recent_timeout_rate", 0.0))
    recent_avoidable_capture_rate = float(progress.get("recent_avoidable_capture_rate", 0.0))
    status = str(progress.get("status", "running"))
    maze_seed = progress.get("maze_seed")
    extra = f" | cycle {active_cycle} | move {episode_steps}"
    if maze_seed is not None:
        extra += f" | seed={int(maze_seed)}"
    if status == "no-progress" or no_progress_steps > 0:
        extra += " | no-progress"
    if progress.get("repeat_loop_detected"):
        extra += " | repeat-loop"
    if progress.get("avoidable_capture"):
        reason = progress.get("avoidable_capture_reason") or "avoidable-capture"
        extra += f" | {reason}"
    return (
        f"training {completed}/{target} ({progress_pct:.1f}%)"
        f" | steps={episode_steps}"
        f" | timesteps={int(progress.get('timesteps', 0))}"
        f" | coverage={coverage:.2f}"
        f" | no_progress={no_progress_steps}"
        f" | peak_no_progress={peak_no_progress_steps}"
        f" | recent_win_rate={recent_win_rate:.2f}"
        f" | recent_stall_rate={recent_stall_rate:.2f}"
        f" | recent_timeout_rate={recent_timeout_rate:.2f}"
        f" | recent_avoidable_capture_rate={recent_avoidable_capture_rate:.2f}"
        f"{extra}"
    )
