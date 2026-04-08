"""Training entry point and checkpoint callback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from maze_rl.config import MazeConfig, RewardConfig, TrainingConfig, maze_config_from_dict, training_config_from_dict
from maze_rl.envs.maze_env import MazeEnv
from maze_rl.policies.model_factory import create_model, load_model_from_checkpoint
from maze_rl.training.checkpointing import CheckpointManager, latest_checkpoint, load_checkpoint_metadata
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
        start_episode: int = 0,
        target_episode: int | None = None,
        save_initial_checkpoint: bool = True,
        stop_event: threading.Event | None = None,
    ) -> None:
        super().__init__()
        self.manager = manager
        self.training_config = training_config
        self.maze_config = maze_config
        self.summary = RollingTrainingSummary()
        self.completed_episodes = start_episode
        self.target_episode = target_episode if target_episode is not None else training_config.episodes
        self.save_initial_checkpoint = save_initial_checkpoint
        self.stop_event = stop_event
        self.last_saved_episode = start_episode if not save_initial_checkpoint else -1

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

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            metrics = info.get("episode_metrics")
            if isinstance(metrics, EpisodeMetrics):
                self.completed_episodes += 1
                self.summary.add(metrics)
                if self.manager.should_save(self.completed_episodes):
                    self._save_checkpoint(episode=self.completed_episodes, timesteps=self.num_timesteps)
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
) -> TrainingArtifacts:
    """Train a fresh model from episode 0."""

    training_config = training_config or TrainingConfig()
    maze_config = maze_config or MazeConfig()
    training_config.log_dir.mkdir(parents=True, exist_ok=True)
    training_config.replay_dir.mkdir(parents=True, exist_ok=True)
    manager = CheckpointManager(training_config=training_config, maze_config=maze_config)
    env = build_training_env(maze_config)
    model = create_model(training_config=training_config, env=env)
    callback = ImmutableCheckpointCallback(manager=manager, training_config=training_config, maze_config=maze_config, stop_event=stop_event)
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
    stop_event: threading.Event | None = None,
) -> TrainingArtifacts:
    """Continue training from the latest available checkpoint, or start fresh if none exists."""

    latest = latest_checkpoint(checkpoint_dir)
    if latest is None:
        base_config = maze_config_for_training_mode(MazeConfig(), training_mode)
        return train_from_scratch(
            TrainingConfig(
                episodes=additional_episodes,
                checkpoint_dir=Path(checkpoint_dir),
                algorithm="maskable_ppo" if training_mode == "maze-only" else "ppo",
                n_steps=64 if training_mode == "maze-only" else 128,
                learning_rate=3e-4 if training_mode == "maze-only" else 2.5e-4,
                ent_coef=0.05 if training_mode == "maze-only" else 0.02,
            ),
            maze_config=base_config,
            stop_event=stop_event,
        )

    latest_episode, checkpoint_path = latest
    metadata = load_checkpoint_metadata(checkpoint_path)
    maze_config = maze_config_for_training_mode(
        maze_config_from_dict(metadata["maze_config"]),
        training_mode,
    )
    training_config = training_config_from_dict(metadata["training_config"])
    training_config = TrainingConfig(
        **{
            **training_config.__dict__,
            "checkpoint_dir": Path(checkpoint_dir),
            "episodes": latest_episode + additional_episodes,
            "algorithm": (
                "maskable_ppo"
                if training_mode == "maze-only"
                else training_config.algorithm
            ),
            "n_steps": 64 if training_mode == "maze-only" else training_config.n_steps,
            "learning_rate": (
                3e-4 if training_mode == "maze-only" else training_config.learning_rate
            ),
            "ent_coef": 0.05 if training_mode == "maze-only" else training_config.ent_coef,
        }
    )
    training_config.log_dir.mkdir(parents=True, exist_ok=True)
    training_config.replay_dir.mkdir(parents=True, exist_ok=True)
    manager = CheckpointManager(training_config=training_config, maze_config=maze_config)
    env = build_training_env(maze_config)
    model = load_model_from_checkpoint(checkpoint_path, env)
    callback = ImmutableCheckpointCallback(
        manager=manager,
        training_config=training_config,
        maze_config=maze_config,
        start_episode=latest_episode,
        target_episode=latest_episode + additional_episodes,
        save_initial_checkpoint=False,
        stop_event=stop_event,
    )
    model.learn(total_timesteps=10_000_000, callback=callback, progress_bar=False, reset_num_timesteps=False)
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
        survival_reward=-0.05,
        exploration_reward=2.5,
        frontier_reward=1.0,
        revisit_penalty=-0.6,
        revisit_depth_penalty=-0.2,
        oscillation_penalty=-1.5,
        dead_end_penalty=-0.3,
        blocked_move_penalty=-2.0,
        exit_progress_reward=2.0,
        safety_gain_reward=1.0,
        safety_loss_penalty=-1.5,
        win_reward=100.0,
        caught_penalty=-50.0,
        timeout_penalty=-10.0,
        stall_penalty=-10.0,
    )
    return MazeConfig(
        **{
            **maze_config.__dict__,
            "rows": 10,
            "cols": 10,
            "max_player_speed": 1,
            "monster_speed": 1,
            "monster_activation_delay": 0,
            "monster_move_interval": 3,
            "max_episode_steps": 120,
            "stall_threshold": 40,
            "curriculum_enabled": False,
            "reward": reward,
        }
    )
