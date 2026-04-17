"""Multi-seed evaluation and curriculum tests."""

from pathlib import Path

from maze_rl.config import MazeConfig, TrainingConfig
from maze_rl.envs.maze_env import MazeEnv
from maze_rl.policies.model_factory import create_model
from maze_rl.training.checkpointing import CheckpointManager
from maze_rl.training.evaluate import evaluate_checkpoint


def test_multi_seed_eval_aggregates_requested_seeds(tmp_path: Path) -> None:
    """Frozen evaluation should support an explicit seed list."""

    training_config = TrainingConfig(checkpoint_dir=tmp_path / "checkpoints")
    maze_config = MazeConfig()
    env = MazeEnv(maze_config)
    model = create_model(training_config=training_config, env=env)
    manager = CheckpointManager(training_config=training_config, maze_config=maze_config)
    checkpoint_path = manager.save(
        model=model,
        episode=0,
        timesteps=0,
        training_summary={"episodes_seen": 0},
        evaluation_summary={"episodes": 0},
    )

    summary = evaluate_checkpoint(checkpoint_path, seeds=[12345, 12346, 12347])

    assert summary.episodes == 3
    assert summary.seeds == [12345, 12346, 12347]
    assert 0.0 <= summary.frontier_reached_rate <= 1.0


def test_training_curriculum_starts_easier_than_frozen_eval() -> None:
    """Training mode should use the easier starting curriculum stage."""

    maze_config = MazeConfig()
    training_env = MazeEnv(maze_config, training_mode=True)
    frozen_env = MazeEnv(maze_config, training_mode=False)

    _, training_info = training_env.reset(seed=123)
    _, frozen_info = frozen_env.reset(seed=123)

    assert training_info["curriculum_stage"] == "bootstrap"
    assert training_info["monster_speed"] < frozen_info["monster_speed"]
    assert training_info["monster_activation_delay"] > frozen_info["monster_activation_delay"]


def test_curriculum_progresses_from_smaller_whole_mazes_to_full_size() -> None:
    """Curriculum stages should grow the active maze while keeping the full observation size fixed."""

    maze_config = MazeConfig()
    env = MazeEnv(maze_config, training_mode=True)

    early_stage = env._resolve_curriculum_stage(0)
    late_stage = env._resolve_curriculum_stage(80)

    assert early_stage.rows < maze_config.rows
    assert early_stage.cols < maze_config.cols
    assert late_stage.rows == maze_config.rows
    assert late_stage.cols == maze_config.cols
    assert env.observation_spec.rows == maze_config.rows
    assert env.observation_spec.cols == maze_config.cols