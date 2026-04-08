"""Checkpoint save and load tests."""

from pathlib import Path

from maze_rl.config import MazeConfig, TrainingConfig
from maze_rl.envs.maze_env import MazeEnv
from maze_rl.policies.model_factory import create_model, load_model_from_checkpoint
from maze_rl.training.checkpointing import CheckpointManager, load_checkpoint_metadata
from maze_rl.training.evaluate import evaluate_model


def test_checkpoint_save_and_load_round_trip(tmp_path: Path) -> None:
    """Checkpoint artifacts should be saved and loadable."""

    training_config = TrainingConfig(checkpoint_dir=tmp_path / "checkpoints")
    maze_config = MazeConfig()
    env = MazeEnv(maze_config)
    model = create_model(training_config=training_config, env=env)
    manager = CheckpointManager(training_config=training_config, maze_config=maze_config)

    evaluation = evaluate_model(model=model, maze_config=maze_config, seed=maze_config.held_out_seed).to_dict()
    checkpoint_path = manager.save(
        model=model,
        episode=0,
        timesteps=0,
        training_summary={"episodes_seen": 0},
        evaluation_summary=evaluation,
    )

    metadata = load_checkpoint_metadata(checkpoint_path)
    loaded_model = load_model_from_checkpoint(checkpoint_path, MazeEnv(maze_config))

    assert checkpoint_path.exists()
    assert checkpoint_path.with_suffix(".json").exists()
    assert metadata["episode"] == 0
    assert loaded_model is not None
