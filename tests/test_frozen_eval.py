"""Frozen evaluation should not mutate checkpoint artifacts."""

from hashlib import sha256
from pathlib import Path

from maze_rl.config import MazeConfig, TrainingConfig
from maze_rl.envs.maze_env import MazeEnv
from maze_rl.policies.model_factory import create_model
from maze_rl.training.checkpointing import CheckpointManager
from maze_rl.training.evaluate import evaluate_checkpoint


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_frozen_evaluation_does_not_mutate_checkpoint_files(tmp_path: Path) -> None:
    """Evaluation should leave checkpoint zip and metadata untouched."""

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
    metadata_path = checkpoint_path.with_suffix(".json")
    before_zip = _digest(checkpoint_path)
    before_json = _digest(metadata_path)

    summary = evaluate_checkpoint(checkpoint_path, seed=maze_config.held_out_seed, episodes=1)

    assert summary.episodes == 1
    assert _digest(checkpoint_path) == before_zip
    assert _digest(metadata_path) == before_json
