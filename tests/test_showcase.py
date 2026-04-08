"""Showcase workflow tests."""

from pathlib import Path

from maze_rl.config import MazeConfig, TrainingConfig
from maze_rl.envs.maze_env import MazeEnv
from maze_rl.policies.model_factory import create_model
from maze_rl.training.checkpointing import CheckpointManager
from maze_rl.training.showcase import format_showcase_table, run_showcase_headless, save_showcase_summary


def test_showcase_skips_missing_checkpoints_and_saves_json(tmp_path: Path) -> None:
    """Headless showcase should skip missing checkpoints and still write a summary."""

    checkpoint_dir = tmp_path / "checkpoints"
    training_config = TrainingConfig(checkpoint_dir=checkpoint_dir)
    maze_config = MazeConfig()
    env = MazeEnv(maze_config)
    model = create_model(training_config=training_config, env=env)
    manager = CheckpointManager(training_config=training_config, maze_config=maze_config)
    manager.save(
        model=model,
        episode=0,
        timesteps=0,
        training_summary={"episodes_seen": 0},
        evaluation_summary={"episodes": 0},
    )

    results = run_showcase_headless(
        checkpoint_dir=checkpoint_dir,
        checkpoints=[0, 200],
        seed=12345,
        max_no_progress_streak=10,
        wall_time_timeout_s=5.0,
    )
    summary_path = save_showcase_summary(results, seed=12345, output_path=tmp_path / "summary.json")
    lines = format_showcase_table(results)

    assert len(results) == 2
    assert results[0].checkpoint == "ckpt 0000"
    assert results[1].status == "missing"
    assert summary_path.exists()
    assert lines[0].startswith("checkpoint | status")
    assert any("ckpt 0200" in line for line in lines)