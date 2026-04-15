"""Training continuation regression tests."""

from pathlib import Path

import pytest

from maze_rl.config import MazeConfig, TrainingConfig, as_serializable_dict
from maze_rl.policies.model_factory import CheckpointCompatibilityError
from maze_rl.training.metrics import RollingTrainingSummary
from maze_rl.training.train import continue_training_from_latest


class _FakeModel:
    def __init__(self) -> None:
        self.seed_calls: list[int] = []
        self.num_timesteps = 321

    def set_random_seed(self, seed: int) -> None:
        self.seed_calls.append(seed)

    def learn(self, total_timesteps: int, callback, progress_bar: bool, reset_num_timesteps: bool) -> None:
        _ = total_timesteps
        _ = callback
        _ = progress_bar
        _ = reset_num_timesteps


def test_continue_training_offsets_and_reapplies_seed(monkeypatch, tmp_path: Path) -> None:
    """Resumed training should reseed the model so it does not replay the original maze stream."""

    checkpoint_path = tmp_path / "checkpoints" / "ckpt_0005.zip"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"placeholder")
    metadata = {
        "maze_config": as_serializable_dict(MazeConfig()),
        "training_config": as_serializable_dict(
            TrainingConfig(seed=7, algorithm="maskable_ppo", checkpoint_dir=checkpoint_path.parent)
        ),
    }
    model = _FakeModel()

    monkeypatch.setattr(
        "maze_rl.training.train.latest_checkpoint",
        lambda _checkpoint_dir: (5, checkpoint_path),
    )
    monkeypatch.setattr(
        "maze_rl.training.train.load_checkpoint_metadata",
        lambda _checkpoint_path: metadata,
    )
    monkeypatch.setattr(
        "maze_rl.training.train.build_training_env",
        lambda _maze_config: object(),
    )
    monkeypatch.setattr(
        "maze_rl.training.train.load_model_from_checkpoint",
        lambda _checkpoint_path, _env: model,
    )

    continue_training_from_latest(additional_episodes=2, checkpoint_dir=checkpoint_path.parent, training_mode="maze-only")

    assert model.seed_calls == [12]


def test_continue_training_rejects_old_plain_ppo_checkpoint_after_real_progress(monkeypatch, tmp_path: Path) -> None:
    """Unsafe legacy PPO checkpoints should fail fast because they cannot honor action masks."""

    checkpoint_path = tmp_path / "checkpoints" / "ckpt_0005.zip"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"placeholder")
    metadata = {
        "maze_config": as_serializable_dict(MazeConfig()),
        "training_config": as_serializable_dict(TrainingConfig(seed=7, algorithm="ppo", checkpoint_dir=checkpoint_path.parent)),
    }

    monkeypatch.setattr(
        "maze_rl.training.train.latest_checkpoint",
        lambda _checkpoint_dir: (5, checkpoint_path),
    )
    monkeypatch.setattr(
        "maze_rl.training.train.load_checkpoint_metadata",
        lambda _checkpoint_path: metadata,
    )

    with pytest.raises(ValueError, match="plain PPO"):
        continue_training_from_latest(additional_episodes=2, checkpoint_dir=checkpoint_path.parent, training_mode="full-monster")


def test_continue_training_restarts_from_scratch_for_episode_zero_plain_ppo_checkpoint(monkeypatch, tmp_path: Path) -> None:
    """A legacy episode-zero checkpoint can be discarded and restarted under MaskablePPO."""

    checkpoint_path = tmp_path / "checkpoints" / "ckpt_0000.zip"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"placeholder")
    metadata = {
        "maze_config": as_serializable_dict(MazeConfig()),
        "training_config": as_serializable_dict(TrainingConfig(seed=7, algorithm="ppo", checkpoint_dir=checkpoint_path.parent)),
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "maze_rl.training.train.latest_checkpoint",
        lambda _checkpoint_dir: (0, checkpoint_path),
    )
    monkeypatch.setattr(
        "maze_rl.training.train.load_checkpoint_metadata",
        lambda _checkpoint_path: metadata,
    )

    def fake_train_from_scratch(training_config, maze_config, stop_event=None, progress_callback=None):
        _ = maze_config
        _ = stop_event
        _ = progress_callback
        captured["algorithm"] = training_config.algorithm
        captured["episodes"] = training_config.episodes
        return object()

    monkeypatch.setattr("maze_rl.training.train.train_from_scratch", fake_train_from_scratch)

    continue_training_from_latest(additional_episodes=2, checkpoint_dir=checkpoint_path.parent, training_mode="full-monster")

    assert captured == {"algorithm": "maskable_ppo", "episodes": 2}


def test_continue_training_falls_back_to_fresh_model_on_incompatible_checkpoint(monkeypatch, tmp_path: Path) -> None:
    """An incompatible checkpoint should be replaced by a fresh model that continues the episode numbering."""

    checkpoint_path = tmp_path / "checkpoints" / "ckpt_0100.zip"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"placeholder")
    metadata = {
        "maze_config": as_serializable_dict(MazeConfig()),
        "training_config": as_serializable_dict(
            TrainingConfig(seed=7, algorithm="maskable_ppo", checkpoint_dir=checkpoint_path.parent)
        ),
    }
    fresh_model = _FakeModel()

    monkeypatch.setattr(
        "maze_rl.training.train.latest_checkpoint",
        lambda _checkpoint_dir: (100, checkpoint_path),
    )
    monkeypatch.setattr(
        "maze_rl.training.train.load_checkpoint_metadata",
        lambda _checkpoint_path: metadata,
    )
    monkeypatch.setattr(
        "maze_rl.training.train.build_training_env",
        lambda _maze_config: object(),
    )
    monkeypatch.setattr(
        "maze_rl.training.train.load_model_from_checkpoint",
        lambda _checkpoint_path, _env: (_ for _ in ()).throw(
            CheckpointCompatibilityError("Observation spaces do not match")
        ),
    )
    monkeypatch.setattr(
        "maze_rl.training.train.create_model",
        lambda training_config, env: fresh_model,
    )

    continue_training_from_latest(
        additional_episodes=5,
        checkpoint_dir=checkpoint_path.parent,
        training_mode="maze-only",
    )

    assert fresh_model.seed_calls == [107]


def test_rolling_training_summary_loads_prior_totals() -> None:
    """Resumed training summaries should preserve all-time episode counters."""

    summary = RollingTrainingSummary()

    summary.load_snapshot(
        {
            "episodes_seen": 125,
            "wins": 44,
            "timeout_count": 8,
            "stall_count": 3,
            "avoidable_capture_count": 6,
        }
    )

    snapshot = summary.snapshot()

    assert snapshot["episodes_seen"] == 125
    assert snapshot["wins"] == 44
    assert snapshot["timeout_count"] == 8
    assert snapshot["stall_count"] == 3
    assert snapshot["avoidable_capture_count"] == 6