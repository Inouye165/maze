"""Checkpoint creation and metadata handling."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from maze_rl.config import MazeConfig, TrainingConfig, as_serializable_dict


class CheckpointManager:
    """Manage immutable checkpoint model files and metadata."""

    def __init__(self, training_config: TrainingConfig, maze_config: MazeConfig) -> None:
        self.training_config = training_config
        self.maze_config = maze_config
        self.training_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def should_save(self, episode: int) -> bool:
        """Return whether the episode should create an immutable checkpoint."""

        if episode in self.training_config.checkpoint_episodes:
            return True
        last_required = max(self.training_config.checkpoint_episodes)
        return episode > last_required and episode % self.training_config.recurring_checkpoint_interval == 0

    def model_path(self, episode: int) -> Path:
        """Model path for a checkpoint episode."""

        return self.training_config.checkpoint_dir / f"ckpt_{episode:04d}.zip"

    def metadata_path(self, episode: int) -> Path:
        """Metadata path for a checkpoint episode."""

        return self.training_config.checkpoint_dir / f"ckpt_{episode:04d}.json"

    def save(
        self,
        model: Any,
        episode: int,
        timesteps: int,
        training_summary: dict[str, Any],
        evaluation_summary: dict[str, Any],
    ) -> Path:
        """Save a model and its metadata."""

        path = self.model_path(episode)
        model.save(str(path))
        metadata = {
            "episode": episode,
            "timesteps": timesteps,
            "algorithm": self.training_config.algorithm,
            "maze_config": as_serializable_dict(self.maze_config),
            "training_config": as_serializable_dict(self.training_config),
            "training_summary": training_summary,
            "evaluation_summary": evaluation_summary,
        }
        self.metadata_path(episode).write_text(json.dumps(_normalize_for_json(metadata), indent=2), encoding="utf-8")
        return path


def load_checkpoint_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load metadata for a checkpoint zip file."""

    checkpoint = Path(checkpoint_path)
    metadata_path = checkpoint.with_suffix(".json")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def checkpoint_is_complete(checkpoint_path: str | Path) -> bool:
    """Return whether a checkpoint zip has its matching metadata file."""

    checkpoint = Path(checkpoint_path)
    return checkpoint.exists() and checkpoint.with_suffix(".json").exists()


def resolve_checkpoint_path(checkpoint_root: str | Path, checkpoint_episode: int) -> Path:
    """Resolve a checkpoint episode number to a zip path."""

    return Path(checkpoint_root) / f"ckpt_{checkpoint_episode:04d}.zip"


def list_checkpoints(checkpoint_root: str | Path) -> list[tuple[int, Path]]:
    """List available checkpoint zip files sorted by episode."""

    root = Path(checkpoint_root)
    items: list[tuple[int, Path]] = []
    for path in sorted(root.glob("ckpt_*.zip")):
        if not checkpoint_is_complete(path):
            continue
        try:
            episode = int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        items.append((episode, path))
    return sorted(items, key=lambda item: item[0])


def latest_checkpoint(checkpoint_root: str | Path) -> tuple[int, Path] | None:
    """Return the latest available checkpoint if one exists."""

    items = list_checkpoints(checkpoint_root)
    return items[-1] if items else None


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__") and not isinstance(value, type):
        return {key: _normalize_for_json(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _normalize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(item) for item in value]
    return value