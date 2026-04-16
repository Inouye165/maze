"""Stable agent backend interface for pluggable RL algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from maze_rl.config import TrainingConfig


@dataclass(frozen=True)
class BackendCapabilities:
    """Declarative description of what an algorithm backend supports."""

    supports_action_masks: bool = False
    supports_recurrent_state: bool = False


class AgentBackend(ABC):
    """Base class for pluggable RL algorithm backends.

    Every backend must implement *create_model*, *load_model*, *predict*,
    and *action_probabilities*.  The ``algorithm_name`` and ``capabilities``
    properties let callers inspect what the backend provides without
    isinstance checks on the underlying SB3 model objects.
    """

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Canonical algorithm identifier stored in checkpoint metadata."""

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Declarative feature flags for this backend."""

    @abstractmethod
    def create_model(self, training_config: TrainingConfig, env: Any) -> Any:
        """Create a fresh untrained model for the given environment."""

    @abstractmethod
    def load_model(self, checkpoint_path: str | Path, env: Any) -> Any:
        """Load trained weights from a checkpoint zip."""

    @abstractmethod
    def predict(
        self,
        model: Any,
        observation: np.ndarray,
        deterministic: bool,
        recurrent_state: Any = None,
        episode_start: np.ndarray | None = None,
        action_masks: np.ndarray | None = None,
    ) -> tuple[int, Any]:
        """Return ``(action, updated_recurrent_state)``."""

    @abstractmethod
    def action_probabilities(
        self,
        model: Any,
        observation: np.ndarray,
        action_masks: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Return per-action probabilities or *None* if unavailable."""
