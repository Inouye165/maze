"""Registry-based model creation, loading, and prediction.

All algorithm-specific logic lives in :mod:`maze_rl.policies.backends`.
This module provides a backend **registry** and thin convenience wrappers
so callers don't need to pick a backend class themselves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from maze_rl.config import TrainingConfig
from maze_rl.policies.agent_interface import AgentBackend


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, AgentBackend] = {}


def register_backend(backend: AgentBackend) -> None:
    """Register a backend instance under its ``algorithm_name``."""

    _REGISTRY[backend.algorithm_name] = backend


def get_backend(algorithm: str) -> AgentBackend:
    """Look up a registered backend by algorithm name."""

    _ensure_builtins()
    try:
        return _REGISTRY[algorithm]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(
            f"Unknown algorithm {algorithm!r}. Registered backends: {available}"
        ) from exc


def available_backends() -> list[str]:
    """Return sorted names of all registered backends."""

    _ensure_builtins()
    return sorted(_REGISTRY)


_BUILTINS_REGISTERED = False


def _ensure_builtins() -> None:
    """Lazily register the three built-in backends on first access."""

    global _BUILTINS_REGISTERED  # noqa: PLW0603
    if _BUILTINS_REGISTERED:
        return
    _BUILTINS_REGISTERED = True

    from maze_rl.policies.backends import (
        MaskablePPOBackend,
        PPOBackend,
        RecurrentPPOBackend,
    )

    for backend in (PPOBackend(), MaskablePPOBackend(), RecurrentPPOBackend()):
        if backend.algorithm_name not in _REGISTRY:
            register_backend(backend)


# ---------------------------------------------------------------------------
# Checkpoint compatibility error (kept here for backward-compat imports)
# ---------------------------------------------------------------------------


class CheckpointCompatibilityError(ValueError):
    """Raised when a saved checkpoint no longer matches the current environment."""


# ---------------------------------------------------------------------------
# Convenience wrappers — same signatures as the old API
# ---------------------------------------------------------------------------


def create_model(training_config: TrainingConfig, env: Any) -> Any:
    """Create a fresh model via the registered backend for *training_config.algorithm*."""

    backend = get_backend(training_config.algorithm)
    return backend.create_model(training_config, env)


def load_model_from_checkpoint(checkpoint_path: str | Path, env: Any) -> Any:
    """Load a checkpoint zip using the algorithm recorded in its sidecar JSON."""

    metadata_path = Path(checkpoint_path).with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    algorithm = metadata["algorithm"]
    backend = get_backend(algorithm)
    try:
        return backend.load_model(checkpoint_path, env)
    except ValueError as error:
        message = str(error)
        if "spaces do not match" in message:
            raise CheckpointCompatibilityError(
                "Checkpoint is incompatible with the current environment shape. "
                "This usually means the observation or action space changed "
                "and the model must be retrained. "
                f"checkpoint={checkpoint_path} | details={message}"
            ) from error
        raise


def predict_action(
    model: Any,
    observation: np.ndarray,
    deterministic: bool,
    recurrent_state: Any = None,
    episode_start: np.ndarray | None = None,
    action_masks: np.ndarray | None = None,
) -> tuple[int, Any]:
    """Predict an action using the backend that matches the model's class."""

    backend = _backend_for_model(model)
    return backend.predict(
        model,
        observation,
        deterministic,
        recurrent_state=recurrent_state,
        episode_start=episode_start,
        action_masks=action_masks,
    )


def action_probabilities(
    model: Any,
    observation: np.ndarray,
    action_masks: np.ndarray | None = None,
) -> np.ndarray | None:
    """Return per-action probabilities via the model's backend."""

    backend = _backend_for_model(model)
    return backend.action_probabilities(model, observation, action_masks=action_masks)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

# Maps SB3 class name → algorithm name for reverse lookup during prediction.
_CLASS_TO_ALGORITHM: dict[str, str] = {
    "PPO": "ppo",
    "MaskablePPO": "maskable_ppo",
    "RecurrentPPO": "recurrent_ppo",
}


def _backend_for_model(model: Any) -> AgentBackend:
    """Resolve the backend that created or can handle *model*."""

    class_name = model.__class__.__name__
    algorithm = _CLASS_TO_ALGORITHM.get(class_name)
    if algorithm is None:
        raise ValueError(
            f"Cannot determine backend for model class {class_name!r}. "
            "Register a backend or use the backend directly."
        )
    return get_backend(algorithm)