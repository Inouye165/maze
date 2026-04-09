"""Model creation and loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO, RecurrentPPO
from stable_baselines3 import PPO

from maze_rl.config import TrainingConfig


def create_model(training_config: TrainingConfig, env: Any) -> Any:
    """Create the requested SB3 model."""

    common_kwargs = {
        "env": env,
        "seed": training_config.seed,
        "learning_rate": training_config.learning_rate,
        "gamma": training_config.gamma,
        "n_steps": training_config.n_steps,
        "batch_size": training_config.batch_size,
        "ent_coef": training_config.ent_coef,
        "verbose": 0,
        "device": "cpu",
    }
    if training_config.algorithm == "maskable_ppo":
        return MaskablePPO("MlpPolicy", **common_kwargs)
    if training_config.algorithm == "ppo":
        return PPO("MlpPolicy", **common_kwargs)
    if training_config.algorithm == "recurrent_ppo":
        return RecurrentPPO("MlpLstmPolicy", **common_kwargs)
    raise ValueError(f"Unsupported algorithm: {training_config.algorithm}")


def load_model_from_checkpoint(checkpoint_path: str | Path, env: Any) -> Any:
    """Load a checkpoint zip with the correct algorithm class."""

    import json

    metadata_path = Path(checkpoint_path).with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    algorithm = metadata["algorithm"]
    if algorithm == "maskable_ppo":
        return MaskablePPO.load(str(checkpoint_path), env=env, device="cpu")
    if algorithm == "ppo":
        return PPO.load(str(checkpoint_path), env=env, device="cpu")
    if algorithm == "recurrent_ppo":
        return RecurrentPPO.load(str(checkpoint_path), env=env, device="cpu")
    raise ValueError(f"Unsupported algorithm in metadata: {algorithm}")


def predict_action(
    model: Any,
    observation: np.ndarray,
    deterministic: bool,
    recurrent_state: Any = None,
    episode_start: np.ndarray | None = None,
    action_masks: np.ndarray | None = None,
) -> tuple[int, Any]:
    """Predict an action for PPO or RecurrentPPO."""

    if model.__class__.__name__ == "RecurrentPPO":
        action, recurrent_state = model.predict(
            observation,
            state=recurrent_state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        return int(action), recurrent_state
    if model.__class__.__name__ == "MaskablePPO":
        action, recurrent_state = model.predict(observation, deterministic=deterministic, action_masks=action_masks)
        return int(action), recurrent_state
    action, recurrent_state = model.predict(observation, deterministic=deterministic)
    return int(action), recurrent_state


def action_probabilities(model: Any, observation: np.ndarray, action_masks: np.ndarray | None = None) -> np.ndarray | None:
    """Return action probabilities when the model exposes a policy distribution."""

    policy = getattr(model, "policy", None)
    if policy is None or not hasattr(policy, "obs_to_tensor") or not hasattr(policy, "get_distribution"):
        return None
    try:
        obs_tensor, _ = policy.obs_to_tensor(observation)
        try:
            distribution = policy.get_distribution(obs_tensor, action_masks=action_masks)
        except TypeError:
            distribution = policy.get_distribution(obs_tensor)
        base_distribution = getattr(distribution, "distribution", distribution)
        probabilities = getattr(base_distribution, "probs", None)
        if probabilities is None:
            return None
        values = np.asarray(probabilities.detach().cpu().numpy()).reshape(-1)
        if action_masks is not None and values.shape == action_masks.shape:
            values = values.copy()
            values[~action_masks] = 0.0
            total = float(values.sum())
            if total > 0.0:
                values /= total
        return values
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None