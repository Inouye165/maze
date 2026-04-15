"""Concrete AgentBackend implementations for supported SB3 algorithms."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO, RecurrentPPO
from stable_baselines3 import PPO

from maze_rl.config import TrainingConfig
from maze_rl.policies.agent_interface import AgentBackend, BackendCapabilities


def _common_model_kwargs(training_config: TrainingConfig, env: Any) -> dict[str, Any]:
    """Shared SB3 constructor keyword arguments."""

    return {
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


def _base_action_probabilities(
    model: Any,
    observation: np.ndarray,
    action_masks: np.ndarray | None = None,
) -> np.ndarray | None:
    """Shared implementation for action-probability extraction."""

    policy = getattr(model, "policy", None)
    if (policy is None
            or not hasattr(policy, "obs_to_tensor")
            or not hasattr(policy, "get_distribution")):
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


class PPOBackend(AgentBackend):
    """Standard PPO backend (stable-baselines3)."""

    @property
    def algorithm_name(self) -> str:
        return "ppo"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities()

    def create_model(self, training_config: TrainingConfig, env: Any) -> Any:
        return PPO("MlpPolicy", **_common_model_kwargs(training_config, env))

    def load_model(self, checkpoint_path: str | Path, env: Any) -> Any:
        return PPO.load(str(checkpoint_path), env=env, device="cpu")

    def predict(
        self,
        model: Any,
        observation: np.ndarray,
        deterministic: bool,
        recurrent_state: Any = None,
        episode_start: np.ndarray | None = None,
        action_masks: np.ndarray | None = None,
    ) -> tuple[int, Any]:
        action, state = model.predict(observation, deterministic=deterministic)
        return int(action), state

    def action_probabilities(
        self,
        model: Any,
        observation: np.ndarray,
        action_masks: np.ndarray | None = None,
    ) -> np.ndarray | None:
        return _base_action_probabilities(model, observation, action_masks)


class MaskablePPOBackend(AgentBackend):
    """MaskablePPO backend (sb3-contrib)."""

    @property
    def algorithm_name(self) -> str:
        return "maskable_ppo"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(supports_action_masks=True)

    def create_model(self, training_config: TrainingConfig, env: Any) -> Any:
        return MaskablePPO("MlpPolicy", **_common_model_kwargs(training_config, env))

    def load_model(self, checkpoint_path: str | Path, env: Any) -> Any:
        return MaskablePPO.load(str(checkpoint_path), env=env, device="cpu")

    def predict(
        self,
        model: Any,
        observation: np.ndarray,
        deterministic: bool,
        recurrent_state: Any = None,
        episode_start: np.ndarray | None = None,
        action_masks: np.ndarray | None = None,
    ) -> tuple[int, Any]:
        action, state = model.predict(
            observation,
            deterministic=deterministic,
            action_masks=action_masks,
        )
        return int(action), state

    def action_probabilities(
        self,
        model: Any,
        observation: np.ndarray,
        action_masks: np.ndarray | None = None,
    ) -> np.ndarray | None:
        return _base_action_probabilities(model, observation, action_masks)


class RecurrentPPOBackend(AgentBackend):
    """RecurrentPPO backend (sb3-contrib)."""

    @property
    def algorithm_name(self) -> str:
        return "recurrent_ppo"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(supports_recurrent_state=True)

    def create_model(self, training_config: TrainingConfig, env: Any) -> Any:
        return RecurrentPPO("MlpLstmPolicy", **_common_model_kwargs(training_config, env))

    def load_model(self, checkpoint_path: str | Path, env: Any) -> Any:
        return RecurrentPPO.load(str(checkpoint_path), env=env, device="cpu")

    def predict(
        self,
        model: Any,
        observation: np.ndarray,
        deterministic: bool,
        recurrent_state: Any = None,
        episode_start: np.ndarray | None = None,
        action_masks: np.ndarray | None = None,
    ) -> tuple[int, Any]:
        action, state = model.predict(
            observation,
            state=recurrent_state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        return int(action), state

    def action_probabilities(
        self,
        model: Any,
        observation: np.ndarray,
        action_masks: np.ndarray | None = None,
    ) -> np.ndarray | None:
        return _base_action_probabilities(model, observation, action_masks)
