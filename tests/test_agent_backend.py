"""Tests for the agent backend abstraction layer."""

from __future__ import annotations

import ast
import inspect
import textwrap

from maze_rl.policies.agent_interface import AgentBackend, BackendCapabilities
from maze_rl.policies.backends import (
    MaskablePPOBackend,
    PPOBackend,
    RecurrentPPOBackend,
)
from maze_rl.policies.model_factory import (
    available_backends,
    get_backend,
    register_backend,
)


# ------------------------------------------------------------------
# Registry tests
# ------------------------------------------------------------------


def test_builtin_backends_registered() -> None:
    """All three built-in algorithms should be available."""

    names = available_backends()
    assert "ppo" in names
    assert "maskable_ppo" in names
    assert "recurrent_ppo" in names


def test_get_backend_returns_correct_type() -> None:
    """get_backend should return the matching concrete class."""

    assert isinstance(get_backend("ppo"), PPOBackend)
    assert isinstance(get_backend("maskable_ppo"), MaskablePPOBackend)
    assert isinstance(get_backend("recurrent_ppo"), RecurrentPPOBackend)


def test_get_backend_unknown_raises_valueerror() -> None:
    """Requesting a non-existent backend should raise ValueError."""

    try:
        get_backend("does_not_exist")
        raise AssertionError("Expected ValueError")
    except ValueError as exc:
        assert "does_not_exist" in str(exc)


def test_custom_backend_registration() -> None:
    """A custom backend can be registered and retrieved."""

    class _DummyBackend(AgentBackend):
        @property
        def algorithm_name(self) -> str:
            return "_test_dummy"

        @property
        def capabilities(self) -> BackendCapabilities:
            return BackendCapabilities()

        def create_model(self, training_config, env):
            pass

        def load_model(self, checkpoint_path, env):
            pass

        def predict(self, model, observation, deterministic, **kw):
            return 0, None

        def action_probabilities(self, model, observation, **kw):
            return None

    register_backend(_DummyBackend())
    assert "_test_dummy" in available_backends()
    assert isinstance(get_backend("_test_dummy"), _DummyBackend)


# ------------------------------------------------------------------
# Capabilities tests
# ------------------------------------------------------------------


def test_ppo_capabilities() -> None:
    """PPO backend should not support masks or recurrence."""

    caps = PPOBackend().capabilities
    assert not caps.supports_action_masks
    assert not caps.supports_recurrent_state


def test_maskable_ppo_capabilities() -> None:
    """MaskablePPO should advertise action mask support."""

    caps = MaskablePPOBackend().capabilities
    assert caps.supports_action_masks
    assert not caps.supports_recurrent_state


def test_recurrent_ppo_capabilities() -> None:
    """RecurrentPPO should advertise recurrent state support."""

    caps = RecurrentPPOBackend().capabilities
    assert not caps.supports_action_masks
    assert caps.supports_recurrent_state


# ------------------------------------------------------------------
# Env import isolation test
# ------------------------------------------------------------------


def test_maze_env_does_not_import_showcase() -> None:
    """maze_env.py must not import from training.showcase at module level.

    Lazy imports inside functions are acceptable but should reference
    policies.action_helpers, not training.showcase.
    """

    source_path = inspect.getfile(
        __import__("maze_rl.envs.maze_env", fromlist=["MazeEnv"])
    )

    with open(source_path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=source_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith(
                "maze_rl.training.showcase"
            ), (
                f"maze_env.py still imports from training.showcase "
                f"(line {node.lineno}): from {node.module} import ..."
            )


# ------------------------------------------------------------------
# Backward-compat re-export tests
# ------------------------------------------------------------------


def test_showcase_reexports_action_helpers() -> None:
    """Key helpers should remain importable from showcase for compat."""

    from maze_rl.training.showcase import (  # noqa: F401
        WAIT_ACTION,
        HeuristicMoveChoice,
        choose_heuristic_action,
        describe_move_choice,
        rank_legal_moves,
        should_override_policy,
    )


def test_model_factory_backward_compat_api() -> None:
    """Legacy model_factory top-level functions must still be importable."""

    from maze_rl.policies.model_factory import (  # noqa: F401
        CheckpointCompatibilityError,
        action_probabilities,
        create_model,
        load_model_from_checkpoint,
        predict_action,
    )


def test_policies_package_exports() -> None:
    """The policies __init__ should export key symbols."""

    from maze_rl.policies import (  # noqa: F401
        AgentBackend,
        BackendCapabilities,
        available_backends,
        create_model,
        get_backend,
        load_model_from_checkpoint,
        predict_action,
        register_backend,
    )
