"""Configuration compatibility tests."""

from maze_rl.config import maze_config_from_dict, training_config_from_dict


def test_maze_config_from_dict_ignores_unknown_reward_fields() -> None:
    """Old checkpoint reward fields should not break config loading."""

    config = maze_config_from_dict(
        {
            "rows": 15,
            "cols": 15,
            "reward": {
                "survival_reward": 0.1,
                "trap_threat_penalty": -4.0,
                "escape_collapse_penalty": -2.0,
            },
            "unexpected_top_level": "ignored",
        }
    )

    assert config.rows == 15
    assert config.cols == 15
    assert config.reward.survival_reward == 0.1


def test_training_config_from_dict_ignores_unknown_fields() -> None:
    """Training config loading should tolerate extra serialized metadata."""

    config = training_config_from_dict(
        {
            "episodes": 123,
            "checkpoint_episodes": [0, 50],
            "obsolete_flag": True,
        }
    )

    assert config.episodes == 123
    assert config.checkpoint_episodes == (0, 50)