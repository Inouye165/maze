"""Training progress formatting tests."""

from maze_rl.training.train import format_training_progress


def test_training_progress_includes_cycle_seed_when_present() -> None:
    """Progress lines should surface the current maze seed for the active cycle."""

    line = format_training_progress(
        {
            "completed_episodes": 2,
            "target_episodes": 5,
            "active_cycle": 3,
            "maze_seed": 987654321,
            "episode_steps": 14,
            "timesteps": 200,
            "coverage": 0.35,
            "no_progress_steps": 0,
            "peak_no_progress_steps": 4,
            "recent_win_rate": 0.5,
            "recent_stall_rate": 0.1,
            "recent_timeout_rate": 0.2,
            "status": "running",
        }
    )

    assert "cycle 3" in line
    assert "seed=987654321" in line