"""Training progress formatting tests."""

from maze_rl.training.metrics import EpisodeMetrics, RollingTrainingSummary
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


def test_rolling_training_summary_tracks_10_100_and_1000_outcome_windows() -> None:
    """Training summaries should expose the larger outcome windows used by the app."""

    summary = RollingTrainingSummary()
    for index in range(120):
        escaped = index < 90
        summary.add(
            EpisodeMetrics(
                outcome="escaped" if escaped else "caught",
                steps=20,
                coverage=0.5,
                revisits=1,
                oscillations=0,
                visible_dead_end_opportunities=0,
                entered_visible_dead_end=0,
                avoided_visible_dead_end=0,
                avoidable_visible_dead_end_penalties_applied=0,
                dead_end_entries=0,
                blocked_moves=0,
                discovered_cells=10,
                reward=1.0,
                maze_seed=index + 1,
                start_monster_distance=4,
                final_monster_distance=2,
                final_exit_distance=1,
                final_player_position=(1, 1),
                final_monster_position=(1, 2),
                final_player_monster_distance=1,
                capture_rule=None,
                time_to_capture=None,
                frontier_cells_visited=1,
                reached_new_frontier=escaped,
                peak_no_progress_steps=0,
                avoidable_capture=False,
                avoidable_capture_reason=None,
                curriculum_stage="full",
                monster_speed=1,
                monster_activation_delay=0,
                escaped=escaped,
                timed_out=False,
                stalled=False,
            )
        )

    snapshot = summary.snapshot()

    assert len(snapshot["recent_10_outcomes"]) == 10
    assert len(snapshot["recent_100_outcomes"]) == 100
    assert len(snapshot["recent_1000_outcomes"]) == 120
    assert snapshot["recent_10_win_rate"] == 0.0
    assert snapshot["recent_100_win_rate"] == 0.7
    assert snapshot["recent_1000_win_rate"] == 0.75
