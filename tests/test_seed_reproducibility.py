"""Seed reproducibility tests."""

from maze_rl.envs.maze_generator import generate_maze


def test_seeded_maze_generation_is_reproducible() -> None:
    """The same seed should generate the same maze."""

    first = generate_maze(seed=1001, rows=15, cols=15)
    second = generate_maze(seed=1001, rows=15, cols=15)
    different = generate_maze(seed=1002, rows=15, cols=15)

    assert first.grid == second.grid
    assert first.player_start == second.player_start
    assert first.exit_position == second.exit_position
    assert first.monster_start == second.monster_start
    assert first.grid != different.grid
