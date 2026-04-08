"""Basic environment contract tests."""

from maze_rl.config import MazeConfig
from maze_rl.envs.entities import MazeLayout, Position
from maze_rl.envs.maze_env import MazeEnv


def test_env_reset_and_step_contract() -> None:
    """The environment should satisfy the Gym reset and step contract."""

    env = MazeEnv(MazeConfig())
    observation, info = env.reset(seed=123)
    assert observation.shape == env.observation_space.shape
    assert info["maze_seed"] == 123

    next_observation, reward, terminated, truncated, step_info = env.step(0)
    assert next_observation.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "coverage" in step_info


def test_repeat_loop_indicator_penalizes_ping_pong_without_progress() -> None:
    """Back-and-forth corner loops should raise a repeat indicator and truncate as a stall."""

    layout = MazeLayout(
        grid=(
            "#######",
            "#.....#",
            "#######",
        ),
        player_start=Position(1, 1),
        monster_start=Position(1, 5),
        exit_position=Position(1, 4),
        seed=321,
    )
    env = MazeEnv(
        MazeConfig(
            max_player_speed=1,
            monster_speed=0,
            monster_activation_delay=999,
            max_episode_steps=50,
            stall_threshold=25,
        ),
        training_mode=False,
    )
    env.reset(options={"layout": layout, "maze_seed": 321})

    last_info = None
    for action in (1, 3, 1, 3, 1, 3):
        _, _, terminated, truncated, info = env.step(action)
        last_info = info
        if terminated or truncated:
            break

    assert last_info is not None
    assert last_info["repeat_move_streak"] >= 5
    assert last_info["peak_repeat_move_streak"] >= 5
    assert last_info["repeat_loop_detected"] is True
    assert last_info["state_snapshot"]["repeat_loop_warning"] is True
    assert last_info["outcome"] == "stall"


def test_visibility_memory_reveals_only_seen_cells_and_blocks_through_walls() -> None:
    """The agent should remember seen cells, hide unknown cells, and not see through walls."""

    layout = MazeLayout(
        grid=(
            "#########",
            "#...#...#",
            "#.......#",
            "#...#...#",
            "#########",
        ),
        player_start=Position(2, 1),
        monster_start=Position(2, 5),
        exit_position=Position(1, 6),
        seed=654,
    )
    env = MazeEnv(MazeConfig(max_player_speed=1, monster_speed=0, monster_activation_delay=999), training_mode=False)
    env.reset(options={"layout": layout, "maze_seed": 654})

    snapshot = env.get_state_snapshot()

    assert snapshot["grid"][2][1] == "."
    assert snapshot["grid"][2][5] == "."
    assert snapshot["grid"][1][4] == "?"
    assert snapshot["grid"][1][6] == "?"
    assert snapshot["monster_visible"] is True
    assert snapshot["exit_seen"] is False


def test_visibility_hides_monster_when_wall_blocks_sightline() -> None:
    """Monster visibility should turn off when a wall blocks the four-cell sight line."""

    layout = MazeLayout(
        grid=(
            "#########",
            "#...#..##",
            "#########",
        ),
        player_start=Position(1, 1),
        monster_start=Position(1, 6),
        exit_position=Position(1, 2),
        seed=655,
    )
    env = MazeEnv(MazeConfig(max_player_speed=1, monster_speed=0, monster_activation_delay=999), training_mode=False)
    env.reset(options={"layout": layout, "maze_seed": 655})

    snapshot = env.get_state_snapshot()

    assert snapshot["grid"][1][4] == "#"
    assert snapshot["grid"][1][5] == "?"
    assert snapshot["grid"][1][6] == "?"
    assert snapshot["monster_visible"] is False
