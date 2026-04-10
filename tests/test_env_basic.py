"""Basic environment contract tests."""

import numpy as np
from collections import deque

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


def test_env_reset_without_seed_draws_from_gym_rng() -> None:
    """An unseeded reset should use the environment RNG instead of episode-index fallback."""

    env = MazeEnv(MazeConfig(train_seed_base=10_000))
    env.reset(seed=123)

    expected_env = MazeEnv(MazeConfig(train_seed_base=10_000))
    expected_env.reset(seed=123)
    expected_seed = int(expected_env.np_random.integers(0, np.iinfo(np.int64).max))

    _, info = env.reset()

    assert info["maze_seed"] == expected_seed
    assert info["maze_seed"] != 10_001


def test_action_masks_block_corner_oscillation_when_better_move_exists() -> None:
    """Masking should drop short-loop backtracking when fresh space is available."""

    env = MazeEnv(
        MazeConfig(
            rows=4,
            cols=5,
            max_player_speed=1,
            monster_speed=0,
            monster_activation_delay=999,
            curriculum_enabled=False,
        ),
        training_mode=True,
    )
    layout = MazeLayout(
        grid=(
            "#####",
            "#..##",
            "#..##",
            "#####",
        ),
        player_start=Position(1, 1),
        monster_start=Position(2, 1),
        exit_position=Position(2, 2),
        seed=77,
    )
    env.reset(seed=77, options={"layout": layout, "maze_seed": 77})
    env.player = Position(1, 2)
    env.visited_counts = {Position(1, 1): 2, Position(1, 2): 2}
    env.path_history = deque(
        [Position(1, 1), Position(1, 2), Position(1, 1), Position(1, 2)],
        maxlen=6,
    )

    masks = env.action_masks()

    assert masks[3] is False
    assert masks[2] is True


def test_action_masks_allow_backtrack_when_it_is_only_exit_from_dead_end() -> None:
    """Masking should not trap the agent by removing the only legal escape move."""

    env = MazeEnv(
        MazeConfig(
            rows=3,
            cols=5,
            max_player_speed=1,
            monster_speed=0,
            monster_activation_delay=999,
            curriculum_enabled=False,
        ),
        training_mode=True,
    )
    layout = MazeLayout(
        grid=(
            "#####",
            "#...#",
            "#####",
        ),
        player_start=Position(1, 3),
        monster_start=Position(1, 1),
        exit_position=Position(1, 1),
        seed=88,
    )
    env.reset(seed=88, options={"layout": layout, "maze_seed": 88})
    env.player = Position(1, 3)
    env.visited_counts = {Position(1, 2): 1, Position(1, 3): 1}
    env.path_history = deque([Position(1, 2), Position(1, 3)], maxlen=6)

    masks = env.action_masks()

    assert masks[3] is True
    assert sum(bool(item) for item in masks) == 1


def test_action_masks_force_flee_move_when_monster_is_visible_in_open_space() -> None:
    """When the monster is visible and escape exists, masking should force the innate flee move."""

    env = MazeEnv(
        MazeConfig(
            rows=5,
            cols=9,
            vision_range=4,
            max_player_speed=1,
            monster_speed=0,
            monster_activation_delay=999,
            curriculum_enabled=False,
        ),
        training_mode=True,
    )
    layout = MazeLayout(
        grid=(
            "#########",
            "#.......#",
            "#.......#",
            "#.......#",
            "#########",
        ),
        player_start=Position(2, 6),
        monster_start=Position(2, 7),
        exit_position=Position(1, 1),
        seed=94,
    )
    env.reset(seed=94, options={"layout": layout, "maze_seed": 94})
    env.visited_counts = {
        Position(2, 5): 2,
        Position(2, 6): 3,
        Position(1, 6): 2,
        Position(3, 6): 2,
        Position(2, 7): 1,
        Position(1, 5): 2,
        Position(3, 5): 2,
    }
    env.path_history = deque([Position(2, 4), Position(2, 5), Position(2, 6)], maxlen=6)

    masks = env.action_masks()

    assert env.get_state_snapshot()["monster_visible"] is True
    assert masks == [False, False, False, True]


def test_action_masks_force_fastest_escape_speed_when_visible_monster_has_clear_corridor() -> None:
    """Visible-monster flee masking should preserve the human speed advantage in long corridors."""

    env = MazeEnv(
        MazeConfig(
            rows=3,
            cols=13,
            vision_range=5,
            max_player_speed=4,
            monster_speed=1,
            monster_activation_delay=0,
            curriculum_enabled=False,
        ),
        training_mode=True,
    )
    layout = MazeLayout(
        grid=(
            "#############",
            "#...........#",
            "#############",
        ),
        player_start=Position(1, 8),
        monster_start=Position(1, 10),
        exit_position=Position(1, 1),
        seed=112,
    )
    env.reset(seed=112, options={"layout": layout, "maze_seed": 112})

    masks = env.action_masks()

    expected_action = 3 * env.config.max_player_speed + (4 - 1)
    assert env.get_state_snapshot()["monster_visible"] is True
    assert sum(bool(item) for item in masks) == 1
    assert masks[expected_action] is True


def test_episode_reports_avoidable_capture_when_clear_escape_was_ignored() -> None:
    """Caught episodes should be flagged when the pre-step state had a clear escape."""

    env = MazeEnv(
        MazeConfig(
            rows=3,
            cols=7,
            vision_range=4,
            max_player_speed=1,
            monster_speed=0,
            monster_activation_delay=999,
            curriculum_enabled=False,
        ),
        training_mode=True,
    )
    layout = MazeLayout(
        grid=(
            "#######",
            "#.....#",
            "#######",
        ),
        player_start=Position(1, 3),
        monster_start=Position(1, 4),
        exit_position=Position(1, 1),
        seed=113,
    )
    env.reset(seed=113, options={"layout": layout, "maze_seed": 113})

    _, _, terminated, truncated, info = env.step(1)

    assert terminated is True
    assert truncated is False
    assert info["outcome"] == "caught"
    assert info["avoidable_capture"] is True
    assert info["avoidable_capture_reason"] == "ignored-clear-escape"
    assert info["episode_metrics"].avoidable_capture is True


def test_repeat_loop_indicator_penalizes_ping_pong_without_progress() -> None:
    """Back-and-forth corner loops should raise a repeat indicator but not end the game."""

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
    assert last_info["outcome"] == "running"


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
    env = MazeEnv(
        MazeConfig(max_player_speed=1, monster_speed=0, monster_activation_delay=999),
        training_mode=False,
    )
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
    env = MazeEnv(
        MazeConfig(max_player_speed=1, monster_speed=0, monster_activation_delay=999),
        training_mode=False,
    )
    env.reset(options={"layout": layout, "maze_seed": 655})

    snapshot = env.get_state_snapshot()

    assert snapshot["grid"][1][4] == "#"
    assert snapshot["grid"][1][5] == "?"
    assert snapshot["grid"][1][6] == "?"
    assert snapshot["player_visible"] is True
    assert snapshot["monster_visible"] is False


def test_last_seen_monster_position_persists_after_rounding_corner() -> None:
    """The player should remember where the monster was last seen after losing sight."""

    layout = MazeLayout(
        grid=(
            "#######",
            "#....##",
            "###.###",
            "##..###",
            "#######",
        ),
        player_start=Position(1, 3),
        monster_start=Position(1, 4),
        exit_position=Position(3, 3),
        seed=657,
    )
    env = MazeEnv(
        MazeConfig(max_player_speed=1, monster_speed=0, monster_activation_delay=999),
        training_mode=False,
    )
    env.reset(options={"layout": layout, "maze_seed": 657})

    before = env.get_state_snapshot()
    env.step(2)
    after = env.get_state_snapshot()

    assert before["monster_visible"] is True
    assert after["monster_visible"] is False
    assert after["last_seen_monster_position"] == (1, 4)
    assert after["turns_since_monster_seen"] == 1


def test_visibility_reaches_one_tile_farther_for_monster_sight() -> None:
    """The player should see a monster one tile farther down a straight corridor."""

    layout = MazeLayout(
        grid=(
            "#########",
            "#......##",
            "#########",
        ),
        player_start=Position(1, 1),
        monster_start=Position(1, 6),
        exit_position=Position(1, 2),
        seed=656,
    )
    env = MazeEnv(
        MazeConfig(
            vision_range=4,
            max_player_speed=1,
            monster_speed=0,
            monster_activation_delay=999,
        ),
        training_mode=False,
    )
    env.reset(options={"layout": layout, "maze_seed": 656})

    snapshot = env.get_state_snapshot()

    assert env.player_vision_range == 5
    assert snapshot["grid"][1][6] == "."
    assert snapshot["monster_visible"] is True


def test_visible_dead_end_detection_and_penalty_on_deterministic_layout() -> None:
    """A clearly visible non-exit dead end should be detected and penalized when avoidable."""

    layout = MazeLayout(
        grid=(
            "#######",
            "##.####",
            "#....##",
            "#######",
        ),
        player_start=Position(2, 2),
        monster_start=Position(2, 1),
        exit_position=Position(1, 2),
        seed=700,
    )
    env = MazeEnv(
        MazeConfig(
            vision_range=4,
            max_player_speed=1,
            monster_speed=0,
            monster_activation_delay=999,
            reward=MazeConfig().reward,
        ),
        training_mode=False,
    )
    env.reset(options={"layout": layout, "maze_seed": 700})

    summaries = env.get_visible_direction_summaries()

    assert summaries[1].enters_visible_dead_end is True
    assert summaries[1].visible_depth == 2
    assert summaries[0].exit_visible is True

    _, _reward, _, _, info = env.step(1)

    assert info["step_visible_dead_end_opportunity"] is True
    assert info["step_entered_visible_dead_end"] is True
    assert info["visible_dead_end_opportunities"] == 1
    assert info["entered_visible_dead_end"] == 1
    assert info["avoidable_visible_dead_end_penalties_applied"] == 1
    assert info["reward_breakdown"].avoidable_visible_dead_end < 0.0


def test_known_dead_end_path_marks_visible_leaf_corridor() -> None:
    """The environment should remember a visible dead-end corridor, not just its tip."""

    layout = MazeLayout(
        grid=(
            "#######",
            "##.####",
            "#....##",
            "#######",
        ),
        player_start=Position(2, 2),
        monster_start=Position(2, 1),
        exit_position=Position(1, 2),
        seed=704,
    )
    env = MazeEnv(
        MazeConfig(
            vision_range=4,
            max_player_speed=1,
            monster_speed=0,
            monster_activation_delay=999,
        ),
        training_mode=False,
    )
    env.reset(options={"layout": layout, "maze_seed": 704})

    snapshot = env.get_state_snapshot()

    assert {(2, 3), (2, 4)}.issubset(
        set(snapshot["known_dead_end_cells"])
    )
    assert (2, 2) not in set(snapshot["known_dead_end_cells"])


def test_wait_step_advances_turn_without_moving_player() -> None:
    """A wait turn should let the monster move while the player stays in place."""

    layout = MazeLayout(
        grid=(
            "#######",
            "#.....#",
            "#######",
        ),
        player_start=Position(1, 1),
        monster_start=Position(1, 5),
        exit_position=Position(1, 4),
        seed=705,
    )
    env = MazeEnv(
        MazeConfig(max_player_speed=1, monster_speed=1, monster_activation_delay=0),
        training_mode=False,
    )
    env.reset(options={"layout": layout, "maze_seed": 705})

    _, _, terminated, truncated, info = env.step_wait()

    assert terminated is False
    assert truncated is False
    assert info["state_snapshot"]["player_position"] == (1, 1)
    assert info["state_snapshot"]["monster_position"] == (1, 4)
    assert info["state_snapshot"]["last_action_kind"] == "wait"
    assert info["state_snapshot"]["last_action_speed"] == 0


def test_visible_dead_end_penalty_is_skipped_when_dead_end_contains_exit() -> None:
    """A visible straight corridor to the exit should not be treated as a penalized dead end."""

    layout = MazeLayout(
        grid=(
            "#######",
            "#######",
            "#....##",
            "#######",
        ),
        player_start=Position(2, 2),
        monster_start=Position(2, 1),
        exit_position=Position(2, 4),
        seed=701,
    )
    env = MazeEnv(
        MazeConfig(
            vision_range=4,
            max_player_speed=1,
            monster_speed=0,
            monster_activation_delay=999,
        ),
        training_mode=False,
    )
    env.reset(options={"layout": layout, "maze_seed": 701})

    summaries = env.get_visible_direction_summaries()

    assert summaries[1].exit_visible is True
    assert summaries[1].enters_visible_dead_end is False

    _, _, _, _, info = env.step(1)

    assert info["entered_visible_dead_end"] == 0
    assert info["avoidable_visible_dead_end_penalties_applied"] == 0
    assert info["reward_breakdown"].avoidable_visible_dead_end == 0.0


def test_visible_dead_end_penalty_is_skipped_when_it_is_only_legal_move() -> None:
    """The avoidable penalty should not fire when the visible dead end is unavoidable."""

    layout = MazeLayout(
        grid=(
            "#####",
            "#####",
            "##..#",
            "#####",
        ),
        player_start=Position(2, 2),
        monster_start=Position(2, 3),
        exit_position=Position(2, 2),
        seed=702,
    )
    env = MazeEnv(
        MazeConfig(
            vision_range=4,
            max_player_speed=1,
            monster_speed=0,
            monster_activation_delay=999,
        ),
        training_mode=False,
    )
    env.reset(options={"layout": layout, "maze_seed": 702})

    summaries = env.get_visible_direction_summaries()

    assert summaries[1].enters_visible_dead_end is True
    assert summaries[0].legal is False
    assert summaries[2].legal is False
    assert summaries[3].legal is False

    _, _, _, _, info = env.step(1)

    assert info["step_visible_dead_end_opportunity"] is False
    assert info["step_entered_visible_dead_end"] is True
    assert info["avoidable_visible_dead_end_penalties_applied"] == 0


def test_observation_includes_per_direction_visibility_features() -> None:
    """Observation tail should include compact per-direction look-ahead features."""

    layout = MazeLayout(
        grid=(
            "#######",
            "##.####",
            "#....##",
            "#######",
        ),
        player_start=Position(2, 2),
        monster_start=Position(2, 1),
        exit_position=Position(1, 2),
        seed=703,
    )
    env = MazeEnv(
        MazeConfig(
            vision_range=4,
            max_player_speed=1,
            monster_speed=0,
            monster_activation_delay=999,
        ),
        training_mode=False,
    )
    observation, _ = env.reset(options={"layout": layout, "maze_seed": 703})

    tail = observation[-env.observation_spec.scalar_features :]
    direction_features = tail[5:21]

    assert observation.shape[0] == env.observation_spec.vector_length
    assert np.isclose(direction_features[0], 1.0)
    assert np.isclose(direction_features[1], 0.2)
    assert np.isclose(direction_features[2], 0.0)
    assert np.isclose(direction_features[3], 1.0)
    assert np.isclose(direction_features[4], 1.0)
    assert np.isclose(direction_features[5], 0.4)
    assert np.isclose(direction_features[6], 1.0)
    assert np.isclose(direction_features[7], 0.0)
