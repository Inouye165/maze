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


def test_env_focused_seed_runs_requested_seed_once_then_jumps_forward() -> None:
    """Focused training should guarantee the requested seed once, then vary future mazes."""

    config = MazeConfig(
        train_seed_base=10_000,
        fixed_maze_seed=222,
        focused_seed_jump_max=1000,
    )
    env = MazeEnv(config, training_mode=True)
    _, first_info = env.reset(seed=123)

    expected_env = MazeEnv(config, training_mode=True)
    _, expected_first_info = expected_env.reset(seed=123)
    _, expected_second_info = expected_env.reset()
    _, expected_third_info = expected_env.reset()

    _, second_info = env.reset()
    _, third_info = env.reset()

    assert first_info["maze_seed"] == 222
    assert first_info["maze_seed"] == expected_first_info["maze_seed"]
    assert second_info["maze_seed"] == expected_second_info["maze_seed"]
    assert third_info["maze_seed"] == expected_third_info["maze_seed"]
    assert 1 <= second_info["maze_seed"] - first_info["maze_seed"] <= 1000
    assert 1 <= third_info["maze_seed"] - second_info["maze_seed"] <= 1000


def test_action_masks_keep_legal_backtrack_when_better_move_exists() -> None:
    """Illegal-only masking must not remove legal backtracking just because a fresh branch exists."""

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

    assert masks[3] is True
    assert masks[2] is True
    assert masks[env.wait_action_index] is True


def test_action_masks_allow_backtrack_when_it_is_only_exit_from_dead_end() -> None:
    """Illegal-only masking should keep the only legal escape move and the wait action available."""

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
    assert masks[env.wait_action_index] is True
    assert sum(bool(item) for item in masks) == 2


def test_action_masks_keep_legal_moves_when_monster_is_visible_in_open_space() -> None:
    """Legal actions must stay available even when the monster is visible."""

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
    assert masks == [True, False, True, True, True]


def test_action_masks_keep_multiple_legal_speeds_when_visible_monster_has_clear_corridor() -> None:
    """Legal speed choices must remain available even in visible-monster corridors."""

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
    assert masks[expected_action] is True
    assert masks[3 * env.config.max_player_speed] is True
    assert masks[env.wait_action_index] is True
    assert sum(bool(item) for item in masks) > 2


def test_observation_keeps_full_human_known_map_as_primary_features() -> None:
    """The observation prefix should remain the full configured remembered map."""

    env = MazeEnv(
        MazeConfig(
            rows=7,
            cols=7,
            monster_speed=0,
            monster_activation_delay=999,
            curriculum_enabled=False,
        ),
        training_mode=False,
    )
    layout = MazeLayout(
        grid=(
            "#####",
            "#...#",
            "#...#",
            "#...#",
            "#####",
        ),
        player_start=Position(1, 1),
        monster_start=Position(3, 3),
        exit_position=Position(3, 1),
        seed=801,
    )

    observation, _ = env.reset(options={"layout": layout, "maze_seed": 801})

    player_index = (1 * env.observation_spec.cols + 1) * env.observation_spec.cell_channels
    unseen_outside_layout_index = (6 * env.observation_spec.cols + 6) * env.observation_spec.cell_channels

    assert env.observation_spec.global_map_length == 7 * 7 * env.observation_spec.cell_channels
    assert observation[player_index + 1] == 1.0
    assert observation[player_index + 3] == 1.0
    assert np.allclose(
        observation[unseen_outside_layout_index : unseen_outside_layout_index + env.observation_spec.cell_channels],
        np.zeros(env.observation_spec.cell_channels, dtype=np.float32),
    )


def test_local_tactical_view_appends_without_removing_global_or_scalar_features() -> None:
    """Optional local tactical features should be inserted between the global map and scalar tail."""

    layout = MazeLayout(
        grid=(
            "#######",
            "#.....#",
            "#.....#",
            "#.....#",
            "#######",
        ),
        player_start=Position(2, 2),
        monster_start=Position(2, 4),
        exit_position=Position(1, 5),
        seed=802,
    )
    base_env = MazeEnv(
        MazeConfig(
            rows=7,
            cols=7,
            monster_speed=0,
            monster_activation_delay=999,
            curriculum_enabled=False,
        ),
        training_mode=False,
    )
    tactical_env = MazeEnv(
        MazeConfig(
            rows=7,
            cols=7,
            enable_local_tactical_view=True,
            local_tactical_radius=1,
            monster_speed=0,
            monster_activation_delay=999,
            curriculum_enabled=False,
        ),
        training_mode=False,
    )

    base_observation, _ = base_env.reset(options={"layout": layout, "maze_seed": 802})
    tactical_observation, _ = tactical_env.reset(options={"layout": layout, "maze_seed": 802})

    assert tactical_env.observation_spec.local_tactical_vector_length == 3 * 3 * tactical_env.observation_spec.cell_channels
    assert np.allclose(
        tactical_observation[: base_env.observation_spec.global_map_length],
        base_observation[: base_env.observation_spec.global_map_length],
    )
    assert np.allclose(
        tactical_observation[-base_env.observation_spec.scalar_features :],
        base_observation[-base_env.observation_spec.scalar_features :],
    )


def test_local_tactical_view_can_encode_last_seen_monster_memory() -> None:
    """The optional local tactical patch should be able to append last-seen monster memory."""

    env = MazeEnv(
        MazeConfig(
            rows=5,
            cols=5,
            enable_local_tactical_view=True,
            local_tactical_radius=1,
            local_tactical_include_monster_memory=True,
            monster_speed=0,
            monster_activation_delay=999,
            curriculum_enabled=False,
        ),
        training_mode=False,
    )
    layout = MazeLayout(
        grid=(
            "#####",
            "#...#",
            "#...#",
            "#...#",
            "#####",
        ),
        player_start=Position(2, 2),
        monster_start=Position(2, 3),
        exit_position=Position(1, 1),
        seed=803,
    )

    env.reset(options={"layout": layout, "maze_seed": 803})
    env.last_seen_monster_position = Position(2, 3)
    observation = env._get_observation()

    patch_start = env.observation_spec.global_map_length
    patch_channels = env.observation_spec.local_tactical_cell_channels
    remembered_monster_cell = patch_start + (1 * 3 + 2) * patch_channels + (patch_channels - 1)

    assert observation.shape[0] == env.observation_spec.vector_length
    assert observation[remembered_monster_cell] == 1.0


def test_episode_reports_avoidable_capture_when_clear_escape_was_ignored() -> None:
    """Stepping directly into the monster should be rejected as an illegal move."""

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

    assert terminated is False
    assert truncated is False
    assert info["outcome"] == "running"
    assert info["blocked_moves"] == 1
    assert info["state_snapshot"]["player_position"] == (1, 3)


def test_no_progress_allows_turns_that_shorten_known_exit_path() -> None:
    """Moving closer to a remembered exit should not count as a stall step."""

    env = MazeEnv(
        MazeConfig(
            rows=3,
            cols=7,
            vision_range=0,
            max_player_speed=1,
            monster_speed=0,
            monster_activation_delay=999,
            stall_threshold=1,
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
        player_start=Position(1, 2),
        monster_start=Position(1, 1),
        exit_position=Position(1, 5),
        seed=120,
    )
    env.reset(seed=120, options={"layout": layout, "maze_seed": 120})
    env.player = Position(1, 2)
    env.monster = Position(1, 1)
    env.seen_open_cells = {
        Position(1, 1),
        Position(1, 2),
        Position(1, 3),
        Position(1, 4),
        Position(1, 5),
    }
    env.visible_open_cells = {Position(1, 1), Position(1, 2), Position(1, 3)}
    env.seen_wall_cells = set()
    env.no_progress_steps = 0

    _, _, terminated, truncated, info = env.step(1)

    assert terminated is False
    assert truncated is False
    assert info["outcome"] == "running"
    assert info["state_snapshot"]["known_exit_path_distance"] == 2
    assert env.no_progress_steps == 0


def test_no_progress_allows_turns_that_shorten_known_frontier_path() -> None:
    """Moving toward a remembered frontier anchor should not count as no progress."""

    env = MazeEnv(
        MazeConfig(
            rows=3,
            cols=7,
            vision_range=0,
            max_player_speed=1,
            monster_speed=0,
            monster_activation_delay=999,
            stall_threshold=1,
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
        player_start=Position(1, 1),
        monster_start=Position(1, 5),
        exit_position=Position(1, 5),
        seed=121,
    )
    env.reset(seed=121, options={"layout": layout, "maze_seed": 121})
    env.player = Position(1, 1)
    env.monster = Position(1, 5)
    env.seen_open_cells = {
        Position(1, 1),
        Position(1, 2),
        Position(1, 3),
    }
    env.visible_open_cells = {Position(1, 1), Position(1, 2)}
    env.seen_wall_cells = set()
    env.no_progress_steps = 0

    _, _, terminated, truncated, info = env.step(1)

    assert terminated is False
    assert truncated is False
    assert info["outcome"] == "running"
    assert info["state_snapshot"]["known_frontier_path_distance"] == 1
    assert info["state_snapshot"]["frontier_anchor_count"] >= 1
    assert env.no_progress_steps == 0


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


def test_repeat_loop_eventually_stalls_episode() -> None:
    """Repeated ping-pong movement should hard-stop the episode instead of training forever."""

    layout = MazeLayout(
        grid=(
            "#######",
            "#.....#",
            "#######",
        ),
        player_start=Position(1, 1),
        monster_start=Position(1, 5),
        exit_position=Position(1, 4),
        seed=654,
    )
    env = MazeEnv(
        MazeConfig(
            max_player_speed=1,
            monster_speed=0,
            monster_activation_delay=999,
            max_episode_steps=100,
            stall_threshold=50,
        ),
        training_mode=False,
    )
    env.reset(options={"layout": layout, "maze_seed": 654})

    final_info = None
    terminated = False
    truncated = False
    for action in (1, 3) * 8:
        _, _, terminated, truncated, info = env.step(action)
        final_info = info
        if terminated or truncated:
            break

    assert final_info is not None
    assert terminated is False
    assert truncated is True
    assert final_info["outcome"] == "stall"
    assert final_info["repeat_loop_detected"] is True
    assert final_info["repeat_loop_stalled"] is True
    assert final_info["episode_metrics"].stalled is True


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


def test_player_is_caught_when_monster_reaches_same_junction_cell() -> None:
    """Once the monster reaches the player's cell, the turn ends in capture."""

    layout = MazeLayout(
        grid=(
            "#######",
            "#######",
            "#######",
            "##...##",
            "###.###",
            "###.###",
            "#######",
        ),
        player_start=Position(4, 3),
        monster_start=Position(3, 4),
        exit_position=Position(5, 3),
        seed=658,
    )
    env = MazeEnv(
        MazeConfig(max_player_speed=1, monster_speed=1, monster_activation_delay=0),
        training_mode=False,
    )
    env.reset(options={"layout": layout, "maze_seed": 658})

    _observation, _reward, terminated, truncated, info = env.step(0)
    masks = env.action_masks()

    assert terminated is True
    assert truncated is False
    assert info["outcome"] == "caught"
    assert info["state_snapshot"]["player_position"] == (3, 3)
    assert info["state_snapshot"]["monster_position"] == (3, 3)
    assert info["state_snapshot"]["monster_visible"] is True
    assert masks[3] is True
    assert any(masks[index] for index in (1, 2))


def test_head_on_corridor_collision_is_immediate_capture() -> None:
    """In a corridor, contact with the monster ends the turn immediately."""

    layout = MazeLayout(
        grid=(
            "#######",
            "#....##",
            "#######",
        ),
        player_start=Position(1, 2),
        monster_start=Position(1, 4),
        exit_position=Position(1, 1),
        seed=659,
    )
    env = MazeEnv(
        MazeConfig(max_player_speed=1, monster_speed=1, monster_activation_delay=0),
        training_mode=False,
    )
    env.reset(options={"layout": layout, "maze_seed": 659})

    _observation, _reward, terminated, truncated, info = env.step(1)

    assert terminated is True
    assert truncated is False
    assert info["outcome"] == "caught"
    assert env.player == Position(1, 3)
    assert env.monster == Position(1, 3)


def test_action_masks_block_moving_into_visible_monster_cell() -> None:
    """The human should never be allowed to step directly into the monster's cell."""

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
        player_start=Position(1, 2),
        monster_start=Position(1, 3),
        exit_position=Position(1, 5),
        seed=660,
    )
    env.reset(options={"layout": layout, "maze_seed": 660})

    masks = env.action_masks()

    assert env.get_state_snapshot()["monster_visible"] is True
    assert masks[1] is False


def test_trapped_in_dead_end_still_caught() -> None:
    """Player is caught when monster blocks the only exit from a dead-end spur."""

    layout = MazeLayout(
        grid=(
            "#####",
            "#.###",
            "#.###",
            "#.###",
            "#####",
        ),
        player_start=Position(3, 1),
        monster_start=Position(1, 1),
        exit_position=Position(1, 1),
        seed=700,
    )
    env = MazeEnv(
        MazeConfig(max_player_speed=1, monster_speed=2, monster_activation_delay=0),
        training_mode=False,
    )
    env.reset(options={"layout": layout, "maze_seed": 700})

    # Player moves north, monster is faster and catches up
    _obs, _rew, terminated, truncated, _info = env.step(0)  # north
    if not terminated:
        _obs, _rew, terminated, truncated, _info = env.step(0)  # north again
    # Eventually the player runs out of room in the corridor
    for _ in range(10):
        if terminated or truncated:
            break
        _obs, _rew, terminated, truncated, _info = env.step(0)
    assert terminated is True


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


def test_known_dead_end_path_marks_branch_route_when_all_forward_branches_die() -> None:
    """A remembered corridor into a branching dead region should be treated as a dead route."""

    layout = MazeLayout(
        grid=(
            "#########",
            "#...#####",
            "###.#####",
            "##...####",
            "#########",
        ),
        player_start=Position(1, 3),
        monster_start=Position(1, 2),
        exit_position=Position(1, 1),
        seed=706,
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
    env.reset(options={"layout": layout, "maze_seed": 706})
    env.seen_open_cells = {
        Position(1, 1),
        Position(1, 2),
        Position(1, 3),
        Position(2, 3),
        Position(3, 2),
        Position(3, 3),
        Position(3, 4),
    }
    env.discovered_cells = len(env.seen_open_cells)
    env._refresh_known_dead_end_paths()

    snapshot = env.get_state_snapshot()
    dead_end_cells = set(snapshot["known_dead_end_cells"])
    summaries = env.get_visible_direction_summaries()

    assert {(2, 3), (3, 2), (3, 3), (3, 4)}.issubset(dead_end_cells)
    assert summaries[2].enters_visible_dead_end is True
    assert summaries[3].exit_visible is True


def test_known_dead_end_branch_region_stays_marked_after_player_walks_away() -> None:
    """Mapped branchy dead regions should stay known even when no longer in view."""

    layout = MazeLayout(
        grid=(
            "###########",
            "#...#######",
            "###.#######",
            "#........##",
            "###.#######",
            "##...######",
            "###########",
        ),
        player_start=Position(3, 8),
        monster_start=Position(3, 1),
        exit_position=Position(3, 7),
        seed=711,
    )
    env = MazeEnv(
        MazeConfig(
            vision_range=2,
            max_player_speed=1,
            monster_speed=0,
            monster_activation_delay=999,
        ),
        training_mode=False,
    )
    env.reset(options={"layout": layout, "maze_seed": 711})
    env.seen_open_cells = {
        Position(1, 1),
        Position(1, 2),
        Position(1, 3),
        Position(2, 3),
        Position(3, 1),
        Position(3, 2),
        Position(3, 3),
        Position(3, 4),
        Position(3, 5),
        Position(3, 6),
        Position(3, 7),
        Position(3, 8),
        Position(4, 3),
        Position(5, 2),
        Position(5, 3),
        Position(5, 4),
    }
    env.discovered_cells = len(env.seen_open_cells)
    env._refresh_known_dead_end_paths()

    dead_end_cells = set(env.get_state_snapshot()["known_dead_end_cells"])

    assert {(3, 3), (2, 3), (1, 3), (1, 2), (1, 1), (4, 3), (5, 2), (5, 3), (5, 4)}.issubset(
        dead_end_cells
    )


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


def test_entering_known_dead_end_under_threat_applies_immediate_penalty() -> None:
    """Threatened dead-end commits should be punished immediately for learning."""

    env = MazeEnv(
        MazeConfig(
            rows=4,
            cols=7,
            max_player_speed=1,
            monster_speed=1,
            monster_activation_delay=0,
            curriculum_enabled=False,
        ),
        training_mode=False,
    )
    layout = MazeLayout(
        grid=(
            "#######",
            "#.....#",
            "###.###",
            "#######",
        ),
        player_start=Position(1, 3),
        monster_start=Position(2, 3),
        exit_position=Position(1, 1),
        seed=130,
    )
    env.reset(seed=130, options={"layout": layout, "maze_seed": 130})
    env.player = Position(1, 3)
    env.visited_counts = {
        Position(1, 3): 1,
        Position(1, 4): 1,
        Position(1, 5): 1,
    }
    env.seen_open_cells = {
        Position(1, 1),
        Position(1, 2),
        Position(1, 3),
        Position(1, 4),
        Position(1, 5),
        Position(2, 3),
    }
    env.known_dead_end_cells = {Position(1, 4), Position(1, 5)}
    env.visible_open_cells = {
        Position(1, 1),
        Position(1, 2),
        Position(1, 3),
        Position(1, 4),
        Position(1, 5),
        Position(2, 3),
    }

    _observation, _reward, terminated, truncated, info = env.step(1)

    assert terminated is False
    assert truncated is False
    assert info["step_entered_visible_dead_end"] is True
    assert info["step_trap_threat_entry"] is True
    assert info["trap_threat_penalties_applied"] == 1
    assert info["reward_breakdown"].trap_threat < 0.0


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


def test_known_dead_end_commit_applies_stronger_deeper_penalty() -> None:
    """Moving deeper into a remembered dead route should incur the extra penalty."""

    env = MazeEnv(
        MazeConfig(
            rows=4,
            cols=7,
            vision_range=4,
            max_player_speed=1,
            monster_speed=0,
            monster_activation_delay=999,
            curriculum_enabled=False,
        ),
        training_mode=False,
    )
    layout = MazeLayout(
        grid=(
            "#######",
            "#.....#",
            "#######",
        ),
        player_start=Position(1, 2),
        monster_start=Position(1, 1),
        exit_position=Position(1, 1),
        seed=160,
    )
    env.reset(seed=160, options={"layout": layout, "maze_seed": 160})
    env.player = Position(1, 2)
    env.visited_counts = {Position(1, 1): 1, Position(1, 2): 1, Position(1, 3): 0}
    env.seen_open_cells = {
        Position(1, 1),
        Position(1, 2),
        Position(1, 3),
        Position(1, 4),
        Position(1, 5),
    }
    env.visible_open_cells = set(env.seen_open_cells)
    env.known_dead_end_cells = {Position(1, 3), Position(1, 4), Position(1, 5)}

    _observation, _reward, terminated, truncated, info = env.step(1)

    assert terminated is False
    assert truncated is False
    assert info["dead_end_entries"] == 1
    assert info["deeper_dead_end_entries"] == 1
    assert info["step_deeper_dead_end_entry"] is True
    assert info["reward_breakdown"].dead_end < 0.0
    assert info["reward_breakdown"].deeper_dead_end < 0.0


def test_backtracking_toward_opening_is_not_penalized_as_dead_end_commit() -> None:
    """Moving out of a remembered dead route should not incur dead-end penalties."""

    env = MazeEnv(
        MazeConfig(
            rows=4,
            cols=7,
            vision_range=4,
            max_player_speed=1,
            monster_speed=0,
            monster_activation_delay=999,
            curriculum_enabled=False,
        ),
        training_mode=False,
    )
    layout = MazeLayout(
        grid=(
            "#######",
            "#.....#",
            "#######",
        ),
        player_start=Position(1, 4),
        monster_start=Position(1, 1),
        exit_position=Position(1, 1),
        seed=161,
    )
    env.reset(seed=161, options={"layout": layout, "maze_seed": 161})
    env.player = Position(1, 4)
    env.visited_counts = {
        Position(1, 1): 1,
        Position(1, 2): 1,
        Position(1, 3): 1,
        Position(1, 4): 1,
    }
    env.seen_open_cells = {
        Position(1, 1),
        Position(1, 2),
        Position(1, 3),
        Position(1, 4),
        Position(1, 5),
    }
    env.visible_open_cells = set(env.seen_open_cells)
    env.known_dead_end_cells = {Position(1, 3), Position(1, 4), Position(1, 5)}

    _observation, _reward, terminated, truncated, info = env.step(3)

    assert terminated is False
    assert truncated is False
    assert info["dead_end_entries"] == 0
    assert info["deeper_dead_end_entries"] == 0
    assert info["step_deeper_dead_end_entry"] is False
    assert info["reward_breakdown"].dead_end == 0.0
    assert info["reward_breakdown"].deeper_dead_end == 0.0


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
