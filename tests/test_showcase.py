"""Showcase workflow tests."""

from collections import deque
from pathlib import Path

import numpy as np

from maze_rl.config import MazeConfig, TrainingConfig
from maze_rl.envs.entities import MazeLayout, Position
from maze_rl.envs.maze_env import MazeEnv
from maze_rl.policies.model_factory import CheckpointCompatibilityError, create_model
from maze_rl.training.checkpointing import CheckpointManager
from maze_rl.training.showcase import (
    WAIT_ACTION,
    choose_heuristic_action,
    describe_move_choice,
    format_showcase_table,
    rank_legal_moves,
    run_showcase_headless,
    save_showcase_summary,
    should_override_policy,
)
import maze_rl.training.showcase as showcase_module


def _build_loop_branch_env() -> MazeEnv:
    """Build a tiny layout where backtracking would create a corner loop."""

    env = MazeEnv(
        MazeConfig(
            rows=4,
            cols=5,
            max_player_speed=1,
            monster_speed=0,
            monster_activation_delay=999,
            curriculum_enabled=False,
        ),
        training_mode=False,
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
    return env


def test_showcase_skips_missing_checkpoints_and_saves_json(tmp_path: Path) -> None:
    """Headless showcase should skip missing checkpoints and still write a summary."""

    checkpoint_dir = tmp_path / "checkpoints"
    training_config = TrainingConfig(checkpoint_dir=checkpoint_dir)
    maze_config = MazeConfig()
    env = MazeEnv(maze_config)
    model = create_model(training_config=training_config, env=env)
    manager = CheckpointManager(training_config=training_config, maze_config=maze_config)
    manager.save(
        model=model,
        episode=0,
        timesteps=0,
        training_summary={"episodes_seen": 0},
        evaluation_summary={"episodes": 0},
    )

    results = run_showcase_headless(
        checkpoint_dir=checkpoint_dir,
        checkpoints=[0, 200],
        seed=12345,
        max_no_progress_streak=10,
        wall_time_timeout_s=5.0,
    )
    summary_path = save_showcase_summary(results, seed=12345, output_path=tmp_path / "summary.json")
    lines = format_showcase_table(results)

    assert len(results) == 2
    assert results[0].checkpoint == "ckpt 0000"
    assert results[1].status == "missing"
    assert summary_path.exists()
    assert lines[0].startswith("checkpoint | status")
    assert any("ckpt 0200" in line for line in lines)


def test_heuristic_action_breaks_corner_oscillation_when_unvisited_move_exists() -> None:
    """The fallback heuristic should leave a local loop when fresh space is available."""

    env = _build_loop_branch_env()

    action = choose_heuristic_action(env)
    choice = describe_move_choice(env, action)

    assert choice is not None
    assert choice.target == Position(2, 2)
    assert choice.visits == 0
    assert choice.immediate_reverse is False


def test_policy_fallback_prefers_unvisited_tile_over_immediate_backtracking() -> None:
    """A weak learned action should be overridden when it backtracks into a short loop."""

    env = _build_loop_branch_env()
    reverse_action = 3
    chosen = describe_move_choice(env, reverse_action)
    best = rank_legal_moves(env)[0]

    assert chosen is not None
    assert chosen.immediate_reverse is True
    assert best.target == Position(2, 2)
    assert should_override_policy(chosen, best, chosen_confidence=0.22, confidence_gap=0.01) is True


def test_heuristic_runs_from_visible_monster_when_exit_path_is_not_safe() -> None:
    """The heuristic should flee when the monster is close and blocks safe progress."""

    env = MazeEnv(
        MazeConfig(
            rows=5,
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
            "#.....#",
            "#######",
        ),
        player_start=Position(1, 3),
        monster_start=Position(1, 4),
        exit_position=Position(1, 5),
        seed=91,
    )
    env.reset(seed=91, options={"layout": layout, "maze_seed": 91})
    env.visited_counts = {
        Position(1, 2): 3,
        Position(1, 3): 2,
        Position(1, 4): 1,
    }

    best = rank_legal_moves(env)[0]

    assert best.target == Position(2, 3)
    assert best.monster_distance_gain > 0


def test_heuristic_commits_to_seen_exit_when_monster_loses_exit_race() -> None:
    """Once the human can see a safe exit route, the heuristic should commit to it."""

    env = MazeEnv(
        MazeConfig(
            rows=5,
            cols=7,
            vision_range=4,
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
            "#.....#",
            "#######",
        ),
        player_start=Position(1, 3),
        monster_start=Position(1, 1),
        exit_position=Position(1, 5),
        seed=141,
    )
    env.reset(seed=141, options={"layout": layout, "maze_seed": 141})
    env.visited_counts = {
        Position(1, 3): 2,
        Position(1, 4): 1,
        Position(2, 3): 0,
    }

    best = rank_legal_moves(env)[0]
    chosen = describe_move_choice(env, 2)

    assert best.target == Position(1, 4)
    assert best.commit_to_exit is True
    assert chosen is not None
    assert should_override_policy(chosen, best, chosen_confidence=0.30, confidence_gap=0.02) is True


def test_heuristic_runs_away_when_monster_enters_extended_corridor_visibility() -> None:
    """The heuristic should flee when the monster appears at the edge of corridor vision."""

    env = MazeEnv(
        MazeConfig(
            rows=3,
            cols=9,
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
            "#########",
            "#.......#",
            "#########",
        ),
        player_start=Position(1, 2),
        monster_start=Position(1, 7),
        exit_position=Position(1, 6),
        seed=92,
    )
    env.reset(seed=92, options={"layout": layout, "maze_seed": 92})

    snapshot = env.get_state_snapshot()
    best = rank_legal_moves(env)[0]

    assert snapshot["monster_visible"] is True
    assert env.player_vision_range == 5
    assert best.target == Position(1, 1)
    assert best.monster_distance_gain > 0


def test_heuristic_avoids_reentering_mapped_branchy_dead_region() -> None:
    """The heuristic should reject an explored dead region even after walking away from it."""

    env = MazeEnv(
        MazeConfig(
            rows=7,
            cols=11,
            vision_range=2,
            max_player_speed=1,
            monster_speed=0,
            monster_activation_delay=999,
            curriculum_enabled=False,
        ),
        training_mode=False,
    )
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
        player_start=Position(3, 4),
        monster_start=Position(3, 1),
        exit_position=Position(3, 7),
        seed=97,
    )
    env.reset(seed=97, options={"layout": layout, "maze_seed": 97})
    env.player = Position(3, 4)
    env.visited_counts = {env.player: 1}
    env.path_history = deque([env.player], maxlen=6)
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
    env._observe_from_player()
    env._refresh_known_dead_end_paths()

    ranked = rank_legal_moves(env)
    best = ranked[0]
    west = next(choice for choice in ranked if choice.target == Position(3, 3))
    east = next(choice for choice in ranked if choice.target == Position(3, 5))

    assert Position(3, 3) in env.known_dead_end_cells
    assert best.target == Position(3, 5)
    assert west.known_dead_end is True
    assert east.known_dead_end is False


def test_heuristic_exits_active_dead_end_section_before_exploring_inside_it() -> None:
    """Once trapped inside a marked section, the heuristic should head for the exit path first."""

    env = MazeEnv(
        MazeConfig(
            rows=7,
            cols=9,
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
            "#########",
            "#########",
            "##....###",
            "####.####",
            "####.####",
            "####...##",
            "#########",
        ),
        player_start=Position(2, 4),
        monster_start=Position(5, 6),
        exit_position=Position(5, 6),
        seed=144,
    )
    env.reset(seed=144, options={"layout": layout, "maze_seed": 144})
    env.player = Position(2, 4)
    env.seen_open_cells = {
        Position(2, 2),
        Position(2, 4),
        Position(2, 3),
        Position(2, 5),
        Position(2, 4),
        Position(2, 3),
        Position(3, 4),
        Position(4, 4),
        Position(5, 4),
        Position(5, 5),
        Position(5, 6),
    }
    env.visible_open_cells = {
        Position(2, 2),
        Position(2, 3),
        Position(2, 4),
        Position(2, 5),
        Position(3, 4),
        Position(4, 4),
        Position(5, 4),
        Position(5, 5),
    }
    env.discovered_cells = len(env.seen_open_cells)
    env.path_history = deque([Position(3, 4), Position(2, 4)], maxlen=6)
    env.visited_counts = {
        Position(2, 4): 2,
        Position(3, 4): 2,
        Position(4, 4): 1,
        Position(5, 4): 1,
    }
    env.known_dead_end_cells = {
        Position(2, 2),
        Position(2, 4),
        Position(2, 3),
        Position(2, 5),
        Position(3, 4),
        Position(4, 4),
        Position(5, 4),
        Position(5, 5),
    }

    ranked = rank_legal_moves(env)
    best = ranked[0]
    action = choose_heuristic_action(env)
    choice = describe_move_choice(env, action)

    assert best.target == Position(3, 4)
    assert action == 2
    assert choice is not None
    assert choice.escaping_dead_end is True
    assert choice.known_dead_end is False


def test_heuristic_does_not_use_hidden_monster_position_without_memory() -> None:
    """The heuristic should not flee from a monster hidden behind a wall it never saw."""

    env = MazeEnv(
        MazeConfig(
            rows=5,
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
            "#..#..#",
            "#..#..#",
            "#.....#",
            "#######",
        ),
        player_start=Position(1, 1),
        monster_start=Position(1, 4),
        exit_position=Position(3, 5),
        seed=96,
    )
    env.reset(seed=96, options={"layout": layout, "maze_seed": 96})

    ranked = rank_legal_moves(env)

    assert env.get_state_snapshot()["monster_visible"] is False
    assert env.get_state_snapshot()["last_seen_monster_position"] is None
    assert ranked[0].monster_distance == 9999


def test_heuristic_prefers_western_escape_over_corner_oscillation_in_open_space() -> None:
    """Fear mode should choose fresh western space over looping in a visible-monster corner."""

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
        training_mode=False,
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
        seed=93,
    )
    env.reset(seed=93, options={"layout": layout, "maze_seed": 93})
    env.visited_counts = {
        Position(2, 6): 3,
        Position(1, 6): 2,
        Position(3, 6): 2,
        Position(2, 7): 1,
    }
    env.path_history = deque(
        [Position(1, 6), Position(2, 6), Position(1, 6), Position(2, 6)],
        maxlen=6,
    )

    ranked = rank_legal_moves(env)
    best = ranked[0]
    north = describe_move_choice(env, 0)

    assert env.get_state_snapshot()["monster_visible"] is True
    assert north is not None
    assert north.short_loop_risk > 0
    assert best.target == Position(2, 5)
    assert best.monster_distance_gain > 0
    assert best.nearest_unvisited_distance == 0


def test_heuristic_allows_immediate_reverse_when_it_is_the_best_escape() -> None:
    """Fear mode should allow reversing into safer western space.

    This should beat lingering near danger.
    """

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
        training_mode=False,
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
    env.path_history = deque(
        [Position(2, 4), Position(2, 5), Position(2, 6)],
        maxlen=6,
    )

    best = rank_legal_moves(env)[0]
    west = describe_move_choice(env, 3)

    assert west is not None
    assert west.immediate_reverse is True
    assert west.monster_distance_gain > 0
    assert env.get_state_snapshot()["monster_visible"] is True
    assert best.target == Position(2, 5)


def test_heuristic_prefers_fastest_visible_escape_when_human_is_faster() -> None:
    """When the monster is visible in a long corridor, the innate heuristic should use speed advantage."""

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
        training_mode=False,
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
        seed=111,
    )
    env.reset(seed=111, options={"layout": layout, "maze_seed": 111})

    best = rank_legal_moves(env)[0]
    action = choose_heuristic_action(env)
    direction, speed = env.decode_action(action)

    assert env.get_state_snapshot()["monster_visible"] is True
    assert best.action == action
    assert direction == 3
    assert speed == 4


def test_heuristic_does_not_wait_when_monster_would_close_distance() -> None:
    """The fallback heuristic should not stand still just to let the monster approach."""

    env = MazeEnv(
        MazeConfig(
            rows=5,
            cols=7,
            vision_range=4,
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
            "#.#...#",
            "###.###",
            "#...#.#",
            "#######",
        ),
        player_start=Position(2, 3),
        monster_start=Position(1, 1),
        exit_position=Position(3, 1),
        seed=95,
    )
    env.reset(seed=95, options={"layout": layout, "maze_seed": 95})
    env.visited_counts = {
        Position(1, 3): 2,
        Position(2, 3): 1,
        Position(3, 3): 2,
    }
    env.known_dead_end_cells = {Position(1, 3), Position(3, 3)}
    env.path_history = deque([Position(1, 3), Position(2, 3)], maxlen=6)

    action = choose_heuristic_action(env)
    choice = describe_move_choice(env, action)

    assert action != WAIT_ACTION
    assert choice is not None
    assert choice.wait_action is False


def test_heuristic_can_return_explicit_wait_action_for_env() -> None:
    """The heuristic should emit the environment wait action id when waiting is best."""

    env = MazeEnv(
        MazeConfig(
            rows=5,
            cols=7,
            vision_range=4,
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
            "#.....#",
            "#######",
        ),
        player_start=Position(2, 3),
        monster_start=Position(1, 1),
        exit_position=Position(3, 5),
        seed=132,
    )
    env.reset(seed=132, options={"layout": layout, "maze_seed": 132})
    env.last_seen_monster_position = Position(1, 1)
    env.turns_since_monster_seen = 1
    env.known_dead_end_cells = {Position(1, 3), Position(3, 3)}
    env.visited_counts = {Position(2, 3): 1, Position(1, 3): 2, Position(3, 3): 2}
    env.path_history = deque([Position(1, 3), Position(2, 3)], maxlen=6)

    action = choose_heuristic_action(env)
    choice = describe_move_choice(env, action)

    assert action == env.wait_action_index
    assert choice is not None
    assert choice.wait_action is True


def test_playback_session_raw_mode_does_not_call_override_logic(monkeypatch) -> None:
    """Raw playback should bypass heuristic override evaluation entirely."""

    env = _build_loop_branch_env()

    monkeypatch.setattr(
        showcase_module,
        "load_checkpoint_for_playback",
        lambda _path: (object(), env, {}),
    )
    monkeypatch.setattr(showcase_module, "predict_action", lambda **_kwargs: (3, None))
    monkeypatch.setattr(
        showcase_module,
        "action_probabilities",
        lambda *_args, **_kwargs: np.array([0.1, 0.1, 0.1, 0.7], dtype=np.float32),
    )
    override_calls = {"count": 0}

    def _count_override(*_args, **_kwargs):
        override_calls["count"] += 1
        return True

    monkeypatch.setattr(showcase_module, "should_override_policy", _count_override)

    session = showcase_module.PlaybackSession(
        checkpoint_path="dummy",
        checkpoint_label="ckpt 0000",
        seed=77,
    )

    assert session.latest_state["policy_kind"] == "trained"
    assert session.latest_state["playback_mode"] == "raw"
    assert session.latest_state["policy_override_enabled"] is False
    assert session.latest_state["policy_decision_label"] == "raw learned policy"

    state, _ = session.advance()

    assert override_calls["count"] == 0
    assert state["policy_kind"] == "trained"
    assert state["playback_mode"] == "raw"
    assert state["policy_override_enabled"] is False
    assert state["policy_override_count"] == 0
    assert state["policy_override_reason"] is None


def test_playback_session_assisted_mode_calls_override_logic(monkeypatch) -> None:
    """Assisted playback should consult and apply the override heuristic when enabled."""

    env = _build_loop_branch_env()

    monkeypatch.setattr(
        showcase_module,
        "load_checkpoint_for_playback",
        lambda _path: (object(), env, {}),
    )
    monkeypatch.setattr(showcase_module, "predict_action", lambda **_kwargs: (3, None))
    monkeypatch.setattr(
        showcase_module,
        "action_probabilities",
        lambda *_args, **_kwargs: np.array([0.1, 0.1, 0.1, 0.7], dtype=np.float32),
    )
    override_calls = {"count": 0}

    def _count_override(*_args, **_kwargs):
        override_calls["count"] += 1
        return True

    monkeypatch.setattr(showcase_module, "should_override_policy", _count_override)

    session = showcase_module.PlaybackSession(
        checkpoint_path="dummy",
        checkpoint_label="ckpt 0000",
        seed=77,
        playback_mode="assisted",
    )

    assert session.latest_state["policy_override_enabled"] is True
    assert session.latest_state["playback_mode"] == "assisted"
    assert session.latest_state["policy_decision_label"] == "assisted learned policy"

    state, _ = session.advance()

    assert override_calls["count"] == 1
    assert state["policy_kind"] == "heuristic-override"
    assert state["playback_mode"] == "assisted"
    assert state["policy_override_enabled"] is True
    assert state["policy_override_count"] == 1
    assert state["policy_override_reason"] in {"heuristic-override", "prefer-unvisited"}
    assert state["policy_decision_label"].startswith("assisted override:")


def test_run_checkpoint_showcase_episode_returns_incompatible_result(monkeypatch) -> None:
    """Showcase should report incompatible checkpoints without raising."""

    def fail_session(**_kwargs):
        raise CheckpointCompatibilityError("Observation spaces do not match")

    monkeypatch.setattr(showcase_module, "PlaybackSession", fail_session)

    result = showcase_module.run_checkpoint_showcase_episode(
        checkpoint_path="dummy.zip",
        checkpoint_label="ckpt 0000",
        seed=12345,
    )

    assert result.status == "incompatible"
    assert result.outcome == "incompatible"
    assert "Observation spaces do not match" in result.notes


def test_fear_mode_avoids_dead_end_even_when_it_gains_monster_distance() -> None:
    """In fear mode a dead-end corridor that increases monster distance must not be
    preferred over a safe alternative, because the monster will corner the agent."""

    env = MazeEnv(
        MazeConfig(
            rows=5,
            cols=8,
            vision_range=4,
            max_player_speed=1,
            monster_speed=1,
            monster_activation_delay=0,
            curriculum_enabled=False,
        ),
        training_mode=False,
    )
    #   01234567
    # 0 ###.####   dead-end pocket at (0,3)
    # 1 #......#   main corridor
    # 2 #.####.#   walls; connectors at (2,1) and (2,6)
    # 3 #......#   south corridor leading to exit
    # 4 ########
    layout = MazeLayout(
        grid=(
            "###.####",
            "#......#",
            "#.####.#",
            "#......#",
            "########",
        ),
        player_start=Position(1, 3),
        monster_start=Position(1, 5),
        exit_position=Position(3, 6),
        seed=180,
    )
    env.reset(seed=180, options={"layout": layout, "maze_seed": 180})

    # Player at junction (1,3); monster visible at (1,5) — fear mode.
    env.player = Position(1, 3)
    env.monster = Position(1, 5)
    env.seen_open_cells = {
        Position(0, 3),
        Position(1, 1), Position(1, 2), Position(1, 3),
        Position(1, 4), Position(1, 5), Position(1, 6),
        Position(2, 1), Position(2, 6),
        Position(3, 1), Position(3, 2), Position(3, 3),
        Position(3, 4), Position(3, 5), Position(3, 6),
    }
    env.visible_open_cells = {
        Position(0, 3),
        Position(1, 1), Position(1, 2), Position(1, 3),
        Position(1, 4), Position(1, 5), Position(1, 6),
    }
    env.visited_counts = {Position(1, 3): 1}
    # (0,3) is a dead-end tip; routes through it from (1,3) commit to a dead end.
    env.known_dead_end_cells = {Position(0, 3)}
    env.path_history = deque([Position(1, 4), Position(1, 3)], maxlen=6)
    env.turns_since_monster_seen = 0
    env.last_seen_monster_position = Position(1, 5)

    ranked = rank_legal_moves(env)
    best = ranked[0]

    # The best choice must NOT be into the dead-end pocket (north to (0,3)).
    assert not best.known_dead_end, (
        f"Fear-mode heuristic chose dead-end cell {best.target} over safe alternatives"
    )
    assert not best.enters_dead_end, (
        f"Fear-mode heuristic entered a dead-end tip at {best.target}"
    )


def test_override_catches_policy_choosing_dead_end_over_safe_move() -> None:
    """The safety override should replace a trained-policy dead-end choice."""

    env = MazeEnv(
        MazeConfig(
            rows=5,
            cols=7,
            vision_range=4,
            max_player_speed=1,
            monster_speed=1,
            monster_activation_delay=0,
            curriculum_enabled=False,
        ),
        training_mode=False,
    )
    #   0123456
    # 0 #######
    # 1 #.....#   corridor; (1,1)-(1,2) are a dead-end pocket
    # 2 #####.#   walls block south; only (2,5) open
    # 3 #.....#   south corridor
    # 4 #######
    layout = MazeLayout(
        grid=(
            "#######",
            "#.....#",
            "#####.#",
            "#.....#",
            "#######",
        ),
        player_start=Position(1, 3),
        monster_start=Position(3, 3),
        exit_position=Position(1, 5),
        seed=181,
    )
    env.reset(seed=181, options={"layout": layout, "maze_seed": 181})
    env.player = Position(1, 3)
    env.monster = Position(3, 3)
    env.seen_open_cells = {
        Position(1, 1), Position(1, 2), Position(1, 3),
        Position(1, 4), Position(1, 5),
        Position(2, 5),
        Position(3, 3), Position(3, 4), Position(3, 5),
    }
    env.visible_open_cells = {
        Position(1, 1), Position(1, 2), Position(1, 3),
        Position(1, 4), Position(1, 5),
    }
    env.visited_counts = {Position(1, 3): 1}
    env.known_dead_end_cells = {Position(1, 1), Position(1, 2)}
    env.path_history = deque([Position(1, 4), Position(1, 3)], maxlen=6)

    # Simulate policy choosing direction 3 (west → dead end at (1,2))
    dead_end_action = 3  # west
    safe_action = 1  # east → (1,4), not a dead end
    chosen = describe_move_choice(env, dead_end_action)
    best = describe_move_choice(env, safe_action)

    from maze_rl.policies.action_helpers import should_override_policy
    assert chosen is not None
    assert best is not None
    assert chosen.known_dead_end is True
    assert best.known_dead_end is False
    assert should_override_policy(chosen, best, 0.6, 0.3) is True
