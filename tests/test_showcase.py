"""Showcase workflow tests."""

from collections import deque
from pathlib import Path

from maze_rl.config import MazeConfig, TrainingConfig
from maze_rl.envs.entities import MazeLayout, Position
from maze_rl.envs.maze_env import MazeEnv
from maze_rl.policies.model_factory import create_model
from maze_rl.training.checkpointing import CheckpointManager
from maze_rl.training.showcase import (
    choose_heuristic_action,
    describe_move_choice,
    format_showcase_table,
    rank_legal_moves,
    run_showcase_headless,
    save_showcase_summary,
    should_override_policy,
)


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
