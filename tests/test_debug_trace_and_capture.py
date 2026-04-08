"""Debug trace, monster motion, and capture regression tests."""

from pathlib import Path

from maze_rl.config import MazeConfig, TrainingConfig
from maze_rl.envs.debug_layouts import build_debug_pursuit_layout
from maze_rl.envs.maze_env import MazeEnv
from maze_rl.policies.model_factory import create_model
from maze_rl.training.checkpointing import CheckpointManager
from maze_rl.training.evaluate import run_frozen_episode
from maze_rl.training.showcase import run_checkpoint_showcase_episode


def test_monster_position_changes_in_debug_layout() -> None:
    """Monster should move in a deterministic pursuit layout."""

    env = MazeEnv(MazeConfig(monster_speed=1, monster_activation_delay=0), training_mode=False)
    env.reset(options={"layout": build_debug_pursuit_layout(), "maze_seed": 999001})
    start_monster = env.get_state_snapshot()["monster_position"]
    env.step(0)
    after_monster = env.get_state_snapshot()["monster_position"]
    assert after_monster != start_monster


def test_render_snapshot_reflects_live_monster_position() -> None:
    """Render state should expose the same live monster state used by the environment."""

    env = MazeEnv(MazeConfig(monster_speed=1, monster_activation_delay=0), training_mode=False)
    env.reset(options={"layout": build_debug_pursuit_layout(), "maze_seed": 999001})
    env.step(0)
    snapshot = env.get_render_state()
    assert snapshot["monster_position"] == env.monster.as_tuple()
    assert snapshot["player_monster_distance"] == abs(env.player.row - env.monster.row) + abs(env.player.col - env.monster.col)


def test_step_info_contains_structured_replay_turn() -> None:
    """Each environment step should emit structured replay data for micro-step playback."""

    env = MazeEnv(MazeConfig(monster_speed=2, monster_activation_delay=0, max_player_speed=2), training_mode=False)
    env.reset(options={"layout": build_debug_pursuit_layout(), "maze_seed": 999001})
    _, _, _, _, info = env.step(1)

    replay_turn = info["replay_turn"]
    assert replay_turn["turn_step"] == 1
    assert replay_turn["player_start_position"]
    assert replay_turn["monster_start_position"]
    assert isinstance(replay_turn["player_path"], tuple)
    assert isinstance(replay_turn["monster_path"], tuple)
    assert isinstance(replay_turn["micro_steps"], list)
    assert replay_turn["final_player_position"] == info["state_snapshot"]["player_position"]
    assert replay_turn["final_monster_position"] == info["state_snapshot"]["monster_position"]


def test_caught_requires_same_cell_and_populates_capture_diagnostics() -> None:
    """Caught should only happen on same-cell collision and should include diagnostics."""

    env = MazeEnv(MazeConfig(monster_speed=1, monster_activation_delay=0), training_mode=False)
    env.reset(options={"layout": build_debug_pursuit_layout(), "maze_seed": 999001})

    last_info = None
    for _ in range(10):
        _, _, terminated, truncated, info = env.step(1)
        last_info = info
        if terminated or truncated:
            break

    assert last_info is not None
    if last_info["outcome"] == "caught":
        diagnostics = last_info["capture_diagnostics"]
        assert diagnostics["final_distance"] == 0
        assert diagnostics["capture_rule"] == "same-cell"
        assert diagnostics["final_player_position"] == diagnostics["final_monster_position"]


def test_showcase_debug_state_contains_updated_monster_positions(tmp_path: Path) -> None:
    """Showcase step callbacks should receive live changing monster positions."""

    checkpoint_dir = tmp_path / "checkpoints"
    training_config = TrainingConfig(checkpoint_dir=checkpoint_dir)
    maze_config = MazeConfig()
    env = MazeEnv(maze_config)
    model = create_model(training_config=training_config, env=env)
    manager = CheckpointManager(training_config=training_config, maze_config=maze_config)
    checkpoint_path = manager.save(
        model=model,
        episode=0,
        timesteps=0,
        training_summary={"episodes_seen": 0},
        evaluation_summary={"episodes": 0},
    )

    monster_positions: list[tuple[int, int]] = []
    run_checkpoint_showcase_episode(
        checkpoint_path=checkpoint_path,
        checkpoint_label="ckpt 0000",
        seed=12345,
        max_no_progress_streak=5,
        wall_time_timeout_s=5.0,
        on_step=lambda state: monster_positions.append(state["monster_position"]) or True,
    )
    assert monster_positions
    assert len(set(monster_positions)) >= 2


def test_debug_trace_eval_path_runs_without_breaking(tmp_path: Path, capsys) -> None:
    """Debug trace should run through the eval path without breaking execution."""

    checkpoint_dir = tmp_path / "checkpoints"
    training_config = TrainingConfig(checkpoint_dir=checkpoint_dir)
    maze_config = MazeConfig()
    env = MazeEnv(maze_config)
    model = create_model(training_config=training_config, env=env)
    manager = CheckpointManager(training_config=training_config, maze_config=maze_config)
    checkpoint_path = manager.save(
        model=model,
        episode=0,
        timesteps=0,
        training_summary={"episodes_seen": 0},
        evaluation_summary={"episodes": 0},
    )

    loaded_env = MazeEnv(maze_config, training_mode=False)
    from maze_rl.policies.model_factory import load_model_from_checkpoint

    loaded_model = load_model_from_checkpoint(checkpoint_path, loaded_env)
    run_frozen_episode(loaded_model, loaded_env, seed=12345, debug_trace=True)
    output = capsys.readouterr().out
    assert "step=" in output