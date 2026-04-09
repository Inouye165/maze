"""Control app controller tests."""

import json
import time
from pathlib import Path

from maze_rl.config import MazeConfig, TrainingConfig
from maze_rl.envs.maze_env import MazeEnv
from maze_rl.policies.model_factory import create_model
from maze_rl.render.control_app import LabAppController, LabControlApp
from maze_rl.training.checkpointing import CheckpointManager
from maze_rl.training.train import maze_config_for_training_mode


def _create_checkpoints(tmp_path: Path, episodes: tuple[int, ...] = (0,)) -> Path:
    """Create one or more lightweight checkpoints for controller tests."""

    checkpoint_dir = tmp_path / "maze_only"
    training_config = TrainingConfig(checkpoint_dir=checkpoint_dir)
    maze_config = MazeConfig()
    env = MazeEnv(maze_config)
    model = create_model(training_config=training_config, env=env)
    manager = CheckpointManager(
        training_config=training_config,
        maze_config=maze_config,
    )
    for episode in episodes:
        manager.save(
            model=model,
            episode=episode,
            timesteps=episode,
            training_summary={"episodes_seen": episode},
            evaluation_summary={"episodes": 1},
        )
    return checkpoint_dir


def test_app_controller_supports_baseline_current_ai_and_replay(tmp_path: Path) -> None:
    """The controller should support baseline, learned runs, and exact replay without the UI loop."""

    _create_checkpoints(tmp_path, episodes=(0,))

    controller = LabAppController(checkpoint_dir=tmp_path)
    assert controller.selected_checkpoint is not None
    controller.set_speed_index(2)
    assert controller.fps in {4, 8, 16}
    controller.start_baseline_legal_mover()
    assert controller.session is not None
    assert controller.current_mode == "baseline-legal-mover"
    controller.pause()
    assert controller.paused is True
    controller.step_once()
    assert controller.last_state is not None
    controller.resume()

    for _ in range(500):
        controller.update()
        if controller.session is None:
            break

    assert controller.last_result is not None
    assert controller.last_recorded_run is not None
    assert controller.last_state is not None
    assert (
        controller.last_state["player_position"]
        == controller.last_result.final_player_position
    )
    assert (
        controller.last_state["monster_position"]
        == controller.last_result.final_monster_position
    )
    assert (
        controller.last_state["rendered_player_position"]
        == controller.last_result.final_player_position
    )
    assert (
        controller.last_state["rendered_monster_position"]
        == controller.last_result.final_monster_position
    )

    first_frame = controller.last_recorded_run.frames[0]
    controller.replay_last_run()
    assert controller.session is not None
    assert controller.last_state == first_frame

    controller.reset()
    controller.start_current_ai_run()
    assert controller.session is not None
    assert controller.current_mode == "current-learned-ai"


def test_basic_tab_uses_play_label_instead_of_marks_play(tmp_path: Path) -> None:
    """The Basic tab should expose one Play button with no legacy Marks wording."""

    _create_checkpoints(tmp_path, episodes=(0,))

    app = LabControlApp(checkpoint_dir=tmp_path)
    labels = app.visible_button_labels()

    assert "Play" in labels
    assert "Marks Play" not in labels


def test_play_runs_innate_mode_without_checkpoint(tmp_path: Path) -> None:
    """Play should fall back to innate behavior when no trained checkpoint exists."""

    controller = LabAppController(checkpoint_dir=tmp_path)

    controller.start_play()

    assert controller.session is not None
    assert controller.current_mode == "baseline-legal-mover"
    assert controller.play_mode_status() == "Mode: Innate"


def test_play_runs_trained_mode_when_checkpoint_exists(tmp_path: Path) -> None:
    """Play should use the trained checkpoint when one is available."""

    _create_checkpoints(tmp_path, episodes=(0,))
    controller = LabAppController(checkpoint_dir=tmp_path)

    controller.start_play()

    assert controller.session is not None
    assert controller.current_mode == "current-learned-ai"
    assert controller.play_mode_status() == "Mode: Trained (ckpt 0000)"


def test_app_controller_uses_mode_specific_checkpoint_directories(tmp_path: Path) -> None:
    """Maze-only and full-monster modes should not share the same checkpoint root."""

    controller = LabAppController(checkpoint_dir=tmp_path)
    assert controller.training_mode == "maze-only"
    assert controller.active_checkpoint_dir == tmp_path / "maze_only"
    manager = CheckpointManager(
        training_config=TrainingConfig(checkpoint_dir=controller.active_checkpoint_dir),
        maze_config=MazeConfig(),
    )
    env = MazeEnv(MazeConfig())
    model = create_model(
        training_config=TrainingConfig(checkpoint_dir=controller.active_checkpoint_dir),
        env=env,
    )
    manager.save(
        model=model,
        episode=0,
        timesteps=0,
        training_summary={"episodes_seen": 0},
        evaluation_summary={"episodes": 1},
    )
    controller.refresh_checkpoints()
    assert controller.selected_checkpoint is not None
    controller.toggle_training_mode()
    assert controller.training_mode == "full-monster"
    assert controller.active_checkpoint_dir == tmp_path / "full_monster"
    assert controller.selected_checkpoint is None


def test_app_controller_compare_milestones_skips_missing_and_runs_existing(tmp_path: Path) -> None:
    """The app compare flow should run existing milestones and skip missing ones."""

    _create_checkpoints(tmp_path, episodes=(0, 50))
    controller = LabAppController(checkpoint_dir=tmp_path)
    controller.compare_pause_s = 0.0
    controller.start_compare_milestones()

    for _ in range(1200):
        controller.update()
        if controller.session is None and not controller.compare_queue:
            break

    assert controller.current_mode == "compare-milestones"
    assert [item.checkpoint for item in controller.compare_results] == [
        "ckpt 0000",
        "ckpt 0050",
        "ckpt 0100",
        "ckpt 0200",
        "ckpt 0500",
        "ckpt 1000",
    ]
    assert any(
        item.checkpoint == "ckpt 0000" and item.status == "ok"
        for item in controller.compare_results
    )
    assert any(
        item.checkpoint == "ckpt 0050" and item.status == "ok"
        for item in controller.compare_results
    )
    assert any(
        item.checkpoint == "ckpt 0100" and item.status == "missing"
        for item in controller.compare_results
    )


def test_app_controller_training_controls_use_app_selected_mode(tmp_path: Path, monkeypatch) -> None:
    """Training controls should use the app-selected training mode and support stop requests."""

    _create_checkpoints(tmp_path, episodes=(0,))
    calls: list[tuple[int, str]] = []

    def fake_continue_training_from_latest(
        additional_episodes: int,
        checkpoint_dir: Path,
        training_mode: str,
        stop_event,
    ) -> None:
        _ = checkpoint_dir
        calls.append((additional_episodes, training_mode))
        end_at = time.monotonic() + 0.05
        while time.monotonic() < end_at and not stop_event.is_set():
            time.sleep(0.005)

    monkeypatch.setattr(
        "maze_rl.render.control_app.continue_training_from_latest",
        fake_continue_training_from_latest,
    )

    controller = LabAppController(checkpoint_dir=tmp_path)
    assert controller.training_mode == "maze-only"
    controller.start_training(50)
    assert controller.training_status == "running"
    deadline = time.monotonic() + 0.2
    while not calls and time.monotonic() < deadline:
        time.sleep(0.005)
    assert calls == [(50, "maze-only")]
    controller.stop_training()
    assert controller.training_status == "stopping"

    while controller.is_training_active:
        controller.update()
        time.sleep(0.005)
    controller.update()

    controller.toggle_training_mode()
    assert controller.training_mode == "full-monster"


def test_maze_learning_mode_simplifies_actions_and_slows_monster() -> None:
    """Maze-learning mode should use 4 directions with a slow monster."""

    maze_only = maze_config_for_training_mode(MazeConfig(), "maze-only")
    env = MazeEnv(maze_only)
    assert maze_only.rows == 19
    assert maze_only.cols == 19
    assert maze_only.max_player_speed == 1
    assert maze_only.monster_speed == 1
    assert maze_only.monster_move_interval == 3
    assert maze_only.max_episode_steps == 280
    assert maze_only.stall_threshold == 90
    assert getattr(env.action_space, "n", None) == 4
    assert maze_only.reward.exit_progress_reward >= 1.5
    assert maze_only.reward.safety_gain_reward >= 1.8
    assert maze_only.reward.safety_loss_penalty <= -3.0
    assert maze_only.reward.win_reward >= 100.0


def test_incomplete_checkpoints_are_ignored_by_app(tmp_path: Path) -> None:
    """The app should ignore orphaned checkpoint zips that have no metadata file."""

    incomplete_dir = tmp_path / "maze_only"
    incomplete_dir.mkdir(parents=True)
    (incomplete_dir / "ckpt_0000.zip").write_bytes(b"placeholder")

    controller = LabAppController(checkpoint_dir=tmp_path)
    assert controller.selected_checkpoint is None
    assert controller.latest_training_summary() is None


def test_app_controller_builds_last_10_last_50_and_all_time_stats(tmp_path: Path) -> None:
    """The simplified dashboard should expose training stats for last 10, last 50, and all time."""

    checkpoint_dir = _create_checkpoints(tmp_path, episodes=(0,))
    metadata_path = checkpoint_dir / "ckpt_0000.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["training_summary"] = {
        "episodes_seen": 60,
        "wins": 36,
        "recent_outcomes": ["escaped"] * 30 + ["caught"] * 20,
        "recent_10_outcomes": ["escaped"] * 7 + ["caught"] * 3,
        "recent_50_outcomes": ["escaped"] * 30 + ["caught"] * 20,
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    controller = LabAppController(checkpoint_dir=tmp_path)
    controller.cycle_input_text = "12"
    cards = controller.training_stat_cards()

    assert controller.parse_cycle_count() == 12
    assert [card.label for card in cards] == ["Last 10", "Last 50", "All Time"]
    assert [(card.cycles, card.wins, card.losses) for card in cards] == [
        (10, 7, 3),
        (50, 30, 20),
        (60, 36, 24),
    ]


def test_reset_training_clears_learned_checkpoints_across_stages(tmp_path: Path) -> None:
    """Reset training should clear saved checkpoint knowledge and return to an untrained state."""

    maze_only_dir = _create_checkpoints(tmp_path, episodes=(0, 50))
    full_monster_dir = tmp_path / "full_monster"
    full_monster_dir.mkdir(parents=True)
    (full_monster_dir / "ckpt_0100.zip").write_bytes(b"placeholder")
    (full_monster_dir / "ckpt_0100.json").write_text("{}", encoding="utf-8")
    (tmp_path / "ckpt_9999.zip").write_bytes(b"placeholder")
    (tmp_path / "ckpt_9999.json").write_text("{}", encoding="utf-8")

    controller = LabAppController(checkpoint_dir=tmp_path)
    assert controller.selected_checkpoint is not None
    assert maze_only_dir.exists()

    controller.reset_training()

    assert controller.selected_checkpoint is None
    assert controller.available_checkpoints == []
    assert controller.last_result is None
    assert controller.last_state is None
    assert controller.training_message == "cleared all learned checkpoints"
    assert not list(tmp_path.rglob("ckpt_*.zip"))
    assert not list(tmp_path.rglob("ckpt_*.json"))


def test_reset_training_returns_play_to_innate_mode(tmp_path: Path) -> None:
    """Reset training should send Play back to innate mode immediately."""

    _create_checkpoints(tmp_path, episodes=(0, 50))
    controller = LabAppController(checkpoint_dir=tmp_path)

    assert controller.play_mode_status() == "Mode: Trained (ckpt 0050)"
    controller.reset_training()
    controller.start_play()

    assert controller.play_mode_status() == "Mode: Innate"
    assert controller.session is not None
    assert controller.current_mode == "baseline-legal-mover"
