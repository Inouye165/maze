"""Control app controller tests."""

import time
from pathlib import Path

from maze_rl.config import MazeConfig, TrainingConfig
from maze_rl.envs.maze_env import MazeEnv
from maze_rl.policies.model_factory import CheckpointCompatibilityError, create_model
from maze_rl.render.control_app import LabAppController, LabControlApp
from maze_rl.training.checkpointing import CheckpointManager
from maze_rl.training.showcase import ShowcaseResult
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
    """The controller should support baseline, learned runs, and exact replay.

    This avoids spinning the full app loop in tests.
    """

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
    assert getattr(controller.session, "allow_policy_override", False) is True


def test_basic_tab_uses_play_label_instead_of_marks_play(tmp_path: Path) -> None:
    """The Basic tab should expose one Play button with no legacy Marks wording."""

    _create_checkpoints(tmp_path, episodes=(0,))

    app = LabControlApp(checkpoint_dir=tmp_path)
    labels = app.visible_button_labels()

    assert "Play" in labels
    assert "Marks Play" not in labels


def test_control_panel_buttons_stay_within_visible_panel(tmp_path: Path) -> None:
    """Visible buttons should fit inside the right-side panel for every tab."""

    _create_checkpoints(tmp_path, episodes=(0,))

    app = LabControlApp(checkpoint_dir=tmp_path)
    allowed_bounds = app.panel_area.inflate(-18, -18)

    for tab_name in ("basic", "review", "advanced"):
        app.active_tab = tab_name
        for button in app.build_buttons():
            assert allowed_bounds.contains(button.rect), (tab_name, button.label, button.rect)


def test_play_runs_innate_mode_without_checkpoint(tmp_path: Path) -> None:
    """Play should fall back to innate behavior when no trained checkpoint exists."""

    controller = LabAppController(checkpoint_dir=tmp_path)

    controller.start_play()

    assert controller.session is not None
    assert controller.current_mode == "baseline-legal-mover"
    assert controller.seed_text == "00001"
    assert controller.seed_ladder_active is True
    assert controller.play_mode_status() == "Policy: Innate"


def test_play_runs_trained_mode_when_checkpoint_exists(tmp_path: Path) -> None:
    """Play should use the trained checkpoint when one is available."""

    _create_checkpoints(tmp_path, episodes=(0,))
    controller = LabAppController(checkpoint_dir=tmp_path)

    controller.start_play()

    assert controller.session is not None
    assert controller.current_mode == "current-learned-ai"
    assert controller.seed_ladder_active is True
    assert controller.play_mode_status() == "Policy: Trained (ckpt 0000)"
    assert getattr(controller.session, "allow_policy_override", False) is True


def test_play_increments_seed_until_loss_then_trains_failed_seed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Seed ladder play should advance on wins, then arm training after a loss."""

    _create_checkpoints(tmp_path, episodes=(0,))
    captured_training_calls: list[tuple[int, str, int]] = []
    outcomes = ["escaped", "caught"]

    def fake_continue_training_from_latest(
        additional_episodes: int,
        checkpoint_dir: Path,
        training_mode: str,
        stop_event,
        fixed_maze_seed: int,
    ) -> None:
        _ = checkpoint_dir
        _ = stop_event
        captured_training_calls.append(
            (additional_episodes, training_mode, fixed_maze_seed)
        )

    class FakePlaybackSession:
        def __init__(
            self,
            checkpoint_path,
            checkpoint_label: str,
            seed: int,
            debug_trace: bool,
            allow_policy_override: bool,
        ) -> None:
            _ = checkpoint_path
            _ = debug_trace
            self.allow_policy_override = allow_policy_override
            self.seed = seed
            self.checkpoint_label = checkpoint_label
            self.latest_state = {
                "grid": ("###", "#.#", "###"),
                "full_grid": ("###", "#.#", "###"),
                "player_position": (1, 1),
                "monster_position": (1, 1),
                "exit_position": (1, 1),
                "seed": seed,
            }

        def advance(self):
            outcome = outcomes.pop(0)
            result = ShowcaseResult(
                checkpoint=self.checkpoint_label,
                status="ok",
                outcome=outcome,
                escape_rate=1.0 if outcome == "escaped" else 0.0,
                coverage=0.5,
                steps=12,
                revisits=1,
                oscillations=0,
                dead_ends=0,
                start_monster_distance=4.0,
                time_to_capture=None,
                frontier_rate=0.4,
                peak_no_progress_streak=0,
                final_player_position=(1, 1),
                final_monster_position=(1, 1),
                final_distance=0,
                capture_rule=None,
                final_state=dict(self.latest_state),
                checkpoint_path="fake",
                seed=self.seed,
            )
            return dict(self.latest_state), result

        def build_recorded_run(self):
            return type("RecordedRunStub", (), {"frames": [dict(self.latest_state)]})()

    monkeypatch.setattr(
        "maze_rl.render.control_app.continue_training_from_latest",
        fake_continue_training_from_latest,
    )
    monkeypatch.setattr("maze_rl.render.control_app.PlaybackSession", FakePlaybackSession)

    controller = LabAppController(checkpoint_dir=tmp_path)
    controller.cycle_input_text = "5"

    controller.start_play()
    controller.update()

    assert controller.seed_text == "00002"
    assert controller.session is not None
    assert controller.seed_ladder_active is True

    controller.update()

    assert captured_training_calls == []
    assert controller.seed_text == "00002"
    assert controller.last_failed_seed == 2
    assert controller.seed_ladder_active is False
    assert controller.pending_training_seed == 2

    controller.cycle_input_text = "7"
    controller.start_play()

    deadline = time.monotonic() + 0.2
    while not captured_training_calls and time.monotonic() < deadline:
        time.sleep(0.005)

    assert captured_training_calls == [(7, "maze-only", 2)]
    assert controller.pending_training_seed is None


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


def test_app_controller_handles_incompatible_checkpoint_without_crashing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The app should surface an incompatible checkpoint instead of crashing."""

    _create_checkpoints(tmp_path, episodes=(0,))

    def fail_playback(**_kwargs):
        raise CheckpointCompatibilityError("Observation spaces do not match")

    monkeypatch.setattr("maze_rl.render.control_app.PlaybackSession", fail_playback)

    controller = LabAppController(checkpoint_dir=tmp_path)
    controller.start_current_ai_run()

    assert controller.session is None
    assert controller.last_state is None
    assert controller.last_result is None
    assert "incompatible" in controller.training_message


def test_app_controller_training_controls_use_app_selected_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Training controls should use the app-selected training mode.

    Stop requests should still interrupt the background run.
    """

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


def test_training_progress_summary_and_render_state_show_live_training_seed_and_map() -> None:
    """The app should surface the active training seed and render the live training maze."""

    controller = LabAppController(checkpoint_dir=Path("unused"))
    controller.training_status = "running"
    controller.training_progress = {
        "maze_seed": 424242,
        "active_cycle": 3,
        "completed_episodes": 2,
        "target_episodes": 5,
        "episode_steps": 12,
        "state_snapshot": {
            "grid": (
                "#####",
                "#...#",
                "#####",
            ),
            "full_grid": (
                "#####",
                "#...#",
                "#####",
            ),
            "player_position": (1, 1),
            "monster_position": (1, 3),
            "exit_position": (1, 2),
            "seed": 424242,
        },
    }

    class _AliveThread:
        def is_alive(self) -> bool:
            return True

    controller.training_thread = _AliveThread()  # type: ignore[assignment]

    assert "seed 424242" in controller.training_progress_summary()
    assert controller.active_training_seed() == 424242
    render_state = controller.render_state()
    assert render_state is not None
    assert render_state["seed"] == 424242
    assert render_state["checkpoint_label"] == "training seed 424242"


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


def test_app_controller_builds_last_10_last_100_and_last_1000_stats(tmp_path: Path) -> None:
    """The dashboard should expose app run stats for last 10, last 100, and last 1000 seeds."""

    controller = LabAppController(checkpoint_dir=tmp_path)
    controller.cycle_input_text = "12"
    controller.run_outcomes.extend(["escaped"] * 700 + ["caught"] * 300)
    controller.total_runs = 1000
    controller.total_wins = 700
    cards = controller.training_stat_cards()

    assert controller.parse_cycle_count() == 12
    assert [card.label for card in cards] == ["Last 10", "Last 100", "Last 1000"]
    assert [(card.cycles, card.wins, card.losses) for card in cards] == [
        (10, 0, 10),
        (100, 0, 100),
        (1000, 700, 300),
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

    assert controller.play_mode_status() == "Policy: Trained (ckpt 0050)"
    controller.reset_training()
    controller.start_play()

    assert controller.play_mode_status() == "Policy: Innate"
    assert controller.session is not None
    assert controller.current_mode == "baseline-legal-mover"
