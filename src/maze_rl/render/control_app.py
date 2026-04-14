"""Local pygame control app for the maze RL lab."""

# pylint: disable=too-many-lines,line-too-long

from __future__ import annotations

from collections import deque
import inspect
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pygame

from maze_rl.config import MazeConfig, maze_config_from_dict
from maze_rl.policies.model_factory import CheckpointCompatibilityError
from maze_rl.render.view_state import (
    viewer_cell_color,
    viewer_dead_end_cells,
    viewer_explored_cells,
    viewer_exit_position,
    viewer_grid,
    viewer_monster_position,
    viewer_policy_badge,
    viewer_player_position,
    viewer_traveled_cells,
    viewer_visible_cells,
)
from maze_rl.training.checkpointing import checkpoint_is_complete, latest_checkpoint, list_checkpoints, load_checkpoint_metadata, resolve_checkpoint_path
from maze_rl.training.showcase import (
    BaselinePlaybackSession,
    PlaybackSession,
    RecordedPlaybackSession,
    RecordedRun,
    ShowcaseResult,
    build_incompatible_result,
    build_missing_result,
)
from maze_rl.training.train import continue_training_from_latest, format_training_progress, maze_config_for_training_mode

QUIT = getattr(pygame, "QUIT", 256)
MOUSEBUTTONDOWN = getattr(pygame, "MOUSEBUTTONDOWN", 1025)
KEYDOWN = getattr(pygame, "KEYDOWN", 768)
K_BACKSPACE = getattr(pygame, "K_BACKSPACE", 8)
K_RETURN = getattr(pygame, "K_RETURN", 13)

MILESTONE_EPISODES = (0, 50, 100, 200, 500, 1000)

APP_BG = (233, 238, 244)
DECOR_WARM = (221, 205, 183)
DECOR_COOL = (198, 217, 230)
DECOR_SOFT = (209, 229, 220)
SURFACE_MAIN = (249, 251, 253)
SURFACE_PANEL = (244, 247, 250)
SURFACE_CARD = (255, 255, 255)
SURFACE_CARD_ALT = (246, 248, 251)
BORDER = (211, 219, 228)
TEXT_PRIMARY = (34, 43, 56)
TEXT_SECONDARY = (88, 99, 113)
TEXT_MUTED = (119, 130, 144)
PRIMARY = (36, 98, 174)
PRIMARY_DARK = (29, 75, 133)
ACCENT = (26, 132, 112)
DANGER = (174, 77, 67)
WARNING = (193, 134, 54)
DISABLED = (206, 213, 221)
CARD_HEADER_HEIGHT = 56


@dataclass
class Button:
    """Clickable button definition."""

    label: str
    rect: pygame.Rect
    action: Callable[[], None]
    enabled: bool = True
    kind: str = "default"
    active: bool = False


@dataclass(frozen=True)
class TrainingStatCard:
    """Compact training performance block shown in the app dashboard."""

    label: str
    cycles: int
    wins: int
    losses: int
    percentage: float


@dataclass(frozen=True)
class CompareRunEntry:
    """One queued compare run entry."""

    checkpoint_episode: int
    checkpoint_path: Path


class LabAppController:
    """State and actions for the app-first local control app."""

    def __init__(self, checkpoint_dir: str | Path = "checkpoints") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.seed_text = "00001"
        self.debug_trace = False
        self.speed_options = [("Slow", 4), ("Normal", 8), ("Fast", 16)]
        self.speed_index = 1
        self.available_checkpoints: list[tuple[int, Path]] = []
        self.selected_index = 0
        self.session: PlaybackSession | BaselinePlaybackSession | RecordedPlaybackSession | None = None
        self.last_recorded_run: RecordedRun | None = None
        self.last_result: ShowcaseResult | None = None
        self.last_state: dict[str, Any] | None = None
        self.paused = False
        self.current_mode = "idle"
        self.selected_mode = "baseline-legal-mover"
        self.training_mode = "maze-only"
        self.training_thread: threading.Thread | None = None
        self.training_stop_event: threading.Event | None = None
        self.training_status = "idle"
        self.training_message = "ready"
        self.training_error: str | None = None
        self.training_progress: dict[str, Any] | None = None
        self.cycle_input_text = "1"
        self.last_training_increment = 100
        self.active_training_seed_override: int | None = None
        self.last_requested_training_seed: int | None = None
        self.seed_ladder_active = False
        self.current_run_seed: int | None = None
        self.last_failed_seed: int | None = None
        self.pending_training_seed: int | None = None
        self.run_outcomes: deque[str] = deque(maxlen=1000)
        self.total_runs = 0
        self.total_wins = 0
        self.compare_queue: list[CompareRunEntry] = []
        self.compare_results: list[ShowcaseResult] = []
        self.compare_pause_until = 0.0
        self.compare_pause_s = 0.9
        self.refresh_checkpoints()
        if self.available_checkpoints:
            self.selected_mode = "current-learned-ai"
            self.selected_index = len(self.available_checkpoints) - 1

    @property
    def selected_checkpoint(self) -> tuple[int, Path] | None:
        """Return the currently selected checkpoint entry, if any."""

        if not self.available_checkpoints:
            return None
        self.selected_index = max(0, min(self.selected_index, len(self.available_checkpoints) - 1))
        return self.available_checkpoints[self.selected_index]

    @property
    def fps(self) -> int:
        """Return the active playback frame rate."""

        return self.speed_options[self.speed_index][1]

    @property
    def active_checkpoint_dir(self) -> Path:
        """Return the checkpoint directory for the active training stage."""

        return self.checkpoint_dir / ("maze_only" if self.training_mode == "maze-only" else "full_monster")

    @property
    def can_start_run(self) -> bool:
        """Return whether the app can start a new playback session."""

        return self.session is None and not self.is_training_active

    @property
    def is_training_active(self) -> bool:
        """Return whether background training is still running."""

        return self.training_thread is not None and self.training_thread.is_alive()

    def refresh_checkpoints(self) -> None:
        """Reload available checkpoints from disk."""

        previous_episode = self.selected_checkpoint[0] if self.selected_checkpoint else None
        self.available_checkpoints = list_checkpoints(self.active_checkpoint_dir)
        if not self.available_checkpoints:
            self.selected_index = 0
            return
        if previous_episode is None:
            self.selected_index = len(self.available_checkpoints) - 1
            return
        for index, (episode, _) in enumerate(self.available_checkpoints):
            if episode == previous_episode:
                self.selected_index = index
                return
        self.selected_index = len(self.available_checkpoints) - 1

    def parse_seed(self) -> int:
        """Return the current numeric seed."""

        try:
            return max(1, int(self.seed_text))
        except ValueError:
            return 1

    @staticmethod
    def format_seed(seed: int) -> str:
        """Return the normalized user-facing seed label."""

        return f"{max(1, int(seed)):05d}"

    def seed_display_text(self, editing: bool = False) -> str:
        """Return the seed text for the current input state."""

        if editing:
            return self.seed_text
        return self.format_seed(self.parse_seed())

    def set_seed_value(self, seed: int) -> None:
        """Persist a normalized numeric seed back to the text field."""

        self.seed_text = self.format_seed(seed)

    def parse_cycle_count(self) -> int:
        """Return the requested training cycle count."""

        try:
            return max(1, int(self.cycle_input_text))
        except ValueError:
            return 1

    def set_speed_index(self, index: int) -> None:
        """Set playback speed by index."""

        self.speed_index = max(0, min(index, len(self.speed_options) - 1))

    def cycle_checkpoint(self, direction: int) -> None:
        """Move the checkpoint selection backward or forward."""

        if not self.available_checkpoints or not self.can_start_run:
            return
        self.selected_index = (self.selected_index + direction) % len(self.available_checkpoints)

    def use_latest_checkpoint(self) -> None:
        """Select the latest available checkpoint."""

        if not self.available_checkpoints or not self.can_start_run:
            return
        self.selected_index = len(self.available_checkpoints) - 1
        self.training_message = f"selected latest checkpoint ckpt {self.available_checkpoints[-1][0]:04d}"

    def set_mode(self, mode: str) -> None:
        """Remember the currently selected app mode."""

        self.selected_mode = mode

    def start_selected_mode(self) -> None:
        """Run the currently selected app mode."""

        if self.selected_mode == "baseline-legal-mover":
            self.start_baseline_legal_mover()
            return
        if self.selected_mode == "current-learned-ai":
            self.start_current_ai_run()
            return
        if self.selected_mode == "compare-milestones":
            self.start_compare_milestones()
            return
        if self.selected_mode == "replay":
            self.replay_last_run()
            return

    def start_baseline_legal_mover(self) -> None:
        """Run the deterministic non-learning legal-move baseline."""

        self._start_baseline_legal_mover(seed=self.parse_seed(), preserve_seed_ladder=False)

    def _start_baseline_legal_mover(self, seed: int, preserve_seed_ladder: bool) -> None:
        """Internal baseline runner with optional seed ladder continuity."""

        if not self.can_start_run:
            self.training_message = "finish training or the current run first"
            return
        if not preserve_seed_ladder:
            self.seed_ladder_active = False
            self.pending_training_seed = None
        maze_config = self._maze_config_for_active_run()
        self._clear_compare_state()
        self.session = BaselinePlaybackSession(
            maze_config=maze_config,
            checkpoint_label="innate",
            seed=seed,
            debug_trace=self.debug_trace,
        )
        self.current_run_seed = seed
        self.current_mode = "baseline-legal-mover"
        self.selected_mode = "baseline-legal-mover"
        self.last_result = None
        self.last_state = self.session.latest_state
        self.paused = False
        self.training_message = f"running innate play on seed {self.format_seed(seed)}"

    def start_current_ai_run(self) -> None:
        """Run one frozen episode using the selected checkpoint."""

        self._start_current_ai_run(seed=self.parse_seed(), preserve_seed_ladder=False)

    def _start_current_ai_run(self, seed: int, preserve_seed_ladder: bool) -> None:
        """Internal trained runner with optional seed ladder continuity."""

        if not self.can_start_run:
            self.training_message = "finish training or the current run first"
            return
        if not preserve_seed_ladder:
            self.seed_ladder_active = False
            self.pending_training_seed = None
        selected = self.selected_checkpoint
        if selected is None:
            self.training_message = "no checkpoint available for current AI"
            return
        episode, checkpoint_path = selected
        self._clear_compare_state()
        try:
            self.session = PlaybackSession(
                checkpoint_path=checkpoint_path,
                checkpoint_label=f"ckpt {episode:04d}",
                seed=seed,
                debug_trace=self.debug_trace,
                allow_policy_override=True,
            )
        except CheckpointCompatibilityError:
            self.session = None
            self.last_result = None
            self.last_state = None
            self.training_message = (
                f"ckpt {episode:04d} is incompatible with the current observation shape; retrain on this branch"
            )
            return
        self.current_run_seed = seed
        self.current_mode = "current-learned-ai"
        self.selected_mode = "current-learned-ai"
        self.last_result = None
        self.last_state = self.session.latest_state
        self.paused = False
        self.training_message = f"running trained play from ckpt {episode:04d} on seed {self.format_seed(seed)}"

    def start_play(self) -> None:
        """Run the seed ladder until the first loss, then train on that failed maze."""

        if self.pending_training_seed is not None:
            cycles = self.parse_cycle_count()
            failed_seed = self.pending_training_seed
            self.pending_training_seed = None
            self.start_training(cycles, fixed_maze_seed=failed_seed)
            if self.training_status == "running":
                self.training_message = (
                    f"training starts on failed seed {self.format_seed(failed_seed)} "
                    f"for {cycles} cycles"
                )
            return

        self.seed_ladder_active = True
        start_seed = self.parse_seed()
        if self.selected_checkpoint is None:
            self._start_baseline_legal_mover(seed=start_seed, preserve_seed_ladder=True)
            if self.session is not None:
                self.training_message = f"seed ladder started at {self.format_seed(start_seed)} using innate policy"
            return
        self._start_current_ai_run(seed=start_seed, preserve_seed_ladder=True)
        if self.session is not None:
            self.training_message = f"seed ladder started at {self.format_seed(start_seed)} using the active checkpoint"

    def start_marks_play(self) -> None:
        """Backward-compatible alias for the plain-English Play action."""

        self.start_play()

    def replay_last_run(self) -> None:
        """Replay the previous recorded event stream exactly."""

        if not self.can_start_run:
            self.training_message = "finish training or the current run first"
            return
        if self.last_recorded_run is None:
            self.training_message = "nothing recorded to replay"
            return
        self.seed_ladder_active = False
        self._clear_compare_state()
        self.session = RecordedPlaybackSession(recorded_run=self.last_recorded_run)
        self.current_mode = "replay"
        self.selected_mode = "replay"
        self.last_result = None
        self.last_state = self.session.latest_state
        self.paused = False
        self.training_message = "replaying last recorded run"

    def start_compare_milestones(self) -> None:
        """Queue milestone checkpoints for sequential in-app comparison."""

        if not self.can_start_run:
            self.training_message = "finish training or the current run first"
            return
        self.seed_ladder_active = False
        seed = self.parse_seed()
        self._clear_compare_state()
        self.compare_results = []
        self.compare_queue = []
        for episode in MILESTONE_EPISODES:
            checkpoint_path = resolve_checkpoint_path(self.active_checkpoint_dir, episode)
            self.compare_queue.append(CompareRunEntry(checkpoint_episode=episode, checkpoint_path=checkpoint_path))
        self.current_mode = "compare-milestones"
        self.selected_mode = "compare-milestones"
        self.last_result = None
        self.paused = False
        self.compare_pause_until = 0.0
        if not self.compare_queue:
            self.training_message = "compare finished: no milestone checkpoints found"
            return
        self.training_message = f"comparing milestones on seed {seed}"
        self._start_next_compare_run()

    def pause(self) -> None:
        """Pause the active replay."""

        self.paused = True

    def resume(self) -> None:
        """Resume the active replay."""

        if self.session is not None:
            self.paused = False

    def reset(self) -> None:
        """Clear current playback and compare state."""

        self.session = None
        self.current_run_seed = None
        self.last_state = None
        self.last_result = None
        self.paused = False
        self.seed_ladder_active = False
        self.pending_training_seed = None
        self.current_mode = "idle"
        self._clear_compare_state()
        self.training_message = "reset"

    def step_once(self) -> None:
        """Advance one playback micro-step."""

        if self.session is None:
            if self.is_training_active:
                return
            self.start_selected_mode()
            self.paused = True
        if self.session is None:
            return
        self._advance_session()
        self.paused = True

    def start_training(self, additional_episodes: int, fixed_maze_seed: int | None = None) -> None:
        """Start background training from the latest checkpoint in the selected training mode."""

        if self.is_training_active or self.session is not None:
            self.training_message = "stop the current run before training"
            return
        self.seed_ladder_active = False
        self.last_training_increment = additional_episodes
        self.active_training_seed_override = fixed_maze_seed
        self.last_requested_training_seed = fixed_maze_seed
        self.pending_training_seed = None
        self.training_stop_event = threading.Event()
        self.training_status = "running"
        self.training_error = None
        self.training_progress = None
        self.current_mode = "training"
        if fixed_maze_seed is None:
            self.training_message = f"training +{additional_episodes} cycles in {self.training_mode}"
        else:
            self.training_message = (
                f"training starts on failed seed {self.format_seed(fixed_maze_seed)} then jumps forward by random 1..1000"
            )

        def _report_progress(progress: dict[str, Any]) -> None:
            self.training_progress = progress
            self.training_message = format_training_progress(progress)

        training_kwargs = {
            "additional_episodes": additional_episodes,
            "checkpoint_dir": self.active_checkpoint_dir,
            "training_mode": self.training_mode,
            "stop_event": self.training_stop_event,
        }
        if (
            fixed_maze_seed is not None
            and "fixed_maze_seed" in inspect.signature(continue_training_from_latest).parameters
        ):
            training_kwargs["fixed_maze_seed"] = fixed_maze_seed
        if "progress_callback" in inspect.signature(continue_training_from_latest).parameters:
            training_kwargs["progress_callback"] = _report_progress

        def _worker() -> None:
            try:
                continue_training_from_latest(**training_kwargs)
            except (OSError, RuntimeError, ValueError) as error:  # pragma: no cover - surfaced in app status
                self.training_error = str(error)
                self.training_message = f"training error: {error}"

        self.training_thread = threading.Thread(target=_worker, daemon=True)
        self.training_thread.start()

    def continue_training(self) -> None:
        """Continue training with the last requested episode increment."""

        self.start_training(self.last_training_increment, fixed_maze_seed=self.last_requested_training_seed)

    def reset_training(self) -> None:
        """Clear learned checkpoints so the app returns to an untrained state."""

        if self.is_training_active:
            self.training_message = "stop training before resetting learned knowledge"
            return
        if self.session is not None:
            self.training_message = "finish or reset the current run before clearing learned knowledge"
            return

        removed_any = False
        if self.checkpoint_dir.exists():
            for path in self.checkpoint_dir.rglob("*"):
                if path.is_file() and path.name.startswith("ckpt_") and path.suffix in {".zip", ".json"}:
                    path.unlink()
                    removed_any = True

        for stage_name in ("maze_only", "full_monster"):
            stage_dir = self.checkpoint_dir / stage_name
            if stage_dir.exists() and not any(stage_dir.iterdir()):
                shutil.rmtree(stage_dir)

        self.selected_index = 0
        self.last_recorded_run = None
        self.last_result = None
        self.last_state = None
        self.training_error = None
        self.training_status = "idle"
        self.training_progress = None
        self.active_training_seed_override = None
        self.last_requested_training_seed = None
        self.current_mode = "idle"
        self.selected_mode = "baseline-legal-mover"
        self.paused = False
        self.seed_ladder_active = False
        self.current_run_seed = None
        self.last_failed_seed = None
        self.pending_training_seed = None
        self.run_outcomes.clear()
        self.total_runs = 0
        self.total_wins = 0
        self._clear_compare_state()
        self.refresh_checkpoints()
        self.training_message = "cleared all learned checkpoints" if removed_any else "no learned checkpoints to clear"

    def stop_training(self) -> None:
        """Request graceful stop for background training."""

        if self.training_stop_event is None or not self.is_training_active:
            return
        self.training_stop_event.set()
        self.training_status = "stopping"
        self.training_message = "stopping training after the current episode"

    def toggle_training_mode(self) -> None:
        """Switch between simplified maze-only and full-monster training."""

        if self.is_training_active:
            return
        self.training_mode = "full-monster" if self.training_mode == "maze-only" else "maze-only"
        self.refresh_checkpoints()
        self.training_message = f"training mode set to {self.training_mode}"

    def update(self) -> None:
        """Advance playback, compare sequencing, and training state."""

        if self.training_thread is not None and not self.training_thread.is_alive():
            finished_seed = self.active_training_seed_override
            self.training_thread = None
            self.training_stop_event = None
            self.refresh_checkpoints()
            if self.training_error is None:
                self.training_status = "idle"
                if finished_seed is None:
                    self.training_message = "training finished"
                else:
                    self.training_message = f"training finished for seed {self.format_seed(finished_seed)}"
                if self.available_checkpoints:
                    self.selected_mode = "current-learned-ai"
                    self.selected_index = len(self.available_checkpoints) - 1
            else:
                self.training_status = "error"
            self.active_training_seed_override = None

        if self.session is not None and not self.paused:
            self._advance_session()

        if self.current_mode == "compare-milestones" and self.session is None and self.compare_queue:
            if time.monotonic() >= self.compare_pause_until:
                self._start_next_compare_run()

    def _advance_session(self) -> None:
        if self.session is None:
            return
        active_session = self.session
        state, result = active_session.advance()
        self.last_state = state
        if result is None:
            return
        self.last_result = result
        if isinstance(active_session, (PlaybackSession, BaselinePlaybackSession)):
            self.last_recorded_run = active_session.build_recorded_run()
        self.session = None
        self.paused = False
        if self.current_mode == "compare-milestones":
            self.compare_results.append(result)
            if self.compare_queue:
                self.compare_pause_until = time.monotonic() + self.compare_pause_s
                self.training_message = f"compare pause after {result.checkpoint}"
            else:
                self.training_message = "compare milestones finished"
            return

        if self.current_mode in {"baseline-legal-mover", "current-learned-ai"}:
            self._record_run_outcome(result)

        if self.seed_ladder_active and self._handle_seed_ladder_result(result):
            return

        self.training_message = f"run finished: {result.outcome}"

    def _handle_seed_ladder_result(self, result: ShowcaseResult) -> bool:
        """Advance the ladder after a win or train on the first loss."""

        run_seed = result.seed if result.seed is not None else self.current_run_seed
        if run_seed is None:
            self.seed_ladder_active = False
            return False

        if result.outcome == "escaped":
            next_seed = run_seed + 1
            self.set_seed_value(next_seed)
            if self.selected_checkpoint is None:
                self._start_baseline_legal_mover(seed=next_seed, preserve_seed_ladder=True)
            else:
                self._start_current_ai_run(seed=next_seed, preserve_seed_ladder=True)
            if self.session is not None:
                self.training_message = (
                    f"seed {self.format_seed(run_seed)} cleared, advancing to {self.format_seed(next_seed)}"
                )
                return True
            self.seed_ladder_active = False
            return False

        self.seed_ladder_active = False
        self.last_failed_seed = run_seed
        self.pending_training_seed = run_seed
        self.set_seed_value(run_seed)
        self.training_message = (
            f"seed {self.format_seed(run_seed)} lost via {result.outcome}; "
            "review the run, choose cycles, then press Play to train"
        )
        return True

    def _record_run_outcome(self, result: ShowcaseResult) -> None:
        """Track app-level performance for the latest real play result."""

        self.total_runs += 1
        if result.outcome == "escaped":
            self.total_wins += 1
        self.run_outcomes.append(result.outcome)
        if result.outcome != "escaped":
            self.last_failed_seed = result.seed if result.seed is not None else self.current_run_seed

    def _outcome_window(self, window_size: int | None) -> list[str]:
        outcomes = list(self.run_outcomes)
        if window_size is None:
            return outcomes
        return outcomes[-window_size:]

    def _build_outcome_stat_card(self, label: str, window_size: int) -> TrainingStatCard:
        outcomes = self._outcome_window(window_size)
        cycles = len(outcomes)
        wins = sum(1 for item in outcomes if item == "escaped")
        losses = cycles - wins
        percentage = wins / cycles if cycles else 0.0
        return TrainingStatCard(label=label, cycles=cycles, wins=wins, losses=losses, percentage=percentage)

    def _start_next_compare_run(self) -> None:
        while self.compare_queue:
            entry = self.compare_queue.pop(0)
            if not checkpoint_is_complete(entry.checkpoint_path):
                self.compare_results.append(build_missing_result(entry.checkpoint_episode, entry.checkpoint_path, self.parse_seed()))
                continue
            try:
                self.session = PlaybackSession(
                    checkpoint_path=entry.checkpoint_path,
                    checkpoint_label=f"ckpt {entry.checkpoint_episode:04d}",
                    seed=self.parse_seed(),
                    debug_trace=self.debug_trace,
                    allow_policy_override=True,
                )
            except CheckpointCompatibilityError as error:
                self.compare_results.append(
                    build_incompatible_result(
                        checkpoint_label=f"ckpt {entry.checkpoint_episode:04d}",
                        checkpoint_path=entry.checkpoint_path,
                        seed=self.parse_seed(),
                        reason=str(error),
                    )
                )
                continue
            self.last_state = self.session.latest_state
            self.last_result = None
            self.training_message = f"compare running ckpt {entry.checkpoint_episode:04d}"
            return
        self.training_message = "compare milestones finished"

    def _clear_compare_state(self) -> None:
        self.compare_queue = []
        self.compare_results = []
        self.compare_pause_until = 0.0

    def _maze_config_for_active_run(self) -> MazeConfig:
        """Resolve maze configuration for baseline playback."""

        selected = self.selected_checkpoint
        if selected is not None:
            metadata = load_checkpoint_metadata(selected[1])
            return maze_config_from_dict(metadata["maze_config"])
        latest = latest_checkpoint(self.active_checkpoint_dir)
        if latest is not None:
            metadata = load_checkpoint_metadata(latest[1])
            return maze_config_from_dict(metadata["maze_config"])
        return maze_config_for_training_mode(MazeConfig(), self.training_mode)

    def mode_label(self) -> str:
        if self.seed_ladder_active:
            return "Seed Ladder"
        labels = {
            "idle": "Idle",
            "baseline-legal-mover": "Innate",
            "current-learned-ai": "Trained",
            "training": "Training",
            "replay": "Replay",
            "compare-milestones": "Compare Milestones",
        }
        return labels.get(self.current_mode, self.current_mode)

    def training_mode_label(self) -> str:
        return "Maze-only learning" if self.training_mode == "maze-only" else "Full monster mode"

    def selected_mode_label(self) -> str:
        labels = {
            "baseline-legal-mover": "Play Innate",
            "current-learned-ai": "Play Trained",
            "replay": "Replay Last Run",
            "compare-milestones": "Auto Compare Milestones",
        }
        return labels.get(self.selected_mode, self.selected_mode)

    def monster_visibility_label(self) -> str:
        """Describe whether monster pressure is active in the current stage."""

        if self.training_mode == "maze-only":
            return "Monster: slow chase (moves every 3rd step)"
        return "Monster pressure ON in full monster mode"

    def compare_progress_label(self) -> str:
        total = len(MILESTONE_EPISODES)
        completed = len(self.compare_results)
        running = 1 if self.current_mode == "compare-milestones" and self.session is not None else 0
        return f"{completed + running}/{total}"

    def has_marks_policy(self) -> bool:
        """Return whether a trained checkpoint is available for Play."""

        return self.selected_checkpoint is not None

    def play_mode_status(self) -> str:
        """Return the small plain-English mode label shown near Play."""

        selected = self.selected_checkpoint
        if selected is None:
            return "Policy: Innate"
        return f"Policy: Trained (ckpt {selected[0]:04d})"

    def all_time_training_card(self) -> TrainingStatCard:
        """Return the compact all-time training summary for the Basic tab."""

        cycles = self.total_runs
        wins = self.total_wins
        losses = max(0, cycles - wins)
        percentage = wins / cycles if cycles else 0.0
        return TrainingStatCard(label="All Runs", cycles=cycles, wins=wins, losses=losses, percentage=percentage)

    def recent_10_outcomes(self) -> list[str]:
        """Return the latest ten outcomes for the compact recent row."""

        return self._outcome_window(10)

    def training_stat_cards(self) -> list[TrainingStatCard]:
        """Return app run outcome cards for last 10, 100, and 1000 seeds."""

        return [
            self._build_outcome_stat_card("Last 10", 10),
            self._build_outcome_stat_card("Last 100", 100),
            self._build_outcome_stat_card("Last 1000", 1000),
        ]

    def primary_status_lines(self) -> list[str]:
        """Return concise status lines for the basic tab."""

        latest_summary = self.latest_training_summary()
        last_outcome = self.last_result.outcome if self.last_result is not None else "none yet"
        cycles_seen = 0 if latest_summary is None else int(latest_summary.get("episodes_seen", 0))
        active_seed = self.active_training_seed()
        pinned_seed = self.last_requested_training_seed
        seed_suffix = "" if active_seed is None else f" | Live train seed: {self.format_seed(active_seed)}"
        pinned_suffix = "dynamic" if pinned_seed is None else self.format_seed(pinned_seed)
        return [
            f"Start seed: {self.format_seed(self.parse_seed())} | Train cycles: {self.parse_cycle_count()}",
            f"{self.play_mode_status()} | Last outcome: {last_outcome}",
            f"Ladder: {'running' if self.seed_ladder_active else 'idle'} | Last failed seed: {self.format_seed(self.last_failed_seed) if self.last_failed_seed is not None else 'none'}",
            f"Training armed: {self.format_seed(self.pending_training_seed) if self.pending_training_seed is not None else 'no'}",
            f"Training mode: {self.training_mode_label()} | Focus seed: {pinned_suffix}",
            f"Training status: {self.training_status} | Cycles completed: {cycles_seen}",
            f"Status: {self.training_message}{seed_suffix}",
        ]

    def training_progress_summary(self) -> str:
        """Return one compact line for the current training progress."""

        progress = self.training_progress
        if not progress:
            return "Training progress: idle"

        session_done = int(progress.get("session_completed_episodes", 0))
        session_total = max(1, int(progress.get("session_target_episodes", 1)))
        percent = 100.0 * session_done / session_total
        active_cycle = int(progress.get("active_cycle", 0))
        episode_steps = int(progress.get("episode_steps", 0))
        parts = [f"Session {session_done}/{session_total} ({percent:.0f}%) | cycle {active_cycle} move {episode_steps}"]

        maze_seed = progress.get("maze_seed")
        if maze_seed is not None:
            parts.append(f"seed {self.format_seed(int(maze_seed))}")

        elapsed = progress.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)):
            minutes = int(elapsed) // 60
            seconds = int(elapsed) % 60
            parts.append(f"time {minutes}m{seconds:02d}s")

        average_seconds = progress.get("average_seconds_per_cycle")
        if isinstance(average_seconds, (int, float)):
            parts.append(f"avg {average_seconds:.1f}s/cycle")

        eta_seconds = progress.get("estimated_remaining_seconds")
        if isinstance(eta_seconds, (int, float)):
            parts.append(f"eta {eta_seconds / 60.0:.1f}m")

        return " | ".join(parts)

    def active_training_seed(self) -> int | None:
        """Return the current live training seed when a training cycle is active."""

        progress = self.training_progress
        if not progress:
            return None
        maze_seed = progress.get("maze_seed")
        return int(maze_seed) if maze_seed is not None else None

    def training_preview_state(self) -> dict[str, Any] | None:
        """Return the live training snapshot to render while training is active."""

        if not self.is_training_active:
            return None
        progress = self.training_progress
        if not progress:
            return None
        snapshot = progress.get("state_snapshot")
        if not isinstance(snapshot, dict):
            return None
        preview = dict(snapshot)
        label_seed = self.active_training_seed()
        if label_seed is None:
            preview["checkpoint_label"] = f"training {self.training_mode}"
        else:
            preview["checkpoint_label"] = f"training seed {self.format_seed(label_seed)}"
        preview["policy_decision_label"] = "live training maze"
        preview["policy_kind"] = "innate"
        return preview

    def render_state(self) -> dict[str, Any] | None:
        """Return the state that should currently drive the main maze panel."""

        training_preview = self.training_preview_state()
        if training_preview is not None:
            return training_preview
        return self.last_state

    def training_progress_ratio(self) -> float:
        """Return a 0..1 completion ratio for the current training session."""

        progress = self.training_progress
        if not progress:
            return 0.0
        session_completed = int(progress.get("session_completed_episodes", 0))
        session_total = max(1, int(progress.get("session_target_episodes", 1)))
        return max(0.0, min(1.0, session_completed / session_total))

    def summary_lines(self) -> list[str]:
        """Return compact app summary lines for the side panel."""

        lines: list[str] = []
        cards = self.training_stat_cards()
        if self.total_runs:
            lines.extend(
                [
                    f"Seed ladder wins: {self.total_wins}/{self.total_runs} ({(self.total_wins / self.total_runs) * 100:.0f}%)",
                    f"Recent win %: 10={cards[0].percentage * 100:.0f}% | 100={cards[1].percentage * 100:.0f}% | 1000={cards[2].percentage * 100:.0f}%",
                ]
            )
        latest_summary = self.latest_training_summary()
        if latest_summary is not None:
            lines.extend(
                [
                    f"Exit rate: {latest_summary.get('win_rate', 0.0):.2f} | Coverage: {latest_summary.get('recent_average_coverage', 0.0):.2f}",
                    f"Discovered: {latest_summary.get('recent_average_discovered_cells', 0.0):.1f} | Frontier count: {latest_summary.get('recent_frontier_expansion_count', 0.0):.1f}",
                    f"Frontier rate: {latest_summary.get('recent_frontier_reached_rate', 0.0):.2f} | Illegal moves: {latest_summary.get('recent_average_illegal_moves', 0.0):.1f}",
                    f"Revisits: {latest_summary.get('recent_average_revisits', 0.0):.1f} | Oscillations: {latest_summary.get('recent_average_oscillations', 0.0):.1f}",
                    f"Timeouts: {latest_summary.get('timeout_count', 0)} | Stalls: {latest_summary.get('stall_count', 0)}",
                    f"Avoidable captures: {latest_summary.get('avoidable_capture_count', 0)} | Recent avoidable rate: {latest_summary.get('recent_avoidable_capture_rate', 0.0):.2f}",
                ]
            )
        if self.last_result is not None:
            lines.extend(
                [
                    f"Last outcome: {self.last_result.outcome}",
                    f"Coverage: {self.last_result.coverage:.2f} | Steps: {self.last_result.steps}",
                    f"Discovered cells: {self.last_result.final_state.get('discovered_cells', 0) if self.last_result.final_state else 0} | Frontier: {self.last_result.frontier_rate:.2f}",
                    f"Oscillations: {self.last_result.oscillations} | Revisits: {self.last_result.revisits} | Repeat streak: {self.last_state.get('repeat_move_streak', 0) if self.last_state else 0}",
                    f"Illegal/blocked: {self.last_state.get('blocked_moves', 0) if self.last_state else 0}",
                    f"Exit/caught: {self.last_result.outcome == 'escaped'} / {self.last_result.outcome == 'caught'}",
                ]
            )
        if self.compare_results:
            lines.append("Compare milestones:")
            for item in self.compare_results[-6:]:
                lines.append(f"{item.checkpoint}: {item.status}/{item.outcome} cov={item.coverage:.2f} steps={item.steps}")
        return lines or ["No run summary yet."]

    def review_lines(self) -> list[str]:
        """Return compact review information for replay and compare tabs."""

        lines: list[str] = []
        if self.total_runs:
            lines.append(
                f"Seed ladder: {self.total_wins}/{self.total_runs} wins | latest failed seed {self.format_seed(self.last_failed_seed) if self.last_failed_seed is not None else 'none'}"
            )
        if self.last_result is not None:
            lines.extend(
                [
                    f"Last run: {self.last_result.outcome} via {self.last_result.checkpoint}",
                    f"Coverage {self.last_result.coverage:.2f} | Steps {self.last_result.steps} | Revisits {self.last_result.revisits}",
                    f"Oscillations {self.last_result.oscillations} | Frontier rate {self.last_result.frontier_rate:.2f}",
                ]
            )
        if self.compare_results:
            lines.append(f"Compare progress: {self.compare_progress_label()}")
            for item in self.compare_results[-4:]:
                lines.append(f"{item.checkpoint}: {item.status} / {item.outcome} / cov {item.coverage:.2f}")
        if not lines:
            return ["Run a game, replay it, or compare milestones from the Review tab."]
        return lines

    def latest_training_summary(self) -> dict[str, Any] | None:
        """Load the latest checkpoint training summary for the active mode."""

        latest = latest_checkpoint(self.active_checkpoint_dir)
        if latest is None:
            return None
        try:
            metadata = load_checkpoint_metadata(latest[1])
        except FileNotFoundError:
            return None
        summary = metadata.get("training_summary")
        return summary if isinstance(summary, dict) else None

    def _active_policy_label(self) -> str:
        """Describe the current play mode in plain English."""

        return self.play_mode_status()

    def _build_training_stat_card(
        self,
        label: str,
        summary: dict[str, Any] | None,
        window_size: int | None,
    ) -> TrainingStatCard:
        if summary is None:
            return TrainingStatCard(label=label, cycles=0, wins=0, losses=0, percentage=0.0)
        if window_size is None:
            cycles = int(summary.get("episodes_seen", 0))
            wins = int(summary.get("wins", 0))
        else:
            outcomes = self._summary_outcomes(summary, window_size)
            cycles = len(outcomes)
            wins = sum(1 for item in outcomes if item == "escaped")
        losses = max(0, cycles - wins)
        percentage = wins / cycles if cycles else 0.0
        return TrainingStatCard(label=label, cycles=cycles, wins=wins, losses=losses, percentage=percentage)

    @staticmethod
    def _summary_outcomes(summary: dict[str, Any], window_size: int) -> list[str]:
        if window_size == 10:
            outcomes = summary.get("recent_10_outcomes")
            if isinstance(outcomes, list):
                return [str(item) for item in outcomes]
        if window_size == 50:
            outcomes = summary.get("recent_50_outcomes", summary.get("recent_outcomes"))
            if isinstance(outcomes, list):
                return [str(item) for item in outcomes][-window_size:]
        outcomes = summary.get("recent_outcomes")
        if isinstance(outcomes, list):
            return [str(item) for item in outcomes][-window_size:]
        return []


class LabControlApp:
    """Pygame control panel app for training and watching the human AI."""

    def __init__(self, checkpoint_dir: str | Path = "checkpoints", auto_quit_ms: int | None = None) -> None:
        self.controller = LabAppController(checkpoint_dir=checkpoint_dir)
        self.auto_quit_ms = auto_quit_ms
        self.window_size = (1460, 900)
        self.game_area = pygame.Rect(24, 24, 884, 852)
        self.panel_area = pygame.Rect(928, 24, 508, 852)
        self.seed_input_rect = pygame.Rect(0, 0, 0, 0)
        self.cycle_input_rect = pygame.Rect(0, 0, 0, 0)
        self.active_input: str | None = None
        self.active_tab = "basic"
        self.buttons: list[Button] = []
        self._running = False
        self.font: Any = None
        self.small_font: Any = None
        self.heading_font: Any = None
        self.title_font: Any = None

    def run(self) -> None:
        """Open the control app loop."""

        pygame.display.init()
        pygame.font.init()
        screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Maze RL Lab")
        self.font = pygame.font.SysFont("bahnschrift", 18)
        self.small_font = pygame.font.SysFont("bahnschrift", 15)
        self.heading_font = pygame.font.SysFont("bahnschrift", 24, bold=True)
        self.title_font = pygame.font.SysFont("cambria", 32, bold=True)
        clock = pygame.time.Clock()
        self._running = True
        started_at = time.monotonic()

        while self._running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self._running = False
                elif event.type == MOUSEBUTTONDOWN:
                    self._handle_click(event.pos)
                elif event.type == KEYDOWN:
                    self._handle_key(event)

            if self.auto_quit_ms is not None and (time.monotonic() - started_at) * 1000 >= self.auto_quit_ms:
                self._running = False

            self.controller.update()
            self.buttons = self.build_buttons()
            self._draw(screen)
            pygame.display.flip()
            clock.tick(self.controller.fps if self.controller.session is not None else 30)

        pygame.display.quit()
        pygame.font.quit()

    def build_buttons(self) -> list[Button]:
        """Return the currently visible buttons for the active tab."""

        return self._build_buttons()

    def visible_button_labels(self) -> list[str]:
        """Return the currently visible button labels for lightweight UI tests."""

        return [button.label for button in self.build_buttons()]

    def _handle_click(self, position: tuple[int, int]) -> None:
        if self.seed_input_rect.collidepoint(position):
            self.active_input = "seed"
            return
        if self.cycle_input_rect.collidepoint(position):
            self.active_input = "cycles"
            return
        if self.active_input == "seed":
            self.controller.set_seed_value(self.controller.parse_seed())
        self.active_input = None
        for button in self.buttons:
            if button.enabled and button.rect.collidepoint(position):
                button.action()
                return

    def _handle_key(self, event: pygame.event.Event) -> None:
        if self.active_input is None:
            return
        if event.key == K_BACKSPACE:
            if self.active_input == "seed":
                self.controller.seed_text = self.controller.seed_text[:-1]
            else:
                self.controller.cycle_input_text = self.controller.cycle_input_text[:-1]
        elif event.key == K_RETURN:
            if self.active_input == "seed":
                self.controller.set_seed_value(self.controller.parse_seed())
            self.active_input = None
        elif event.unicode.isdigit():
            if self.active_input == "seed" and len(self.controller.seed_text) < 10:
                if self.controller.seed_text in {"", "00001"}:
                    self.controller.seed_text = event.unicode
                else:
                    self.controller.seed_text += event.unicode
            if self.active_input == "cycles" and len(self.controller.cycle_input_text) < 6:
                self.controller.cycle_input_text += event.unicode

    def _panel_layout(self) -> dict[str, pygame.Rect]:
        panel_x = self.panel_area.x + 18
        panel_width = self.panel_area.width - 36
        return {
            "tabs": pygame.Rect(panel_x, self.panel_area.y + 76, panel_width, 42),
            "content": pygame.Rect(panel_x, self.panel_area.y + 134, panel_width, self.panel_area.height - 154),
        }

    def _basic_tab_sections(self, rect: pygame.Rect) -> dict[str, pygame.Rect]:
        gap = 10
        controls_height = 214
        stats_height = 188
        progress_height = 116
        top = rect.y
        controls_rect = pygame.Rect(rect.x, top, rect.width, controls_height)
        top = controls_rect.bottom + gap
        stats_rect = pygame.Rect(rect.x, top, rect.width, stats_height)
        top = stats_rect.bottom + gap
        progress_rect = pygame.Rect(rect.x, top, rect.width, progress_height)
        top = progress_rect.bottom + gap
        status_rect = pygame.Rect(rect.x, top, rect.width, max(100, rect.bottom - top))
        return {
            "controls": controls_rect,
            "stats": stats_rect,
            "progress": progress_rect,
            "status": status_rect,
        }

    def _review_tab_sections(self, rect: pygame.Rect) -> dict[str, pygame.Rect]:
        gap = 14
        review_height = 214
        summary_height = 202
        review_rect = pygame.Rect(rect.x, rect.y, rect.width, review_height)
        summary_top = review_rect.bottom + gap
        summary_rect = pygame.Rect(rect.x, summary_top, rect.width, summary_height)
        detail_top = summary_rect.bottom + gap
        detail_rect = pygame.Rect(rect.x, detail_top, rect.width, max(150, rect.bottom - detail_top))
        return {
            "review": review_rect,
            "summary": summary_rect,
            "detail": detail_rect,
        }

    def _advanced_tab_sections(self, rect: pygame.Rect) -> dict[str, pygame.Rect]:
        gap = 14
        control_height = 172
        status_height = 196
        control_rect = pygame.Rect(rect.x, rect.y, rect.width, control_height)
        status_top = control_rect.bottom + gap
        status_rect = pygame.Rect(rect.x, status_top, rect.width, status_height)
        summary_top = status_rect.bottom + gap
        summary_rect = pygame.Rect(rect.x, summary_top, rect.width, max(180, rect.bottom - summary_top))
        return {
            "control": control_rect,
            "status": status_rect,
            "summary": summary_rect,
        }

    @staticmethod
    def _card_body_rect(rect: pygame.Rect, inset_x: int = 14, inset_bottom: int = 14) -> pygame.Rect:
        return pygame.Rect(
            rect.x + inset_x,
            rect.y + CARD_HEADER_HEIGHT,
            rect.width - inset_x * 2,
            max(0, rect.height - CARD_HEADER_HEIGHT - inset_bottom),
        )

    def _build_buttons(self) -> list[Button]:
        layout = self._panel_layout()
        buttons: list[Button] = []
        row_height = 34
        gap = 10
        tabs = layout["tabs"]
        tab_width = (tabs.width - 2 * gap) // 3
        tab_names = [("basic", "Basic"), ("review", "Review"), ("advanced", "Advanced")]
        for index, (tab_key, tab_label) in enumerate(tab_names):
            buttons.append(
                Button(
                    tab_label,
                    pygame.Rect(tabs.x + index * (tab_width + gap), tabs.y, tab_width, tabs.height),
                    lambda key=tab_key: self._set_tab(key),
                    kind="tab",
                    active=self.active_tab == tab_key,
                )
            )

        content = layout["content"]
        if self.active_tab == "basic":
            sections = self._basic_tab_sections(content)
            controls_rect = sections["controls"]
            body_rect = self._card_body_rect(controls_rect)
            left = body_rect.x
            gap = 6
            play_width = 146
            stop_width = 78
            small_width = (body_rect.width - play_width - stop_width - 3 * gap) // 2
            button_y = body_rect.y + 96
            buttons.extend(
                [
                    Button(
                        "Play",
                        pygame.Rect(left, button_y, play_width, 40),
                        self.controller.start_play,
                        enabled=self.controller.can_start_run,
                        kind="primary",
                    ),
                    Button(
                        "Stop",
                        pygame.Rect(left + play_width + gap, button_y, stop_width, 40),
                        self.controller.stop_training,
                        enabled=self.controller.is_training_active,
                        kind="danger",
                    ),
                    Button(
                        "Train Seed",
                        pygame.Rect(left + play_width + gap + stop_width + gap, button_y, small_width, 40),
                        lambda: self.controller.start_training(
                            self.controller.parse_cycle_count(),
                            fixed_maze_seed=self.controller.parse_seed(),
                        ),
                        enabled=not self.controller.is_training_active and self.controller.session is None,
                        kind="accent",
                    ),
                    Button(
                        "Reset Model",
                        pygame.Rect(left + play_width + gap + stop_width + gap + small_width + gap, button_y, small_width, 40),
                        self.controller.reset_training,
                        enabled=not self.controller.is_training_active and self.controller.session is None,
                        kind="danger",
                    ),
                ]
            )
        elif self.active_tab == "review":
            sections = self._review_tab_sections(content)
            review_rect = sections["review"]
            buttons.extend(
                [
                    Button(
                        "Pause",
                        pygame.Rect(review_rect.x + 12, review_rect.y + 72, 102, row_height),
                        self.controller.pause,
                        enabled=self.controller.session is not None and not self.controller.paused,
                    ),
                    Button(
                        "Resume",
                        pygame.Rect(review_rect.x + 126, review_rect.y + 72, 102, row_height),
                        self.controller.resume,
                        enabled=self.controller.session is not None and self.controller.paused,
                    ),
                    Button(
                        "Step",
                        pygame.Rect(review_rect.x + 240, review_rect.y + 72, 102, row_height),
                        self.controller.step_once,
                        enabled=not self.controller.is_training_active,
                    ),
                    Button(
                        "Reset Run",
                        pygame.Rect(review_rect.x + 354, review_rect.y + 72, 110, row_height),
                        self.controller.reset,
                        enabled=self.controller.session is not None or self.controller.last_state is not None,
                    ),
                    Button(
                        "Replay Last Run",
                        pygame.Rect(review_rect.x + 12, review_rect.y + 122, 220, 40),
                        self.controller.replay_last_run,
                        enabled=self.controller.can_start_run and self.controller.last_recorded_run is not None,
                        kind="primary",
                    ),
                    Button(
                        "Compare Milestones",
                        pygame.Rect(review_rect.x + 244, review_rect.y + 122, 220, 40),
                        self.controller.start_compare_milestones,
                        enabled=self.controller.can_start_run,
                        kind="accent",
                    ),
                    Button(
                        "Previous Ckpt",
                        pygame.Rect(review_rect.x + 12, review_rect.y + 170, 140, row_height),
                        lambda: self.controller.cycle_checkpoint(-1),
                        enabled=bool(self.controller.available_checkpoints) and self.controller.can_start_run,
                    ),
                    Button(
                        "Next Ckpt",
                        pygame.Rect(review_rect.x + 164, review_rect.y + 170, 140, row_height),
                        lambda: self.controller.cycle_checkpoint(1),
                        enabled=bool(self.controller.available_checkpoints) and self.controller.can_start_run,
                    ),
                    Button(
                        "Latest",
                        pygame.Rect(review_rect.x + 316, review_rect.y + 170, 148, row_height),
                        self.controller.use_latest_checkpoint,
                        enabled=bool(self.controller.available_checkpoints) and self.controller.can_start_run,
                    ),
                ]
            )
        else:
            sections = self._advanced_tab_sections(content)
            control_rect = sections["control"]
            buttons.extend(
                [
                    Button(
                        "Continue Training",
                        pygame.Rect(control_rect.x + 12, control_rect.y + 72, 172, 40),
                        self.controller.continue_training,
                        enabled=not self.controller.is_training_active and self.controller.session is None,
                        kind="accent",
                    ),
                    Button(
                        "Stop Training",
                        pygame.Rect(control_rect.x + 196, control_rect.y + 72, 132, 40),
                        self.controller.stop_training,
                        enabled=self.controller.is_training_active,
                    ),
                    Button(
                        "Toggle Train Mode",
                        pygame.Rect(control_rect.x + 340, control_rect.y + 72, 124, 40),
                        self.controller.toggle_training_mode,
                        enabled=not self.controller.is_training_active,
                    ),
                    Button(
                        "Watch Innate",
                        pygame.Rect(control_rect.x + 12, control_rect.y + 124, 140, row_height),
                        self.controller.start_baseline_legal_mover,
                        enabled=self.controller.can_start_run,
                    ),
                    Button(
                        "Play Trained",
                        pygame.Rect(control_rect.x + 164, control_rect.y + 124, 140, row_height),
                        self.controller.start_current_ai_run,
                        enabled=self.controller.can_start_run and self.controller.has_marks_policy(),
                    ),
                    Button(
                        "Step Once",
                        pygame.Rect(control_rect.x + 316, control_rect.y + 124, 148, row_height),
                        self.controller.step_once,
                        enabled=not self.controller.is_training_active,
                    ),
                ]
            )
        return buttons

    def _draw(self, screen: Any) -> None:
        screen.fill(APP_BG)
        pygame.draw.circle(screen, DECOR_WARM, (96, 108), 156)
        pygame.draw.circle(screen, DECOR_SOFT, (1330, 148), 132)
        pygame.draw.circle(screen, DECOR_COOL, (1236, 734), 172)
        self._draw_surface_panel(screen, self.game_area, SURFACE_MAIN)
        self._draw_surface_panel(screen, self.panel_area, SURFACE_PANEL)
        self._draw_game_area(screen)
        self._draw_panel(screen)

    def _draw_game_area(self, screen: Any) -> None:
        inner = self.game_area.inflate(-18, -18)
        pygame.draw.rect(screen, SURFACE_CARD, inner, border_radius=24)
        state = self.controller.render_state()
        if state is None or not state.get("grid"):
            title = self.title_font.render("Seed Ladder Ready", True, TEXT_PRIMARY)
            screen.blit(title, (self.game_area.x + 30, self.game_area.y + 30))
            subtitle = self.font.render(
                "Play starts at the chosen seed, advances on every win, and trains the first maze that beats the policy.",
                True,
                TEXT_SECONDARY,
            )
            screen.blit(subtitle, (self.game_area.x + 30, self.game_area.y + 72))
            tip_lines = [
                "Use the basic tab for the production flow: set a start seed and a train count, then let the app work.",
                "Review and Advanced keep replay, comparison, and checkpoint controls available without crowding the main surface.",
            ]
            self._draw_wrapped_text(
                screen,
                self.heading_font,
                tip_lines,
                TEXT_SECONDARY,
                pygame.Rect(self.game_area.x + 30, self.game_area.y + 146, self.game_area.width - 60, 120),
                line_gap=8,
            )
            return

        grid = viewer_grid(state)
        visible_cells = viewer_visible_cells(state)
        explored_cells = viewer_explored_cells(state)
        traveled_cells = viewer_traveled_cells(state)
        dead_end_cells = viewer_dead_end_cells(state)
        rows = len(grid)
        cols = len(grid[0])
        cell_size = min((self.game_area.width - 64) // cols, (self.game_area.height - 224) // rows)
        offset_x = self.game_area.x + 30
        offset_y = self.game_area.y + 30
        board_rect = pygame.Rect(offset_x, offset_y, cols * cell_size, rows * cell_size)

        for row_index, row in enumerate(grid):
            for col_index, cell in enumerate(row):
                rect = pygame.Rect(offset_x + col_index * cell_size, offset_y + row_index * cell_size, cell_size - 1, cell_size - 1)
                position = (row_index, col_index)
                color = viewer_cell_color(
                    cell,
                    position in visible_cells,
                    is_explored=position in explored_cells,
                    is_traveled=position in traveled_cells,
                    is_dead_end=position in dead_end_cells,
                )
                pygame.draw.rect(screen, color, rect)

        player_visible = bool(state.get("player_visible", True))
        rendered_player = viewer_player_position(state)
        rendered_monster = viewer_monster_position(state)
        exit_position = viewer_exit_position(state)
        npc_monster_visible = bool(state.get("monster_visible", True))
        if player_visible and rendered_player is not None and rendered_monster is not None and rendered_player == rendered_monster:
            self._draw_overlap_entities(screen, board_rect, cell_size, rendered_player)
        else:
            if player_visible and rendered_player is not None:
                self._draw_entity(screen, board_rect, cell_size, rendered_player, (60, 110, 220), "H", "Human AI")
            if rendered_monster is not None:
                self._draw_entity(screen, board_rect, cell_size, rendered_monster, (205, 60, 60), "M", "Monster")
        if exit_position is not None:
            self._draw_entity(screen, board_rect, cell_size, exit_position, (77, 145, 95), "E", "Exit")

        overlay_rect = pygame.Rect(self.game_area.x + 24, board_rect.bottom + 18, self.game_area.width - 48, self.game_area.bottom - board_rect.bottom - 32)
        pygame.draw.rect(screen, SURFACE_CARD_ALT, overlay_rect, border_radius=18)
        pygame.draw.rect(screen, BORDER, overlay_rect, width=1, border_radius=18)
        badge_label, badge_color, badge_text_color = viewer_policy_badge(state)
        badge_rect = pygame.Rect(overlay_rect.x + 18, overlay_rect.y + 14, min(380, max(190, len(badge_label) * 10)), 30)
        pygame.draw.rect(screen, badge_color, badge_rect, border_radius=10)
        badge_text = self.font.render(badge_label, True, badge_text_color)
        screen.blit(badge_text, (badge_rect.x + 10, badge_rect.y + 5))
        training_preview = self.controller.training_preview_state() is not None
        overlay_lines = [
            f"Mode: {'Training Preview' if training_preview else self.controller.mode_label()} | Seed: {self.controller.format_seed(int(state.get('seed', 1)))} | Checkpoint: {state.get('checkpoint_label')}",
            f"Turn: {state.get('turn_step', state.get('steps'))} | Outcome: {state.get('outcome')} | Coverage: {state.get('coverage', 0.0):.2f} | Score: {state.get('game_score', 0)}",
            f"Human: {rendered_player if player_visible else 'hidden'} | Monster: {rendered_monster} | Exit: {exit_position}",
            f"Seen cells: {state.get('discovered_cells', 0)} | Vision-shaded cells: {len(visible_cells)} | NPC sees monster: {npc_monster_visible} | Exit seen: {state.get('exit_seen', False)}",
            f"Distance: {state.get('player_monster_distance')} | Repeat streak: {state.get('repeat_move_streak', 0)} | Micro-step: {state.get('current_micro_step', 0)}/{state.get('micro_step_count', 0)} | Speed: {self.controller.speed_options[self.controller.speed_index][0]}",
            (
                f"Policy: {state.get('policy_kind', 'trained')} | Override enabled: {state.get('policy_override_enabled', False)} | "
                f"Override count: {state.get('policy_override_count', 0)} | Reason: {state.get('policy_override_reason', 'n/a')}"
            ),
        ]
        if training_preview:
            overlay_lines.append("This is the live maze being used for the current training cycle. It updates as the cycle and seed change.")
        if self.controller.last_result is not None:
            overlay_lines.append(
                f"Last finished run: {self.controller.last_result.outcome} | Steps {self.controller.last_result.steps} | Revisits {self.controller.last_result.revisits} | Oscillations {self.controller.last_result.oscillations} | Peak repeat {state.get('peak_repeat_move_streak', 0)}"
            )
        if self.controller.debug_trace:
            overlay_lines.extend(
                [
                    f"Debug rendered H={state.get('rendered_player_position')} | rendered M={state.get('rendered_monster_position')}",
                    f"Debug committed H={state.get('committed_player_position')} | committed M={state.get('committed_monster_position')}",
                    f"Debug actor={state.get('micro_actor')} | phase={state.get('micro_phase')} | action={state.get('action_index')} dir={state.get('action_direction')} speed={state.get('action_speed')}",
                ]
            )
        self._draw_wrapped_text(screen, self.font, overlay_lines, (57, 63, 75), pygame.Rect(overlay_rect.x + 18, overlay_rect.y + 50, overlay_rect.width - 36, overlay_rect.height - 64), line_gap=4)

    def _draw_panel(self, screen: Any) -> None:
        layout = self._panel_layout()
        clip_rect = self.panel_area.inflate(-10, -10)
        previous_clip = screen.get_clip()
        screen.set_clip(clip_rect)

        title = self.title_font.render("Maze Lab Control", True, TEXT_PRIMARY)
        screen.blit(title, (self.panel_area.x + 18, self.panel_area.y + 18))
        subtitle = self.small_font.render("Professional seed ladder flow with focused retraining on the first failed maze.", True, TEXT_MUTED)
        screen.blit(subtitle, (self.panel_area.x + 18, self.panel_area.y + 52))

        if self.active_tab == "basic":
            self._draw_basic_tab(screen, layout["content"])
        elif self.active_tab == "review":
            self._draw_review_tab(screen, layout["content"])
        else:
            self._draw_advanced_tab(screen, layout["content"])

        for button in self.buttons:
            self._draw_button(screen, button)
        screen.set_clip(previous_clip)

    def _draw_basic_tab(self, screen: Any, rect: pygame.Rect) -> None:
        sections = self._basic_tab_sections(rect)
        controls_rect = sections["controls"]
        stats_rect = sections["stats"]
        progress_rect = sections["progress"]
        status_rect = sections["status"]

        self._draw_card(screen, controls_rect, "Seed Ladder")
        controls_body = self._card_body_rect(controls_rect)
        self.seed_input_rect = pygame.Rect(controls_body.x, controls_body.y + 18, 124, 36)
        self.cycle_input_rect = pygame.Rect(controls_body.x + 146, controls_body.y + 18, 104, 36)
        self._draw_input(screen, self.seed_input_rect, "Start Seed", self.controller.seed_display_text(self.active_input == "seed"), self.active_input == "seed")
        self._draw_input(screen, self.cycle_input_rect, "Cycles", self.controller.cycle_input_text, self.active_input == "cycles")

        mode_surface = self.heading_font.render(self.controller.play_mode_status(), True, PRIMARY_DARK)
        screen.blit(mode_surface, (controls_body.x, controls_body.y + 62))

        self._draw_card(screen, stats_rect, "Run Performance")
        self._draw_basic_training_summary(screen, stats_rect)

        self._draw_card(screen, progress_rect, "Training Progress")
        self._draw_training_progress(screen, progress_rect)

        self._draw_card(screen, status_rect, "Status")
        status_body = self._card_body_rect(status_rect)
        self._draw_wrapped_text(screen, self.font, self.controller.primary_status_lines(), (56, 64, 76), status_body, line_gap=4)

    def _draw_review_tab(self, screen: Any, rect: pygame.Rect) -> None:
        sections = self._review_tab_sections(rect)
        review_rect = sections["review"]
        summary_rect = sections["summary"]
        detail_rect = sections["detail"]

        self._draw_card(screen, review_rect, "Review")
        review_body = self._card_body_rect(review_rect)
        lines = [
            "Review puts the extra tools behind a tab so the default screen stays focused.",
            "Pause, step, replay, and compare checkpoints on the same seed.",
        ]
        self._draw_wrapped_text(screen, self.small_font, lines, (101, 109, 120), pygame.Rect(review_body.x, review_body.y, review_body.width, 32), line_gap=2)

        self._draw_card(screen, summary_rect, "Review Summary")
        summary_body = self._card_body_rect(summary_rect)
        self._draw_wrapped_text(screen, self.font, self.controller.review_lines(), (56, 64, 76), summary_body, line_gap=4)

        self._draw_card(screen, detail_rect, "Run Details")
        detail_body = self._card_body_rect(detail_rect)
        self._draw_wrapped_text(screen, self.small_font, self.controller.summary_lines(), (73, 81, 93), detail_body, line_gap=4)

    def _draw_advanced_tab(self, screen: Any, rect: pygame.Rect) -> None:
        sections = self._advanced_tab_sections(rect)
        control_rect = sections["control"]
        status_rect = sections["status"]
        summary_rect = sections["summary"]

        self._draw_card(screen, control_rect, "Advanced Controls")
        control_body = self._card_body_rect(control_rect)
        lines = [
            f"Training mode: {self.controller.training_mode_label()}",
            self.controller.monster_visibility_label(),
            f"Checkpoint root: {self.controller.active_checkpoint_dir}",
            f"Playback speed: {self.controller.speed_options[self.controller.speed_index][0]} ({self.controller.fps} fps)",
        ]
        self._draw_wrapped_text(screen, self.small_font, lines, (101, 109, 120), control_body, line_gap=4)

        self._draw_card(screen, status_rect, "Checkpoint State")
        status_body = self._card_body_rect(status_rect)
        selected = self.controller.selected_checkpoint
        latest_entry: tuple[int, Path] | None = self.controller.available_checkpoints[-1] if self.controller.available_checkpoints else None
        selected_label = "none" if selected is None else f"ckpt {selected[0]:04d}"
        latest_label = "none"
        if latest_entry is not None:
            latest_episode, _ = latest_entry
            latest_label = f"ckpt {latest_episode:04d}"
        status_lines = [
            f"Selected checkpoint: {selected_label}",
            f"Latest checkpoint: {latest_label}",
            f"Training status: {self.controller.training_status}",
            f"Compare progress: {self.controller.compare_progress_label()}",
            f"Message: {self.controller.training_message}",
        ]
        self._draw_wrapped_text(screen, self.font, status_lines, (56, 64, 76), status_body, line_gap=4)

        self._draw_card(screen, summary_rect, "Telemetry")
        summary_body = self._card_body_rect(summary_rect)
        self._draw_wrapped_text(screen, self.small_font, self.controller.summary_lines(), (73, 81, 93), summary_body, line_gap=4)

    def _draw_training_cards(self, screen: Any, rect: pygame.Rect) -> None:
        cards = self.controller.training_stat_cards()
        gap = 8
        card_width = (rect.width - 28 - 2 * gap) // 3
        top = rect.y
        for index, card in enumerate(cards):
            card_rect = pygame.Rect(rect.x + 14 + index * (card_width + gap), top, card_width, 58)
            pygame.draw.rect(screen, SURFACE_CARD_ALT, card_rect, border_radius=16)
            pygame.draw.rect(screen, BORDER, card_rect, width=1, border_radius=16)
            title = self.small_font.render(card.label, True, TEXT_SECONDARY)
            screen.blit(title, (card_rect.x + 10, card_rect.y + 4))
            percentage = self.heading_font.render(f"{card.percentage * 100:.0f}%", True, TEXT_PRIMARY)
            screen.blit(percentage, (card_rect.x + 10, card_rect.y + 18))
            summary = self.small_font.render(f"{card.cycles} seeds | {card.wins}W {card.losses}L", True, TEXT_SECONDARY)
            screen.blit(summary, (card_rect.x + 10, card_rect.y + 40))

    def _draw_basic_training_summary(self, screen: Any, rect: pygame.Rect) -> None:
        """Draw the ladder win-rate windows and recent seed outcomes."""

        card = self.controller.all_time_training_card()
        body_rect = self._card_body_rect(rect)
        label_color = TEXT_SECONDARY
        value_color = TEXT_PRIMARY
        labels = ["All runs", "Wins", "Losses", "Win %"]
        values = [str(card.cycles), str(card.wins), str(card.losses), f"{card.percentage * 100:.0f}%"]
        column_width = (rect.width - 28) // 4
        for index, (label, value) in enumerate(zip(labels, values, strict=True)):
            item_x = body_rect.x + index * column_width
            label_surface = self.small_font.render(label, True, label_color)
            value_surface = self.heading_font.render(value, True, value_color)
            screen.blit(label_surface, (item_x, body_rect.y))
            screen.blit(value_surface, (item_x, body_rect.y + 18))

        self._draw_training_cards(screen, pygame.Rect(rect.x, body_rect.y + 48, rect.width, 62))

        recent = self.controller.recent_10_outcomes()
        recent_label = self.small_font.render("Last 10", True, label_color)
        recent_y = body_rect.y + 112
        screen.blit(recent_label, (body_rect.x, recent_y))
        if recent:
            symbols = " ".join("W" if item == "escaped" else "L" for item in recent)
            recent_surface = self.small_font.render(symbols, True, value_color)
        else:
            recent_surface = self.small_font.render("No completed seeds yet", True, value_color)
        screen.blit(recent_surface, (body_rect.x + 60, recent_y))

    def _draw_training_progress(self, screen: Any, rect: pygame.Rect) -> None:
        """Draw the live training progress bar and timing hint."""

        progress = self.controller.training_progress
        ratio = self.controller.training_progress_ratio()
        body_rect = self._card_body_rect(rect)
        label_surface = self.small_font.render(self.controller.training_progress_summary(), True, TEXT_PRIMARY)
        screen.blit(label_surface, (body_rect.x, body_rect.y))

        bar_rect = pygame.Rect(body_rect.x, body_rect.y + 20, body_rect.width, 18)
        pygame.draw.rect(screen, SURFACE_CARD_ALT, bar_rect, border_radius=9)
        fill_width = int(bar_rect.width * ratio)
        if fill_width > 0:
            fill_color = PRIMARY
            if progress is not None and progress.get("status") == "no-progress":
                fill_color = WARNING
            fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, max(8, fill_width), bar_rect.height)
            pygame.draw.rect(screen, fill_color, fill_rect, border_radius=9)
        pygame.draw.rect(screen, BORDER, bar_rect, width=1, border_radius=9)

        if progress is None:
            hint = "Start training to watch the bar fill and the cycle age update."
        else:
            session_done = int(progress.get("session_completed_episodes", 0))
            session_total = max(1, int(progress.get("session_target_episodes", 1)))
            elapsed = progress.get("elapsed_seconds")
            elapsed_label = ""
            if isinstance(elapsed, (int, float)):
                minutes = int(elapsed) // 60
                seconds = int(elapsed) % 60
                elapsed_label = f" | elapsed {minutes}m{seconds:02d}s"
            eta_seconds = progress.get("estimated_remaining_seconds")
            eta_label = f" | eta {eta_seconds / 60.0:.1f}m" if isinstance(eta_seconds, (int, float)) else ""
            hint = f"Session {session_done}/{session_total}{elapsed_label}{eta_label} | status {progress.get('status', 'running')}"
        hint_surface = self.small_font.render(hint, True, TEXT_MUTED)
        screen.blit(hint_surface, (body_rect.x, body_rect.y + 42))

    def _draw_input(self, screen: Any, rect: pygame.Rect, label: str, value: str, active: bool) -> None:
        label_surface = self.small_font.render(label, True, TEXT_SECONDARY)
        screen.blit(label_surface, (rect.x, rect.y - 18))
        pygame.draw.rect(screen, SURFACE_CARD, rect, border_radius=10)
        pygame.draw.rect(screen, PRIMARY if active else BORDER, rect, width=2 if active else 1, border_radius=10)
        value_surface = self.font.render(value or "", True, TEXT_PRIMARY)
        screen.blit(value_surface, (rect.x + 10, rect.y + 5))

    def _draw_button(self, screen: Any, button: Button) -> None:
        if button.kind == "tab":
            fill = PRIMARY_DARK if button.active else SURFACE_CARD_ALT
            text_color = (248, 250, 252) if button.active else TEXT_PRIMARY
        elif not button.enabled:
            fill = DISABLED
            text_color = TEXT_MUTED
        elif button.kind == "primary":
            fill = PRIMARY
            text_color = (250, 251, 252)
        elif button.kind == "danger":
            fill = DANGER
            text_color = (251, 248, 242)
        elif button.kind == "accent":
            fill = ACCENT
            text_color = (251, 248, 242)
        elif button.active:
            fill = ACCENT
            text_color = (248, 249, 246)
        else:
            fill = SURFACE_CARD_ALT
            text_color = TEXT_PRIMARY
        pygame.draw.rect(screen, fill, button.rect, border_radius=10)
        pygame.draw.rect(screen, BORDER, button.rect, width=1, border_radius=10)
        text = self.small_font.render(button.label, True, text_color)
        screen.blit(text, (button.rect.centerx - text.get_width() / 2, button.rect.centery - text.get_height() / 2))

    def _draw_card(self, screen: Any, rect: pygame.Rect, title: str) -> None:
        shadow_rect = rect.move(0, 4)
        pygame.draw.rect(screen, (222, 228, 235), shadow_rect, border_radius=18)
        pygame.draw.rect(screen, SURFACE_CARD, rect, border_radius=18)
        pygame.draw.rect(screen, BORDER, rect, width=1, border_radius=18)
        heading = self.heading_font.render(title, True, TEXT_PRIMARY)
        screen.blit(heading, (rect.x + 14, rect.y + 14))

    @staticmethod
    def _draw_surface_panel(screen: Any, rect: pygame.Rect, fill: tuple[int, int, int]) -> None:
        shadow_rect = rect.move(0, 6)
        pygame.draw.rect(screen, (214, 222, 230), shadow_rect, border_radius=28)
        pygame.draw.rect(screen, fill, rect, border_radius=28)
        pygame.draw.rect(screen, BORDER, rect, width=1, border_radius=28)

    def _set_tab(self, tab_key: str) -> None:
        self.active_tab = tab_key
        self.active_input = None

    def _draw_entity(
        self,
        screen: Any,
        board_rect: pygame.Rect,
        cell_size: int,
        position: tuple[int, int],
        color: tuple[int, int, int],
        glyph: str,
        label: str,
    ) -> None:
        rect = pygame.Rect(
            board_rect.x + position[1] * cell_size + 3,
            board_rect.y + position[0] * cell_size + 3,
            cell_size - 6,
            cell_size - 6,
        )
        pygame.draw.rect(screen, color, rect, border_radius=8)
        glyph_text = self.heading_font.render(glyph, True, (255, 255, 255))
        screen.blit(glyph_text, (rect.centerx - glyph_text.get_width() / 2, rect.centery - glyph_text.get_height() / 2))
        label_surface = self.small_font.render(label, True, (248, 249, 250))
        pill_width = label_surface.get_width() + 12
        pill = pygame.Rect(rect.x, max(board_rect.y - 2, rect.y - 22), pill_width, 18)
        if pill.right > board_rect.right:
            pill.x = board_rect.right - pill.width
        pygame.draw.rect(screen, (10, 12, 18), pill, border_radius=8)
        pygame.draw.rect(screen, color, pill, width=1, border_radius=8)
        screen.blit(label_surface, (pill.x + 6, pill.y + 2))

    def _draw_overlap_entities(self, screen: Any, board_rect: pygame.Rect, cell_size: int, position: tuple[int, int]) -> None:
        rect = pygame.Rect(
            board_rect.x + position[1] * cell_size + 3,
            board_rect.y + position[0] * cell_size + 3,
            cell_size - 6,
            cell_size - 6,
        )
        left_half = pygame.Rect(rect.x, rect.y, rect.width // 2, rect.height)
        right_half = pygame.Rect(rect.x + rect.width // 2, rect.y, rect.width - rect.width // 2, rect.height)
        pygame.draw.rect(screen, (60, 110, 220), left_half, border_top_left_radius=8, border_bottom_left_radius=8)
        pygame.draw.rect(screen, (205, 60, 60), right_half, border_top_right_radius=8, border_bottom_right_radius=8)
        left_text = self.heading_font.render("H", True, (255, 255, 255))
        right_text = self.heading_font.render("M", True, (255, 255, 255))
        screen.blit(left_text, (left_half.centerx - left_text.get_width() / 2, rect.centery - left_text.get_height() / 2))
        screen.blit(right_text, (right_half.centerx - right_text.get_width() / 2, rect.centery - right_text.get_height() / 2))
        label_surface = self.small_font.render("Human AI + Monster", True, (248, 249, 250))
        pill_width = label_surface.get_width() + 12
        pill = pygame.Rect(rect.x, max(board_rect.y - 2, rect.y - 22), pill_width, 18)
        if pill.right > board_rect.right:
            pill.x = board_rect.right - pill.width
        pygame.draw.rect(screen, (10, 12, 18), pill, border_radius=8)
        pygame.draw.rect(screen, (205, 60, 60), pill, width=1, border_radius=8)
        screen.blit(label_surface, (pill.x + 6, pill.y + 2))

    def _draw_wrapped_text(
        self,
        screen: Any,
        font: Any,
        lines: list[str],
        color: tuple[int, int, int],
        rect: pygame.Rect,
        line_gap: int = 2,
    ) -> None:
        y = rect.y
        for line in lines:
            for wrapped in self._wrap_line(font, line, rect.width):
                if y + font.get_height() > rect.bottom:
                    return
                surface = font.render(wrapped, True, color)
                screen.blit(surface, (rect.x, y))
                y += font.get_height() + line_gap

    @staticmethod
    def _wrap_line(font: Any, text: str, width: int) -> list[str]:
        if not text:
            return [""]
        words = text.split()
        if not words:
            return [text]
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if font.size(candidate)[0] <= width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines


def run_app(checkpoint_dir: str | Path = "checkpoints", auto_quit_ms: int | None = None) -> None:
    """Launch the local control app."""

    LabControlApp(checkpoint_dir=checkpoint_dir, auto_quit_ms=auto_quit_ms).run()
