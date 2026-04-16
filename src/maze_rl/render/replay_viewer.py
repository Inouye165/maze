"""Minimal pygame replay viewer."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pygame

from maze_rl.render.view_state import (
    viewer_cell_color,
    viewer_dead_end_cells,
    viewer_exit_color,
    viewer_explored_cells,
    viewer_exit_position,
    viewer_grid,
    viewer_monster_position,
    viewer_policy_badge,
    viewer_player_position,
    viewer_traveled_cells,
    viewer_visible_cells,
)
from maze_rl.training.showcase import ShowcaseResult, run_checkpoint_showcase_episode

QUIT = getattr(pygame, "QUIT", 256)


class ReplayViewer:
    """Watch one checkpoint run to completion on one maze seed."""

    def __init__(self, cell_size: int = 32, margin: int = 16) -> None:
        self.cell_size = cell_size
        self.margin = margin
        self._screen: Any = None
        self._font: Any = None
        self._clock: Any = None

    def watch(
        self,
        checkpoint_path: str | Path,
        seed: int,
        fps: int = 10,
        max_no_progress_streak: int = 25,
        wall_time_timeout_s: float = 30.0,
        debug_trace: bool = False,
        allow_policy_override: bool = False,
    ) -> str:
        """Render a frozen run and return the final outcome."""

        checkpoint_label = Path(checkpoint_path).stem.replace("_", " ")
        self._ensure_window(checkpoint_path)
        result = run_checkpoint_showcase_episode(
            checkpoint_path=checkpoint_path,
            checkpoint_label=checkpoint_label,
            seed=seed,
            max_no_progress_streak=max_no_progress_streak,
            wall_time_timeout_s=wall_time_timeout_s,
            on_step=lambda state: self._render_frame(state, checkpoint_path, fps),
            debug_trace=debug_trace,
            allow_policy_override=allow_policy_override,
        )
        self._draw(self._screen, self._coerce_state(result, seed), self._font, checkpoint_path, outcome=result.outcome)
        pygame.display.flip()
        time.sleep(1.5)
        pygame.display.quit()
        pygame.font.quit()
        self._screen = None
        return result.outcome

    def showcase(
        self,
        checkpoint_entries: list[tuple[int, str | Path]],
        seed: int,
        fps: int = 10,
        pause_ms: int = 1500,
        max_no_progress_streak: int = 25,
        wall_time_timeout_s: float = 30.0,
        debug_trace: bool = False,
        allow_policy_override: bool = False,
    ) -> list[ShowcaseResult]:
        """Render a sequential autoplay showcase in one window."""

        results: list[ShowcaseResult] = []
        active_path = next((Path(path) for _, path in checkpoint_entries if Path(path).exists()), None)
        if active_path is None:
            return results
        self._ensure_window(active_path)
        for checkpoint_episode, checkpoint_path in checkpoint_entries:
            path = Path(checkpoint_path)
            if not path.exists():
                results.append(
                    ShowcaseResult(
                        checkpoint=f"ckpt {checkpoint_episode:04d}",
                        status="missing",
                        outcome="skipped",
                        escape_rate=0.0,
                        coverage=0.0,
                        steps=0,
                        revisits=0,
                        oscillations=0,
                        dead_ends=0,
                        start_monster_distance=None,
                        time_to_capture=None,
                        frontier_rate=0.0,
                        peak_no_progress_streak=0,
                        checkpoint_path=str(path),
                        seed=seed,
                        notes="checkpoint missing",
                    )
                )
                continue
            result = run_checkpoint_showcase_episode(
                checkpoint_path=path,
                checkpoint_label=f"ckpt {checkpoint_episode:04d}",
                seed=seed,
                max_no_progress_streak=max_no_progress_streak,
                wall_time_timeout_s=wall_time_timeout_s,
                on_step=lambda state, current_path=path: self._render_frame(state, current_path, fps),
                debug_trace=debug_trace,
                allow_policy_override=allow_policy_override,
            )
            results.append(result)
            self._draw(self._screen, self._coerce_state(result, seed), self._font, path, outcome=result.outcome)
            pygame.display.flip()
            self._wait_with_events(pause_ms)
        pygame.display.quit()
        pygame.font.quit()
        self._screen = None
        return results

    def _ensure_window(self, checkpoint_path: str | Path) -> None:
        metadata = Path(checkpoint_path).with_suffix(".json")
        if not metadata.exists():
            raise FileNotFoundError(f"Missing checkpoint metadata: {metadata}")

        meta = json.loads(metadata.read_text(encoding="utf-8"))
        maze_config = meta["maze_config"]
        pygame.display.init()
        pygame.font.init()
        width = maze_config["cols"] * self.cell_size + self.margin * 2
        height = maze_config["rows"] * self.cell_size + self.margin * 2 + 160
        self._screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Maze RL Lab Showcase")
        self._font = pygame.font.SysFont("consolas", 20)
        self._clock = pygame.time.Clock()

    def _render_frame(self, state: dict[str, Any], checkpoint_path: str | Path, fps: int) -> bool:
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
        self._draw(self._screen, state, self._font, checkpoint_path, outcome=state.get("outcome"))
        pygame.display.flip()
        self._clock.tick(fps)
        return True

    def _wait_with_events(self, pause_ms: int) -> None:
        wait_until = time.monotonic() + pause_ms / 1000.0
        while time.monotonic() < wait_until:
            for event in pygame.event.get():
                if event.type == QUIT:
                    return
            time.sleep(0.02)

    def _coerce_state(self, result: ShowcaseResult, seed: int) -> dict[str, Any]:
        if result.final_state is not None:
            state = dict(result.final_state)
            state.update(
                {
                    "checkpoint_label": result.checkpoint,
                    "outcome": result.outcome,
                    "peak_no_progress_streak": result.peak_no_progress_streak,
                    "start_monster_distance": result.start_monster_distance,
                    "time_to_capture": result.time_to_capture,
                    "frontier_rate": result.frontier_rate,
                }
            )
            return state
        return {
            "grid": tuple(),
            "player": None,
            "monster": None,
            "exit": None,
            "player_position": None,
            "player_visible": True,
            "monster_position": None,
            "player_monster_distance": None,
            "seed": seed,
            "steps": result.steps,
            "coverage": result.coverage,
            "revisits": result.revisits,
            "oscillations": result.oscillations,
            "dead_end_entries": result.dead_ends,
            "blocked_moves": 0,
            "reward": 0.0,
            "checkpoint_label": result.checkpoint,
            "outcome": result.outcome,
            "peak_no_progress_streak": result.peak_no_progress_streak,
            "start_monster_distance": result.start_monster_distance,
            "time_to_capture": result.time_to_capture,
            "frontier_rate": result.frontier_rate,
        }

    def _draw(
        self,
        screen: Any,
        state: dict[str, Any],
        font: Any,
        checkpoint_path: str | Path,
        outcome: str | None = None,
    ) -> None:
        screen.fill((18, 20, 28))
        grid = viewer_grid(state)
        visible_cells = viewer_visible_cells(state)
        explored_cells = viewer_explored_cells(state)
        traveled_cells = viewer_traveled_cells(state)
        dead_end_cells = viewer_dead_end_cells(state)
        if grid:
            for row_index, row in enumerate(grid):
                for col_index, cell in enumerate(row):
                    rect = pygame.Rect(
                        self.margin + col_index * self.cell_size,
                        self.margin + row_index * self.cell_size,
                        self.cell_size - 1,
                        self.cell_size - 1,
                    )
                    position = (row_index, col_index)
                    color = viewer_cell_color(
                        cell,
                        position in visible_cells,
                        is_explored=position in explored_cells,
                        is_traveled=position in traveled_cells,
                        is_dead_end=position in dead_end_cells,
                    )
                    pygame.draw.rect(screen, color, rect)

            entities: list[tuple[tuple[int, int] | None, tuple[int, int, int]]] = [
                (viewer_exit_position(state), viewer_exit_color(state)),
            ]
            if state.get("player_visible", True):
                entities.append((viewer_player_position(state), (60, 110, 220)))
            entities.insert(1, (viewer_monster_position(state), (205, 60, 60)))
            for position, color in entities:
                if position is None:
                    continue
                rect = pygame.Rect(
                    self.margin + position[1] * self.cell_size + 4,
                    self.margin + position[0] * self.cell_size + 4,
                    self.cell_size - 8,
                    self.cell_size - 8,
                )
                pygame.draw.rect(screen, color, rect, border_radius=6)

        info_top = self.margin + len(grid) * self.cell_size + 12
        monster_text = viewer_monster_position(state)
        badge_label, badge_color, badge_text_color = viewer_policy_badge(state)
        badge = pygame.Rect(self.margin, info_top, min(380, max(180, len(badge_label) * 10)), 28)
        pygame.draw.rect(screen, badge_color, badge, border_radius=10)
        badge_text = font.render(badge_label, True, badge_text_color)
        screen.blit(badge_text, (badge.x + 10, badge.y + 4))
        info_top += 36
        lines = [
            f"checkpoint: {state.get('checkpoint_label', Path(checkpoint_path).name)}",
            f"seed: {state['seed']} | steps: {state['steps']} | coverage: {state['coverage']:.2f}",
            (
                f"player: {viewer_player_position(state)} | "
                f"vision cells: {len(visible_cells)} | "
                f"monster: {monster_text} | "
                f"npc sees monster: {state.get('monster_visible', True)}"
            ),
            (
                f"revisits: {state['revisits']} | "
                f"oscillations: {state['oscillations']} | "
                f"dead ends: {state['dead_end_entries']}"
            ),
            (
                f"blocked: {state['blocked_moves']} | reward: {state['reward']:.2f} | "
                f"frontier rate: {state.get('frontier_rate', 0.0):.2f}"
            ),
            (
                f"start monster distance: {state.get('start_monster_distance', 'n/a')} | "
                f"peak no-progress: {state.get('peak_no_progress_streak', 0)} | "
                f"capture: {state.get('capture_rule')}"
            ),
            (
                f"policy: {state.get('policy_kind', 'trained')} | "
                f"override enabled: {state.get('policy_override_enabled', False)} | "
                f"override count: {state.get('policy_override_count', 0)} | "
                f"reason: {state.get('policy_override_reason', 'n/a')}"
            ),
        ]
        if outcome:
            lines.append(
                f"final outcome: {outcome} | "
                f"time to capture: {state.get('time_to_capture', 'n/a')}"
            )
        for index, line in enumerate(lines):
            text = font.render(line, True, (240, 242, 245))
            screen.blit(text, (self.margin, info_top + index * 22))
