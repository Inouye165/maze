"""Custom Gymnasium environment for the maze RL lab."""

# pylint: disable=line-too-long

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from maze_rl.config import CurriculumStage, MazeConfig
from maze_rl.training.metrics import EpisodeMetrics

from .entities import MazeLayout, Position, ReplayMicroStep, ReplayTurnEvent
from .maze_generator import generate_maze
from .observation import ObservationSpec, build_observation_space, encode_observation
from .rewards import StepEvent, compute_reward


DIRECTION_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
REPEAT_LOOP_TERMINATION_STREAK = 12

# Game score constants (integer points displayed alongside rewards)
SCORE_NEW_CELL = 10
SCORE_FRONTIER = 5
SCORE_SURVIVAL = 1
SCORE_REVISIT = -2
SCORE_OSCILLATION = -3
SCORE_DEAD_END = -5
SCORE_BLOCKED = -2
SCORE_ESCAPE_BONUS = 200
SCORE_CAUGHT_PENALTY = -100
SCORE_TIMEOUT_PENALTY = -50
SCORE_STALL_PENALTY = -40


@dataclass(frozen=True)
class VisibleDirectionSummary:
    """Visibility summary for one legal direction from the current player position."""

    direction: int
    legal: bool
    visible_depth: int
    exit_visible: bool
    enters_visible_dead_end: bool


class MazeEnv(gym.Env[np.ndarray, int]):
    """Maze environment with deterministic maze seeds and fixed monster speed."""

    metadata = {"render_modes": ["human"], "render_fps": 12}

    def __init__(self, config: MazeConfig | None = None, training_mode: bool = True) -> None:
        super().__init__()
        self.config = config or MazeConfig()
        self.training_mode = training_mode
        self.observation_spec = ObservationSpec(rows=self.config.rows, cols=self.config.cols)
        self.observation_space = build_observation_space(self.observation_spec)
        self.action_space = gym.spaces.Discrete(4 * self.config.max_player_speed)
        self._episode_index = 0
        self._latest_seed = self.config.train_seed_base
        self._active_rows = self.config.rows
        self._active_cols = self.config.cols
        self._active_monster_speed = self.config.monster_speed
        self._active_monster_activation_delay = self.config.monster_activation_delay
        self._active_monster_move_interval = self.config.monster_move_interval
        self._active_max_episode_steps = self.config.max_episode_steps
        self._active_stall_threshold = self.config.stall_threshold
        self._active_stage = self._resolve_curriculum_stage(0)
        self.layout: MazeLayout | None = None
        self.player: Position | None = None
        self.monster: Position | None = None
        self.last_action_direction: int | None = None
        self.last_action_speed: int = 1
        self.step_count = 0
        self.no_progress_steps = 0
        self.total_reward = 0.0
        self.visited_counts: dict[Position, int] = {}
        self.unique_visited: set[Position] = set()
        self.seen_open_cells: set[Position] = set()
        self.seen_wall_cells: set[Position] = set()
        self.known_dead_end_cells: set[Position] = set()
        self.visible_open_cells: set[Position] = set()
        self.visible_wall_cells: set[Position] = set()
        self.last_seen_monster_position: Position | None = None
        self.turns_since_monster_seen: int | None = None
        self.dead_end_entries = 0
        self.visible_dead_end_opportunities = 0
        self.entered_visible_dead_end = 0
        self.avoided_visible_dead_end = 0
        self.avoidable_visible_dead_end_penalties_applied = 0
        self.revisits = 0
        self.oscillations = 0
        self.blocked_moves = 0
        self.frontier_cells_visited = 0
        self.discovered_cells = 0
        self.start_monster_distance = 0
        self.peak_no_progress_steps = 0
        self.repeat_move_streak = 0
        self.peak_repeat_move_streak = 0
        self.last_outcome = "running"
        self.last_capture_rule: str | None = None
        self._last_avoidable_capture = False
        self._last_avoidable_capture_reason: str | None = None
        self.last_action_index: int | None = None
        self.path_history: deque[Position] = deque(maxlen=6)
        self._scheduled_focus_seed: int | None = None
        self._game_score = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        fixed_layout = options.get("layout")
        maze_seed = options.get("maze_seed")
        if maze_seed is None:
            if self.config.fixed_maze_seed is not None:
                if self._scheduled_focus_seed is None:
                    self._scheduled_focus_seed = int(self.config.fixed_maze_seed)
                maze_seed = self._scheduled_focus_seed
                jump_max = max(1, int(self.config.focused_seed_jump_max))
                jump = int(self.np_random.integers(1, jump_max + 1))
                self._scheduled_focus_seed += jump
            else:
                maze_seed = (
                    seed
                    if seed is not None
                    else int(self.np_random.integers(0, np.iinfo(np.int64).max))
                )
        elif self.config.fixed_maze_seed is not None:
            self._scheduled_focus_seed = int(maze_seed)
        self._latest_seed = int(maze_seed)
        self._active_stage = self._resolve_curriculum_stage(self._episode_index)
        self.layout = (
            fixed_layout
            if fixed_layout is not None
            else generate_maze(
                self._latest_seed,
                self._active_rows,
                self._active_cols,
                vision_range=self.config.vision_range,
                max_player_speed=self.config.max_player_speed,
                monster_speed=self._active_monster_speed,
                monster_activation_delay=self._active_monster_activation_delay,
                monster_move_interval=self._active_monster_move_interval,
                max_episode_steps=self._active_max_episode_steps,
            )
        )
        self._active_rows = self.layout.rows
        self._active_cols = self.layout.cols
        self.player = self.layout.player_start
        self.monster = self.layout.monster_start
        self.last_action_direction = None
        self.last_action_speed = 1
        self.last_action_index = None
        self.step_count = 0
        self.no_progress_steps = 0
        self.total_reward = 0.0
        self.visited_counts = {self.player: 1}
        self.unique_visited = {self.player}
        self.seen_open_cells = set()
        self.seen_wall_cells = set()
        self.known_dead_end_cells = set()
        self.visible_open_cells = set()
        self.visible_wall_cells = set()
        self.last_seen_monster_position = None
        self.turns_since_monster_seen = None
        self.dead_end_entries = 0
        self.visible_dead_end_opportunities = 0
        self.entered_visible_dead_end = 0
        self.avoided_visible_dead_end = 0
        self.avoidable_visible_dead_end_penalties_applied = 0
        self.revisits = 0
        self.oscillations = 0
        self.blocked_moves = 0
        self.frontier_cells_visited = 0
        self.discovered_cells = 0
        self.path_history = deque([self.player], maxlen=6)
        self.start_monster_distance = self._distance(self.player, self.monster)
        self._observe_from_player()
        self._refresh_known_dead_end_paths()
        self.peak_no_progress_steps = 0
        self.repeat_move_streak = 0
        self.peak_repeat_move_streak = 0
        self.last_outcome = "running"
        self.last_capture_rule = None
        self._last_avoidable_capture = False
        self._last_avoidable_capture_reason = None
        self._game_score = 0
        self._episode_index += 1
        observation = self._get_observation()
        return observation, {
            "maze_seed": self._latest_seed,
            "curriculum_stage": self._active_stage.label,
            "monster_speed": self._active_monster_speed,
            "monster_activation_delay": self._active_monster_activation_delay,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.layout is None or self.player is None or self.monster is None:
            raise RuntimeError("Environment must be reset before stepping.")

        direction_index, player_speed = self.decode_action(action)
        return self._step_decoded(direction_index=direction_index, player_speed=player_speed, action_index=int(action))

    def step_wait(self) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one turn without moving the player so the monster can react."""

        if self.layout is None or self.player is None or self.monster is None:
            raise RuntimeError("Environment must be reset before stepping.")
        return self._step_decoded(direction_index=None, player_speed=0, action_index=None)

    def _step_decoded(
        self,
        direction_index: int | None,
        player_speed: int,
        action_index: int | None,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one decoded turn, optionally with a player wait action."""

        self.last_action_index = action_index
        terminated = False
        truncated = False
        reason = "running"
        new_cells = 0
        revisits = 0
        oscillations = 0
        dead_end_entries = 0
        blocked_moves = 0
        frontier_cells = 0
        revisit_depth = 0
        player_can_continue = True
        previous_exit_distance = self._distance(self.player, self.layout.exit_position)
        previous_monster_distance = self._distance(self.player, self.monster)
        direction_summaries = self._visible_direction_summaries()
        chosen_direction_summary = (
            direction_summaries[direction_index] if direction_index is not None else None
        )
        step_visible_dead_end_opportunity = self._has_avoidable_visible_dead_end(direction_summaries)
        step_entered_visible_dead_end = (
            chosen_direction_summary.enters_visible_dead_end if chosen_direction_summary is not None else False
        )
        step_avoided_visible_dead_end = step_visible_dead_end_opportunity and not step_entered_visible_dead_end
        step_escape_move_available = False
        step_best_escape_action: int | None = None
        step_avoidable_capture = False
        step_avoidable_capture_reason: str | None = None
        avoidable_visible_dead_end_entries = 0
        player_start = self.player
        monster_start = self.monster
        player_path: list[Position] = []
        monster_path: list[Position] = []
        micro_steps: list[ReplayMicroStep] = []
        capture_event: dict[str, Any] | None = None

        from maze_rl.training.showcase import rank_legal_moves

        ranked_moves = rank_legal_moves(self)
        if ranked_moves:
            best_escape = ranked_moves[0]
            if best_escape.monster_distance_gain > 0:
                step_escape_move_available = True
                step_best_escape_action = best_escape.action

        if step_visible_dead_end_opportunity:
            self.visible_dead_end_opportunities += 1
        if step_entered_visible_dead_end:
            self.entered_visible_dead_end += 1
        if step_avoided_visible_dead_end:
            self.avoided_visible_dead_end += 1
        if step_visible_dead_end_opportunity and step_entered_visible_dead_end:
            avoidable_visible_dead_end_entries = 1
            self.avoidable_visible_dead_end_penalties_applied += 1

        if direction_index is None:
            micro_steps.append(
                ReplayMicroStep(
                    actor="player",
                    index=0,
                    position=self.player.as_tuple(),
                    phase="player-wait",
                )
            )

        # Phase 1: Human moves first – complete all player substeps before the monster acts
        for substep in range(player_speed if direction_index is not None else 0):
            if not player_can_continue or terminated or truncated:
                break
            moved, revisited, entered_dead_end, visit_depth = self._move_player(direction_index)
            if moved:
                player_path.append(self.player)
                micro_steps.append(
                    ReplayMicroStep(
                        actor="player",
                        index=len(player_path),
                        position=self.player.as_tuple(),
                        phase=f"player-{substep + 1}",
                    )
                )
            if moved and self.player not in self.unique_visited:
                self.unique_visited.add(self.player)
                new_cells += 1
            if revisited:
                revisits += 1
                self.revisits += 1
                revisit_depth = max(revisit_depth, visit_depth)
            if entered_dead_end:
                dead_end_entries += 1
                self.dead_end_entries += 1
            if not moved:
                blocked_moves += 1
                self.blocked_moves += 1
                player_can_continue = False
            if self.player == self.monster:
                terminated = True
                reason = "caught"
                self.last_capture_rule = "same-cell"
                capture_event = self._build_capture_event("player", substep + 1)
            if self.player == self.layout.exit_position:
                terminated = True
                reason = "escaped"

        # Phase 2: Monster acts after the human has finished moving
        if self._monster_is_active():
            for substep in range(self._active_monster_speed):
                if terminated or truncated:
                    break
                monster_previous_position = self.monster
                moved = self._move_monster()
                monster_passed_through_intersection = False
                if moved and self.player == self.monster:
                    monster_passed_through_intersection = self._apply_intersection_escape_window(
                        player_start,
                        monster_previous_position,
                    )
                if moved:
                    monster_path.append(self.monster)
                    micro_steps.append(
                        ReplayMicroStep(
                            actor="monster",
                            index=len(monster_path),
                            position=self.monster.as_tuple(),
                            phase=(
                                f"monster-{substep + 1}-pass"
                                if monster_passed_through_intersection
                                else f"monster-{substep + 1}"
                            ),
                        )
                    )
                if self.player == self.monster:
                    dodge_target = self._reactive_dodge(monster_previous_position)
                    if dodge_target is not None:
                        self.player = dodge_target
                        self.path_history.append(self.player)
                        prev_visits = self.visited_counts.get(self.player, 0)
                        self.visited_counts[self.player] = prev_visits + 1
                        player_path.append(self.player)
                        micro_steps.append(
                            ReplayMicroStep(
                                actor="player",
                                index=len(player_path),
                                position=self.player.as_tuple(),
                                phase=f"player-dodge-{substep + 1}",
                            )
                        )
                        if self.player not in self.unique_visited:
                            self.unique_visited.add(self.player)
                            new_cells += 1
                        elif prev_visits > 0:
                            revisits += 1
                            self.revisits += 1
                        if self.player == self.layout.exit_position:
                            terminated = True
                            reason = "escaped"
                        break  # monster loses remaining substeps after dodge
                    else:
                        terminated = True
                        reason = "caught"
                        self.last_capture_rule = "same-cell"
                        if micro_steps:
                            last_step = micro_steps[-1]
                            micro_steps[-1] = ReplayMicroStep(
                                actor=last_step.actor,
                                index=last_step.index,
                                position=last_step.position,
                                phase=last_step.phase,
                                capture=True,
                            )
                        capture_event = self._build_capture_event("monster", substep + 1)

        self.step_count += 1
        oscillation_severity = self._oscillation_severity()
        if oscillation_severity > 0:
            oscillations = 1
            self.oscillations += 1

        newly_seen_open_cells = self._observe_from_player()
        self._refresh_known_dead_end_paths()
        frontier_cells = max(0, newly_seen_open_cells - new_cells)
        self.frontier_cells_visited += frontier_cells

        if oscillation_severity > 0 and new_cells == 0 and frontier_cells == 0:
            self.repeat_move_streak += 1
        elif new_cells > 0 or frontier_cells > 0:
            self.repeat_move_streak = 0
        else:
            self.repeat_move_streak = max(0, self.repeat_move_streak - 1)
        self.peak_repeat_move_streak = max(self.peak_repeat_move_streak, self.repeat_move_streak)
        if self.repeat_move_streak > 0:
            oscillation_severity += self.repeat_move_streak

        if new_cells == 0 and frontier_cells == 0:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0
        self.peak_no_progress_steps = max(self.peak_no_progress_steps, self.no_progress_steps)

        repeat_loop_detected = self.repeat_move_streak >= 5 and new_cells == 0 and frontier_cells == 0
        repeat_loop_stalled = (
            self.repeat_move_streak >= REPEAT_LOOP_TERMINATION_STREAK
            and new_cells == 0
            and frontier_cells == 0
        )

        if not terminated:
            if repeat_loop_stalled or self.no_progress_steps >= self._active_stall_threshold:
                truncated = True
                reason = "stall"
            elif self.step_count >= self._active_max_episode_steps:
                truncated = True
                reason = "timeout"

        exit_progress_delta = float(previous_exit_distance - self._distance(self.player, self.layout.exit_position))
        monster_distance_delta = float(self._distance(self.player, self.monster) - previous_monster_distance)
        reward = compute_reward(
            self.config.reward,
            StepEvent(
                new_cells=new_cells,
                frontier_cells=frontier_cells,
                revisits=revisits,
                revisit_depth=revisit_depth,
                oscillations=oscillations,
                oscillation_severity=oscillation_severity,
                dead_end_entries=dead_end_entries,
                avoidable_visible_dead_end_entries=avoidable_visible_dead_end_entries,
                blocked_moves=blocked_moves,
                exit_progress_delta=exit_progress_delta,
                monster_distance_delta=monster_distance_delta,
                reached_exit=reason == "escaped",
                caught=reason == "caught",
                timeout=reason == "timeout",
                stalled=reason == "stall",
            ),
        )
        self.total_reward += reward.total

        # Accumulate visible game score
        step_score = (
            SCORE_NEW_CELL * new_cells
            + SCORE_FRONTIER * frontier_cells
            + SCORE_SURVIVAL
            + SCORE_REVISIT * revisits
            + SCORE_OSCILLATION * oscillation_severity
            + SCORE_DEAD_END * dead_end_entries
            + SCORE_BLOCKED * blocked_moves
        )
        if reason == "escaped":
            step_score += SCORE_ESCAPE_BONUS
        elif reason == "caught":
            step_score += SCORE_CAUGHT_PENALTY
        elif reason == "timeout":
            step_score += SCORE_TIMEOUT_PENALTY
        elif reason == "stall":
            step_score += SCORE_STALL_PENALTY
        self._game_score += step_score

        self.last_action_direction = direction_index
        self.last_action_speed = player_speed
        self.last_outcome = reason
        if reason == "caught" and step_escape_move_available:
            chosen_action = -1 if action_index is None else int(action_index)
            if step_best_escape_action is not None and chosen_action != step_best_escape_action:
                step_avoidable_capture = True
                step_avoidable_capture_reason = "ignored-clear-escape"
        self._last_avoidable_capture = step_avoidable_capture
        self._last_avoidable_capture_reason = step_avoidable_capture_reason

        observation = self._get_observation()
        state_snapshot = self.get_state_snapshot()
        info: dict[str, Any] = {
            "maze_seed": self._latest_seed,
            "coverage": self.coverage,
            "revisits": self.revisits,
            "oscillations": self.oscillations,
            "no_progress_steps": self.no_progress_steps,
            "dead_end_entries": self.dead_end_entries,
            "visible_dead_end_opportunities": self.visible_dead_end_opportunities,
            "entered_visible_dead_end": self.entered_visible_dead_end,
            "avoided_visible_dead_end": self.avoided_visible_dead_end,
            "avoidable_visible_dead_end_penalties_applied": self.avoidable_visible_dead_end_penalties_applied,
            "step_visible_dead_end_opportunity": step_visible_dead_end_opportunity,
            "step_entered_visible_dead_end": step_entered_visible_dead_end,
            "step_avoided_visible_dead_end": step_avoided_visible_dead_end,
            "blocked_moves": self.blocked_moves,
            "illegal_moves": self.blocked_moves,
            "outcome": reason,
            "reward_breakdown": reward,
            "start_monster_distance": self.start_monster_distance,
            "time_to_capture": self.step_count if reason == "caught" else None,
            "frontier_cells_visited": self.frontier_cells_visited,
            "discovered_cells": self.discovered_cells,
            "reached_new_frontier": frontier_cells > 0,
            "peak_no_progress_steps": self.peak_no_progress_steps,
            "repeat_move_streak": self.repeat_move_streak,
            "peak_repeat_move_streak": self.peak_repeat_move_streak,
            "repeat_loop_detected": repeat_loop_detected,
            "repeat_loop_stalled": repeat_loop_stalled,
            "escape_move_available": step_escape_move_available,
            "best_escape_action": step_best_escape_action,
            "avoidable_capture": step_avoidable_capture,
            "avoidable_capture_reason": step_avoidable_capture_reason,
            "curriculum_stage": self._active_stage.label,
            "monster_speed": self._active_monster_speed,
            "monster_activation_delay": self._active_monster_activation_delay,
            "monster_enabled": self._active_monster_speed > 0 and self._active_monster_activation_delay <= self._active_max_episode_steps,
            "action_count": int(self.action_space.n),
            "state_snapshot": state_snapshot,
            "capture_diagnostics": self._capture_diagnostics(),
            "replay_turn": ReplayTurnEvent(
                turn_step=self.step_count,
                action_index=-1 if action_index is None else action_index,
                action_direction=direction_index,
                action_speed=player_speed,
                player_start_position=player_start.as_tuple(),
                player_path=tuple(position.as_tuple() for position in player_path),
                monster_start_position=monster_start.as_tuple(),
                monster_path=tuple(position.as_tuple() for position in monster_path),
                final_player_position=self.player.as_tuple(),
                final_monster_position=self.monster.as_tuple(),
                capture_event=capture_event,
                capture_rule=self.last_capture_rule,
                outcome=reason,
                micro_steps=tuple(micro_steps),
            ).to_dict(),
        }
        if terminated or truncated:
            info["episode_metrics"] = self._build_episode_metrics(reason)
            info["episode_end"] = True
        return observation, reward.total, terminated, truncated, info

    @property
    def coverage(self) -> float:
        """Coverage ratio for the current episode."""

        if self.layout is None:
            return 0.0
        return len(self.unique_visited) / max(1, self.layout.open_cell_count)

    def decode_action(self, action: int) -> tuple[int, int]:
        """Decode the flattened discrete action into direction and speed."""

        direction = int(action) // self.config.max_player_speed
        speed = int(action) % self.config.max_player_speed + 1
        if not 0 <= direction < 4:
            raise ValueError(f"Invalid direction index: {direction}")
        return direction, speed

    def action_masks(self) -> list[bool]:
        """Return a legal-action mask for algorithms that support masking."""

        if self.player is None:
            return [True] * int(self.action_space.n)
        from maze_rl.training.showcase import _project_action_target, describe_move_choice, rank_legal_moves, should_override_policy

        masks: list[bool] = []
        for action in range(int(self.action_space.n)):
            _first_step_target, _target, completed_full_speed = _project_action_target(self, action)
            masks.append(completed_full_speed)
        ranked_moves = rank_legal_moves(self)
        best_move = ranked_moves[0] if ranked_moves else None
        if best_move is None:
            return masks
        if self.monster is not None and self.monster in self.visible_open_cells and best_move.monster_distance_gain > 0:
            flee_masks = [False] * int(self.action_space.n)
            if 0 <= best_move.action < len(flee_masks):
                flee_masks[best_move.action] = True
                return flee_masks
        filtered_masks = list(masks)
        for action, allowed in enumerate(masks):
            if not allowed:
                continue
            chosen_move = describe_move_choice(self, action)
            if chosen_move is None:
                continue
            if should_override_policy(chosen_move, best_move, chosen_confidence=None, confidence_gap=None):
                filtered_masks[action] = False
        if any(filtered_masks):
            return filtered_masks
        return masks

    def get_render_state(self) -> dict[str, Any]:
        """Return a lightweight snapshot for the local viewer."""

        return self.get_state_snapshot()

    def render(self) -> dict[str, Any]:
        """Return the current render snapshot for tooling that calls Gym render."""

        return self.get_render_state()

    def get_visible_direction_summaries(self) -> tuple[VisibleDirectionSummary, ...]:
        """Return one-step visibility summaries for debug tooling and tests."""

        return self._visible_direction_summaries()

    def path_distance(self, start: Position, goal: Position) -> int:
        """Return the shortest legal path distance between two positions."""

        path = self._shortest_path(start, goal)
        if not path:
            return 9999
        if len(path) == 1 and start != goal:
            return 9999
        return max(0, len(path) - 1)

    @property
    def player_vision_range(self) -> int:
        """Return the effective player visibility range used for line-of-sight reveal."""

        return self.config.vision_range + 1

    def get_state_snapshot(self) -> dict[str, Any]:
        """Return the single source of truth snapshot for render and debug tooling."""

        if self.layout is None or self.player is None or self.monster is None:
            raise RuntimeError("Environment must be reset before rendering.")
        return {
            "grid": self._knowledge_grid(),
            "full_grid": self.layout.grid,
            "player": self.player,
            "monster": self.monster,
            "exit": self.layout.exit_position,
            "player_position": self.player.as_tuple(),
            "player_visible": True,
            "monster_position": self.monster.as_tuple(),
            "exit_position": self.layout.exit_position.as_tuple(),
            "monster_visible": self.monster in self.visible_open_cells,
            "last_seen_monster_position": (
                self.last_seen_monster_position.as_tuple()
                if self.last_seen_monster_position is not None
                else None
            ),
            "turns_since_monster_seen": self.turns_since_monster_seen,
            "exit_seen": self.layout.exit_position in self.seen_open_cells,
            "visible_cells": [position.as_tuple() for position in sorted(self.visible_open_cells, key=lambda item: (item.row, item.col))],
            "explored_cells": [position.as_tuple() for position in sorted(self.seen_open_cells, key=lambda item: (item.row, item.col))],
            "traveled_cells": [position.as_tuple() for position in sorted(self.unique_visited, key=lambda item: (item.row, item.col))],
            "known_dead_end_cells": [position.as_tuple() for position in sorted(self.known_dead_end_cells, key=lambda item: (item.row, item.col))],
            "player_monster_distance": self._distance(self.player, self.monster),
            "seed": self._latest_seed,
            "curriculum_stage": self._active_stage.label,
            "monster_speed": self._active_monster_speed,
            "monster_activation_delay": self._active_monster_activation_delay,
            "steps": self.step_count,
            "coverage": self.coverage,
            "revisits": self.revisits,
            "oscillations": self.oscillations,
            "dead_end_entries": self.dead_end_entries,
            "blocked_moves": self.blocked_moves,
            "illegal_moves": self.blocked_moves,
            "discovered_cells": self.discovered_cells,
            "repeat_move_streak": self.repeat_move_streak,
            "peak_repeat_move_streak": self.peak_repeat_move_streak,
            "repeat_loop_warning": self.repeat_move_streak >= 3,
            "visible_dead_end_opportunities": self.visible_dead_end_opportunities,
            "entered_visible_dead_end": self.entered_visible_dead_end,
            "avoided_visible_dead_end": self.avoided_visible_dead_end,
            "avoidable_visible_dead_end_penalties_applied": self.avoidable_visible_dead_end_penalties_applied,
            "reward": self.total_reward,
            "outcome": self.last_outcome,
            "last_action_index": self.last_action_index,
            "last_action_direction": self.last_action_direction,
            "last_action_speed": self.last_action_speed,
            "last_action_kind": "wait" if self.last_action_direction is None and self.last_action_speed == 0 else "move",
            "capture_rule": self.last_capture_rule,
            "monster_enabled": self._active_monster_speed > 0 and self._active_monster_activation_delay <= self._active_max_episode_steps,
            "wait_supported": True,
            "action_count": int(self.action_space.n),
            "game_score": self._game_score,
        }

    def _get_observation(self) -> np.ndarray:
        if self.layout is None or self.player is None or self.monster is None:
            raise RuntimeError("Environment must be reset before encoding observations.")
        revisit_ratio = min(1.0, self.revisits / max(1, self.step_count + 1))
        oscillation_ratio = min(1.0, (self.oscillations + self.repeat_move_streak) / max(1, self.step_count + 1))
        dead_end_ratio = min(1.0, self.dead_end_entries / max(1, self.step_count + 1))
        return encode_observation(
            spec=self.observation_spec,
            layout=self.layout,
            player=self.player,
            monster=self.monster,
            visited_counts=self.visited_counts,
            seen_open_cells=self.seen_open_cells,
            seen_wall_cells=self.seen_wall_cells,
            visible_open_cells=self.visible_open_cells,
            step_count=self.step_count,
            max_episode_steps=self._active_max_episode_steps,
            last_direction=self.last_action_direction,
            last_speed=self.last_action_speed,
            coverage=self.coverage,
            revisit_ratio=revisit_ratio,
            oscillation_ratio=oscillation_ratio,
            dead_end_ratio=dead_end_ratio,
            direction_features=self._observation_direction_features(),
        )

    def _move_player(self, direction_index: int) -> tuple[bool, bool, bool, int]:
        delta_row, delta_col = DIRECTION_DELTAS[direction_index]
        candidate = self.player.shifted(delta_row, delta_col)
        if self._is_wall(candidate):
            self.path_history.append(self.player)
            return False, False, False, 0
        self.player = candidate
        self.path_history.append(self.player)
        previous_visits = self.visited_counts.get(self.player, 0)
        self.visited_counts[self.player] = previous_visits + 1
        revisited = previous_visits > 0
        entered_dead_end = self._is_dead_end(self.player) and self.player != self.layout.exit_position
        return True, revisited, entered_dead_end, previous_visits

    def _move_monster(self) -> bool:
        path = self._shortest_path(self.monster, self.player)
        if len(path) >= 2:
            self.monster = path[1]
            return True
        return False

    def _apply_intersection_escape_window(
        self,
        player_previous_position: Position,
        monster_previous_position: Position,
    ) -> bool:
        """Let the monster pass through a junction instead of causing a surprise corner capture."""

        current_position = self.player
        if current_position is None or self.monster is None or self.layout is None:
            return False
        if player_previous_position == current_position:
            return False

        player_delta = (
            current_position.row - player_previous_position.row,
            current_position.col - player_previous_position.col,
        )
        monster_delta = (
            current_position.row - monster_previous_position.row,
            current_position.col - monster_previous_position.col,
        )
        if player_delta == monster_delta:
            return False
        if player_delta == (-monster_delta[0], -monster_delta[1]):
            return False
        if player_delta[0] * monster_delta[0] + player_delta[1] * monster_delta[1] != 0:
            return False

        exits = [
            neighbor
            for delta_row, delta_col in DIRECTION_DELTAS
            if not self._is_wall(neighbor := current_position.shifted(delta_row, delta_col))
        ]
        if len(exits) < 3:
            return False

        pass_through_position = current_position.shifted(monster_delta[0], monster_delta[1])
        if self._is_wall(pass_through_position):
            return False
        if pass_through_position == player_previous_position:
            return False

        alternative_escapes = [
            position
            for position in exits
            if position != pass_through_position and position != monster_previous_position
        ]
        if not alternative_escapes:
            return False

        self.monster = pass_through_position
        return True

    def _reactive_dodge(self, monster_previous_position: Position) -> Position | None:
        """When the monster lands on the player, pick the best adjacent escape cell.

        Returns the dodge target position, or ``None`` if the player is truly
        trapped (no open adjacent cell).  The caller is responsible for moving
        the player and recording micro-steps.
        """
        if self.player is None or self.layout is None or self.monster is None:
            return None

        candidates: list[Position] = []
        for dr, dc in DIRECTION_DELTAS:
            nb = self.player.shifted(dr, dc)
            if not self._is_wall(nb) and nb != monster_previous_position:
                candidates.append(nb)

        if not candidates:
            return None

        exit_pos = self.layout.exit_position

        def _rank(target: Position) -> tuple[bool, int, int]:
            is_known_dead = self.is_known_dead_route_target(target)
            visits = self.visited_counts.get(target, 0)
            exit_dist = self._distance(target, exit_pos)
            return (is_known_dead, exit_dist, visits)

        candidates.sort(key=_rank)
        return candidates[0]

    def _observe_from_player(self) -> int:
        """Reveal visible cells from the player and return newly seen open-cell count."""

        newly_seen_open_cells = 0
        self.visible_open_cells = {self.player}
        self.visible_wall_cells = set()
        if self.player not in self.seen_open_cells:
            self.seen_open_cells.add(self.player)
            newly_seen_open_cells += 1

        for delta_row, delta_col in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            current = self.player
            for _ in range(self.player_vision_range):
                current = current.shifted(delta_row, delta_col)
                if (
                    current.row < 0
                    or current.col < 0
                    or current.row >= self.layout.rows
                    or current.col >= self.layout.cols
                ):
                    break
                if self._is_wall(current):
                    self.visible_wall_cells.add(current)
                    self.seen_wall_cells.add(current)
                    break
                self.visible_open_cells.add(current)
                if current not in self.seen_open_cells:
                    self.seen_open_cells.add(current)
                    newly_seen_open_cells += 1

        if self.monster in self.visible_open_cells:
            self.last_seen_monster_position = self.monster
            self.turns_since_monster_seen = 0
        elif self.last_seen_monster_position is not None:
            self.turns_since_monster_seen = 1 if self.turns_since_monster_seen is None else self.turns_since_monster_seen + 1

        self.discovered_cells = len(self.seen_open_cells)
        return newly_seen_open_cells

    def _refresh_known_dead_end_paths(self) -> None:
        """Mark fully known leaf corridors as dead-end paths in the remembered map."""

        if self.layout is None:
            self.known_dead_end_cells = set()
            return

        open_neighbors: dict[Position, set[Position]] = {}
        unknown_neighbor_counts: dict[Position, int] = {}
        for position in self.seen_open_cells:
            neighbors: set[Position] = set()
            unknown_neighbors = 0
            for delta_row, delta_col in DIRECTION_DELTAS:
                candidate = position.shifted(delta_row, delta_col)
                if (
                    candidate.row < 0
                    or candidate.col < 0
                    or candidate.row >= self.layout.rows
                    or candidate.col >= self.layout.cols
                ):
                    continue
                if self._is_wall(candidate):
                    continue
                if candidate in self.seen_open_cells:
                    neighbors.add(candidate)
                    continue
                unknown_neighbors += 1
            open_neighbors[position] = neighbors
            unknown_neighbor_counts[position] = unknown_neighbors

        remaining_neighbors = {
            position: set(neighbors) for position, neighbors in open_neighbors.items()
        }
        original_degrees = {
            position: len(neighbors) for position, neighbors in open_neighbors.items()
        }
        dead_end_cells: set[Position] = set()
        queue: deque[Position] = deque(
            position
            for position, neighbors in remaining_neighbors.items()
            if position != self.layout.exit_position
            and unknown_neighbor_counts[position] == 0
            and len(neighbors) <= 1
        )

        while queue:
            position = queue.popleft()
            if position in dead_end_cells:
                continue
            dead_end_cells.add(position)
            for neighbor in list(remaining_neighbors.get(position, set())):
                remaining_neighbors[neighbor].discard(position)
                if (
                    neighbor not in dead_end_cells
                    and neighbor != self.layout.exit_position
                    and original_degrees.get(neighbor, 0) <= 2
                    and unknown_neighbor_counts[neighbor] == 0
                    and len(remaining_neighbors[neighbor]) <= 1
                ):
                    queue.append(neighbor)

        dead_end_cells.update(self._all_known_dead_route_cells())
        self.known_dead_end_cells = dead_end_cells

    def _all_known_dead_route_cells(self) -> set[Position]:
        """Return all remembered routes that only lead into fully known dead branches."""

        if self.layout is None:
            return set()

        known_cells: set[Position] = set()
        memo: dict[tuple[Position, Position], set[Position] | None] = {}
        for parent in self.seen_open_cells:
            for delta_row, delta_col in DIRECTION_DELTAS:
                child = parent.shifted(delta_row, delta_col)
                if child not in self.seen_open_cells or self._is_wall(child):
                    continue
                route = self._dead_route_from(child, parent, memo, set())
                if route is not None:
                    known_cells.update(route)
        return known_cells

    def is_known_dead_route_target(self, target: Position, parent: Position | None = None) -> bool:
        """Return whether moving from parent into target commits to a known dead route."""

        if self.layout is None:
            return False
        route_parent = self.player if parent is None else parent
        if route_parent is None or target not in self.seen_open_cells:
            return False
        return bool(self._known_dead_route_cells(target, route_parent))

    def _visible_dead_end_path_cells(self) -> set[Position]:
        """Return remembered forward routes that are known to terminate in dead ends."""

        if self.player is None or self.layout is None:
            return set()

        known_cells: set[Position] = set()
        for summary in self._visible_direction_summaries():
            if not summary.legal or not summary.enters_visible_dead_end:
                continue
            delta_row, delta_col = DIRECTION_DELTAS[summary.direction]
            current = self.player.shifted(delta_row, delta_col)
            known_cells.update(self._known_dead_route_cells(current, self.player))
        return known_cells

    def _known_dead_route_cells(self, start: Position, parent: Position) -> set[Position]:
        """Return cells in a remembered route that can only end in known dead branches."""

        if self.layout is None or start not in self.seen_open_cells:
            return set()

        memo: dict[tuple[Position, Position], set[Position] | None] = {}
        route = self._dead_route_from(start, parent, memo, set())
        return set() if route is None else route

    def _dead_route_from(
        self,
        position: Position,
        parent: Position,
        memo: dict[tuple[Position, Position], set[Position] | None],
        active: set[tuple[Position, Position]],
    ) -> set[Position] | None:
        """Return the remembered subtree when every forward continuation is a dead route."""

        if self.layout is None:
            return None
        if position == self.layout.exit_position:
            return None

        key = (position, parent)
        if key in memo:
            return memo[key]
        if key in active:
            memo[key] = None
            return None

        active.add(key)
        children: list[Position] = []
        for delta_row, delta_col in DIRECTION_DELTAS:
            candidate = position.shifted(delta_row, delta_col)
            if candidate == parent or self._is_wall(candidate):
                continue
            if candidate not in self.seen_open_cells:
                active.remove(key)
                memo[key] = None
                return None
            children.append(candidate)

        if not children:
            result: set[Position] | None = {position}
        else:
            result = {position}
            for child in children:
                child_route = self._dead_route_from(child, position, memo, active)
                if child_route is None:
                    result = None
                    break
                result.update(child_route)

        active.remove(key)
        memo[key] = result
        return result

    def _shortest_path(self, start: Position, goal: Position) -> list[Position]:
        queue: deque[Position] = deque([start])
        parents: dict[Position, Position | None] = {start: None}
        while queue:
            current = queue.popleft()
            if current == goal:
                break
            for delta_row, delta_col in DIRECTION_DELTAS:
                candidate = current.shifted(delta_row, delta_col)
                if self._is_wall(candidate) or candidate in parents:
                    continue
                parents[candidate] = current
                queue.append(candidate)
        if goal not in parents:
            return [start]
        path: list[Position] = []
        node: Position | None = goal
        while node is not None:
            path.append(node)
            node = parents[node]
        path.reverse()
        return path

    def _is_wall(self, position: Position) -> bool:
        return (
            position.row < 0
            or position.col < 0
            or position.row >= self.layout.rows
            or position.col >= self.layout.cols
            or self.layout.grid[position.row][position.col] == "#"
        )

    def _is_dead_end(self, position: Position) -> bool:
        exits = 0
        for delta_row, delta_col in DIRECTION_DELTAS:
            neighbor = position.shifted(delta_row, delta_col)
            if not self._is_wall(neighbor):
                exits += 1
        return exits <= 1

    def _visible_direction_summaries(self) -> tuple[VisibleDirectionSummary, ...]:
        """Summarize what the player can see in each cardinal direction."""

        return tuple(self._summarize_visible_direction(direction) for direction in range(4))

    def _summarize_visible_direction(self, direction_index: int) -> VisibleDirectionSummary:
        """Describe a straight visible corridor from the current state."""

        delta_row, delta_col = DIRECTION_DELTAS[direction_index]
        candidate = self.player.shifted(delta_row, delta_col)
        if self._is_wall(candidate):
            return VisibleDirectionSummary(
                direction=direction_index,
                legal=False,
                visible_depth=0,
                exit_visible=False,
                enters_visible_dead_end=False,
            )

        remembered_dead_route = bool(self._known_dead_route_cells(candidate, self.player))

        visible_depth = 0
        current = candidate
        while current in self.visible_open_cells:
            visible_depth += 1
            if current == self.layout.exit_position:
                return VisibleDirectionSummary(
                    direction=direction_index,
                    legal=True,
                    visible_depth=visible_depth,
                    exit_visible=True,
                    enters_visible_dead_end=False,
                )
            if self._is_dead_end(current):
                return VisibleDirectionSummary(
                    direction=direction_index,
                    legal=True,
                    visible_depth=visible_depth,
                    exit_visible=False,
                    enters_visible_dead_end=True,
                )
            if self._has_side_branch(current, direction_index):
                break
            current = current.shifted(delta_row, delta_col)

        return VisibleDirectionSummary(
            direction=direction_index,
            legal=True,
            visible_depth=visible_depth,
            exit_visible=False,
            enters_visible_dead_end=remembered_dead_route,
        )

    def _has_side_branch(self, position: Position, direction_index: int) -> bool:
        """Return whether a straight corridor cell opens sideways."""

        reverse_direction = (direction_index + 2) % 4
        for candidate_direction, (delta_row, delta_col) in enumerate(DIRECTION_DELTAS):
            if candidate_direction in {direction_index, reverse_direction}:
                continue
            if not self._is_wall(position.shifted(delta_row, delta_col)):
                return True
        return False

    def _has_avoidable_visible_dead_end(
        self,
        direction_summaries: tuple[VisibleDirectionSummary, ...],
    ) -> bool:
        """Return whether the player can see an avoidable non-exit dead-end move."""

        has_visible_dead_end = any(summary.legal and summary.enters_visible_dead_end for summary in direction_summaries)
        has_safe_alternative = any(summary.legal and not summary.enters_visible_dead_end for summary in direction_summaries)
        return has_visible_dead_end and has_safe_alternative

    def _observation_direction_features(self) -> tuple[float, ...]:
        """Return compact per-direction visibility features for the policy observation."""

        features: list[float] = []
        for summary in self._visible_direction_summaries():
            features.extend(
                [
                    1.0 if summary.legal else 0.0,
                    min(1.0, summary.visible_depth / max(1, self.player_vision_range)),
                    1.0 if summary.enters_visible_dead_end else 0.0,
                    1.0 if summary.exit_visible else 0.0,
                ]
            )
        return tuple(features)

    @staticmethod
    def _distance(left: Position, right: Position) -> int:
        return abs(left.row - right.row) + abs(left.col - right.col)

    def _knowledge_grid(self) -> tuple[str, ...]:
        """Return the remembered maze view with unknown cells hidden."""

        rows: list[str] = []
        for row_index in range(self.layout.rows):
            cells: list[str] = []
            for col_index in range(self.layout.cols):
                position = Position(row_index, col_index)
                if position in self.seen_wall_cells:
                    cells.append("#")
                elif position in self.seen_open_cells:
                    cells.append(".")
                else:
                    cells.append("?")
            rows.append("".join(cells))
        return tuple(rows)

    def _build_episode_metrics(self, outcome: str) -> EpisodeMetrics:
        return EpisodeMetrics(
            outcome=outcome,
            steps=self.step_count,
            coverage=self.coverage,
            revisits=self.revisits,
            oscillations=self.oscillations,
            visible_dead_end_opportunities=self.visible_dead_end_opportunities,
            entered_visible_dead_end=self.entered_visible_dead_end,
            avoided_visible_dead_end=self.avoided_visible_dead_end,
            avoidable_visible_dead_end_penalties_applied=self.avoidable_visible_dead_end_penalties_applied,
            dead_end_entries=self.dead_end_entries,
            blocked_moves=self.blocked_moves,
            discovered_cells=self.discovered_cells,
            reward=self.total_reward,
            maze_seed=self._latest_seed,
            start_monster_distance=self.start_monster_distance,
            final_monster_distance=self._distance(self.player, self.monster),
            final_exit_distance=self._distance(self.player, self.layout.exit_position),
            final_player_position=self.player.as_tuple(),
            final_monster_position=self.monster.as_tuple(),
            final_player_monster_distance=self._distance(self.player, self.monster),
            capture_rule=self.last_capture_rule,
            time_to_capture=self.step_count if outcome == "caught" else None,
            frontier_cells_visited=self.frontier_cells_visited,
            reached_new_frontier=self.frontier_cells_visited > 0,
            peak_no_progress_steps=self.peak_no_progress_steps,
            avoidable_capture=self._last_avoidable_capture,
            avoidable_capture_reason=self._last_avoidable_capture_reason,
            curriculum_stage=self._active_stage.label,
            monster_speed=self._active_monster_speed,
            monster_activation_delay=self._active_monster_activation_delay,
            escaped=outcome == "escaped",
            timed_out=outcome == "timeout",
            stalled=outcome == "stall",
        )

    def _capture_diagnostics(self) -> dict[str, Any]:
        """Return capture diagnostics for debugging and summaries."""

        if self.player is None or self.monster is None:
            return {}
        return {
            "final_player_position": self.player.as_tuple(),
            "final_monster_position": self.monster.as_tuple(),
            "final_distance": self._distance(self.player, self.monster),
            "capture_rule": self.last_capture_rule,
        }

    def _build_capture_event(self, actor: str, substep_index: int) -> dict[str, Any]:
        """Build a structured capture event for replay and debug tooling."""

        return {
            "actor": actor,
            "substep_index": substep_index,
            "position": self.player.as_tuple(),
            "capture_rule": self.last_capture_rule,
        }

    def _resolve_curriculum_stage(self, episode_index: int) -> CurriculumStage:
        if not self.training_mode or not self.config.curriculum_enabled or not self.config.curriculum:
            self._active_rows = self.config.rows
            self._active_cols = self.config.cols
            self._active_monster_speed = self.config.monster_speed
            self._active_monster_activation_delay = self.config.monster_activation_delay
            self._active_monster_move_interval = self.config.monster_move_interval
            self._active_max_episode_steps = self.config.max_episode_steps
            self._active_stall_threshold = self.config.stall_threshold
            return CurriculumStage(
                start_episode=0,
                rows=self.config.rows,
                cols=self.config.cols,
                monster_speed=self.config.monster_speed,
                monster_activation_delay=self.config.monster_activation_delay,
                monster_move_interval=self.config.monster_move_interval,
                max_episode_steps=self.config.max_episode_steps,
                stall_threshold=self.config.stall_threshold,
                label="frozen",
            )

        active_stage = self.config.curriculum[0]
        for stage in self.config.curriculum:
            if episode_index >= stage.start_episode:
                active_stage = stage
            else:
                break
        self._active_rows = active_stage.rows
        self._active_cols = active_stage.cols
        self._active_monster_speed = active_stage.monster_speed
        self._active_monster_activation_delay = active_stage.monster_activation_delay
        self._active_monster_move_interval = getattr(active_stage, 'monster_move_interval', 1)
        self._active_max_episode_steps = active_stage.max_episode_steps
        self._active_stall_threshold = active_stage.stall_threshold
        return active_stage

    def _monster_is_active(self) -> bool:
        if self.step_count < self._active_monster_activation_delay:
            return False
        if self._active_monster_move_interval > 1 and self.step_count % self._active_monster_move_interval != 0:
            return False
        return True

    def _oscillation_severity(self) -> int:
        severity = 0
        if len(self.path_history) >= 3 and self.path_history[-1] == self.path_history[-3]:
            severity += 1
        if len(self.path_history) >= 4 and self.path_history[-1] == self.path_history[-3] and self.path_history[-2] == self.path_history[-4]:
            severity += 1
        if len(self.path_history) >= 6:
            recent = list(self.path_history)[-6:]
            if recent[0] == recent[2] == recent[4] and recent[1] == recent[3] == recent[5] and recent[0] != recent[1]:
                severity += 2
        return severity
