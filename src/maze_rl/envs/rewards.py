"""Reward shaping for the maze environment."""

from __future__ import annotations

from dataclasses import dataclass

from maze_rl.config import RewardConfig


@dataclass(frozen=True)
class StepEvent:
    """Signals collected during one environment step."""

    new_cells: int
    frontier_cells: int
    revisits: int
    revisit_depth: int
    oscillations: int
    oscillation_severity: int
    dead_end_entries: int
    avoidable_visible_dead_end_entries: int
    blocked_moves: int
    exit_progress_delta: float
    monster_distance_delta: float
    reached_exit: bool
    caught: bool
    timeout: bool
    stalled: bool


@dataclass(frozen=True)
class RewardBreakdown:
    """Named reward components for debugging and logging."""

    total: float
    exploration: float
    revisit: float
    oscillation: float
    dead_end: float
    avoidable_visible_dead_end: float
    blocking: float
    progress: float
    safety: float
    terminal: float
    survival: float


def compute_reward(config: RewardConfig, event: StepEvent) -> RewardBreakdown:
    """Compute the shaped reward for one step."""

    exploration = config.exploration_reward * event.new_cells + config.frontier_reward * event.frontier_cells
    revisit = config.revisit_penalty * event.revisits + config.revisit_depth_penalty * event.revisit_depth
    oscillation = config.oscillation_penalty * event.oscillation_severity * event.oscillations
    dead_end = config.dead_end_penalty * event.dead_end_entries
    avoidable_visible_dead_end = (
        config.avoidable_visible_dead_end_penalty * event.avoidable_visible_dead_end_entries
    )
    blocking = config.blocked_move_penalty * event.blocked_moves
    progress = config.exit_progress_reward * event.exit_progress_delta

    safety = 0.0
    if event.monster_distance_delta > 0:
        safety += config.safety_gain_reward * event.monster_distance_delta
    elif event.monster_distance_delta < 0:
        safety += (-config.safety_loss_penalty) * event.monster_distance_delta

    terminal = 0.0
    if event.reached_exit:
        terminal += config.win_reward
    if event.caught:
        terminal += config.caught_penalty
    if event.timeout:
        terminal += config.timeout_penalty
    if event.stalled:
        terminal += config.stall_penalty

    survival = config.survival_reward
    total = (
        exploration
        + revisit
        + oscillation
        + dead_end
        + avoidable_visible_dead_end
        + blocking
        + progress
        + safety
        + terminal
        + survival
    )
    return RewardBreakdown(
        total=total,
        exploration=exploration,
        revisit=revisit,
        oscillation=oscillation,
        dead_end=dead_end,
        avoidable_visible_dead_end=avoidable_visible_dead_end,
        blocking=blocking,
        progress=progress,
        safety=safety,
        terminal=terminal,
        survival=survival,
    )