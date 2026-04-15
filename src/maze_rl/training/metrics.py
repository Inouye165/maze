"""Training and evaluation metrics."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class EpisodeMetrics:
    """Terminal metrics for one episode."""

    outcome: str
    steps: int
    coverage: float
    revisits: int
    oscillations: int
    visible_dead_end_opportunities: int
    entered_visible_dead_end: int
    avoided_visible_dead_end: int
    avoidable_visible_dead_end_penalties_applied: int
    dead_end_entries: int
    blocked_moves: int
    discovered_cells: int
    reward: float
    maze_seed: int
    start_monster_distance: int
    final_monster_distance: int
    final_exit_distance: int
    final_player_position: tuple[int, int]
    final_monster_position: tuple[int, int]
    final_player_monster_distance: int
    capture_rule: str | None
    time_to_capture: int | None
    frontier_cells_visited: int
    reached_new_frontier: bool
    peak_no_progress_steps: int
    avoidable_capture: bool
    avoidable_capture_reason: str | None
    curriculum_stage: str
    monster_speed: int
    monster_activation_delay: int
    escaped: bool
    timed_out: bool
    stalled: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert the metrics to a serializable dictionary."""

        return asdict(self)


class RollingTrainingSummary:
    """Maintain compact aggregate metrics during training."""

    def __init__(self, window_size: int = 1000) -> None:
        self.window_size = window_size
        self.total_episodes = 0
        self.total_wins = 0
        self.total_timeouts = 0
        self.total_stalls = 0
        self.total_avoidable_captures = 0
        self.episodes: deque[EpisodeMetrics] = deque(maxlen=window_size)

    def add(self, metrics: EpisodeMetrics) -> None:
        """Add one completed episode."""

        self.total_episodes += 1
        if metrics.outcome == "escaped":
            self.total_wins += 1
        if metrics.outcome == "timeout":
            self.total_timeouts += 1
        if metrics.outcome == "stall":
            self.total_stalls += 1
        if metrics.avoidable_capture:
            self.total_avoidable_captures += 1
        self.episodes.append(metrics)

    def load_snapshot(self, snapshot: dict[str, Any] | None) -> None:
        """Hydrate aggregate counters from a previously saved training summary."""

        if not isinstance(snapshot, dict):
            return

        self.total_episodes = int(snapshot.get("episodes_seen", 0))
        self.total_wins = int(snapshot.get("wins", 0))
        self.total_timeouts = int(snapshot.get("timeout_count", 0))
        self.total_stalls = int(snapshot.get("stall_count", 0))
        self.total_avoidable_captures = int(snapshot.get("avoidable_capture_count", 0))

    def snapshot(self) -> dict[str, Any]:
        """Build a serializable training summary."""

        window = list(self.episodes)
        recent_outcomes = [item.outcome for item in window]
        recent_10_outcomes = recent_outcomes[-10:]
        recent_50_outcomes = recent_outcomes[-50:]
        recent_100_outcomes = recent_outcomes[-100:]
        recent_1000_outcomes = recent_outcomes[-1000:]

        def _win_rate(outcomes: list[str]) -> float:
            if not outcomes:
                return 0.0
            return sum(1 for item in outcomes if item == "escaped") / len(outcomes)

        return {
            "episodes_seen": self.total_episodes,
            "wins": self.total_wins,
            "win_rate": self.total_wins / max(1, self.total_episodes),
            "timeout_count": self.total_timeouts,
            "stall_count": self.total_stalls,
            "avoidable_capture_count": self.total_avoidable_captures,
            "recent_window_size": self.window_size,
            "recent_count": len(window),
            "recent_win_rate": mean([item.outcome == "escaped" for item in window]) if window else 0.0,
            "recent_average_coverage": mean([item.coverage for item in window]) if window else 0.0,
            "recent_average_discovered_cells": mean([item.discovered_cells for item in window]) if window else 0.0,
            "recent_average_steps": mean([item.steps for item in window]) if window else 0.0,
            "recent_average_revisits": mean([item.revisits for item in window]) if window else 0.0,
            "recent_average_oscillations": mean([item.oscillations for item in window]) if window else 0.0,
            "recent_average_visible_dead_end_opportunities": (
                mean([item.visible_dead_end_opportunities for item in window]) if window else 0.0
            ),
            "recent_average_entered_visible_dead_ends": (
                mean([item.entered_visible_dead_end for item in window]) if window else 0.0
            ),
            "recent_average_avoided_visible_dead_ends": (
                mean([item.avoided_visible_dead_end for item in window]) if window else 0.0
            ),
            "recent_average_dead_ends": mean([item.dead_end_entries for item in window]) if window else 0.0,
            "recent_average_illegal_moves": mean([item.blocked_moves for item in window]) if window else 0.0,
            "recent_average_time_to_capture": (
                mean(
                    [
                        item.time_to_capture
                        for item in window
                        if item.time_to_capture is not None
                    ]
                )
                if any(item.time_to_capture is not None for item in window)
                else None
            ),
            "recent_frontier_expansion_count": (
                mean([item.frontier_cells_visited for item in window]) if window else 0.0
            ),
            "recent_frontier_reached_rate": (
                mean([item.reached_new_frontier for item in window]) if window else 0.0
            ),
            "recent_timeout_rate": mean([item.timed_out for item in window]) if window else 0.0,
            "recent_stall_rate": mean([item.stalled for item in window]) if window else 0.0,
            "recent_avoidable_capture_rate": (
                mean([item.avoidable_capture for item in window]) if window else 0.0
            ),
            "recent_outcomes": recent_outcomes,
            "recent_10_outcomes": recent_10_outcomes,
            "recent_50_outcomes": recent_50_outcomes,
            "recent_100_outcomes": recent_100_outcomes,
            "recent_1000_outcomes": recent_1000_outcomes,
            "recent_10_win_rate": _win_rate(recent_10_outcomes),
            "recent_100_win_rate": _win_rate(recent_100_outcomes),
            "recent_1000_win_rate": _win_rate(recent_1000_outcomes),
        }
