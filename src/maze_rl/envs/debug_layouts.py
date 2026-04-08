"""Deterministic layouts for debugging pursuit and capture behavior."""

from __future__ import annotations

from .entities import MazeLayout, Position


def build_debug_pursuit_layout() -> MazeLayout:
    """Return a small fixed maze for deterministic pursuit debugging."""

    return MazeLayout(
        grid=(
            "#######",
            "#.....#",
            "#.###.#",
            "#.....#",
            "#######",
        ),
        player_start=Position(1, 1),
        monster_start=Position(3, 5),
        exit_position=Position(1, 5),
        seed=999_001,
    )