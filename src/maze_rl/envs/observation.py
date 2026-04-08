"""Observation encoding and shape validation."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from .entities import MazeLayout, Position


@dataclass(frozen=True)
class ObservationSpec:
    """Single source of truth for the observation vector."""

    rows: int
    cols: int
    cell_channels: int = 6
    scalar_features: int = 13
    visit_clip: int = 4

    @property
    def vector_length(self) -> int:
        """Total flattened observation size."""

        return self.rows * self.cols * self.cell_channels + self.scalar_features


def build_observation_space(spec: ObservationSpec) -> gym.spaces.Box:
    """Construct the Gymnasium observation space."""

    return gym.spaces.Box(
        low=0.0,
        high=1.0,
        shape=(spec.vector_length,),
        dtype=np.float32,
    )


def validate_observation(observation: np.ndarray, spec: ObservationSpec) -> np.ndarray:
    """Fail loudly on any observation shape drift."""

    if observation.shape != (spec.vector_length,):
        raise ValueError(
            f"Observation length mismatch. Expected {spec.vector_length}, got {observation.shape}."
        )
    return observation.astype(np.float32, copy=False)


def _normalized_distance(left: Position, right: Position, rows: int, cols: int) -> float:
    maximum = max(1, rows + cols)
    return min(1.0, (abs(left.row - right.row) + abs(left.col - right.col)) / maximum)


def encode_observation(
    spec: ObservationSpec,
    layout: MazeLayout,
    player: Position,
    monster: Position,
    visited_counts: dict[Position, int],
    seen_open_cells: set[Position],
    seen_wall_cells: set[Position],
    visible_open_cells: set[Position],
    step_count: int,
    max_episode_steps: int,
    last_direction: int | None,
    last_speed: int,
    coverage: float,
    revisit_ratio: float,
    oscillation_ratio: float,
    dead_end_ratio: float,
) -> np.ndarray:
    """Encode the current environment state into a fixed vector."""

    cell_features: list[float] = []
    for row_index in range(spec.rows):
        for col_index in range(spec.cols):
            position = Position(row_index, col_index)
            in_layout = row_index < layout.rows and col_index < layout.cols
            known_wall = in_layout and position in seen_wall_cells
            known_open = in_layout and position in seen_open_cells
            visits = (
                min(visited_counts.get(position, 0), spec.visit_clip) / spec.visit_clip
                if known_open
                else 0.0
            )
            cell_features.extend(
                [
                    1.0 if known_wall else 0.0,
                    1.0 if known_open else 0.0,
                    visits,
                    1.0 if known_open and position == player else 0.0,
                    1.0
                    if position == monster and position in visible_open_cells
                    else 0.0,
                    1.0
                    if position == layout.exit_position and position in seen_open_cells
                    else 0.0,
                ]
            )

    direction_one_hot = [0.0, 0.0, 0.0, 0.0]
    if last_direction is not None and 0 <= last_direction < 4:
        direction_one_hot[last_direction] = 1.0

    scalar_features = [
        min(1.0, step_count / max(1, max_episode_steps)),
        coverage,
        revisit_ratio,
        oscillation_ratio,
        dead_end_ratio,
        *direction_one_hot,
        min(1.0, last_speed / 10.0),
        _normalized_distance(player, layout.exit_position, layout.rows, layout.cols),
        _normalized_distance(player, monster, layout.rows, layout.cols),
        min(1.0, visited_counts.get(player, 0) / spec.visit_clip),
    ]
    observation = np.asarray(cell_features + scalar_features, dtype=np.float32)
    return validate_observation(observation, spec)
