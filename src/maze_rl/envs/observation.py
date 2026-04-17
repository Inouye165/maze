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
    scalar_features: int = 29
    visit_clip: int = 4
    local_tactical_enabled: bool = False
    local_tactical_radius: int = 2
    local_tactical_include_monster_memory: bool = False

    @property
    def global_map_length(self) -> int:
        """Flattened length of the full remembered-map features."""

        return self.rows * self.cols * self.cell_channels

    @property
    def local_tactical_side_length(self) -> int:
        """Width and height of the optional local tactical patch."""

        if not self.local_tactical_enabled:
            return 0
        return self.local_tactical_radius * 2 + 1

    @property
    def local_tactical_cell_channels(self) -> int:
        """Per-cell feature count for the optional local tactical patch."""

        return self.cell_channels + (1 if self.local_tactical_include_monster_memory else 0)

    @property
    def local_tactical_vector_length(self) -> int:
        """Flattened length of the optional local tactical patch."""

        side_length = self.local_tactical_side_length
        if side_length == 0:
            return 0
        return side_length * side_length * self.local_tactical_cell_channels

    @property
    def vector_length(self) -> int:
        """Total flattened observation size."""

        return self.global_map_length + self.local_tactical_vector_length + self.scalar_features


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


def _encode_global_map_features(
    spec: ObservationSpec,
    layout: MazeLayout,
    player: Position,
    monster: Position,
    visited_counts: dict[Position, int],
    seen_open_cells: set[Position],
    seen_wall_cells: set[Position],
    visible_open_cells: set[Position],
) -> list[float]:
    """Encode the full human-known map without collapsing to a local crop."""

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
                    1.0 if position == monster and position in visible_open_cells else 0.0,
                    1.0 if position == layout.exit_position and position in seen_open_cells else 0.0,
                ]
            )
    return cell_features


def _encode_local_tactical_features(
    spec: ObservationSpec,
    layout: MazeLayout,
    player: Position,
    monster: Position,
    last_seen_monster_position: Position | None,
    visited_counts: dict[Position, int],
    seen_open_cells: set[Position],
    seen_wall_cells: set[Position],
    visible_open_cells: set[Position],
) -> list[float]:
    """Encode an optional local tactical patch centered on the player."""

    if not spec.local_tactical_enabled:
        return []

    local_features: list[float] = []
    for row_offset in range(-spec.local_tactical_radius, spec.local_tactical_radius + 1):
        for col_offset in range(-spec.local_tactical_radius, spec.local_tactical_radius + 1):
            position = player.shifted(row_offset, col_offset)
            in_layout = 0 <= position.row < layout.rows and 0 <= position.col < layout.cols
            known_wall = in_layout and position in seen_wall_cells
            known_open = in_layout and position in seen_open_cells
            visits = (
                min(visited_counts.get(position, 0), spec.visit_clip) / spec.visit_clip
                if known_open
                else 0.0
            )
            local_features.extend(
                [
                    1.0 if known_wall else 0.0,
                    1.0 if known_open else 0.0,
                    visits,
                    1.0 if position == player else 0.0,
                    1.0 if position == monster and position in visible_open_cells else 0.0,
                    1.0 if position == layout.exit_position and position in seen_open_cells else 0.0,
                ]
            )
            if spec.local_tactical_include_monster_memory:
                local_features.append(
                    1.0 if last_seen_monster_position is not None and position == last_seen_monster_position else 0.0
                )
    return local_features


def _encode_scalar_features(
    spec: ObservationSpec,
    layout: MazeLayout,
    player: Position,
    monster: Position,
    visited_counts: dict[Position, int],
    step_count: int,
    max_episode_steps: int,
    last_direction: int | None,
    last_speed: int,
    coverage: float,
    revisit_ratio: float,
    oscillation_ratio: float,
    dead_end_ratio: float,
    direction_features: tuple[float, ...],
) -> list[float]:
    """Encode scalar/global rollout metrics appended after map features."""

    direction_one_hot = [0.0, 0.0, 0.0, 0.0]
    if last_direction is not None and 0 <= last_direction < 4:
        direction_one_hot[last_direction] = 1.0

    return [
        min(1.0, step_count / max(1, max_episode_steps)),
        coverage,
        revisit_ratio,
        oscillation_ratio,
        dead_end_ratio,
        *direction_features,
        *direction_one_hot,
        min(1.0, last_speed / 10.0),
        _normalized_distance(player, layout.exit_position, layout.rows, layout.cols),
        _normalized_distance(player, monster, layout.rows, layout.cols),
        min(1.0, visited_counts.get(player, 0) / spec.visit_clip),
    ]


def encode_observation(
    spec: ObservationSpec,
    layout: MazeLayout,
    player: Position,
    monster: Position,
    last_seen_monster_position: Position | None,
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
    direction_features: tuple[float, ...],
) -> np.ndarray:
    """Encode the current environment state into a fixed vector."""

    global_map_features = _encode_global_map_features(
        spec=spec,
        layout=layout,
        player=player,
        monster=monster,
        visited_counts=visited_counts,
        seen_open_cells=seen_open_cells,
        seen_wall_cells=seen_wall_cells,
        visible_open_cells=visible_open_cells,
    )
    local_tactical_features = _encode_local_tactical_features(
        spec=spec,
        layout=layout,
        player=player,
        monster=monster,
        last_seen_monster_position=last_seen_monster_position,
        visited_counts=visited_counts,
        seen_open_cells=seen_open_cells,
        seen_wall_cells=seen_wall_cells,
        visible_open_cells=visible_open_cells,
    )
    scalar_features = _encode_scalar_features(
        spec=spec,
        layout=layout,
        player=player,
        monster=monster,
        visited_counts=visited_counts,
        step_count=step_count,
        max_episode_steps=max_episode_steps,
        last_direction=last_direction,
        last_speed=last_speed,
        coverage=coverage,
        revisit_ratio=revisit_ratio,
        oscillation_ratio=oscillation_ratio,
        dead_end_ratio=dead_end_ratio,
        direction_features=direction_features,
    )
    observation = np.asarray(global_map_features + local_tactical_features + scalar_features, dtype=np.float32)
    return validate_observation(observation, spec)
