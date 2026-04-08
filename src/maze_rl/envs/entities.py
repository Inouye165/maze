"""Core entities and immutable layout structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Position:
    """Grid position."""

    row: int
    col: int

    def shifted(self, delta_row: int, delta_col: int) -> "Position":
        """Return a moved copy of the current position."""

        return Position(self.row + delta_row, self.col + delta_col)

    def as_tuple(self) -> tuple[int, int]:
        """Return the position as a plain tuple."""

        return (self.row, self.col)


@dataclass(frozen=True, slots=True)
class MazeLayout:
    """Static maze layout for one seeded episode."""

    grid: tuple[str, ...]
    player_start: Position
    monster_start: Position
    exit_position: Position
    seed: int

    @property
    def rows(self) -> int:
        """Maze row count."""

        return len(self.grid)

    @property
    def cols(self) -> int:
        """Maze column count."""

        return len(self.grid[0]) if self.grid else 0

    @property
    def open_cell_count(self) -> int:
        """Count traversable cells."""

        return sum(cell != "#" for row in self.grid for cell in row)


@dataclass(frozen=True, slots=True)
class ReplayMicroStep:
    """One animated micro-step inside a full environment turn."""

    actor: str
    index: int
    position: tuple[int, int]
    phase: str
    capture: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert the micro-step to a serializable dictionary."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReplayTurnEvent:
    """Structured replay event for one environment turn."""

    turn_step: int
    action_index: int
    action_direction: int
    action_speed: int
    player_start_position: tuple[int, int]
    player_path: tuple[tuple[int, int], ...]
    monster_start_position: tuple[int, int]
    monster_path: tuple[tuple[int, int], ...]
    final_player_position: tuple[int, int]
    final_monster_position: tuple[int, int]
    capture_event: dict[str, Any] | None
    capture_rule: str | None
    outcome: str
    micro_steps: tuple[ReplayMicroStep, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert the replay event to a serializable dictionary."""

        payload = asdict(self)
        payload["micro_steps"] = [step.to_dict() for step in self.micro_steps]
        return payload
