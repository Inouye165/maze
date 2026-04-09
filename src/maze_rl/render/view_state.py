"""Helpers for rendering the maze from the viewer's perspective."""

from __future__ import annotations

from typing import Any, Mapping


VISIBLE_WALL_COLOR = (88, 98, 112)
DIM_WALL_COLOR = (126, 134, 145)
VISIBLE_FLOOR_COLOR = (246, 243, 236)
DIM_FLOOR_COLOR = (220, 212, 198)


def viewer_grid(state: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the full maze layout for the human viewer when available."""

    full_grid = state.get("full_grid")
    if isinstance(full_grid, tuple):
        return full_grid
    if isinstance(full_grid, list):
        return tuple(str(row) for row in full_grid)
    grid = state.get("grid")
    if isinstance(grid, tuple):
        return grid
    if isinstance(grid, list):
        return tuple(str(row) for row in grid)
    return tuple()


def viewer_visible_cells(state: Mapping[str, Any]) -> set[tuple[int, int]]:
    """Return the cells currently inside the human agent's sight range."""

    raw_cells = state.get("visible_cells", [])
    visible: set[tuple[int, int]] = set()
    if not isinstance(raw_cells, list):
        return visible
    for item in raw_cells:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            visible.add((int(item[0]), int(item[1])))
    return visible


def viewer_cell_color(cell: str, is_visible: bool) -> tuple[int, int, int]:
    """Return the human-view color for one maze cell."""

    if cell == "#":
        return VISIBLE_WALL_COLOR if is_visible else DIM_WALL_COLOR
    return VISIBLE_FLOOR_COLOR if is_visible else DIM_FLOOR_COLOR


def viewer_player_position(state: Mapping[str, Any]) -> tuple[int, int] | None:
    """Return the player position to render for the viewer."""

    position = state.get("rendered_player_position", state.get("player_position"))
    return _normalize_position(position)


def viewer_monster_position(state: Mapping[str, Any]) -> tuple[int, int] | None:
    """Return the monster position to render for the viewer."""

    position = state.get("rendered_monster_position", state.get("monster_position"))
    return _normalize_position(position)


def viewer_exit_position(state: Mapping[str, Any]) -> tuple[int, int] | None:
    """Return the exit position to render for the viewer."""

    return _normalize_position(state.get("exit_position"))


def _normalize_position(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    row = getattr(value, "row", None)
    col = getattr(value, "col", None)
    if isinstance(row, int) and isinstance(col, int):
        return (row, col)
    return None
