"""Helpers for rendering the maze from the viewer's perspective."""

from __future__ import annotations

from typing import Any, Mapping


VISIBLE_WALL_COLOR = (88, 98, 112)
DIM_WALL_COLOR = (126, 134, 145)
VISIBLE_FLOOR_COLOR = (246, 243, 236)
DIM_FLOOR_COLOR = (234, 227, 213)
UNEXPLORED_FLOOR_COLOR = (206, 198, 186)
TRAVELED_FLOOR_COLOR = (214, 196, 150)
VISIBLE_DEAD_END_COLOR = (220, 170, 94)
DIM_DEAD_END_COLOR = (178, 139, 88)
EXIT_COLOR = (77, 145, 95)
SEEN_EXIT_COLOR = (219, 183, 86)


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


def viewer_explored_cells(state: Mapping[str, Any]) -> set[tuple[int, int]]:
    """Return the cells the human agent has already explored or seen."""

    return _normalize_position_list(state.get("explored_cells", []))


def viewer_dead_end_cells(state: Mapping[str, Any]) -> set[tuple[int, int]]:
    """Return the cells the human agent has classified as dead-end path."""

    return _normalize_position_list(state.get("known_dead_end_cells", []))


def viewer_traveled_cells(state: Mapping[str, Any]) -> set[tuple[int, int]]:
    """Return the cells the human agent has physically traversed."""

    return _normalize_position_list(state.get("traveled_cells", []))


def viewer_cell_color(
    cell: str,
    is_visible: bool,
    is_explored: bool = False,
    is_traveled: bool = False,
    is_dead_end: bool = False,
) -> tuple[int, int, int]:
    """Return the human-view color for one maze cell."""

    if cell == "#":
        return VISIBLE_WALL_COLOR if is_visible else DIM_WALL_COLOR
    if is_dead_end:
        return VISIBLE_DEAD_END_COLOR if is_visible else DIM_DEAD_END_COLOR
    if is_visible:
        return VISIBLE_FLOOR_COLOR
    if is_traveled:
        return TRAVELED_FLOOR_COLOR
    if is_explored:
        return DIM_FLOOR_COLOR
    return UNEXPLORED_FLOOR_COLOR


def viewer_player_position(state: Mapping[str, Any]) -> tuple[int, int] | None:
    """Return the player position to render for the viewer."""

    position = state.get("rendered_player_position", state.get("player_position"))
    return _normalize_position(position)


def viewer_monster_position(state: Mapping[str, Any]) -> tuple[int, int] | None:
    """Return the monster position to render for the viewer."""

    position = state.get("rendered_monster_position", state.get("monster_position"))
    return _normalize_position(position)


def viewer_policy_badge(
    state: Mapping[str, Any],
) -> tuple[str, tuple[int, int, int], tuple[int, int, int]]:
    """Return a concise decision badge label plus background and text colors."""

    label = str(state.get("policy_decision_label", "trained policy"))
    policy_kind = str(state.get("policy_kind", "trained"))
    if policy_kind == "heuristic-override":
        return (label, (207, 120, 54), (255, 247, 238))
    if policy_kind == "innate":
        return (label, (79, 123, 92), (244, 250, 244))
    if bool(state.get("policy_override_enabled", False)):
        return (label, (64, 94, 137), (242, 246, 252))
    return (label, (96, 102, 112), (243, 244, 246))


def viewer_exit_position(state: Mapping[str, Any]) -> tuple[int, int] | None:
    """Return the exit position to render for the viewer."""

    return _normalize_position(state.get("exit_position"))


def viewer_exit_color(state: Mapping[str, Any]) -> tuple[int, int, int]:
    """Return the exit marker color, brightening once the human has seen it."""

    return SEEN_EXIT_COLOR if bool(state.get("exit_seen", False)) else EXIT_COLOR


def _normalize_position(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    row = getattr(value, "row", None)
    col = getattr(value, "col", None)
    if isinstance(row, int) and isinstance(col, int):
        return (row, col)
    return None


def _normalize_position_list(values: Any) -> set[tuple[int, int]]:
    normalized: set[tuple[int, int]] = set()
    if not isinstance(values, list):
        return normalized
    for value in values:
        position = _normalize_position(value)
        if position is not None:
            normalized.add(position)
    return normalized
