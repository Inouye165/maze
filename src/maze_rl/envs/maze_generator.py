"""Deterministic maze generation."""

from __future__ import annotations

from collections import deque
from random import Random

from .entities import MazeLayout, Position


def _ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _neighbors_for_carving(cell_row: int, cell_col: int) -> list[tuple[int, int, int, int]]:
    return [
        (cell_row - 2, cell_col, cell_row - 1, cell_col),
        (cell_row, cell_col + 2, cell_row, cell_col + 1),
        (cell_row + 2, cell_col, cell_row + 1, cell_col),
        (cell_row, cell_col - 2, cell_row, cell_col - 1),
    ]


def _in_bounds(row: int, col: int, rows: int, cols: int) -> bool:
    return 0 <= row < rows and 0 <= col < cols


def _open_positions(grid: list[list[str]]) -> list[Position]:
    positions: list[Position] = []
    for row_index, row in enumerate(grid):
        for col_index, value in enumerate(row):
            if value != "#":
                positions.append(Position(row_index, col_index))
    return positions


def _extra_connection_walls(grid: list[list[str]]) -> list[tuple[int, int]]:
    """Return removable walls that would create additional routes."""

    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    candidates: list[tuple[int, int]] = []
    for row_index in range(1, rows - 1):
        for col_index in range(1, cols - 1):
            if grid[row_index][col_index] != "#":
                continue
            if row_index % 2 == 1 and col_index % 2 == 0:
                if grid[row_index][col_index - 1] == "." and grid[row_index][col_index + 1] == ".":
                    candidates.append((row_index, col_index))
            elif row_index % 2 == 0 and col_index % 2 == 1:
                if grid[row_index - 1][col_index] == "." and grid[row_index + 1][col_index] == ".":
                    candidates.append((row_index, col_index))
    return candidates


def _carve_extra_connections(grid: list[list[str]], rng: Random) -> None:
    """Open a few additional walls so the maze has multiple routes."""

    candidates = _extra_connection_walls(grid)
    if not candidates:
        return
    extra_connections = max(1, ((len(grid) - 2) * (len(grid[0]) - 2)) // 170)
    rng.shuffle(candidates)
    for row_index, col_index in candidates[:extra_connections]:
        grid[row_index][col_index] = "."


def _bfs_distances(grid: tuple[str, ...], start: Position) -> dict[Position, int]:
    distances: dict[Position, int] = {start: 0}
    queue: deque[Position] = deque([start])
    while queue:
        current = queue.popleft()
        for delta_row, delta_col in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            candidate = current.shifted(delta_row, delta_col)
            if not _in_bounds(candidate.row, candidate.col, len(grid), len(grid[0])):
                continue
            if grid[candidate.row][candidate.col] == "#" or candidate in distances:
                continue
            distances[candidate] = distances[current] + 1
            queue.append(candidate)
    return distances


def _shortest_path(
    grid: tuple[str, ...],
    start: Position,
    goal: Position,
) -> list[Position]:
    """Return the shortest path between two open cells."""

    parents: dict[Position, Position | None] = {start: None}
    queue: deque[Position] = deque([start])
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for delta_row, delta_col in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            candidate = current.shifted(delta_row, delta_col)
            if not _in_bounds(candidate.row, candidate.col, len(grid), len(grid[0])):
                continue
            if grid[candidate.row][candidate.col] == "#" or candidate in parents:
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


def _has_line_of_sight(
    grid: tuple[str, ...],
    viewer: Position,
    target: Position,
    vision_range: int,
) -> bool:
    """Return whether the target is visible under the maze sight rules."""

    if viewer.row == target.row:
        distance = abs(viewer.col - target.col)
        if distance > vision_range:
            return False
        step = 1 if target.col > viewer.col else -1
        for col_index in range(viewer.col + step, target.col, step):
            if grid[viewer.row][col_index] == "#":
                return False
        return True
    if viewer.col == target.col:
        distance = abs(viewer.row - target.row)
        if distance > vision_range:
            return False
        step = 1 if target.row > viewer.row else -1
        for row_index in range(viewer.row + step, target.row, step):
            if grid[row_index][viewer.col] == "#":
                return False
        return True
    return False


def _choose_exit_and_monster(
    grid: tuple[str, ...],
    start: Position,
    vision_range: int,
) -> tuple[Position, Position]:
    """Choose an exit and monster pair that guarantees at least one sighting."""

    distances_from_start = _bfs_distances(grid, start)
    max_dist = max(distances_from_start.values())
    target_dist = max(2, int(max_dist * 0.72))
    exit_candidates = sorted(
        distances_from_start,
        key=lambda position: (
            abs(distances_from_start[position] - target_dist),
            -distances_from_start[position],
        ),
    )
    open_positions = [position for position in distances_from_start if position != start]

    for exit_position in exit_candidates:
        path_to_exit = _shortest_path(grid, start, exit_position)
        visible_monsters = [
            position
            for position in open_positions
            if position != exit_position
            and any(
                _has_line_of_sight(grid, path_position, position, vision_range)
                for path_position in path_to_exit
            )
        ]
        if not visible_monsters:
            continue
        def _score_visible_monster(position: Position) -> tuple[int, int]:
            return (
                distances_from_start.get(position, 0),
                abs(position.row - exit_position.row)
                + abs(position.col - exit_position.col),
            )

        monster_start = max(
            visible_monsters,
            key=_score_visible_monster,
        )
        return exit_position, monster_start

    fallback_exit = exit_candidates[0]
    def _score_fallback_monster(position: Position) -> tuple[int, int]:
        return (
            distances_from_start.get(position, 0),
            abs(position.row - fallback_exit.row)
            + abs(position.col - fallback_exit.col),
        )

    fallback_monster = max(
        (position for position in open_positions if position != fallback_exit),
        key=_score_fallback_monster,
    )
    return fallback_exit, fallback_monster


def generate_maze(
    seed: int,
    rows: int,
    cols: int,
    vision_range: int = 4,
) -> MazeLayout:
    """Generate a deterministic maze for the given seed."""

    rng = Random(seed)
    rows = max(5, _ensure_odd(rows))
    cols = max(5, _ensure_odd(cols))
    grid = [["#" for _ in range(cols)] for _ in range(rows)]
    start = Position(1, 1)
    stack = [start]
    grid[start.row][start.col] = "."

    while stack:
        current = stack[-1]
        options = _neighbors_for_carving(current.row, current.col)
        rng.shuffle(options)
        carved = False
        for next_row, next_col, wall_row, wall_col in options:
            if not (1 <= next_row < rows - 1 and 1 <= next_col < cols - 1):
                continue
            if grid[next_row][next_col] != "#":
                continue
            grid[wall_row][wall_col] = "."
            grid[next_row][next_col] = "."
            stack.append(Position(next_row, next_col))
            carved = True
            break
        if not carved:
            stack.pop()

    _carve_extra_connections(grid, rng)

    grid_tuple = tuple("".join(row) for row in grid)
    exit_position, monster_start = _choose_exit_and_monster(
        grid_tuple,
        start,
        vision_range,
    )

    return MazeLayout(
        grid=grid_tuple,
        player_start=start,
        monster_start=monster_start,
        exit_position=exit_position,
        seed=seed,
    )
