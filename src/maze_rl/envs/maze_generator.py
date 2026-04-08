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


def generate_maze(seed: int, rows: int, cols: int) -> MazeLayout:
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

    grid_tuple = tuple("".join(row) for row in grid)
    distances_from_start = _bfs_distances(grid_tuple, start)

    # Place exit at moderate distance – not the absolute farthest cell,
    # which would force the player through nearly every corridor.
    max_dist = max(distances_from_start.values())
    target_dist = max(2, int(max_dist * 0.6))
    exit_candidates = sorted(
        distances_from_start,
        key=lambda p: abs(distances_from_start[p] - target_dist),
    )
    exit_position = exit_candidates[0]

    candidates = _open_positions(grid)
    monster_start = max(
        (position for position in candidates if position not in {start, exit_position}),
        key=lambda position: distances_from_start.get(position, 0) + abs(position.row - exit_position.row) + abs(position.col - exit_position.col),
    )

    return MazeLayout(
        grid=grid_tuple,
        player_start=start,
        monster_start=monster_start,
        exit_position=exit_position,
        seed=seed,
    )