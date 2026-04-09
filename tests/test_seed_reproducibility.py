"""Seed reproducibility tests."""

from collections import deque

from maze_rl.envs.entities import Position
from maze_rl.envs.maze_generator import generate_maze


def _edge_count(grid: tuple[str, ...]) -> int:
    edges = 0
    for row_index, row in enumerate(grid):
        for col_index, cell in enumerate(row):
            if cell == "#":
                continue
            for delta_row, delta_col in ((1, 0), (0, 1)):
                next_row = row_index + delta_row
                next_col = col_index + delta_col
                if 0 <= next_row < len(grid) and 0 <= next_col < len(row):
                    if grid[next_row][next_col] != "#":
                        edges += 1
    return edges


def _open_cell_count(grid: tuple[str, ...]) -> int:
    return sum(cell != "#" for row in grid for cell in row)


def _dead_end_count(grid: tuple[str, ...]) -> int:
    count = 0
    for row_index, row in enumerate(grid):
        for col_index, cell in enumerate(row):
            if cell == "#":
                continue
            exits = 0
            for delta_row, delta_col in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                next_row = row_index + delta_row
                next_col = col_index + delta_col
                if 0 <= next_row < len(grid) and 0 <= next_col < len(row):
                    if grid[next_row][next_col] != "#":
                        exits += 1
            if exits <= 1:
                count += 1
    return count


def _shortest_path(
    grid: tuple[str, ...],
    start: Position,
    goal: Position,
) -> list[Position]:
    parents: dict[Position, Position | None] = {start: None}
    queue: deque[Position] = deque([start])
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for delta_row, delta_col in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            candidate = Position(current.row + delta_row, current.col + delta_col)
            if not (0 <= candidate.row < len(grid) and 0 <= candidate.col < len(grid[0])):
                continue
            if grid[candidate.row][candidate.col] == "#" or candidate in parents:
                continue
            parents[candidate] = current
            queue.append(candidate)
    path: list[Position] = []
    node: Position | None = goal
    while node is not None:
        path.append(node)
        node = parents.get(node)
    path.reverse()
    return path


def _has_line_of_sight(
    grid: tuple[str, ...],
    viewer: Position,
    target: Position,
    vision_range: int,
) -> bool:
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


def test_seeded_maze_generation_is_reproducible() -> None:
    """The same seed should generate the same maze."""

    first = generate_maze(seed=1001, rows=15, cols=15)
    second = generate_maze(seed=1001, rows=15, cols=15)
    different = generate_maze(seed=1002, rows=15, cols=15)

    assert first.grid == second.grid
    assert first.player_start == second.player_start
    assert first.exit_position == second.exit_position
    assert first.monster_start == second.monster_start
    assert first.grid != different.grid


def test_generated_maze_contains_loops_for_multiple_paths() -> None:
    """Generated mazes should include extra connections so routes are not single-path only."""

    layout = generate_maze(seed=1001, rows=19, cols=19)

    assert _edge_count(layout.grid) >= _open_cell_count(layout.grid)


def test_generated_maze_preserves_many_dead_ends() -> None:
    """Generated mazes should stay branchy enough to retain many dead ends."""

    layout = generate_maze(seed=1001, rows=19, cols=19)

    assert _dead_end_count(layout.grid) >= 11


def test_generated_exit_route_sees_monster_at_least_once() -> None:
    """The shortest route to the exit should expose the monster at least once."""

    layout = generate_maze(seed=1001, rows=19, cols=19, vision_range=4)
    path = _shortest_path(layout.grid, layout.player_start, layout.exit_position)

    assert any(
        _has_line_of_sight(layout.grid, position, layout.monster_start, 4)
        for position in path
    )
