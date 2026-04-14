"""Deterministic maze generation."""

from __future__ import annotations

from collections import deque
from random import Random

from .entities import MazeLayout, Position


DIRECTION_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


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
        for delta_row, delta_col in DIRECTION_DELTAS:
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
        for delta_row, delta_col in DIRECTION_DELTAS:
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


def _move_player(
    grid: tuple[str, ...],
    player: Position,
    direction: tuple[int, int],
) -> tuple[Position, bool]:
    candidate = player.shifted(direction[0], direction[1])
    if not _in_bounds(candidate.row, candidate.col, len(grid), len(grid[0])):
        return player, False
    if grid[candidate.row][candidate.col] == "#":
        return player, False
    return candidate, True


def _monster_is_active(turn_index: int, monster_activation_delay: int, monster_move_interval: int) -> bool:
    if turn_index < monster_activation_delay:
        return False
    if monster_move_interval > 1 and turn_index % monster_move_interval != 0:
        return False
    return True


def _turn_phase(
    turn_index: int,
    monster_activation_delay: int,
    monster_move_interval: int,
) -> tuple[int, int]:
    if turn_index < monster_activation_delay:
        return (0, turn_index)
    return (1, turn_index % max(1, monster_move_interval))


def _simulate_turn(
    grid: tuple[str, ...],
    player: Position,
    monster: Position,
    exit_position: Position,
    direction_index: int,
    speed: int,
    turn_index: int,
    monster_speed: int,
    monster_activation_delay: int,
    monster_move_interval: int,
) -> tuple[Position, Position, str]:
    player_current = player
    monster_current = monster
    player_can_continue = True

    for substep in range(max(1, speed, monster_speed)):
        if player_can_continue and substep < speed:
            player_current, moved = _move_player(
                grid,
                player_current,
                DIRECTION_DELTAS[direction_index],
            )
            if not moved:
                player_can_continue = False
            if player_current == monster_current:
                return player_current, monster_current, "caught"
            if player_current == exit_position:
                return player_current, monster_current, "escaped"

        if substep < monster_speed and _monster_is_active(turn_index, monster_activation_delay, monster_move_interval):
            path = _shortest_path(grid, monster_current, player_current)
            if len(path) >= 2:
                monster_current = path[1]
            if player_current == monster_current:
                return player_current, monster_current, "caught"

    return player_current, monster_current, "running"


def _path_segment(
    path: list[Position],
    start_index: int,
    max_player_speed: int,
) -> tuple[int, int, int]:
    current = path[start_index]
    next_position = path[start_index + 1]
    direction = (next_position.row - current.row, next_position.col - current.col)
    direction_index = DIRECTION_DELTAS.index(direction)
    end_index = start_index + 1
    while end_index < len(path) - 1 and end_index - start_index < max_player_speed:
        candidate = path[end_index + 1]
        candidate_direction = (
            candidate.row - path[end_index].row,
            candidate.col - path[end_index].col,
        )
        if candidate_direction != direction:
            break
        end_index += 1
    speed = end_index - start_index
    return direction_index, speed, end_index


def _layout_is_winnable(
    grid: tuple[str, ...],
    player_start: Position,
    monster_start: Position,
    exit_position: Position,
    *,
    max_player_speed: int = 1,
    monster_speed: int = 1,
    monster_activation_delay: int = 0,
    monster_move_interval: int = 1,
    max_episode_steps: int = 120,
) -> bool:
    path_to_exit = _shortest_path(grid, player_start, exit_position)
    if not path_to_exit or path_to_exit[-1] != exit_position:
        return False

    player = player_start
    monster = monster_start
    turn_index = 0
    path_index = 0

    while path_index < len(path_to_exit) - 1:
        if turn_index >= max_episode_steps:
            return False
        direction_index, speed, next_index = _path_segment(
            path_to_exit,
            path_index,
            max_player_speed,
        )
        player, monster, outcome = _simulate_turn(
            grid,
            player,
            monster,
            exit_position,
            direction_index,
            speed,
            turn_index,
            monster_speed,
            monster_activation_delay,
            monster_move_interval,
        )
        if outcome == "escaped":
            return True
        if outcome == "caught":
            return False
        path_index = next_index
        turn_index += 1

    return player == exit_position


def _choose_exit_and_monster(
    grid: tuple[str, ...],
    start: Position,
    vision_range: int,
    *,
    max_player_speed: int = 1,
    monster_speed: int = 1,
    monster_activation_delay: int = 0,
    monster_move_interval: int = 1,
    max_episode_steps: int = 120,
) -> tuple[Position, Position]:
    """Choose an exit and monster pair that guarantees at least one sighting."""

    max_exit_candidates = 24
    max_monster_candidates = 24

    distances_from_start = _bfs_distances(grid, start)
    max_dist = max(distances_from_start.values())
    target_dist = max(2, int(max_dist * 0.72))
    exit_candidates = sorted(
        distances_from_start,
        key=lambda position: (
            abs(distances_from_start[position] - target_dist),
            -distances_from_start[position],
        ),
    )[:max_exit_candidates]
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
        fallback_monsters = [position for position in open_positions if position != exit_position]

        def _score_visible_monster(position: Position) -> tuple[int, int]:
            return (
                distances_from_start.get(position, 0),
                abs(position.row - exit_position.row)
                + abs(position.col - exit_position.col),
            )

        candidate_groups = [
            sorted(
                visible_monsters,
                key=_score_visible_monster,
                reverse=True,
            )[:max_monster_candidates],
            sorted(
                fallback_monsters,
                key=_score_visible_monster,
                reverse=True,
            )[:max_monster_candidates],
        ]
        for candidates in candidate_groups:
            for monster_start in candidates:
                if _layout_is_winnable(
                    grid,
                    start,
                    monster_start,
                    exit_position,
                    max_player_speed=max_player_speed,
                    monster_speed=monster_speed,
                    monster_activation_delay=monster_activation_delay,
                    monster_move_interval=monster_move_interval,
                    max_episode_steps=max_episode_steps,
                ):
                    return exit_position, monster_start

    fallback_exit = exit_candidates[0]
    def _score_fallback_monster(position: Position) -> tuple[int, int]:
        return (
            distances_from_start.get(position, 0),
            abs(position.row - fallback_exit.row)
            + abs(position.col - fallback_exit.col),
        )
    fallback_monsters = sorted(
        (position for position in open_positions if position != fallback_exit),
        key=_score_fallback_monster,
        reverse=True,
    )[:max_monster_candidates]
    for fallback_monster in fallback_monsters:
        if _layout_is_winnable(
            grid,
            start,
            fallback_monster,
            fallback_exit,
            max_player_speed=max_player_speed,
            monster_speed=monster_speed,
            monster_activation_delay=monster_activation_delay,
            monster_move_interval=monster_move_interval,
            max_episode_steps=max_episode_steps,
        ):
            return fallback_exit, fallback_monster
    raise ValueError("Unable to generate a winnable maze layout for the requested rules")


def generate_maze(
    seed: int,
    rows: int,
    cols: int,
    vision_range: int = 4,
    max_player_speed: int = 1,
    monster_speed: int = 1,
    monster_activation_delay: int = 0,
    monster_move_interval: int = 1,
    max_episode_steps: int = 120,
) -> MazeLayout:
    """Generate a deterministic maze for the given seed."""

    rows = max(5, _ensure_odd(rows))
    cols = max(5, _ensure_odd(cols))
    start = Position(1, 1)
    for attempt in range(12):
        attempt_seed = seed + attempt * 104_729
        rng = Random(attempt_seed)
        grid = [["#" for _ in range(cols)] for _ in range(rows)]
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
        try:
            exit_position, monster_start = _choose_exit_and_monster(
                grid_tuple,
                start,
                vision_range,
                max_player_speed=max_player_speed,
                monster_speed=monster_speed,
                monster_activation_delay=monster_activation_delay,
                monster_move_interval=monster_move_interval,
                max_episode_steps=max_episode_steps,
            )
        except ValueError:
            continue

        return MazeLayout(
            grid=grid_tuple,
            player_start=start,
            monster_start=monster_start,
            exit_position=exit_position,
            seed=seed,
        )

    raise ValueError("Unable to generate a winnable maze after deterministic retries")
