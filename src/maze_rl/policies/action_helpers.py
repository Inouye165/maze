"""Tactical decision helpers for legal-action masking and move ranking.

These utilities analyse the live ``MazeEnv`` state to rank legal moves,
detect loops, project action targets, and decide when a safety override
should replace a weak policy choice.  They are intentionally free of any
dependency on training, showcase, or render modules so that **both** the
environment and high-level playback code can share them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from maze_rl.envs.entities import Position
from maze_rl.envs.maze_env import MazeEnv


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIRECTION_DELTAS: list[tuple[int, int]] = [(-1, 0), (0, 1), (1, 0), (0, -1)]
WAIT_ACTION: int = -1
WAIT_DIRECTION: int = 4


# ---------------------------------------------------------------------------
# Action identity helpers
# ---------------------------------------------------------------------------


def wait_action_for_env(env: MazeEnv) -> int:
    """Return the explicit wait action id for the given environment."""

    return int(getattr(env, "wait_action_index", WAIT_ACTION))


def is_wait_action(env: MazeEnv, action: int) -> bool:
    """Return whether *action* represents a wait turn for this environment."""

    return int(action) in {WAIT_ACTION, wait_action_for_env(env)}


# ---------------------------------------------------------------------------
# Heuristic move descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeuristicMoveChoice:
    """One legal move ranked by the playback exploration heuristic."""

    action: int
    direction: int
    speed: int
    target: Position
    visits: int
    nearest_unvisited_distance: int
    exit_distance: int
    monster_distance: int
    monster_distance_gain: int
    direct_monster_escape: bool
    immediate_reverse: bool
    short_loop_risk: int
    dead_end_escape_distance: int
    escaping_dead_end: bool
    enters_dead_end: bool
    known_dead_end: bool
    commit_to_exit: bool
    wait_action: bool


# ---------------------------------------------------------------------------
# Move description / ranking / override
# ---------------------------------------------------------------------------


def describe_move_choice(env: MazeEnv, action: int) -> HeuristicMoveChoice | None:
    """Describe one legal move using the same features as the fallback ranking."""

    if env.player is None or env.layout is None or env.monster is None:
        return None
    active_dead_end_component = _active_dead_end_component(env)
    current_escape_distance = _dead_end_escape_distance(
        env,
        env.player,
        active_dead_end_component,
    )
    threat_position = _threat_reference_position(env)
    if is_wait_action(env, action):
        current_monster_distance = (
            _path_distance(env, env.player, threat_position)
            if threat_position is not None
            else 9999
        )
        return HeuristicMoveChoice(
            action=wait_action_for_env(env),
            direction=WAIT_DIRECTION,
            speed=0,
            target=env.player,
            visits=0,
            nearest_unvisited_distance=_nearest_unvisited_distance(
                env,
                env.player,
            ),
            exit_distance=_path_distance(env, env.player, env.layout.exit_position),
            monster_distance=current_monster_distance,
            monster_distance_gain=0,
            direct_monster_escape=False,
            immediate_reverse=False,
            short_loop_risk=0,
            dead_end_escape_distance=current_escape_distance,
            escaping_dead_end=False,
            enters_dead_end=False,
            known_dead_end=False,
            commit_to_exit=False,
            wait_action=True,
        )
    direction, speed = env.decode_action(action)
    first_step_target, target, completed_full_speed = project_action_target(env, action)
    if first_step_target is None or target is None or not completed_full_speed:
        return None
    previous_position = _previous_position(env)
    exit_distance = _path_distance(env, target, env.layout.exit_position)
    current_monster_distance = (
        _path_distance(env, env.player, threat_position)
        if threat_position is not None
        else 9999
    )
    monster_distance = (
        _path_distance(env, target, threat_position)
        if threat_position is not None
        else 9999
    )
    target_escape_distance = _dead_end_escape_distance(
        env,
        target,
        active_dead_end_component,
    )
    escaping_dead_end = (
        bool(active_dead_end_component)
        and target_escape_distance < current_escape_distance
    )
    known_dead_end = env.is_known_dead_route_target(target)
    if escaping_dead_end:
        known_dead_end = False
    commit_to_exit = _should_commit_to_exit(env, target, exit_distance)
    return HeuristicMoveChoice(
        action=action,
        direction=direction,
        speed=speed,
        target=target,
        visits=env.visited_counts.get(target, 0),
        nearest_unvisited_distance=_nearest_unvisited_distance(env, target),
        exit_distance=exit_distance,
        monster_distance=monster_distance,
        monster_distance_gain=monster_distance - current_monster_distance,
        direct_monster_escape=_is_direct_monster_escape(env, target),
        immediate_reverse=previous_position is not None and first_step_target == previous_position,
        short_loop_risk=_short_loop_risk(env, target),
        dead_end_escape_distance=target_escape_distance,
        escaping_dead_end=escaping_dead_end,
        enters_dead_end=_is_dead_end_target(env, target),
        known_dead_end=known_dead_end,
        commit_to_exit=commit_to_exit,
        wait_action=False,
    )


def rank_legal_moves(env: MazeEnv) -> list[HeuristicMoveChoice]:
    """Rank legal one-step moves to favor coverage and avoid local loops."""

    action_count = int(getattr(env.action_space, "n", 0))
    wait_action = wait_action_for_env(env)
    choices = [
        choice
        for action in range(action_count)
        if action != wait_action
        if (choice := describe_move_choice(env, action)) is not None
    ]
    if _should_offer_wait(env, choices) and not any(choice.wait_action for choice in choices):
        wait_choice = describe_move_choice(env, wait_action)
        if wait_choice is not None:
            choices.append(wait_choice)
    exit_choice = _best_exit_commitment_choice(choices)
    fear_mode = _fear_mode(env, choices, exit_choice)
    return sorted(
        choices,
        key=lambda choice: _choice_priority(choice, fear_mode, exit_choice),
    )


def choose_heuristic_action(env: MazeEnv) -> int:
    """Choose the best deterministic fallback action for the current state."""

    ranked = rank_legal_moves(env)
    return ranked[0].action if ranked else wait_action_for_env(env)


def should_override_policy(
    chosen: HeuristicMoveChoice | None,
    best: HeuristicMoveChoice | None,
    chosen_confidence: float | None,
    confidence_gap: float | None,
) -> bool:
    """Decide whether the heuristic should replace a weak or loop-prone policy move."""

    if chosen is None or best is None or chosen.direction == best.direction:
        return False
    if best.commit_to_exit and not chosen.commit_to_exit:
        return True
    if (
        best.commit_to_exit
        and chosen.commit_to_exit
        and chosen.exit_distance > best.exit_distance
    ):
        return True
    if best.escaping_dead_end and not chosen.escaping_dead_end:
        return True
    if (
        (chosen.known_dead_end or chosen.enters_dead_end)
        and not (best.known_dead_end or best.enters_dead_end)
    ):
        return True
    if chosen.visits > 0 and best.visits == 0:
        return True
    if chosen.immediate_reverse and not best.immediate_reverse:
        return True
    if chosen.short_loop_risk >= 6 and chosen.short_loop_risk > best.short_loop_risk:
        return True
    if (not chosen.direct_monster_escape
            and best.direct_monster_escape
            and best.monster_distance_gain > 0):
        return True
    if chosen.monster_distance <= 1 and best.monster_distance > chosen.monster_distance:
        return True
    if chosen.monster_distance_gain < 0 <= best.monster_distance_gain:
        return True
    if chosen.monster_distance_gain < best.monster_distance_gain and best.monster_distance_gain > 0:
        return True
    if (chosen.nearest_unvisited_distance > best.nearest_unvisited_distance
            and best.visits <= chosen.visits):
        return True
    if chosen_confidence is not None and confidence_gap is not None:
        if chosen_confidence < 0.45 or confidence_gap < 0.08:
            return True
    return False


def policy_confidence(
    probabilities: np.ndarray | None,
    chosen_action: int,
) -> tuple[float | None, float | None]:
    """Return chosen-action confidence and the margin over the next-best action."""

    if probabilities is None or chosen_action >= len(probabilities):
        return None, None
    ordered = np.sort(probabilities)[::-1]
    chosen_confidence = float(probabilities[chosen_action])
    top_gap = float(ordered[0] - ordered[1]) if len(ordered) > 1 else chosen_confidence
    return chosen_confidence, top_gap


def policy_decision_label(
    policy_kind: str,
    override_enabled: bool,
    override_reason: str | None,
) -> str:
    """Return a short human-readable label for the current decision source."""

    if policy_kind == "heuristic-override":
        reason = override_reason.replace("-", " ") if override_reason else "safety override"
        return f"safety override: {reason}"
    if policy_kind == "innate":
        return "innate heuristic"
    if override_enabled:
        return "trained policy with safety net"
    return "trained policy"


# ---------------------------------------------------------------------------
# Action target projection (public — used by env action_masks)
# ---------------------------------------------------------------------------


def project_action_target(
    env: MazeEnv,
    action: int,
) -> tuple[Position | None, Position | None, bool]:
    """Return first-step target, final target, and whether the full speed completes legally."""

    if env.player is None:
        return None, None, False
    if is_wait_action(env, action):
        return None, env.player, True
    direction, speed = env.decode_action(action)
    delta_row, delta_col = DIRECTION_DELTAS[direction]
    current = env.player
    first_step_target: Position | None = None
    for step_index in range(speed):
        candidate = current.shifted(delta_row, delta_col)
        if is_wall_position(env, candidate) or candidate == env.monster:
            return first_step_target, current, False
        current = candidate
        if step_index == 0:
            first_step_target = current
    return first_step_target, current, True


# ---------------------------------------------------------------------------
# Geometry / pathfinding helpers (public where env needs them)
# ---------------------------------------------------------------------------


def is_wall_position(env: MazeEnv, position: Position) -> bool:
    """Return whether a position is outside the maze or blocked by a wall."""

    layout = env.layout
    if layout is None:
        return True
    return (
        position.row < 0
        or position.col < 0
        or position.row >= layout.rows
        or position.col >= layout.cols
        or layout.grid[position.row][position.col] == "#"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _previous_position(env: MazeEnv) -> Position | None:
    history = list(env.path_history)
    if len(history) < 2:
        return None
    return history[-2]


def _nearest_unvisited_distance(env: MazeEnv, start: Position) -> int:
    """Measure how quickly a move can reach unexplored space."""

    if start in env.seen_open_cells and env.visited_counts.get(start, 0) == 0:
        return 0
    queue: deque[tuple[Position, int]] = deque([(start, 0)])
    seen = {start}
    while queue:
        current, distance = queue.popleft()
        for candidate in _known_neighbors(env, current):
            if candidate in seen:
                continue
            if candidate in env.seen_open_cells and env.visited_counts.get(candidate, 0) == 0:
                return distance + 1
            seen.add(candidate)
            queue.append((candidate, distance + 1))
    return 9999


def _path_distance(env: MazeEnv, start: Position, goal: Position) -> int:
    """Return the shortest path distance through the known map only."""

    if start == goal:
        return 0
    if start not in env.seen_open_cells or goal not in env.seen_open_cells:
        return 9999

    queue: deque[tuple[Position, int]] = deque([(start, 0)])
    seen = {start}
    while queue:
        current, distance = queue.popleft()
        if current == goal:
            return distance
        for candidate in _known_neighbors(env, current):
            if candidate in seen:
                continue
            seen.add(candidate)
            queue.append((candidate, distance + 1))
    return 9999


def _known_neighbors(env: MazeEnv, position: Position) -> tuple[Position, ...]:
    """Return traversable neighbors that exist in the agent's remembered map."""

    neighbors: list[Position] = []
    for delta_row, delta_col in DIRECTION_DELTAS:
        candidate = position.shifted(delta_row, delta_col)
        if candidate in env.seen_open_cells:
            neighbors.append(candidate)
    return tuple(neighbors)


def _short_loop_risk(env: MazeEnv, candidate: Position) -> int:
    """Score how strongly a move resembles a short back-and-forth loop."""

    history = list(env.path_history)
    risk = 0
    if len(history) >= 2 and candidate == history[-2]:
        risk += 6
    if len(history) >= 4 and history[-1] == history[-3] and candidate == history[-2]:
        risk += 10
    if len(history) >= 4 and candidate == history[-4]:
        risk += 2
    return risk


def _active_dead_end_component(env: MazeEnv) -> set[Position]:
    """Return the connected known-dead region currently containing the player."""

    if env.player is None or env.player not in env.known_dead_end_cells:
        return set()

    component: set[Position] = set()
    queue: deque[Position] = deque([env.player])
    while queue:
        current = queue.popleft()
        if current in component or current not in env.known_dead_end_cells:
            continue
        component.add(current)
        for candidate in _known_neighbors(env, current):
            if candidate in env.known_dead_end_cells and candidate not in component:
                queue.append(candidate)
    return component


def _dead_end_escape_distance(
    env: MazeEnv,
    start: Position,
    active_component: set[Position],
) -> int:
    """Return steps from *start* to the nearest remembered cell.

    The destination must lie outside the player's active remembered dead-end region.
    """

    if not active_component or start not in active_component:
        return 0

    queue: deque[tuple[Position, int]] = deque([(start, 0)])
    seen = {start}
    while queue:
        current, distance = queue.popleft()
        for candidate in _known_neighbors(env, current):
            if candidate in seen:
                continue
            if candidate not in active_component:
                return distance + 1
            seen.add(candidate)
            queue.append((candidate, distance + 1))
    return 9999


def _is_dead_end_target(env: MazeEnv, target: Position) -> bool:
    """Return whether the target is a dead-end cell, excluding the maze exit."""

    if env.layout is not None and target == env.layout.exit_position:
        return False
    exits = sum(
        1 for dr, dc in DIRECTION_DELTAS if not is_wall_position(env, target.shifted(dr, dc))
    )
    return exits <= 1


def _is_direct_monster_escape(env: MazeEnv, candidate: Position) -> bool:
    """Return whether a move heads directly away from the monster's strongest approach axis."""

    threat_position = _threat_reference_position(env)
    if env.player is None or threat_position is None:
        return False
    row_delta = threat_position.row - env.player.row
    col_delta = threat_position.col - env.player.col
    candidate_row_delta = candidate.row - env.player.row
    candidate_col_delta = candidate.col - env.player.col

    if abs(col_delta) >= abs(row_delta) and col_delta != 0:
        return candidate_col_delta == (-1 if col_delta > 0 else 1)
    if row_delta != 0:
        return candidate_row_delta == (-1 if row_delta > 0 else 1)
    return False


def _best_exit_commitment_choice(
    choices: list[HeuristicMoveChoice],
) -> HeuristicMoveChoice | None:
    """Return the best move that safely commits to a seen exit path."""

    exit_choices = [
        choice
        for choice in choices
        if choice.commit_to_exit and not choice.wait_action
    ]
    if not exit_choices:
        return None
    return min(
        exit_choices,
        key=lambda choice: (
            choice.exit_distance,
            choice.monster_distance <= 1,
            choice.monster_distance_gain < 0,
            choice.visits > 0,
            choice.visits,
            -choice.speed,
            choice.direction,
        ),
    )


def _fear_mode(
    env: MazeEnv,
    choices: list[HeuristicMoveChoice],
    exit_choice: HeuristicMoveChoice | None,
) -> bool:
    """Return whether the heuristic should prioritize running from the monster."""

    threat_position = _threat_reference_position(env)
    if env.player is None or threat_position is None or env.layout is None or not choices:
        return False
    if exit_choice is not None:
        return False
    current_monster_distance = _path_distance(env, env.player, threat_position)
    monster_visible = env.monster in env.visible_open_cells
    if current_monster_distance <= 2:
        return True
    has_escape_gain = any(choice.monster_distance_gain > 0 for choice in choices)
    if monster_visible and current_monster_distance <= max(5, env.player_vision_range + 1):
        return True
    if _monster_recently_seen(env) and current_monster_distance <= 4 and has_escape_gain:
        return True
    if monster_visible and has_escape_gain:
        current_exit_distance = _path_distance(env, env.player, env.layout.exit_position)
        has_safe_exit_progress = any(
            choice.exit_distance < current_exit_distance and choice.monster_distance_gain >= 0
            for choice in choices
        )
        return not has_safe_exit_progress
    return False


def _should_offer_wait(env: MazeEnv, choices: list[HeuristicMoveChoice]) -> bool:
    """Return whether a tactical wait is better than committing to a bad branch."""

    threat_position = _threat_reference_position(env)
    if env.player is None or threat_position is None or env.layout is None or not choices:
        return False
    current_monster_distance = _path_distance(env, env.player, threat_position)
    if current_monster_distance <= 6:
        return False
    if env.player in env.known_dead_end_cells:
        return False
    if env.monster in env.visible_open_cells:
        return False
    if not _monster_recently_seen(env):
        return False
    current_exit_distance = _path_distance(env, env.player, env.layout.exit_position)
    safe_unvisited = any(
        not choice.known_dead_end
        and not choice.enters_dead_end
        and not choice.wait_action
        and choice.visits == 0
        and choice.monster_distance_gain >= 0
        for choice in choices
    )
    safe_exit_progress = any(
        not choice.known_dead_end
        and not choice.enters_dead_end
        and not choice.wait_action
        and choice.exit_distance < current_exit_distance
        and choice.monster_distance_gain >= 0
        for choice in choices
    )
    risky_commit = any(
        choice.known_dead_end
        or choice.enters_dead_end
        or choice.immediate_reverse
        or choice.monster_distance_gain < 0
        for choice in choices
    )
    if safe_unvisited or safe_exit_progress or not risky_commit:
        return False
    current_threat_distance = _path_distance(env, env.player, threat_position)
    for choice in choices:
        if choice.wait_action or choice.known_dead_end or choice.enters_dead_end:
            continue
        if choice.visits == 0 and choice.monster_distance_gain >= 0:
            return False
        if choice.exit_distance < current_exit_distance and choice.monster_distance_gain >= 0:
            return False
    return current_threat_distance >= max(5, env.player_vision_range)


def _monster_recently_seen(env: MazeEnv) -> bool:
    """Return whether the monster was seen recently enough to maintain threat memory."""

    turns_since_seen = getattr(env, "turns_since_monster_seen", None)
    return turns_since_seen is not None and turns_since_seen <= 3


def _threat_reference_position(env: MazeEnv) -> Position | None:
    """Return the visible monster or a recent last-seen threat position."""

    if env.monster is not None and env.monster in env.visible_open_cells:
        return env.monster
    if _monster_recently_seen(env):
        return getattr(env, "last_seen_monster_position", None)
    return None


def _choice_priority(
    choice: HeuristicMoveChoice,
    fear_mode: bool,
    exit_choice: HeuristicMoveChoice | None,
) -> tuple[Any, ...]:
    """Return a deterministic sort key for heuristic playback choices."""

    commits_to_exit = exit_choice is not None and choice.action == exit_choice.action
    if fear_mode:
        dead_end_trap = choice.known_dead_end or choice.enters_dead_end
        return (
            not commits_to_exit,
            choice.monster_distance <= 1,
            choice.monster_distance_gain < 0,
            dead_end_trap,
            not choice.wait_action and choice.monster_distance_gain == 0,
            -choice.monster_distance_gain,
            -choice.monster_distance,
            choice.dead_end_escape_distance,
            not choice.escaping_dead_end,
            -choice.speed,
            choice.nearest_unvisited_distance,
            choice.visits > 0,
            choice.visits,
            not choice.direct_monster_escape,
            choice.immediate_reverse and choice.monster_distance_gain <= 0,
            choice.short_loop_risk if choice.monster_distance_gain <= 0 else 0,
            choice.exit_distance,
            choice.direction,
        )
    return (
        not commits_to_exit,
        choice.dead_end_escape_distance,
        not choice.escaping_dead_end,
        choice.known_dead_end,
        choice.monster_distance <= 1,
        choice.enters_dead_end,
        choice.visits > 0,
        choice.visits,
        choice.nearest_unvisited_distance,
        -choice.speed,
        choice.wait_action,
        choice.immediate_reverse,
        choice.short_loop_risk,
        choice.monster_distance_gain < 0,
        choice.exit_distance,
        -choice.monster_distance,
        choice.direction,
    )


def _should_commit_to_exit(env: MazeEnv, target: Position, exit_distance: int) -> bool:
    """Return whether moving to *target* should commit the agent to the seen exit."""

    if env.player is None or env.layout is None:
        return False
    if env.layout.exit_position not in env.seen_open_cells:
        return False
    current_exit_distance = _path_distance(env, env.player, env.layout.exit_position)
    if current_exit_distance in {0, 9999} or exit_distance >= current_exit_distance:
        return False
    if target == env.layout.exit_position:
        return True
    threat_position = _threat_reference_position(env)
    if threat_position is None:
        return True
    if target == threat_position:
        return False
    monster_exit_distance = _path_distance(env, threat_position, env.layout.exit_position)
    if monster_exit_distance == 9999:
        return True
    if monster_exit_distance == 0:
        return False
    player_speed = max(1, int(getattr(env.config, "max_player_speed", 1)))
    monster_speed = max(
        1,
        int(
            getattr(
                env,
                "_active_monster_speed",
                getattr(env.config, "monster_speed", 1),
            )
        ),
    )
    player_turns_remaining = (exit_distance + player_speed - 1) // player_speed
    monster_turns_to_exit = (monster_exit_distance + monster_speed - 1) // monster_speed
    return player_turns_remaining <= monster_turns_to_exit
