"""Viewer-render state regression tests."""

from maze_rl.render.view_state import (
    viewer_cell_color,
    viewer_dead_end_cells,
    viewer_exit_color,
    viewer_explored_cells,
    viewer_grid,
    viewer_monster_position,
    viewer_policy_badge,
    viewer_player_position,
    viewer_traveled_cells,
    viewer_visible_cells,
)


def test_viewer_uses_full_grid_instead_of_npc_knowledge_grid() -> None:
    """The human viewer should see the full maze layout, not the NPC fog-of-war grid."""

    state = {
        "grid": (
            "#####",
            "#?.?#",
            "#####",
        ),
        "full_grid": (
            "#####",
            "#...#",
            "#####",
        ),
    }

    assert viewer_grid(state) == state["full_grid"]


def test_viewer_visibility_cells_are_parsed_for_shading() -> None:
    """Visible cells should be available to tint the currently seen area."""

    state = {"visible_cells": [(1, 1), [1, 2], (2, 2)]}

    assert viewer_visible_cells(state) == {(1, 1), (1, 2), (2, 2)}
    assert viewer_cell_color(".", True) != viewer_cell_color(".", False)
    assert viewer_cell_color("#", True) != viewer_cell_color("#", False)


def test_viewer_keeps_monster_drawn_even_when_npc_cannot_see_it() -> None:
    """Viewer rendering should keep the monster visible without changing NPC vision state."""

    state = {
        "player_position": (1, 1),
        "monster_position": (1, 6),
        "rendered_player_position": (1, 2),
        "rendered_monster_position": (1, 5),
        "monster_visible": False,
    }

    assert viewer_player_position(state) == (1, 2)
    assert viewer_monster_position(state) == (1, 5)
    assert state["monster_visible"] is False


def test_viewer_colors_distinguish_explored_and_dead_end_cells() -> None:
    """Viewer shading should separate unexplored floor, explored floor, and known dead ends."""

    state = {
        "explored_cells": [(1, 1), (1, 2)],
        "known_dead_end_cells": [(1, 2)],
    }

    assert viewer_explored_cells(state) == {(1, 1), (1, 2)}
    assert viewer_dead_end_cells(state) == {(1, 2)}
    assert viewer_cell_color(".", False, is_explored=False, is_dead_end=False) != viewer_cell_color(
        ".", False, is_explored=True, is_dead_end=False
    )
    assert viewer_cell_color(".", False, is_explored=True, is_dead_end=False) != viewer_cell_color(
        ".", False, is_explored=True, is_dead_end=True
    )


def test_viewer_traveled_cells_are_distinct_from_explored_cells() -> None:
    """Traveled cells should be available for a stronger path tint than mere exploration."""

    state = {
        "explored_cells": [(1, 1), (1, 2), (1, 3)],
        "traveled_cells": [(1, 1), (1, 2)],
    }

    assert viewer_explored_cells(state) == {(1, 1), (1, 2), (1, 3)}
    assert viewer_traveled_cells(state) == {(1, 1), (1, 2)}
    assert viewer_cell_color(".", False, is_explored=True, is_traveled=False) != viewer_cell_color(
        ".", False, is_explored=True, is_traveled=True
    )


def test_viewer_policy_badge_reports_override_state() -> None:
    """Viewer badge should summarize whether policy safety override is active."""

    label, _bg, _fg = viewer_policy_badge(
        {
            "policy_kind": "heuristic-override",
            "policy_decision_label": "safety override: break loop",
        }
    )

    assert label == "safety override: break loop"


def test_viewer_exit_color_brightens_after_exit_is_seen() -> None:
    """The exit marker should change color once the human has seen the exit."""

    unseen_state = {"exit_seen": False}
    seen_state = {"exit_seen": True}

    assert viewer_exit_color(unseen_state) != viewer_exit_color(seen_state)
