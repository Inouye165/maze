"""Viewer-render state regression tests."""

from maze_rl.render.view_state import (
    viewer_cell_color,
    viewer_grid,
    viewer_monster_position,
    viewer_player_position,
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
