"""CLI parser coverage for observation and playback flags."""

from __future__ import annotations

import pytest

from maze_rl.cli import build_parser


def test_train_parser_accepts_local_tactical_view_flags() -> None:
    """Training CLI should expose additive local tactical view flags."""

    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--episodes",
            "10",
            "--enable-local-tactical-view",
            "--local-tactical-radius",
            "3",
            "--local-tactical-include-monster-memory",
        ]
    )

    assert args.command == "train"
    assert args.enable_local_tactical_view is True
    assert args.local_tactical_radius == 3
    assert args.local_tactical_include_monster_memory is True


def test_train_parser_accepts_explicit_curriculum_stages() -> None:
    """Training CLI should parse explicit whole-maze curriculum stages."""

    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--curriculum-stage",
            "0:9:9:1:10:60:20:1:bootstrap",
            "--curriculum-stage",
            "80:15:15:6:1:100:15:1:full",
        ]
    )

    assert args.curriculum_stage is not None
    assert args.curriculum_stage[0].rows == 9
    assert args.curriculum_stage[0].label == "bootstrap"
    assert args.curriculum_stage[1].rows == 15
    assert args.curriculum_stage[1].label == "full"


def test_watch_parser_defaults_to_raw_playback_mode() -> None:
    """Watch mode should default to honest learned-policy playback."""

    parser = build_parser()
    args = parser.parse_args(["watch", "--checkpoint", "checkpoints/ckpt_0000.zip", "--seed", "12345"])

    assert args.playback_mode == "raw"
    assert args.allow_policy_override is False


def test_showcase_parser_accepts_assisted_and_heuristic_modes() -> None:
    """Showcase CLI should allow explicit assisted and heuristic playback modes."""

    parser = build_parser()
    assisted_args = parser.parse_args(
        ["showcase", "--checkpoints", "0", "50", "--seed", "12345", "--playback-mode", "assisted"]
    )
    heuristic_args = parser.parse_args(
        ["showcase", "--checkpoints", "0", "50", "--seed", "12345", "--playback-mode", "heuristic"]
    )

    assert assisted_args.playback_mode == "assisted"
    assert heuristic_args.playback_mode == "heuristic"


def test_train_parser_rejects_invalid_curriculum_stage() -> None:
    """Malformed curriculum stage strings should fail during argument parsing."""

    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--curriculum-stage", "0:9:9"])