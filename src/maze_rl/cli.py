"""Command-line interface for Maze RL Lab."""

from __future__ import annotations

import argparse
from pathlib import Path

from maze_rl.config import MazeConfig, TrainingConfig
from maze_rl.policies.model_factory import CheckpointCompatibilityError
from maze_rl.render.control_app import run_app
from maze_rl.render.replay_viewer import ReplayViewer
from maze_rl.training.checkpointing import resolve_checkpoint_path
from maze_rl.training.evaluate import evaluate_checkpoint
from maze_rl.training.showcase import format_showcase_table, run_showcase_headless, save_showcase_summary
from maze_rl.training.train import format_training_progress, train_from_scratch


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(description="Maze RL Lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train from scratch")
    train_parser.add_argument("--episodes", type=int, default=500)
    train_parser.add_argument("--algorithm", choices=["ppo", "maskable_ppo", "recurrent_ppo"], default="ppo")
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument("--held-out-seed", type=int, default=12_345)
    train_parser.add_argument("--rows", type=int, default=15)
    train_parser.add_argument("--cols", type=int, default=15)
    train_parser.add_argument("--checkpoint-interval", type=int, default=250)
    train_parser.add_argument("--disable-curriculum", action="store_true")

    eval_parser = subparsers.add_parser("eval", help="Evaluate one checkpoint")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_seed_group = eval_parser.add_mutually_exclusive_group(required=True)
    eval_seed_group.add_argument("--seed", type=int)
    eval_seed_group.add_argument("--seeds", nargs="+", type=int)
    eval_parser.add_argument("--episodes", type=int, default=1)
    eval_parser.add_argument("--debug-trace", action="store_true")

    watch_parser = subparsers.add_parser("watch", help="Watch one checkpoint")
    watch_parser.add_argument("--checkpoint", required=True)
    watch_parser.add_argument("--seed", type=int, required=True)
    watch_parser.add_argument("--fps", type=int, default=10)
    watch_parser.add_argument("--debug-trace", action="store_true")
    watch_parser.add_argument("--allow-policy-override", action="store_true")

    compare_parser = subparsers.add_parser("compare", help="Compare multiple checkpoints on one seed")
    compare_parser.add_argument("--checkpoints", nargs="+", type=int, required=True)
    compare_parser.add_argument("--seed", type=int, required=True)
    compare_parser.add_argument("--checkpoint-dir", default="checkpoints")

    showcase_parser = subparsers.add_parser("showcase", help="Sequential checkpoint playback on one held-out seed")
    showcase_parser.add_argument("--checkpoints", nargs="+", type=int, required=True)
    showcase_parser.add_argument("--seed", type=int, required=True)
    showcase_parser.add_argument("--checkpoint-dir", default="checkpoints")
    showcase_parser.add_argument("--headless", action="store_true")
    showcase_parser.add_argument("--pause-ms", type=int, default=1500)
    showcase_parser.add_argument("--fps", type=int, default=10)
    showcase_parser.add_argument("--max-no-progress-streak", type=int, default=25)
    showcase_parser.add_argument("--wall-timeout-s", type=float, default=30.0)
    showcase_parser.add_argument("--save-summary-json", default=None)
    showcase_parser.add_argument("--debug-trace", action="store_true")
    showcase_parser.add_argument("--allow-policy-override", action="store_true")

    app_parser = subparsers.add_parser("app", help="Open the local control panel app")
    app_parser.add_argument("--checkpoint-dir", default="checkpoints")
    app_parser.add_argument("--auto-quit-ms", type=int, default=None)

    return parser


def main() -> None:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        maze_config = MazeConfig(rows=args.rows, cols=args.cols, held_out_seed=args.held_out_seed)
        if args.disable_curriculum:
            maze_config = MazeConfig(
                rows=args.rows,
                cols=args.cols,
                held_out_seed=args.held_out_seed,
                curriculum_enabled=False,
                reward=maze_config.reward,
            )
        training_config = TrainingConfig(
            episodes=args.episodes,
            algorithm=args.algorithm,
            seed=args.seed,
            recurring_checkpoint_interval=args.checkpoint_interval,
        )
        last_progress_line: str | None = None

        def _report_progress(progress: dict[str, object]) -> None:
            nonlocal last_progress_line
            line = format_training_progress(progress)
            if line == last_progress_line:
                return
            last_progress_line = line
            print(line)

        artifacts = train_from_scratch(
            training_config=training_config,
            maze_config=maze_config,
            progress_callback=_report_progress,
        )
        print(f"training complete | episodes={artifacts.final_episode_count} | timesteps={artifacts.total_timesteps}")
        print(f"checkpoints: {artifacts.checkpoint_dir}")
        return

    if args.command == "eval":
        try:
            summary = evaluate_checkpoint(
                args.checkpoint,
                seed=args.seed,
                episodes=args.episodes,
                seeds=args.seeds,
                debug_trace=args.debug_trace,
            )
        except CheckpointCompatibilityError as error:
            parser.exit(2, f"checkpoint compatibility error: {error}\n")
        print(f"checkpoint: {summary.checkpoint}")
        if summary.seeds:
            print(f"seeds: {summary.seeds} | episodes: {summary.episodes}")
        else:
            print(f"seed: {summary.seed} | episodes: {summary.episodes}")
        print(f"outcomes: {summary.outcomes}")
        print(
            "metrics: "
            f"escape_rate={summary.escape_rate:.2f} "
            f"coverage={summary.average_coverage:.2f} "
            f"steps={summary.average_steps:.1f} "
            f"revisits={summary.average_revisits:.1f} "
            f"oscillations={summary.average_oscillations:.1f} "
            f"dead_ends={summary.average_dead_end_entries:.1f} "
            f"start_monster_distance={summary.average_start_monster_distance:.1f} "
            f"time_to_capture={summary.average_time_to_capture if summary.average_time_to_capture is not None else 'n/a'} "
            f"frontier_rate={summary.frontier_reached_rate:.2f} "
            f"peak_no_progress={summary.average_peak_no_progress_steps:.1f}"
        )
        return

    if args.command == "watch":
        try:
            outcome = ReplayViewer().watch(
                args.checkpoint,
                seed=args.seed,
                fps=args.fps,
                debug_trace=args.debug_trace,
                allow_policy_override=args.allow_policy_override,
            )
        except CheckpointCompatibilityError as error:
            parser.exit(2, f"checkpoint compatibility error: {error}\n")
        print(f"final outcome: {outcome}")
        return

    if args.command == "compare":
        for checkpoint_episode in args.checkpoints:
            checkpoint_path = resolve_checkpoint_path(args.checkpoint_dir, checkpoint_episode)
            if not Path(checkpoint_path).exists():
                print(f"ckpt {checkpoint_episode:04d} | missing | skipping")
                continue
            try:
                summary = evaluate_checkpoint(checkpoint_path, seed=args.seed, episodes=1)
            except (CheckpointCompatibilityError, FileNotFoundError) as error:
                print(f"ckpt {checkpoint_episode:04d} | unavailable | {error} | skipping")
                continue
            print(
                f"ckpt {checkpoint_episode:04d} | outcome={summary.outcomes} | "
                f"escape_rate={summary.escape_rate:.2f} | coverage={summary.average_coverage:.2f} | "
                f"steps={summary.average_steps:.1f} | revisits={summary.average_revisits:.1f} | "
                f"oscillations={summary.average_oscillations:.1f} | dead_ends={summary.average_dead_end_entries:.1f} | "
                f"frontier_rate={summary.frontier_reached_rate:.2f} | time_to_capture={summary.average_time_to_capture if summary.average_time_to_capture is not None else 'n/a'}"
            )
        return

    if args.command == "showcase":
        checkpoint_entries = [
            (checkpoint_episode, resolve_checkpoint_path(args.checkpoint_dir, checkpoint_episode))
            for checkpoint_episode in args.checkpoints
        ]
        if args.headless:
            try:
                results = run_showcase_headless(
                    checkpoint_dir=args.checkpoint_dir,
                    checkpoints=args.checkpoints,
                    seed=args.seed,
                    max_no_progress_streak=args.max_no_progress_streak,
                    wall_time_timeout_s=args.wall_timeout_s,
                    debug_trace=args.debug_trace,
                    allow_policy_override=args.allow_policy_override,
                )
            except CheckpointCompatibilityError as error:
                parser.exit(2, f"checkpoint compatibility error: {error}\n")
        else:
            try:
                results = ReplayViewer().showcase(
                    checkpoint_entries=checkpoint_entries,
                    seed=args.seed,
                    fps=args.fps,
                    pause_ms=args.pause_ms,
                    max_no_progress_streak=args.max_no_progress_streak,
                    wall_time_timeout_s=args.wall_timeout_s,
                    debug_trace=args.debug_trace,
                    allow_policy_override=args.allow_policy_override,
                )
            except CheckpointCompatibilityError as error:
                parser.exit(2, f"checkpoint compatibility error: {error}\n")
            missing = {item.checkpoint for item in results if item.status == "missing"}
            for checkpoint_episode in args.checkpoints:
                label = f"ckpt {checkpoint_episode:04d}"
                if label in missing:
                    print(f"{label} | missing | skipping")
        summary_path = save_showcase_summary(results, seed=args.seed, output_path=args.save_summary_json)
        for line in format_showcase_table(results):
            print(line)
        print(f"summary_json: {summary_path}")
        return

    if args.command == "app":
        run_app(checkpoint_dir=args.checkpoint_dir, auto_quit_ms=args.auto_quit_ms)
        return

    parser.error(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
