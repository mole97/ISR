"""CLI for non-destructive ISR acceleration of LeRobot 2.1 datasets."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_conversion_progress(
    completed: int,
    total: int,
    *,
    elapsed_seconds: float,
) -> str:
    """Format deterministic episode progress with a running-average ETA."""
    if completed < 1 or total < completed:
        raise ValueError("progress must satisfy 1 <= completed <= total")
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise ValueError("elapsed_seconds must be finite and nonnegative")
    eta_seconds = elapsed_seconds / completed * (total - completed)
    return (
        f"Episode {completed}/{total} ({100.0 * completed / total:.1f}%) | "
        f"elapsed {_format_duration(elapsed_seconds)} | ETA {_format_duration(eta_seconds)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="isr-accelerate-lerobot",
        description="Create a new time-compressed LeRobot 2.1 dataset without modifying the input.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Read-only source LeRobot 2.1 dataset root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output dataset root; it must not exist.",
    )
    parser.add_argument(
        "--mode",
        choices=("va", "vf"),
        required=True,
        help="VA uses velocity/acceleration; VF uses velocity/force events.",
    )
    parser.add_argument(
        "--target-retention",
        type=float,
        default=0.5,
        help="Requested output/source frame ratio before hard retention constraints (default: 0.5).",
    )
    parser.add_argument(
        "--max-skip",
        type=int,
        default=4,
        help="Maximum source-frame gap between adjacent output frames (default: 4).",
    )
    parser.add_argument(
        "--free-contact-seconds",
        type=float,
        default=1.0,
        help="Contact-free episode prefix used to calibrate VF force noise (default: 1.0).",
    )
    parser.add_argument(
        "--gripper-change-tolerance",
        type=_positive_float,
        default=1e-4,
        help="Minimum per-frame gripper change forced as an event (default: 1e-4).",
    )
    parser.add_argument(
        "--video-workers",
        type=_positive_int,
        default=4,
        help="Maximum concurrent video encoders per episode (default: 4).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        from isr.lerobot_v21 import convert_dataset
    except ModuleNotFoundError as error:
        if error.name in {"pyarrow", "pandas"}:
            raise SystemExit(
                "LeRobot conversion dependencies are missing; "
                "install with: pip install -e '.[lerobot]'"
            ) from error
        raise
    started = time.monotonic()

    def report_progress(completed: int, total: int) -> None:
        print(
            format_conversion_progress(
                completed,
                total,
                elapsed_seconds=time.monotonic() - started,
            ),
            file=sys.stderr,
            flush=True,
        )

    summary = convert_dataset(
        arguments.input,
        arguments.output,
        mode=arguments.mode,
        target_retention=arguments.target_retention,
        max_skip=arguments.max_skip,
        free_contact_seconds=arguments.free_contact_seconds,
        gripper_change_tolerance=arguments.gripper_change_tolerance,
        video_workers=arguments.video_workers,
        progress_callback=report_progress,
    )
    print(
        json.dumps(
            {
                "input": str(summary.input_root),
                "output": str(summary.output_root),
                "mode": summary.mode,
                "episodes": summary.episodes,
                "source_frames": summary.total_source_frames,
                "output_frames": summary.total_output_frames,
                "achieved_speedup": summary.achieved_speedup,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
