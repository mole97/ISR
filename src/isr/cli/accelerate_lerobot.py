"""CLI for non-destructive ISR acceleration of LeRobot 2.1 datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


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
    summary = convert_dataset(
        arguments.input,
        arguments.output,
        mode=arguments.mode,
        target_retention=arguments.target_retention,
        max_skip=arguments.max_skip,
        free_contact_seconds=arguments.free_contact_seconds,
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
