from __future__ import annotations

import pytest

from isr.cli.accelerate_lerobot import build_parser


def test_parser_accepts_va_conversion() -> None:
    arguments = build_parser().parse_args(
        [
            "--input",
            "/datasets/source",
            "--output",
            "/datasets/output-va",
            "--mode",
            "va",
        ]
    )

    assert arguments.mode == "va"
    assert arguments.target_retention == pytest.approx(0.5)
    assert arguments.max_skip == 4


def test_parser_accepts_vf_force_calibration() -> None:
    arguments = build_parser().parse_args(
        [
            "--input",
            "/datasets/source",
            "--output",
            "/datasets/output-vf",
            "--mode",
            "vf",
            "--free-contact-seconds",
            "0.5",
        ]
    )

    assert arguments.mode == "vf"
    assert arguments.free_contact_seconds == pytest.approx(0.5)


def test_parser_rejects_unknown_mode() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--input", "/datasets/source", "--output", "/datasets/output", "--mode", "unknown"]
        )
