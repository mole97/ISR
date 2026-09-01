from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from isr.cli import accelerate_lerobot

build_parser = accelerate_lerobot.build_parser
format_conversion_progress = accelerate_lerobot.format_conversion_progress


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
    assert arguments.gripper_change_tolerance == pytest.approx(1e-4)
    assert arguments.video_workers == 4


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


def test_parser_accepts_conversion_performance_overrides() -> None:
    arguments = build_parser().parse_args(
        [
            "--input",
            "/datasets/source",
            "--output",
            "/datasets/output",
            "--mode",
            "va",
            "--gripper-change-tolerance",
            "0.001",
            "--video-workers",
            "2",
        ]
    )

    assert arguments.gripper_change_tolerance == pytest.approx(0.001)
    assert arguments.video_workers == 2


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--gripper-change-tolerance", "0"),
        ("--gripper-change-tolerance", "-0.1"),
        ("--video-workers", "0"),
        ("--video-workers", "-1"),
    ],
)
def test_parser_rejects_nonpositive_conversion_options(option: str, value: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--input",
                "/datasets/source",
                "--output",
                "/datasets/output",
                "--mode",
                "va",
                option,
                value,
            ]
        )


def test_format_conversion_progress_reports_elapsed_and_eta() -> None:
    assert format_conversion_progress(3, 10, elapsed_seconds=12.0) == (
        "Episode 3/10 (30.0%) | elapsed 00:12 | ETA 00:28"
    )


def test_main_forwards_options_and_keeps_progress_out_of_json_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_convert(input_root, output_root, **kwargs):
        captured.update(kwargs)
        kwargs["progress_callback"](1, 2)
        return SimpleNamespace(
            input_root=Path(input_root),
            output_root=Path(output_root),
            mode=kwargs["mode"],
            episodes=2,
            total_source_frames=100,
            total_output_frames=50,
            achieved_speedup=2.0,
        )

    monotonic_values = iter([100.0, 112.0])
    monkeypatch.setattr("isr.lerobot_v21.convert_dataset", fake_convert)
    monkeypatch.setattr(
        accelerate_lerobot,
        "time",
        SimpleNamespace(monotonic=lambda: next(monotonic_values)),
        raising=False,
    )

    result = accelerate_lerobot.main(
        [
            "--input",
            "/datasets/source",
            "--output",
            "/datasets/output",
            "--mode",
            "va",
            "--gripper-change-tolerance",
            "0.001",
            "--video-workers",
            "2",
        ]
    )

    output = capsys.readouterr()
    assert result == 0
    assert captured["gripper_change_tolerance"] == pytest.approx(0.001)
    assert captured["video_workers"] == 2
    assert "Episode 1/2" in output.err
    assert json.loads(output.out)["achieved_speedup"] == pytest.approx(2.0)
