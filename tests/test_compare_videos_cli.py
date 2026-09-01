from __future__ import annotations

from pathlib import Path

import pytest

from isr.cli.compare_videos import (
    build_parser,
    fit_image_size,
    handle_drop_data,
    scaled_font_size,
)


def test_parser_accepts_explicit_dataset_triplet() -> None:
    arguments = build_parser().parse_args(
        [
            "--original-root",
            "/datasets/task",
            "--va-root",
            "/datasets/task_isr_va",
            "--vf-root",
            "/datasets/task_isr_vf",
            "--episode",
            "12",
            "--camera",
            "observation.images.left_wrist_view",
        ]
    )

    assert arguments.original_root == Path("/datasets/task")
    assert arguments.va_root == Path("/datasets/task_isr_va")
    assert arguments.vf_root == Path("/datasets/task_isr_vf")
    assert arguments.episode == 12
    assert arguments.camera == "observation.images.left_wrist_view"


def test_parser_defaults_to_empty_drop_window() -> None:
    arguments = build_parser().parse_args([])

    assert arguments.original_root is None
    assert arguments.va_root is None
    assert arguments.vf_root is None
    assert arguments.episode == 0
    assert arguments.camera == "observation.images.third_view"
    assert arguments.ui_scale == pytest.approx(2.4)


def test_parser_accepts_custom_ui_scale() -> None:
    arguments = build_parser().parse_args(["--ui-scale", "1.8"])

    assert arguments.ui_scale == pytest.approx(1.8)


def test_parser_rejects_nonpositive_ui_scale() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--ui-scale", "0"])


def test_handle_drop_data_loads_first_path_and_returns_copy_action() -> None:
    loaded: list[Path] = []

    action = handle_drop_data(
        "{/datasets/task one/episode_000001.parquet} /datasets/second.mp4",
        splitlist=lambda _data: (
            "/datasets/task one/episode_000001.parquet",
            "/datasets/second.mp4",
        ),
        load_path=loaded.append,
    )

    assert loaded == [Path("/datasets/task one/episode_000001.parquet")]
    assert action == "copy"


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        ((224, 224), (400, 500), (400, 400)),
        ((640, 480), (400, 400), (400, 300)),
        ((1080, 1920), (500, 300), (169, 300)),
    ],
)
def test_fit_image_size_scales_up_or_down_without_changing_aspect_ratio(
    source: tuple[int, int],
    target: tuple[int, int],
    expected: tuple[int, int],
) -> None:
    assert fit_image_size(source, target) == expected


def test_fit_image_size_rejects_nonpositive_dimensions() -> None:
    with pytest.raises(ValueError, match="positive"):
        fit_image_size((224, 0), (400, 400))


@pytest.mark.parametrize(
    ("font_size", "scale", "expected"),
    [(10, 1.5, 15), (9, 1.5, 14), (-12, 1.5, -18)],
)
def test_scaled_font_size_preserves_point_or_pixel_units(
    font_size: int,
    scale: float,
    expected: int,
) -> None:
    assert scaled_font_size(font_size, scale) == expected
