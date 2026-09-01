from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from isr.episode_compare import (
    DatasetTriplet,
    common_video_keys,
    derive_dataset_triplet,
    extract_episode_index,
    find_dataset_root,
    load_source_indices,
    map_playback_frames,
    nearest_retained_frame,
    resolve_video_path,
)


def _write_dataset_root(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "chunks_size": 1000,
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observation.images.third_view": {"dtype": "video"},
            "observation.images.left_wrist_view": {"dtype": "video"},
            "observation.state.eef_pose": {"dtype": "float32"},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))


def _write_triplet(tmp_path: Path) -> DatasetTriplet:
    original = tmp_path / "task"
    va = tmp_path / "task_isr_va"
    vf = tmp_path / "task_isr_vf"
    for root in (original, va, vf):
        _write_dataset_root(root)
    return DatasetTriplet(original=original, va=va, vf=vf)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("episode_000123.parquet", 123),
        ("episode_000007.mp4", 7),
        ("prefix/episode_42/", 42),
    ],
)
def test_extract_episode_index_accepts_episode_paths(name: str, expected: int) -> None:
    assert extract_episode_index(name) == expected


def test_extract_episode_index_rejects_path_without_episode() -> None:
    with pytest.raises(ValueError, match="episode index"):
        extract_episode_index("/datasets/task")


def test_find_dataset_root_walks_up_from_episode_video(tmp_path: Path) -> None:
    triplet = _write_triplet(tmp_path)
    video = (
        triplet.original
        / "videos/chunk-000/observation.images.third_view/episode_000123.mp4"
    )
    video.parent.mkdir(parents=True)
    video.touch()

    assert find_dataset_root(video) == triplet.original.resolve()


def test_derive_triplet_from_va_episode_file(tmp_path: Path) -> None:
    triplet = _write_triplet(tmp_path)
    parquet = triplet.va / "data/chunk-000/episode_000123.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.touch()

    assert derive_dataset_triplet(parquet) == DatasetTriplet(
        original=triplet.original.resolve(),
        va=triplet.va.resolve(),
        vf=triplet.vf.resolve(),
    )


def test_derive_triplet_accepts_explicit_roots(tmp_path: Path) -> None:
    triplet = _write_triplet(tmp_path)

    result = derive_dataset_triplet(
        triplet.original,
        original_root=triplet.original,
        va_root=triplet.va,
        vf_root=triplet.vf,
    )

    assert result == DatasetTriplet(
        original=triplet.original.resolve(),
        va=triplet.va.resolve(),
        vf=triplet.vf.resolve(),
    )


def test_derive_triplet_rejects_missing_sibling(tmp_path: Path) -> None:
    original = tmp_path / "task"
    _write_dataset_root(original)

    with pytest.raises(FileNotFoundError, match="VA dataset"):
        derive_dataset_triplet(original)


def test_common_video_keys_returns_only_keys_shared_by_all_roots(tmp_path: Path) -> None:
    triplet = _write_triplet(tmp_path)
    vf_info_path = triplet.vf / "meta/info.json"
    vf_info = json.loads(vf_info_path.read_text())
    vf_info["features"].pop("observation.images.left_wrist_view")
    vf_info_path.write_text(json.dumps(vf_info))

    assert common_video_keys(triplet) == ["observation.images.third_view"]


def test_resolve_video_path_formats_chunk_episode_and_camera(tmp_path: Path) -> None:
    triplet = _write_triplet(tmp_path)
    expected = (
        triplet.original
        / "videos/chunk-001/observation.images.third_view/episode_001234.mp4"
    )
    expected.parent.mkdir(parents=True)
    expected.touch()

    assert (
        resolve_video_path(
            triplet.original,
            episode_index=1234,
            video_key="observation.images.third_view",
        )
        == expected.resolve()
    )


def test_load_source_indices_reads_requested_manifest_episode(tmp_path: Path) -> None:
    triplet = _write_triplet(tmp_path)
    records = [
        {"episode_index": 0, "selected_source_indices": [0, 2, 4]},
        {"episode_index": 1, "selected_source_indices": [0, 1, 4, 7]},
    ]
    (triplet.va / "meta/selection_manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )

    np.testing.assert_array_equal(
        load_source_indices(triplet.va, episode_index=1),
        [0, 1, 4, 7],
    )


def test_load_source_indices_rejects_non_monotonic_mapping(tmp_path: Path) -> None:
    triplet = _write_triplet(tmp_path)
    (triplet.vf / "meta/selection_manifest.jsonl").write_text(
        json.dumps(
            {"episode_index": 3, "selected_source_indices": [0, 4, 4, 8]}
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        load_source_indices(triplet.vf, episode_index=3)


@pytest.mark.parametrize(
    ("source_frame", "expected_output_frame"),
    [(0, 0), (3, 1), (4, 2), (6, 2), (8, 3)],
)
def test_nearest_retained_frame_uses_left_side_for_ties(
    source_frame: int,
    expected_output_frame: int,
) -> None:
    selected = np.asarray([0, 2, 5, 8])

    assert nearest_retained_frame(selected, source_frame) == expected_output_frame


def test_map_playback_frames_source_mode_aligns_by_recorded_source_frame() -> None:
    result = map_playback_frames(
        "source",
        tick=5,
        source_frame_count=10,
        va_source_indices=np.asarray([0, 2, 4, 6, 9]),
        vf_source_indices=np.asarray([0, 1, 5, 7, 9]),
    )

    assert (result.original_index, result.va_index, result.vf_index) == (5, 2, 2)
    assert (result.original_source, result.va_source, result.vf_source) == (5, 4, 5)


def test_map_playback_frames_native_mode_clamps_shorter_outputs() -> None:
    result = map_playback_frames(
        "native",
        tick=7,
        source_frame_count=10,
        va_source_indices=np.asarray([0, 2, 4, 6, 9]),
        vf_source_indices=np.asarray([0, 1, 5, 9]),
    )

    assert (result.original_index, result.va_index, result.vf_index) == (7, 4, 3)
    assert (result.original_source, result.va_source, result.vf_source) == (7, 9, 9)


def test_map_playback_frames_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        map_playback_frames(
            "progress",
            tick=0,
            source_frame_count=2,
            va_source_indices=np.asarray([0, 1]),
            vf_source_indices=np.asarray([0, 1]),
        )
