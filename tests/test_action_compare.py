from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from isr.action_compare import (  # noqa: E402
    action_dimension_names,
    action_plot_series,
    common_action_keys,
    load_action_comparison,
    resolve_episode_data_path,
)
from isr.episode_compare import DatasetTriplet  # noqa: E402


def _fixed_list(values: np.ndarray):
    return pa.array(values.tolist(), type=pa.list_(pa.float32(), values.shape[1]))


def _write_dataset(
    root: Path,
    *,
    eef_pose: np.ndarray,
    joint_position: np.ndarray,
    selected_source_indices: list[int] | None,
) -> None:
    (root / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "actions.eef_pose": {
                "dtype": "float32",
                "shape": [10],
                "names": [["x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6", "gripper"]],
            },
            "actions.joint_position": {
                "dtype": "float32",
                "shape": [8],
                "names": [["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper"]],
            },
            "observation.state.eef_pose": {"dtype": "float32", "shape": [10]},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    parquet = root / "data/chunk-000/episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "actions.eef_pose": _fixed_list(eef_pose),
                "actions.joint_position": _fixed_list(joint_position),
            }
        ),
        parquet,
    )
    if selected_source_indices is not None:
        record = {
            "episode_index": 0,
            "selected_source_indices": selected_source_indices,
        }
        (root / "meta/selection_manifest.jsonl").write_text(json.dumps(record) + "\n")


@pytest.fixture
def action_triplet(tmp_path: Path) -> DatasetTriplet:
    original = tmp_path / "task"
    va = tmp_path / "task_isr_va"
    vf = tmp_path / "task_isr_vf"
    eef = np.arange(50, dtype=np.float32).reshape(5, 10)
    joint = np.arange(40, dtype=np.float32).reshape(5, 8)
    va_selected = [0, 2, 4]
    vf_selected = [0, 1, 3, 4]
    _write_dataset(
        original,
        eef_pose=eef,
        joint_position=joint,
        selected_source_indices=None,
    )
    _write_dataset(
        va,
        eef_pose=eef[va_selected],
        joint_position=joint[va_selected],
        selected_source_indices=va_selected,
    )
    _write_dataset(
        vf,
        eef_pose=eef[vf_selected],
        joint_position=joint[vf_selected],
        selected_source_indices=vf_selected,
    )
    return DatasetTriplet(original=original, va=va, vf=vf)


def test_common_action_keys_returns_intersection(action_triplet: DatasetTriplet) -> None:
    assert common_action_keys(action_triplet) == [
        "actions.eef_pose",
        "actions.joint_position",
    ]


def test_resolve_episode_data_path_formats_chunk_and_episode(tmp_path: Path) -> None:
    root = tmp_path / "task"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "chunks_size": 1000,
                "data_path": (
                    "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
                ),
            }
        )
    )
    expected = root / "data/chunk-001/episode_001234.parquet"
    expected.parent.mkdir(parents=True)
    expected.touch()

    assert resolve_episode_data_path(root, episode_index=1234) == expected.resolve()


def test_load_action_comparison_reads_values_and_source_mappings(
    action_triplet: DatasetTriplet,
) -> None:
    comparison = load_action_comparison(
        action_triplet,
        episode_index=0,
        action_key="actions.eef_pose",
    )

    assert comparison.action_key == "actions.eef_pose"
    assert comparison.fps == pytest.approx(30.0)
    assert comparison.original.shape == (5, 10)
    assert comparison.va.shape == (3, 10)
    assert comparison.vf.shape == (4, 10)
    np.testing.assert_array_equal(comparison.original_source_indices, [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(comparison.va_source_indices, [0, 2, 4])
    np.testing.assert_array_equal(comparison.vf_source_indices, [0, 1, 3, 4])
    np.testing.assert_array_equal(comparison.va, comparison.original[[0, 2, 4]])


def test_load_action_comparison_rejects_manifest_row_mismatch(
    action_triplet: DatasetTriplet,
) -> None:
    manifest = action_triplet.va / "meta/selection_manifest.jsonl"
    manifest.write_text(
        json.dumps({"episode_index": 0, "selected_source_indices": [0, 4]}) + "\n"
    )

    with pytest.raises(ValueError, match="VA action rows"):
        load_action_comparison(
            action_triplet,
            episode_index=0,
            action_key="actions.eef_pose",
        )


def test_action_dimension_names_reads_nested_lerobot_names(
    action_triplet: DatasetTriplet,
) -> None:
    assert action_dimension_names(
        action_triplet.original,
        action_key="actions.eef_pose",
        dimension_count=10,
    ) == ("x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6", "gripper")


def test_action_dimension_names_falls_back_when_names_are_missing(
    action_triplet: DatasetTriplet,
) -> None:
    info_path = action_triplet.original / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["actions.joint_position"].pop("names")
    info_path.write_text(json.dumps(info))

    assert action_dimension_names(
        action_triplet.original,
        action_key="actions.joint_position",
        dimension_count=8,
    ) == tuple(f"dim_{index}" for index in range(8))


def test_action_dimension_names_rejects_wrong_name_count(
    action_triplet: DatasetTriplet,
) -> None:
    info_path = action_triplet.original / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["actions.eef_pose"]["names"] = [["x", "y"]]
    info_path.write_text(json.dumps(info))

    with pytest.raises(ValueError, match="dimension names"):
        action_dimension_names(
            action_triplet.original,
            action_key="actions.eef_pose",
            dimension_count=10,
        )


def test_action_plot_series_uses_independent_compressed_times(
    action_triplet: DatasetTriplet,
) -> None:
    comparison = load_action_comparison(
        action_triplet,
        episode_index=0,
        action_key="actions.eef_pose",
    )

    plot = action_plot_series(comparison, dimension=0, axis_mode="compressed")

    np.testing.assert_allclose(plot.original_time, np.arange(5) / 30)
    np.testing.assert_allclose(plot.va_time, np.arange(3) / 30)
    np.testing.assert_allclose(plot.vf_time, np.arange(4) / 30)
    np.testing.assert_array_equal(plot.va_values, comparison.va[:, 0])
    assert plot.dimension_name == "x"


def test_action_plot_series_uses_recorded_source_times(
    action_triplet: DatasetTriplet,
) -> None:
    comparison = load_action_comparison(
        action_triplet,
        episode_index=0,
        action_key="actions.joint_position",
    )

    plot = action_plot_series(comparison, dimension=7, axis_mode="source")

    np.testing.assert_allclose(plot.original_time, np.arange(5) / 30)
    np.testing.assert_allclose(plot.va_time, np.asarray([0, 2, 4]) / 30)
    np.testing.assert_allclose(plot.vf_time, np.asarray([0, 1, 3, 4]) / 30)
    assert plot.dimension_name == "gripper"


def test_action_plot_series_rejects_invalid_dimension_or_mode(
    action_triplet: DatasetTriplet,
) -> None:
    comparison = load_action_comparison(
        action_triplet,
        episode_index=0,
        action_key="actions.eef_pose",
    )

    with pytest.raises(ValueError, match="dimension"):
        action_plot_series(comparison, dimension=10, axis_mode="compressed")
    with pytest.raises(ValueError, match="axis_mode"):
        action_plot_series(comparison, dimension=0, axis_mode="progress")
