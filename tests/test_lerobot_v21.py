from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from isr.lerobot_v21 import (  # noqa: E402
    _episode_selection,
    compute_episode_stats,
    convert_dataset,
    ensure_distinct_paths,
    probe_video,
    resample_episode_table,
    rewrite_info,
    rewrite_video,
)


def _fixed_list(values: list[list[float]], width: int):
    return pa.array(values, type=pa.list_(pa.float32(), width))


@pytest.fixture
def tiny_table():
    frame_count = 5
    state = [[float(index), 0, 0, 1, 0, 0, 0, 1, 0, 0.0] for index in range(frame_count)]
    action = [[100 + float(index), 0, 0, 1, 0, 0, 0, 1, 0, index] for index in range(frame_count)]
    return pa.table(
        {
            "observation.state.eef_pose": _fixed_list(state, 10),
            "observation.state.left_wrist_left_tactile": _fixed_list([[0, 0, 0, index] for index in range(5)], 4),
            "actions.eef_pose": _fixed_list(action, 10),
            "timestamp": pa.array(np.arange(frame_count) / 30, type=pa.float32()),
            "frame_index": pa.array(np.arange(frame_count), type=pa.int64()),
            "episode_index": pa.array(np.zeros(frame_count), type=pa.int64()),
            "index": pa.array(np.arange(20, 20 + frame_count), type=pa.int64()),
            "task_index": pa.array(np.zeros(frame_count), type=pa.int64()),
            "sampling.speed_force_weight": pa.array(np.ones(frame_count), type=pa.float32()),
        }
    )


def test_resample_table_preserves_action_state_phase(tiny_table) -> None:
    selected = np.asarray([0, 2, 4], dtype=np.int64)

    output = resample_episode_table(tiny_table, selected, fps=30.0, global_start=7)

    expected = tiny_table.take(pa.array(selected))
    assert output["observation.state.eef_pose"].to_pylist() == expected["observation.state.eef_pose"].to_pylist()
    assert output["actions.eef_pose"].to_pylist() == expected["actions.eef_pose"].to_pylist()
    assert output["observation.state.left_wrist_left_tactile"].to_pylist() == expected[
        "observation.state.left_wrist_left_tactile"
    ].to_pylist()
    assert output["source_frame_index"].to_pylist() == [0, 2, 4]


def test_episode_selection_forces_gripper_changes_from_second_arm() -> None:
    frame_count = 6
    first_arm = [
        [float(index), 0, 0, 1, 0, 0, 0, 1, 0, 0.0]
        for index in range(frame_count)
    ]
    second_arm = [
        [0, float(index), 0, 1, 0, 0, 0, 1, 0, float(index >= 3)]
        for index in range(frame_count)
    ]
    table = pa.table(
        {
            "observation.state.eef_pose": _fixed_list(
                [left + right for left, right in zip(first_arm, second_arm)],
                20,
            )
        }
    )

    _, manifest = _episode_selection(
        table,
        mode="va",
        fps=30.0,
        target_retention=0.5,
        max_skip=4,
        free_contact_seconds=1.0,
    )

    assert {2, 3}.issubset(manifest["forced_source_indices"])


def test_episode_selection_ignores_gripper_noise_below_tolerance() -> None:
    gripper = [0.0, 5e-5, 0.0, 0.01, 0.01, 0.01]
    poses = [
        [float(index), 0, 0, 1, 0, 0, 0, 1, 0, gripper[index]]
        for index in range(len(gripper))
    ]
    table = pa.table({"observation.state.eef_pose": _fixed_list(poses, 10)})

    _, manifest = _episode_selection(
        table,
        mode="va",
        fps=30.0,
        target_retention=0.5,
        max_skip=4,
        free_contact_seconds=1.0,
        gripper_change_tolerance=1e-4,
    )

    assert manifest["forced_source_indices"] == [2, 3]


def test_resample_table_rebuilds_time_and_indices(tiny_table) -> None:
    output = resample_episode_table(tiny_table, np.asarray([0, 2, 4]), fps=20.0, global_start=7)

    np.testing.assert_allclose(output["timestamp"].to_numpy(), [0.0, 0.05, 0.1])
    assert output["frame_index"].to_pylist() == [0, 1, 2]
    assert output["index"].to_pylist() == [7, 8, 9]
    assert "sampling.speed_force_weight" not in output.column_names


@pytest.mark.parametrize("selected", [np.asarray([0, 0]), np.asarray([2, 1]), np.asarray([-1, 2]), np.asarray([0, 5])])
def test_resample_table_rejects_invalid_selected_indices(tiny_table, selected) -> None:
    with pytest.raises(ValueError, match="selected_indices"):
        resample_episode_table(tiny_table, selected, fps=30.0, global_start=0)


def test_rewrite_info_removes_stale_sampling_contract_and_adds_audit() -> None:
    info = {
        "codebase_version": "v2.1",
        "total_episodes": 2,
        "total_frames": 10,
        "features": {
            "actions.eef_pose": {"dtype": "float32", "shape": [10]},
            "sampling.speed_force_weight": {"dtype": "float32", "shape": [1]},
        },
        "speed_force_sampling": {"weight_key": "sampling.speed_force_weight"},
    }

    output = rewrite_info(info, total_frames=6, mode="vf", target_retention=0.5, max_skip=4)

    assert output["total_frames"] == 6
    assert "sampling.speed_force_weight" not in output["features"]
    assert "speed_force_sampling" not in output
    assert output["features"]["source_frame_index"] == {"dtype": "int64", "shape": [1], "names": None}
    assert output["trajectory_acceleration"]["mode"] == "vf"
    assert output["trajectory_acceleration"]["gripper_change_tolerance"] == pytest.approx(1e-4)
    assert info["total_frames"] == 10


def test_compute_episode_stats_uses_resampled_values(tiny_table) -> None:
    output = resample_episode_table(tiny_table, np.asarray([0, 2, 4]), fps=30.0, global_start=0)

    stats = compute_episode_stats(output)

    pose_stats = stats["observation.state.eef_pose"]
    assert pose_stats["min"][0] == 0.0
    assert pose_stats["max"][0] == 4.0
    assert pose_stats["mean"][0] == pytest.approx(2.0)
    assert pose_stats["count"] == [3]
    assert stats["timestamp"]["max"] == pytest.approx([2 / 30])
    assert stats["source_frame_index"]["max"] == [4.0]
    assert "sampling.speed_force_weight" not in stats


def test_ensure_distinct_paths_rejects_same_or_nested_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="different"):
        ensure_distinct_paths(source, source)
    with pytest.raises(ValueError, match="inside"):
        ensure_distinct_paths(source, source / "output")


def test_ensure_distinct_paths_does_not_create_or_modify_paths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    marker = source / "marker.json"
    marker.write_text(json.dumps({"untouched": True}))
    output = tmp_path / "new-output"

    resolved_source, resolved_output = ensure_distinct_paths(source, output)

    assert resolved_source == source.resolve()
    assert resolved_output == output.resolve()
    assert not output.exists()
    assert json.loads(marker.read_text()) == {"untouched": True}


def test_convert_dataset_rejects_existing_output_before_writing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("untouched")

    with pytest.raises(FileExistsError, match="already exists"):
        convert_dataset(
            source,
            output,
            mode="va",
            target_retention=0.5,
            max_skip=4,
        )

    assert marker.read_text() == "untouched"


@pytest.mark.parametrize("video_workers", [0, -1])
def test_convert_dataset_rejects_nonpositive_video_workers(
    tmp_path: Path,
    video_workers: int,
) -> None:
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="video_workers"):
        convert_dataset(
            source,
            tmp_path / "output",
            mode="va",
            target_retention=0.5,
            max_skip=4,
            video_workers=video_workers,
        )


def test_rewrite_video_selects_exact_source_frames(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    destination = tmp_path / "output.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=32x32:rate=30:duration=0.2",
            "-frames:v",
            "6",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )

    rewrite_video(
        source,
        destination,
        np.asarray([0, 2, 5]),
        fps=30.0,
        expected_source_frame_count=6,
    )

    source_info = probe_video(source)
    output_info = probe_video(destination)
    assert source_info.frame_count == 6
    assert output_info.frame_count == 3
    assert output_info.fps == pytest.approx(30.0)


def test_rewrite_video_rejects_unexpected_source_frame_count(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    destination = tmp_path / "output.mp4"
    _write_test_video(source, frame_count=6)

    with pytest.raises(ValueError, match="source video has 6 frames, expected 7"):
        rewrite_video(
            source,
            destination,
            np.asarray([0, 2, 5]),
            fps=30.0,
            expected_source_frame_count=7,
        )


def _write_test_video(path: Path, frame_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=32x32:rate=30:duration={frame_count / 30}",
            "-frames:v",
            str(frame_count),
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def _write_tiny_lerobot_dataset(root: Path, table) -> None:
    (root / "meta").mkdir(parents=True)
    data_path = root / "data/chunk-000/episode_000000.parquet"
    data_path.parent.mkdir(parents=True)
    pq.write_table(table, data_path)
    video_features = {
        key: {
            "dtype": "video",
            "shape": [32, 32, 3],
            "names": ["height", "width", "channels"],
            "info": {"video.fps": 30, "video.codec": "h264", "video.pix_fmt": "yuv420p"},
        }
        for key in ("observation.images.third_view", "observation.images.left_wrist_view")
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "test",
        "fps": 30,
        "total_episodes": 1,
        "total_frames": table.num_rows,
        "total_tasks": 1,
        "total_videos": 2,
        "total_chunks": 1,
        "chunks_size": 1000,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state.eef_pose": {"dtype": "float32", "shape": [10], "names": None},
            "observation.state.left_wrist_left_tactile": {"dtype": "float32", "shape": [4], "names": None},
            "actions.eef_pose": {"dtype": "float32", "shape": [10], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "sampling.speed_force_weight": {"dtype": "float32", "shape": [1], "names": None},
            **video_features,
        },
        "speed_force_sampling": {"weight_key": "sampling.speed_force_weight"},
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "test"}) + "\n")
    (root / "meta/episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": ["test"], "length": table.num_rows}) + "\n"
    )
    source_stats = {
        "episode_index": 0,
        "stats": {
            key: {
                "min": [[[0.0]], [[0.0]], [[0.0]]],
                "max": [[[1.0]], [[1.0]], [[1.0]]],
                "mean": [[[0.5]], [[0.5]], [[0.5]]],
                "std": [[[0.1]], [[0.1]], [[0.1]]],
                "count": [1],
            }
            for key in video_features
        },
    }
    (root / "meta/episodes_stats.jsonl").write_text(json.dumps(source_stats) + "\n")
    for key in video_features:
        _write_test_video(root / f"videos/chunk-000/{key}/episode_000000.mp4", table.num_rows)


def test_convert_dataset_encodes_videos_concurrently_and_reports_progress(
    tmp_path: Path,
    tiny_table,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "output"
    _write_tiny_lerobot_dataset(source, tiny_table)
    lock = threading.Lock()
    encoders_started = threading.Barrier(2)
    active = 0
    maximum_active = 0
    progress: list[tuple[int, int]] = []

    def fake_rewrite_video(
        source_path,
        destination_path,
        selected_indices,
        *,
        fps,
        expected_source_frame_count,
    ) -> None:
        nonlocal active, maximum_active
        assert expected_source_frame_count == tiny_table.num_rows
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        encoders_started.wait(timeout=2.0)
        Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
        Path(destination_path).touch()
        with lock:
            active -= 1

    monkeypatch.setattr("isr.lerobot_v21.rewrite_video", fake_rewrite_video)

    convert_dataset(
        source,
        destination,
        mode="va",
        target_retention=0.6,
        max_skip=2,
        video_workers=2,
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )

    assert maximum_active == 2
    assert progress == [(1, 1)]


@pytest.mark.parametrize("mode", ["va", "vf"])
def test_convert_dataset_writes_new_aligned_dataset_without_mutating_input(
    tmp_path: Path,
    tiny_table,
    mode: str,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / mode
    _write_tiny_lerobot_dataset(source, tiny_table)
    source_info_before = (source / "meta/info.json").read_bytes()
    source_parquet_before = (source / "data/chunk-000/episode_000000.parquet").read_bytes()

    summary = convert_dataset(
        source,
        destination,
        mode=mode,
        target_retention=0.6,
        max_skip=2,
        free_contact_seconds=0.05,
    )

    assert destination.is_dir()
    assert summary.total_source_frames == 5
    assert 3 <= summary.total_output_frames < 5
    output_info = json.loads((destination / "meta/info.json").read_text())
    assert output_info["total_frames"] == summary.total_output_frames
    assert output_info["trajectory_acceleration"]["mode"] == mode
    output_table = pq.read_table(destination / "data/chunk-000/episode_000000.parquet")
    selected = np.asarray(output_table["source_frame_index"].to_pylist())
    expected_actions = tiny_table.take(pa.array(selected))["actions.eef_pose"].to_pylist()
    assert output_table["actions.eef_pose"].to_pylist() == expected_actions
    assert "sampling.speed_force_weight" not in output_table.column_names
    for key in ("observation.images.third_view", "observation.images.left_wrist_view"):
        video = destination / f"videos/chunk-000/{key}/episode_000000.mp4"
        assert probe_video(video).frame_count == output_table.num_rows
    assert (source / "meta/info.json").read_bytes() == source_info_before
    assert (source / "data/chunk-000/episode_000000.parquet").read_bytes() == source_parquet_before
