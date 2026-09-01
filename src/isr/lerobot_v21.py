"""Non-destructive helpers for rewriting accelerated LeRobot 2.1 datasets."""

from __future__ import annotations

import copy
import dataclasses
import json
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .trajectory_acceleration import compute_va_priority
from .trajectory_acceleration import compute_vf_priority
from .trajectory_acceleration import find_gripper_change_indices
from .trajectory_acceleration import select_isr_indices


SAMPLING_WEIGHT_KEY = "sampling.speed_force_weight"
SOURCE_FRAME_KEY = "source_frame_index"


@dataclasses.dataclass(frozen=True)
class VideoInfo:
    frame_count: int
    fps: float
    width: int
    height: int


@dataclasses.dataclass(frozen=True)
class ConversionSummary:
    input_root: Path
    output_root: Path
    mode: str
    episodes: int
    total_source_frames: int
    total_output_frames: int

    @property
    def achieved_speedup(self) -> float:
        return self.total_source_frames / self.total_output_frames


def ensure_distinct_paths(input_root: str | Path, output_root: str | Path) -> tuple[Path, Path]:
    """Resolve and validate non-overlapping input/output dataset roots without writing."""
    source = Path(input_root).resolve()
    output = Path(output_root).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"input dataset directory does not exist: {source}")
    if source == output:
        raise ValueError("input and output dataset paths must be different")
    if source in output.parents:
        raise ValueError("output dataset must not be inside the input dataset")
    if output in source.parents:
        raise ValueError("output dataset must not contain the input dataset")
    return source, output


def _replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    position = table.schema.get_field_index(name)
    if position < 0:
        return table.append_column(name, values)
    return table.set_column(position, name, values)


def resample_episode_table(
    table: pa.Table,
    selected_indices: np.ndarray,
    *,
    fps: float,
    global_start: int,
) -> pa.Table:
    """Take the same source rows from every signal and rebuild temporal indices."""
    selected = np.asarray(selected_indices, dtype=np.int64)
    if (
        selected.ndim != 1
        or selected.size == 0
        or np.any(selected < 0)
        or np.any(selected >= table.num_rows)
        or np.any(np.diff(selected) <= 0)
    ):
        raise ValueError("selected_indices must be a nonempty, strictly increasing in-bounds vector")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    if global_start < 0:
        raise ValueError("global_start must be nonnegative")

    source_frame = (
        np.asarray(table["frame_index"].combine_chunks().to_numpy(), dtype=np.int64)[selected]
        if "frame_index" in table.column_names
        else selected
    )
    output = table.take(pa.array(selected, type=pa.int64()))
    if SAMPLING_WEIGHT_KEY in output.column_names:
        output = output.drop_columns([SAMPLING_WEIGHT_KEY])
    frame_count = selected.size
    output = _replace_column(
        output,
        "timestamp",
        pa.array(np.arange(frame_count, dtype=np.float32) / np.float32(fps), type=pa.float32()),
    )
    output = _replace_column(output, "frame_index", pa.array(np.arange(frame_count), type=pa.int64()))
    output = _replace_column(
        output,
        "index",
        pa.array(np.arange(global_start, global_start + frame_count), type=pa.int64()),
    )
    output = _replace_column(output, SOURCE_FRAME_KEY, pa.array(source_frame, type=pa.int64()))
    return output.replace_schema_metadata(None)


def rewrite_info(
    info: dict[str, Any],
    *,
    total_frames: int,
    mode: str,
    target_retention: float,
    max_skip: int,
) -> dict[str, Any]:
    """Return updated metadata without mutating the source dictionary."""
    if mode not in {"va", "vf"}:
        raise ValueError("mode must be 'va' or 'vf'")
    if total_frames < 1:
        raise ValueError("total_frames must be positive")
    output = copy.deepcopy(info)
    output["total_frames"] = int(total_frames)
    features = output.setdefault("features", {})
    features.pop(SAMPLING_WEIGHT_KEY, None)
    features[SOURCE_FRAME_KEY] = {"dtype": "int64", "shape": [1], "names": None}
    output.pop("speed_force_sampling", None)
    output["trajectory_acceleration"] = {
        "schema_version": 1,
        "mode": mode,
        "target_retention": float(target_retention),
        "max_skip": int(max_skip),
        "source_frame_key": SOURCE_FRAME_KEY,
        "temporal_policy": "all parquet signals and video frames use identical selected source indices",
    }
    return output


def compute_episode_stats(table: pa.Table) -> dict[str, dict[str, list[float] | list[int]]]:
    """Compute LeRobot-style population statistics for state/action columns."""
    stats: dict[str, dict[str, list[float] | list[int]]] = {}
    for name in table.column_names:
        if name == SAMPLING_WEIGHT_KEY:
            continue
        column = table[name].combine_chunks()
        try:
            values = np.asarray(column.to_pylist(), dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"numeric statistics require a finite [T, D] column: {name}")
        stats[name] = {
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "count": [int(values.shape[0])],
        }
    return stats


def probe_video(path: str | Path) -> VideoInfo:
    """Read exact primary-video metadata with FFprobe."""
    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(f"video does not exist: {video_path}")
    process = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,nb_read_frames",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(process.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected one primary video stream in {video_path}")
    stream = streams[0]
    count_value = stream.get("nb_frames")
    if count_value in (None, "N/A"):
        count_value = stream.get("nb_read_frames")
    if count_value in (None, "N/A"):
        raise ValueError(f"could not determine frame count for {video_path}")
    return VideoInfo(
        frame_count=int(count_value),
        fps=float(Fraction(stream["r_frame_rate"])),
        width=int(stream["width"]),
        height=int(stream["height"]),
    )


def rewrite_video(
    source: str | Path,
    destination: str | Path,
    selected_indices: np.ndarray,
    *,
    fps: float,
) -> None:
    """Encode exactly the selected source frames on a new fixed-rate timeline."""
    selected = np.asarray(selected_indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0 or np.any(selected < 0) or np.any(np.diff(selected) <= 0):
        raise ValueError("selected_indices must be nonempty, nonnegative, and strictly increasing")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    source_path = Path(source)
    destination_path = Path(destination)
    source_info = probe_video(source_path)
    if selected[-1] >= source_info.frame_count:
        raise ValueError(
            f"selected frame {selected[-1]} exceeds source video length {source_info.frame_count}"
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    select_expression = "+".join(f"eq(n\\,{int(index)})" for index in selected)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-vf",
            f"select={select_expression},setpts=N/({float(fps):.12g}*TB)",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            f"{float(fps):.12g}",
            "-frames:v",
            str(selected.size),
            str(destination_path),
        ],
        check=True,
    )
    destination_info = probe_video(destination_path)
    if destination_info.frame_count != selected.size:
        raise ValueError(
            f"rewritten video has {destination_info.frame_count} frames, expected {selected.size}: {destination_path}"
        )
    if not np.isclose(destination_info.fps, fps, rtol=0.0, atol=1e-6):
        raise ValueError(f"rewritten video FPS {destination_info.fps} does not match requested FPS {fps}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"required metadata file does not exist: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))


def _format_dataset_path(
    template: str,
    *,
    episode_index: int,
    chunks_size: int,
    video_key: str | None = None,
) -> Path:
    arguments: dict[str, Any] = {
        "episode_chunk": episode_index // chunks_size,
        "episode_index": episode_index,
    }
    if video_key is not None:
        arguments["video_key"] = video_key
    return Path(template.format(**arguments))


def _video_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("info.json features must be an object")
    return sorted(
        key
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    )


def _force_magnitudes(table: pa.Table) -> np.ndarray:
    keys = [
        key
        for key in table.column_names
        if key.startswith("observation.state.") and "tactile" in key
    ]
    if not keys:
        raise ValueError("VF acceleration requires at least one observation.state.*tactile force field")
    sensors: list[np.ndarray] = []
    for key in keys:
        values = np.asarray(table[key].to_pylist(), dtype=np.float64)
        if values.ndim != 2 or values.shape[1] < 4:
            raise ValueError(f"force field {key!r} must have shape [T, >=4]")
        sensors.append(values[:, 3])
    return np.stack(sensors, axis=1)


def _episode_selection(
    table: pa.Table,
    *,
    mode: str,
    fps: float,
    target_retention: float,
    max_skip: int,
    free_contact_seconds: float,
) -> tuple[list[int], dict[str, Any]]:
    if "observation.state.eef_pose" not in table.column_names:
        raise ValueError("episode parquet is missing observation.state.eef_pose")
    poses = np.asarray(table["observation.state.eef_pose"].to_pylist(), dtype=np.float64)
    if mode == "va":
        signals = compute_va_priority(poses, fps=fps)
    elif mode == "vf":
        signals = compute_vf_priority(
            poses,
            _force_magnitudes(table),
            fps=fps,
            free_contact_seconds=free_contact_seconds,
        )
    else:
        raise ValueError("mode must be 'va' or 'vf'")
    gripper_forced = find_gripper_change_indices(poses[:, 9::10])
    forced = np.unique(np.concatenate((signals.forced_indices, gripper_forced)))
    selected = select_isr_indices(
        signals.priority,
        target_retention=target_retention,
        max_skip=max_skip,
        forced_indices=forced,
    )
    return selected, {
        "source_frames": int(table.num_rows),
        "output_frames": len(selected),
        "speedup": table.num_rows / len(selected),
        "selected_source_indices": selected,
        "forced_source_indices": forced.tolist(),
    }


def _merge_episode_stats(
    table: pa.Table,
    *,
    source_stats: dict[str, Any],
    video_keys: list[str],
) -> dict[str, Any]:
    stats = compute_episode_stats(table)
    for key in video_keys:
        if key in source_stats:
            stats[key] = copy.deepcopy(source_stats[key])
    return stats


def convert_dataset(
    input_root: str | Path,
    output_root: str | Path,
    *,
    mode: str,
    target_retention: float,
    max_skip: int,
    free_contact_seconds: float = 1.0,
) -> ConversionSummary:
    """Create one accelerated LeRobot 2.1 dataset in a new sibling directory."""
    source, output = ensure_distinct_paths(input_root, output_root)
    if output.exists():
        raise FileExistsError(f"output dataset already exists: {output}")
    info_path = source / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"input dataset metadata does not exist: {info_path}")
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"expected LeRobot v2.1, got {info.get('codebase_version')!r}")
    fps = float(info.get("fps", 0.0))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("dataset FPS must be positive")
    total_episodes = info.get("total_episodes")
    chunks_size = info.get("chunks_size", 1000)
    data_template = info.get("data_path")
    video_template = info.get("video_path")
    if not isinstance(total_episodes, int) or total_episodes < 1:
        raise ValueError("total_episodes must be a positive integer")
    if not isinstance(chunks_size, int) or chunks_size < 1:
        raise ValueError("chunks_size must be a positive integer")
    if not isinstance(data_template, str) or not isinstance(video_template, str):
        raise ValueError("data_path and video_path templates are required")

    episode_records = {
        record["episode_index"]: record
        for record in _read_jsonl(source / "meta/episodes.jsonl")
    }
    source_episode_stats = {
        record["episode_index"]: record.get("stats", {})
        for record in _read_jsonl(source / "meta/episodes_stats.jsonl")
    }
    video_keys = _video_keys(info)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    total_source_frames = 0
    total_output_frames = 0
    new_episode_records: list[dict[str, Any]] = []
    new_episode_stats: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    try:
        (staging / "meta").mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "meta/tasks.jsonl", staging / "meta/tasks.jsonl")
        for episode_index in range(total_episodes):
            relative_data = _format_dataset_path(
                data_template,
                episode_index=episode_index,
                chunks_size=chunks_size,
            )
            source_data = source / relative_data
            if not source_data.is_file():
                raise FileNotFoundError(f"episode parquet does not exist: {source_data}")
            source_table = pq.read_table(source_data)
            selected, manifest = _episode_selection(
                source_table,
                mode=mode,
                fps=fps,
                target_retention=target_retention,
                max_skip=max_skip,
                free_contact_seconds=free_contact_seconds,
            )
            output_table = resample_episode_table(
                source_table,
                np.asarray(selected, dtype=np.int64),
                fps=fps,
                global_start=total_output_frames,
            )
            destination_data = staging / relative_data
            destination_data.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(output_table, destination_data, compression="zstd")
            for video_key in video_keys:
                relative_video = _format_dataset_path(
                    video_template,
                    episode_index=episode_index,
                    chunks_size=chunks_size,
                    video_key=video_key,
                )
                source_video = source / relative_video
                if probe_video(source_video).frame_count != source_table.num_rows:
                    raise ValueError(f"video/parquet frame mismatch: {source_video}")
                rewrite_video(source_video, staging / relative_video, np.asarray(selected), fps=fps)

            total_source_frames += source_table.num_rows
            total_output_frames += output_table.num_rows
            source_episode_record = episode_records.get(episode_index)
            if source_episode_record is None:
                raise ValueError(f"episodes.jsonl is missing episode {episode_index}")
            new_record = copy.deepcopy(source_episode_record)
            new_record["length"] = output_table.num_rows
            new_episode_records.append(new_record)
            new_episode_stats.append(
                {
                    "episode_index": episode_index,
                    "stats": _merge_episode_stats(
                        output_table,
                        source_stats=source_episode_stats.get(episode_index, {}),
                        video_keys=video_keys,
                    ),
                }
            )
            manifests.append({"episode_index": episode_index, "mode": mode, **manifest})

        _write_jsonl(staging / "meta/episodes.jsonl", new_episode_records)
        _write_jsonl(staging / "meta/episodes_stats.jsonl", new_episode_stats)
        _write_jsonl(staging / "meta/selection_manifest.jsonl", manifests)
        output_info = rewrite_info(
            info,
            total_frames=total_output_frames,
            mode=mode,
            target_retention=target_retention,
            max_skip=max_skip,
        )
        output_info["trajectory_acceleration"].update(
            {
                "input_root": str(source),
                "total_source_frames": total_source_frames,
                "total_output_frames": total_output_frames,
                "achieved_speedup": total_source_frames / total_output_frames,
                "video_stats_policy": "copied from source episode; numeric parquet stats recomputed",
            }
        )
        (staging / "meta/info.json").write_text(json.dumps(output_info, ensure_ascii=False, indent=2) + "\n")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return ConversionSummary(
        input_root=source,
        output_root=output,
        mode=mode,
        episodes=total_episodes,
        total_source_frames=total_source_frames,
        total_output_frames=total_output_frames,
    )
