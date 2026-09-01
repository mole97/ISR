"""Read-only discovery and synchronization helpers for episode video comparison."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


_EPISODE_PATTERN = re.compile(r"episode_(\d+)")


@dataclasses.dataclass(frozen=True)
class DatasetTriplet:
    """Original and two accelerated LeRobot dataset roots."""

    original: Path
    va: Path
    vf: Path


@dataclasses.dataclass(frozen=True)
class PlaybackFrames:
    """Video-frame indices and the source frames represented by them."""

    original_index: int
    va_index: int
    vf_index: int
    original_source: int
    va_source: int
    vf_source: int


def extract_episode_index(path: str | Path) -> int:
    """Extract the integer in an ``episode_<digits>`` path component."""
    matches = _EPISODE_PATTERN.findall(str(path))
    if not matches:
        raise ValueError(f"path does not contain an episode index: {path}")
    return int(matches[-1])


def find_dataset_root(path: str | Path) -> Path:
    """Find the nearest parent containing LeRobot ``meta/info.json``."""
    candidate = Path(path).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / "meta/info.json").is_file():
            return directory
    raise FileNotFoundError(f"could not find a LeRobot dataset root above: {path}")


def _derived_roots(discovered: Path) -> DatasetTriplet:
    name = discovered.name
    if name.endswith("_isr_va"):
        base = name.removesuffix("_isr_va")
        original = discovered.with_name(base)
        return DatasetTriplet(
            original=original,
            va=discovered,
            vf=discovered.with_name(f"{base}_isr_vf"),
        )
    if name.endswith("_isr_vf"):
        base = name.removesuffix("_isr_vf")
        original = discovered.with_name(base)
        return DatasetTriplet(
            original=original,
            va=discovered.with_name(f"{base}_isr_va"),
            vf=discovered,
        )
    return DatasetTriplet(
        original=discovered,
        va=discovered.with_name(f"{name}_isr_va"),
        vf=discovered.with_name(f"{name}_isr_vf"),
    )


def derive_dataset_triplet(
    dropped_path: str | Path,
    *,
    original_root: str | Path | None = None,
    va_root: str | Path | None = None,
    vf_root: str | Path | None = None,
) -> DatasetTriplet:
    """Resolve and validate the three read-only dataset roots."""
    derived = _derived_roots(find_dataset_root(dropped_path))
    roots = DatasetTriplet(
        original=Path(original_root).resolve() if original_root else derived.original,
        va=Path(va_root).resolve() if va_root else derived.va,
        vf=Path(vf_root).resolve() if vf_root else derived.vf,
    )
    for label, root in (
        ("original dataset", roots.original),
        ("VA dataset", roots.va),
        ("VF dataset", roots.vf),
    ):
        if not (root / "meta/info.json").is_file():
            raise FileNotFoundError(f"{label} metadata does not exist: {root}")
    return roots


def _read_info(root: Path) -> dict[str, Any]:
    path = root / "meta/info.json"
    if not path.is_file():
        raise FileNotFoundError(f"dataset metadata does not exist: {path}")
    return json.loads(path.read_text())


def common_video_keys(roots: DatasetTriplet) -> list[str]:
    """Return sorted video feature keys present in all three datasets."""
    key_sets: list[set[str]] = []
    for root in (roots.original, roots.va, roots.vf):
        features = _read_info(root).get("features", {})
        key_sets.append(
            {
                key
                for key, feature in features.items()
                if isinstance(feature, dict) and feature.get("dtype") == "video"
            }
        )
    common = sorted(set.intersection(*key_sets))
    if not common:
        raise ValueError("the three datasets do not share a video feature")
    return common


def resolve_video_path(
    root: str | Path,
    *,
    episode_index: int,
    video_key: str,
) -> Path:
    """Resolve one existing episode video from its LeRobot path template."""
    dataset_root = Path(root).resolve()
    info = _read_info(dataset_root)
    features = info.get("features", {})
    if features.get(video_key, {}).get("dtype") != "video":
        raise ValueError(f"dataset has no video feature {video_key!r}: {dataset_root}")
    chunks_size = int(info.get("chunks_size", 1000))
    template = info.get("video_path")
    if not isinstance(template, str):
        raise ValueError(f"dataset has no video_path template: {dataset_root}")
    relative = template.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
        video_key=video_key,
    )
    path = (dataset_root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"episode video does not exist: {path}")
    return path


def load_source_indices(root: str | Path, *, episode_index: int) -> np.ndarray:
    """Load the output-frame to source-frame mapping for an accelerated episode."""
    path = Path(root).resolve() / "meta/selection_manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"selection manifest does not exist: {path}")
    record: dict[str, Any] | None = None
    for line in path.read_text().splitlines():
        candidate = json.loads(line)
        if candidate.get("episode_index") == episode_index:
            record = candidate
            break
    if record is None:
        raise ValueError(f"selection manifest has no episode {episode_index}: {path}")
    selected = np.asarray(record.get("selected_source_indices"), dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0 or np.any(np.diff(selected) <= 0):
        raise ValueError(
            f"episode {episode_index} source indices must be nonempty and strictly increasing"
        )
    if np.any(selected < 0):
        raise ValueError(f"episode {episode_index} source indices must be nonnegative")
    return selected


def _validated_source_indices(values: np.ndarray, *, label: str) -> np.ndarray:
    selected = np.asarray(values, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0 or np.any(np.diff(selected) <= 0):
        raise ValueError(f"{label} source indices must be nonempty and strictly increasing")
    if np.any(selected < 0):
        raise ValueError(f"{label} source indices must be nonnegative")
    return selected


def nearest_retained_frame(selected_source_indices: np.ndarray, source_frame: int) -> int:
    """Return the output index closest to a source frame, preferring the left tie."""
    selected = _validated_source_indices(selected_source_indices, label="retained")
    target = int(source_frame)
    right = int(np.searchsorted(selected, target, side="left"))
    if right == 0:
        return 0
    if right == selected.size:
        return selected.size - 1
    left = right - 1
    if target - int(selected[left]) <= int(selected[right]) - target:
        return left
    return right


def map_playback_frames(
    mode: str,
    *,
    tick: int,
    source_frame_count: int,
    va_source_indices: np.ndarray,
    vf_source_indices: np.ndarray,
) -> PlaybackFrames:
    """Map one UI tick to original, VA, and VF frame indices."""
    if mode not in {"source", "native"}:
        raise ValueError("playback mode must be 'source' or 'native'")
    if source_frame_count < 1:
        raise ValueError("source_frame_count must be positive")
    if tick < 0:
        raise ValueError("tick must be nonnegative")
    va_selected = _validated_source_indices(va_source_indices, label="VA")
    vf_selected = _validated_source_indices(vf_source_indices, label="VF")
    original_index = min(int(tick), source_frame_count - 1)
    if mode == "source":
        va_index = nearest_retained_frame(va_selected, original_index)
        vf_index = nearest_retained_frame(vf_selected, original_index)
    else:
        va_index = min(int(tick), va_selected.size - 1)
        vf_index = min(int(tick), vf_selected.size - 1)
    return PlaybackFrames(
        original_index=original_index,
        va_index=va_index,
        vf_index=vf_index,
        original_source=original_index,
        va_source=int(va_selected[va_index]),
        vf_source=int(vf_selected[vf_index]),
    )
