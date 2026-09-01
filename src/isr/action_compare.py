"""Read-only action loading and plot preparation for LeRobot comparisons."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from .episode_compare import DatasetTriplet, load_source_indices


@dataclasses.dataclass(frozen=True)
class ActionComparison:
    """Action arrays and output-to-source mappings for one episode."""

    action_key: str
    dimension_names: tuple[str, ...]
    fps: float
    original: np.ndarray
    va: np.ndarray
    vf: np.ndarray
    original_source_indices: np.ndarray
    va_source_indices: np.ndarray
    vf_source_indices: np.ndarray


@dataclasses.dataclass(frozen=True)
class ActionPlotSeries:
    """One action dimension on compressed or original-source time axes."""

    dimension_name: str
    axis_mode: str
    original_time: np.ndarray
    original_values: np.ndarray
    va_time: np.ndarray
    va_values: np.ndarray
    vf_time: np.ndarray
    vf_values: np.ndarray


def _read_info(root: Path) -> dict[str, Any]:
    path = root / "meta/info.json"
    if not path.is_file():
        raise FileNotFoundError(f"dataset metadata does not exist: {path}")
    return json.loads(path.read_text())


def common_action_keys(roots: DatasetTriplet) -> list[str]:
    """Return sorted ``actions.*`` features shared by all three datasets."""
    key_sets: list[set[str]] = []
    for root in (roots.original, roots.va, roots.vf):
        features = _read_info(root).get("features", {})
        key_sets.append(
            {
                key
                for key, feature in features.items()
                if key.startswith("actions.")
                and isinstance(feature, dict)
                and feature.get("dtype") != "video"
            }
        )
    common = sorted(set.intersection(*key_sets))
    if not common:
        raise ValueError("the three datasets do not share an action feature")
    return common


def action_dimension_names(
    root: str | Path,
    *,
    action_key: str,
    dimension_count: int,
) -> tuple[str, ...]:
    """Read LeRobot feature names or generate stable dimension labels."""
    if dimension_count < 1:
        raise ValueError("dimension_count must be positive")
    features = _read_info(Path(root).resolve()).get("features", {})
    feature = features.get(action_key)
    if not isinstance(feature, dict):
        raise ValueError(f"dataset has no action feature {action_key!r}")
    names = feature.get("names")
    if names is None:
        return tuple(f"dim_{index}" for index in range(dimension_count))
    if (
        isinstance(names, list)
        and len(names) == 1
        and isinstance(names[0], list)
    ):
        names = names[0]
    if (
        not isinstance(names, list)
        or len(names) != dimension_count
        or not all(isinstance(name, str) and name for name in names)
    ):
        raise ValueError(
            f"action feature {action_key!r} dimension names must contain "
            f"{dimension_count} strings"
        )
    return tuple(names)


def resolve_episode_data_path(root: str | Path, *, episode_index: int) -> Path:
    """Resolve one existing episode Parquet from the LeRobot path template."""
    dataset_root = Path(root).resolve()
    info = _read_info(dataset_root)
    chunks_size = int(info.get("chunks_size", 1000))
    template = info.get("data_path")
    if not isinstance(template, str):
        raise ValueError(f"dataset has no data_path template: {dataset_root}")
    relative = template.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )
    path = (dataset_root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"episode Parquet does not exist: {path}")
    return path


def _read_action(path: Path, action_key: str, *, label: str) -> np.ndarray:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Action comparison requires PyArrow; install with: pip install pyarrow"
        ) from error
    table = pq.read_table(path, columns=[action_key])
    values = np.asarray(table[action_key].to_pylist(), dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"{label} action {action_key!r} must have shape [T, D]")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} action {action_key!r} must be finite")
    return values


def load_action_comparison(
    roots: DatasetTriplet,
    *,
    episode_index: int,
    action_key: str,
) -> ActionComparison:
    """Load one action field from original, VA, and VF episode Parquets."""
    if action_key not in common_action_keys(roots):
        raise ValueError(f"action feature is not shared by all datasets: {action_key}")
    arrays = {
        label: _read_action(
            resolve_episode_data_path(root, episode_index=episode_index),
            action_key,
            label=label,
        )
        for label, root in (
            ("original", roots.original),
            ("VA", roots.va),
            ("VF", roots.vf),
        )
    }
    original = arrays["original"]
    va = arrays["VA"]
    vf = arrays["VF"]
    if va.shape[1] != original.shape[1] or vf.shape[1] != original.shape[1]:
        raise ValueError("original, VA, and VF action dimensions must match")
    va_source = load_source_indices(roots.va, episode_index=episode_index)
    vf_source = load_source_indices(roots.vf, episode_index=episode_index)
    if va.shape[0] != va_source.size:
        raise ValueError(
            f"VA action rows ({va.shape[0]}) do not match source indices ({va_source.size})"
        )
    if vf.shape[0] != vf_source.size:
        raise ValueError(
            f"VF action rows ({vf.shape[0]}) do not match source indices ({vf_source.size})"
        )
    if va_source[-1] >= original.shape[0] or vf_source[-1] >= original.shape[0]:
        raise ValueError("accelerated action source indices exceed the original episode")
    fps = float(_read_info(roots.original).get("fps", 0.0))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("dataset FPS must be positive")
    return ActionComparison(
        action_key=action_key,
        dimension_names=action_dimension_names(
            roots.original,
            action_key=action_key,
            dimension_count=original.shape[1],
        ),
        fps=fps,
        original=original,
        va=va,
        vf=vf,
        original_source_indices=np.arange(original.shape[0], dtype=np.int64),
        va_source_indices=va_source,
        vf_source_indices=vf_source,
    )


def action_plot_series(
    comparison: ActionComparison,
    *,
    dimension: int,
    axis_mode: str,
) -> ActionPlotSeries:
    """Return one action dimension on compressed or source-aligned seconds."""
    dimension_count = comparison.original.shape[1]
    if dimension < 0 or dimension >= dimension_count:
        raise ValueError(f"action dimension must be within [0, {dimension_count})")
    if axis_mode not in {"compressed", "source"}:
        raise ValueError("axis_mode must be 'compressed' or 'source'")
    if axis_mode == "compressed":
        original_time = np.arange(comparison.original.shape[0]) / comparison.fps
        va_time = np.arange(comparison.va.shape[0]) / comparison.fps
        vf_time = np.arange(comparison.vf.shape[0]) / comparison.fps
    else:
        original_time = comparison.original_source_indices / comparison.fps
        va_time = comparison.va_source_indices / comparison.fps
        vf_time = comparison.vf_source_indices / comparison.fps
    return ActionPlotSeries(
        dimension_name=comparison.dimension_names[dimension],
        axis_mode=axis_mode,
        original_time=original_time,
        original_values=comparison.original[:, dimension],
        va_time=va_time,
        va_values=comparison.va[:, dimension],
        vf_time=vf_time,
        vf_values=comparison.vf[:, dimension],
    )
