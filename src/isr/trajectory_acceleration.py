"""Information signals and bounded ISR selection for trajectory acceleration."""

from __future__ import annotations

import dataclasses

import numpy as np

from .pose import compute_pose_kinematics


@dataclasses.dataclass(frozen=True)
class PrioritySignals:
    priority: np.ndarray
    speed_score: np.ndarray
    secondary_score: np.ndarray
    forced_indices: np.ndarray
    raw_secondary: np.ndarray


def robust_normalize(
    values: np.ndarray,
    *,
    lower_quantile: float = 0.1,
    upper_quantile: float = 0.9,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Map a finite vector to [0, 1] with clipped quantile scaling."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("values must be a nonempty vector")
    if not np.all(np.isfinite(data)):
        raise ValueError("values must be finite")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    lower, upper = np.quantile(data, [lower_quantile, upper_quantile])
    scale = float(upper - lower)
    if scale <= tolerance:
        lower = float(np.min(data))
        upper = float(np.max(data))
        scale = upper - lower
        if scale <= tolerance:
            return np.zeros_like(data)
    return np.clip((data - lower) / scale, 0.0, 1.0)


def _combined_motion_score(linear: np.ndarray, angular: np.ndarray) -> np.ndarray:
    return 0.5 * (robust_normalize(linear) + robust_normalize(angular))


def compute_va_priority(eef_pose: np.ndarray, *, fps: float) -> PrioritySignals:
    """Compute equal-weight velocity/acceleration information for one arm."""
    kinematics = compute_pose_kinematics(eef_pose, fps=fps)
    speed_score = _combined_motion_score(kinematics.linear_speed, kinematics.angular_speed)
    acceleration_score = _combined_motion_score(
        kinematics.linear_acceleration,
        kinematics.angular_acceleration,
    )
    return PrioritySignals(
        priority=0.5 * speed_score + 0.5 * acceleration_score,
        speed_score=speed_score,
        secondary_score=acceleration_score,
        forced_indices=np.empty((0,), dtype=np.int64),
        raw_secondary=0.5 * (kinematics.linear_acceleration + kinematics.angular_acceleration),
    )


def _force_event_signal(
    force_magnitudes: np.ndarray,
    *,
    fps: float,
    free_contact_seconds: float,
    ema_alpha: float,
    noise_sigma_multiplier: float,
) -> np.ndarray:
    force = np.asarray(force_magnitudes, dtype=np.float64)
    if force.ndim == 1:
        force = force[:, None]
    if force.ndim != 2 or force.shape[0] == 0:
        raise ValueError("force_magnitudes must have shape [T, sensors]")
    if not np.all(np.isfinite(force)):
        raise ValueError("force_magnitudes must be finite")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    if not np.isfinite(free_contact_seconds) or free_contact_seconds <= 0:
        raise ValueError("free_contact_seconds must be positive")
    if not 0.0 < ema_alpha <= 1.0:
        raise ValueError("ema_alpha must be within (0, 1]")
    if not np.isfinite(noise_sigma_multiplier) or noise_sigma_multiplier < 0:
        raise ValueError("noise_sigma_multiplier must be nonnegative")

    free_frames = min(force.shape[0], max(1, round(free_contact_seconds * fps)))
    bias = np.median(force[:free_frames], axis=0)
    debiased = np.maximum(force - bias[None, :], 0.0)
    filtered = np.empty_like(debiased)
    filtered[0] = debiased[0]
    for index in range(1, filtered.shape[0]):
        filtered[index] = ema_alpha * debiased[index] + (1.0 - ema_alpha) * filtered[index - 1]
    derivative = np.zeros_like(filtered)
    derivative[1:] = np.diff(filtered, axis=0) * fps
    threshold = noise_sigma_multiplier * np.std(derivative[:free_frames], axis=0)
    return np.max(np.maximum(np.abs(derivative) - threshold[None, :], 0.0), axis=1)


def _force_peak_indices(event: np.ndarray, *, quantile: float, padding: int) -> np.ndarray:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("force_event_quantile must be within [0, 1]")
    if padding < 0:
        raise ValueError("force_event_padding must be nonnegative")
    positive = event[event > 0]
    if positive.size == 0:
        return np.empty((0,), dtype=np.int64)
    threshold = float(np.quantile(positive, quantile))
    peaks: list[int] = []
    for index, value in enumerate(event):
        left = event[index - 1] if index > 0 else -np.inf
        right = event[index + 1] if index + 1 < event.size else -np.inf
        if value >= threshold and value >= left and value >= right:
            peaks.append(index)
    expanded = {
        neighbor
        for peak in peaks
        for neighbor in range(max(0, peak - padding), min(event.size, peak + padding + 1))
    }
    return np.asarray(sorted(expanded), dtype=np.int64)


def compute_vf_priority(
    eef_pose: np.ndarray,
    force_magnitudes: np.ndarray,
    *,
    fps: float,
    free_contact_seconds: float = 1.0,
    ema_alpha: float = 0.2,
    noise_sigma_multiplier: float = 3.0,
    force_event_quantile: float = 0.9,
    force_event_padding: int = 1,
) -> PrioritySignals:
    """Compute equal-weight velocity/force-event information for one arm."""
    kinematics = compute_pose_kinematics(eef_pose, fps=fps)
    speed_score = _combined_motion_score(kinematics.linear_speed, kinematics.angular_speed)
    force_event = _force_event_signal(
        force_magnitudes,
        fps=fps,
        free_contact_seconds=free_contact_seconds,
        ema_alpha=ema_alpha,
        noise_sigma_multiplier=noise_sigma_multiplier,
    )
    force_score = robust_normalize(force_event)
    return PrioritySignals(
        priority=0.5 * speed_score + 0.5 * force_score,
        speed_score=speed_score,
        secondary_score=force_score,
        forced_indices=_force_peak_indices(
            force_event,
            quantile=force_event_quantile,
            padding=force_event_padding,
        ),
        raw_secondary=force_event,
    )


def find_gripper_change_indices(gripper: np.ndarray, *, tolerance: float = 1e-6) -> np.ndarray:
    """Return both sides of every transition in one or more gripper columns."""
    values = np.asarray(gripper, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("gripper must have shape [T] or [T, arm_count]")
    if not np.all(np.isfinite(values)):
        raise ValueError("gripper must be finite")
    changes = np.flatnonzero(np.any(np.abs(np.diff(values, axis=0)) > tolerance, axis=1))
    if changes.size == 0:
        return np.empty((0,), dtype=np.int64)
    return np.unique(np.concatenate((changes, changes + 1))).astype(np.int64)


def _select_with_spacing(
    density: np.ndarray,
    *,
    spacing: float,
    max_skip: int,
    forced_indices: np.ndarray,
) -> list[int]:
    prefix = np.concatenate(([0.0], np.cumsum(density, dtype=np.float64)))
    selected: list[int] = []
    for interval_index in range(forced_indices.size - 1):
        start = int(forced_indices[interval_index])
        end = int(forced_indices[interval_index + 1])
        costs = np.full((end - start + 1,), np.inf, dtype=np.float64)
        previous = np.full((end - start + 1,), -1, dtype=np.int64)
        costs[0] = 0.0
        for current in range(start + 1, end + 1):
            current_local = current - start
            first_candidate = max(start, current - max_skip)
            for candidate in range(first_candidate, current):
                candidate_local = candidate - start
                information = prefix[current + 1] - prefix[candidate + 1]
                total = costs[candidate_local] + (information - spacing) ** 2
                if total < costs[current_local] - 1e-15:
                    costs[current_local] = total
                    previous[current_local] = candidate
        path = [end]
        cursor = end
        while cursor > start:
            cursor = int(previous[cursor - start])
            if cursor < start:
                raise RuntimeError("bounded ISR path is disconnected")
            path.append(cursor)
        path.reverse()
        if selected:
            selected.extend(path[1:])
        else:
            selected.extend(path)
    return selected


def select_isr_indices(
    priority: np.ndarray,
    *,
    target_retention: float,
    max_skip: int,
    forced_indices: np.ndarray | None = None,
    priority_epsilon: float = 0.05,
    bisection_steps: int = 48,
) -> list[int]:
    """Select a deterministic bounded ISR path close to a target retention."""
    values = np.asarray(priority, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("priority must be a nonempty vector")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("priority must be finite and nonnegative")
    if not 0.0 < target_retention <= 1.0:
        raise ValueError("target_retention must be within (0, 1]")
    if max_skip < 1:
        raise ValueError("max_skip must be positive")
    if priority_epsilon <= 0:
        raise ValueError("priority_epsilon must be positive")

    frame_count = values.size
    if frame_count == 1:
        return [0]
    extra_forced = (
        np.empty((0,), dtype=np.int64)
        if forced_indices is None
        else np.asarray(forced_indices, dtype=np.int64)
    )
    if extra_forced.ndim != 1 or np.any(extra_forced < 0) or np.any(extra_forced >= frame_count):
        raise ValueError("forced_indices must lie within the trajectory")
    forced = np.unique(np.concatenate(([0, frame_count - 1], extra_forced))).astype(np.int64)
    density = values + priority_epsilon
    target_count = max(forced.size, int(round(frame_count * target_retention)), 2)

    lower = max(float(np.min(density)) * 1e-3, 1e-12)
    upper = float(np.sum(density)) + 1.0
    best: list[int] | None = None
    best_key: tuple[int, int, int] | None = None
    for _ in range(bisection_steps):
        spacing = 0.5 * (lower + upper)
        candidate = _select_with_spacing(
            density,
            spacing=spacing,
            max_skip=max_skip,
            forced_indices=forced,
        )
        count = len(candidate)
        key = (abs(count - target_count), 0 if count >= target_count else 1, count)
        if best_key is None or key < best_key:
            best = candidate
            best_key = key
        if count > target_count:
            lower = spacing
        elif count < target_count:
            upper = spacing
        else:
            best = candidate
            break
    if best is None:
        raise RuntimeError("bounded ISR selection produced no path")
    return best
