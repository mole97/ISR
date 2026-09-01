from __future__ import annotations

import numpy as np
import pytest

from isr.trajectory_acceleration import (
    compute_va_priority,
    compute_vf_priority,
    find_gripper_change_indices,
    robust_normalize,
    select_isr_indices,
)


def _straight_poses(frame_count: int, *, step: float = 0.01) -> np.ndarray:
    poses = np.zeros((frame_count, 10), dtype=np.float64)
    poses[:, 0] = np.arange(frame_count) * step
    poses[:, 3] = 1.0
    poses[:, 7] = 1.0
    return poses


def test_robust_normalize_clips_quantile_outliers() -> None:
    values = np.asarray([-100.0, 0.0, 5.0, 10.0, 100.0])

    result = robust_normalize(values, lower_quantile=0.2, upper_quantile=0.8)

    assert result[0] == 0.0
    assert result[-1] == 1.0
    assert 0.0 < result[2] < 1.0


def test_robust_normalize_constant_signal_is_zero() -> None:
    np.testing.assert_array_equal(robust_normalize(np.ones(10)), np.zeros(10))


def test_va_priority_peaks_at_velocity_change() -> None:
    poses = _straight_poses(20, step=0.0)
    poses[10:, 0] = np.arange(10) * 0.02

    signals = compute_va_priority(poses, fps=30.0)

    assert np.argmax(signals.secondary_score) in range(9, 12)
    assert signals.priority.shape == (20,)


def test_vf_priority_marks_force_step_and_forces_context() -> None:
    force = np.r_[np.zeros(10), np.ones(10) * 5.0][:, None]

    signals = compute_vf_priority(
        _straight_poses(20),
        force,
        fps=30.0,
        free_contact_seconds=0.2,
        force_event_padding=1,
    )

    assert np.argmax(signals.secondary_score) in range(10, 13)
    assert {9, 10, 11}.issubset(set(signals.forced_indices.tolist()))


def test_va_and_vf_priorities_support_two_concatenated_arms() -> None:
    first_arm = _straight_poses(20)
    second_arm = _straight_poses(20, step=0.02)
    poses = np.concatenate((first_arm, second_arm), axis=1)
    force = np.r_[np.zeros(10), np.ones(10) * 5.0][:, None]

    va = compute_va_priority(poses, fps=30.0)
    vf = compute_vf_priority(
        poses,
        force,
        fps=30.0,
        free_contact_seconds=0.2,
    )

    assert va.priority.shape == (20,)
    assert vf.priority.shape == (20,)
    assert np.all(np.isfinite(va.priority))
    assert np.all(np.isfinite(vf.priority))


def test_find_gripper_change_indices_keeps_both_sides() -> None:
    gripper = np.asarray([0.0, 0.0, 1.0, 1.0])

    assert find_gripper_change_indices(gripper).tolist() == [1, 2]


def test_find_gripper_change_indices_unions_changes_from_both_arms() -> None:
    grippers = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    assert find_gripper_change_indices(grippers).tolist() == [1, 2, 3]


def test_selector_keeps_boundaries_forced_indices_and_max_skip() -> None:
    selected = select_isr_indices(
        np.zeros(31),
        target_retention=0.2,
        max_skip=4,
        forced_indices=np.asarray([13]),
    )

    assert selected[0] == 0
    assert selected[-1] == 30
    assert 13 in selected
    assert max(np.diff(selected)) <= 4


def test_selector_is_deterministic() -> None:
    priority = np.linspace(0.0, 1.0, 40)

    first = select_isr_indices(priority, target_retention=0.5, max_skip=6)
    second = select_isr_indices(priority, target_retention=0.5, max_skip=6)

    assert first == second


def test_selector_is_denser_in_high_information_region() -> None:
    priority = np.r_[np.full(20, 0.05), np.ones(20)]

    selected = np.asarray(select_isr_indices(priority, target_retention=0.5, max_skip=8))

    low_count = np.count_nonzero(selected < 20)
    high_count = np.count_nonzero(selected >= 20)
    assert high_count > low_count


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_retention": 0.0, "max_skip": 4}, "target_retention"),
        ({"target_retention": 0.5, "max_skip": 0}, "max_skip"),
    ],
)
def test_selector_rejects_invalid_configuration(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        select_isr_indices(np.ones(5), **kwargs)
