from __future__ import annotations

import numpy as np
import pytest

from isr.pose import compute_pose_kinematics, rotation_6d_to_matrix


def _pose(*, xyz=(0.0, 0.0, 0.0), rotation_6d=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0), gripper=0.0):
    return np.asarray([*xyz, *rotation_6d, gripper], dtype=np.float64)


def test_rotation_6d_identity() -> None:
    matrix = rotation_6d_to_matrix(np.asarray([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]))

    np.testing.assert_allclose(matrix[0], np.eye(3), atol=1e-12)


def test_rotation_6d_orthonormalizes_noisy_columns() -> None:
    matrix = rotation_6d_to_matrix(np.asarray([[2.0, 0.0, 0.0, 1.0, 3.0, 0.0]]))[0]

    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
    assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_rotation_6d_rejects_degenerate_columns() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        rotation_6d_to_matrix(np.asarray([[1.0, 0.0, 0.0, 2.0, 0.0, 0.0]]))


def test_pose_kinematics_reports_translation_and_quarter_turn() -> None:
    poses = np.stack(
        [
            _pose(),
            _pose(
                xyz=(0.5, 0.0, 0.0),
                rotation_6d=(0.0, 1.0, 0.0, -1.0, 0.0, 0.0),
            ),
        ]
    )

    result = compute_pose_kinematics(poses, fps=2.0)

    np.testing.assert_allclose(result.linear_speed, [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(result.angular_speed, [0.0, np.pi], atol=1e-12)


def test_pose_kinematics_constant_velocity_has_zero_acceleration() -> None:
    poses = np.stack([_pose(xyz=(float(index), 0.0, 0.0)) for index in range(5)])

    result = compute_pose_kinematics(poses, fps=1.0)

    np.testing.assert_allclose(result.linear_acceleration, np.zeros(5), atol=1e-12)
    np.testing.assert_allclose(result.angular_acceleration, np.zeros(5), atol=1e-12)


def test_pose_kinematics_supports_two_concatenated_arms() -> None:
    poses = np.stack(
        [
            np.concatenate(
                (
                    _pose(xyz=(float(index), 0.0, 0.0)),
                    _pose(xyz=(0.0, 2.0 * index, 0.0)),
                )
            )
            for index in range(4)
        ]
    )

    result = compute_pose_kinematics(poses, fps=1.0)

    assert result.linear_velocity.shape == (4, 6)
    assert result.angular_velocity.shape == (4, 6)
    np.testing.assert_allclose(
        result.linear_speed,
        [0.0, np.sqrt(5.0), np.sqrt(5.0), np.sqrt(5.0)],
        atol=1e-12,
    )
    np.testing.assert_allclose(result.linear_acceleration, np.zeros(4), atol=1e-12)


def test_pose_kinematics_requires_dimension_multiple_of_ten() -> None:
    with pytest.raises(ValueError, match=r"10 \* arm_count"):
        compute_pose_kinematics(np.zeros((3, 9)), fps=30.0)


def test_pose_kinematics_rejects_nonpositive_fps() -> None:
    with pytest.raises(ValueError, match="fps"):
        compute_pose_kinematics(np.stack([_pose(), _pose()]), fps=0.0)
