"""SE(3) helpers for one or more concatenated 10-D end-effector poses.

Each arm uses ``[x, y, z, r1, ..., r6, gripper]``.  The six rotation values
are the first two columns of a rotation matrix, concatenated by column.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class PoseKinematics:
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    linear_speed: np.ndarray
    angular_speed: np.ndarray
    linear_acceleration: np.ndarray
    angular_acceleration: np.ndarray


def rotation_6d_to_matrix(values: np.ndarray) -> np.ndarray:
    """Recover proper rotation matrices with Gram--Schmidt orthogonalization."""
    rotation = np.asarray(values, dtype=np.float64)
    if rotation.ndim < 1 or rotation.shape[-1] != 6:
        raise ValueError(f"rotation values must end in six dimensions, got {rotation.shape}")
    if not np.all(np.isfinite(rotation)):
        raise ValueError("rotation values must be finite")

    first = rotation[..., :3]
    second = rotation[..., 3:6]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm <= 1e-12):
        raise ValueError("rotation representation has a degenerate first column")
    basis_1 = first / first_norm
    second_orthogonal = second - np.sum(basis_1 * second, axis=-1, keepdims=True) * basis_1
    second_norm = np.linalg.norm(second_orthogonal, axis=-1, keepdims=True)
    if np.any(second_norm <= 1e-12):
        raise ValueError("rotation representation has degenerate parallel columns")
    basis_2 = second_orthogonal / second_norm
    basis_3 = np.cross(basis_1, basis_2)
    return np.stack((basis_1, basis_2, basis_3), axis=-1)


def rotation_log_vector(matrix: np.ndarray) -> np.ndarray:
    """Return axis-angle logarithm vectors for proper rotation matrices."""
    rotations = np.asarray(matrix, dtype=np.float64)
    if rotations.ndim < 2 or rotations.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrices must end in [3, 3], got {rotations.shape}")
    trace = np.trace(rotations, axis1=-2, axis2=-1)
    angle = np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    skew = np.stack(
        (
            rotations[..., 2, 1] - rotations[..., 1, 2],
            rotations[..., 0, 2] - rotations[..., 2, 0],
            rotations[..., 1, 0] - rotations[..., 0, 1],
        ),
        axis=-1,
    )
    result = np.zeros_like(skew)
    regular = np.abs(np.sin(angle)) > 1e-8
    if np.any(regular):
        scale = angle[regular] / (2.0 * np.sin(angle[regular]))
        result[regular] = skew[regular] * scale[..., None]
    small = ~regular & (angle < 1e-6)
    if np.any(small):
        result[small] = 0.5 * skew[small]
    near_pi = ~regular & ~small
    for flat_index in np.flatnonzero(near_pi):
        flat_rotations = rotations.reshape((-1, 3, 3))
        flat_angles = angle.reshape(-1)
        current = flat_rotations[flat_index]
        axis = np.sqrt(np.maximum((np.diag(current) + 1.0) * 0.5, 0.0))
        largest = int(np.argmax(axis))
        if axis[largest] <= 1e-8:
            raise ValueError("could not recover rotation axis near pi")
        if largest == 0:
            axis[1] = np.copysign(axis[1], current[0, 1] + current[1, 0])
            axis[2] = np.copysign(axis[2], current[0, 2] + current[2, 0])
        elif largest == 1:
            axis[0] = np.copysign(axis[0], current[0, 1] + current[1, 0])
            axis[2] = np.copysign(axis[2], current[1, 2] + current[2, 1])
        else:
            axis[0] = np.copysign(axis[0], current[0, 2] + current[2, 0])
            axis[1] = np.copysign(axis[1], current[1, 2] + current[2, 1])
        result.reshape((-1, 3))[flat_index] = axis / np.linalg.norm(axis) * flat_angles[flat_index]
    return result


def compute_pose_kinematics(eef_pose: np.ndarray, *, fps: float) -> PoseKinematics:
    """Compute combined motion for a ``[T, 10 * arm_count]`` EEF trajectory."""
    poses = np.asarray(eef_pose, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] < 10 or poses.shape[1] % 10 != 0:
        raise ValueError(
            "eef_pose must have shape [T, 10 * arm_count] with arm_count >= 1, "
            f"got {poses.shape}"
        )
    if poses.shape[0] == 0:
        raise ValueError("eef_pose must contain at least one frame")
    if not np.all(np.isfinite(poses)):
        raise ValueError("eef_pose must be finite")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    frame_count = poses.shape[0]
    arm_count = poses.shape[1] // 10
    poses_by_arm = poses.reshape(frame_count, arm_count, 10)
    rotations = rotation_6d_to_matrix(poses_by_arm[:, :, 3:9])
    linear_velocity = np.zeros((frame_count, arm_count, 3), dtype=np.float64)
    angular_velocity = np.zeros((frame_count, arm_count, 3), dtype=np.float64)
    if frame_count > 1:
        linear_velocity[1:] = np.diff(poses_by_arm[:, :, :3], axis=0) * fps
        relative = np.swapaxes(rotations[:-1], -1, -2) @ rotations[1:]
        angular_velocity[1:] = rotation_log_vector(relative) * fps

    linear_velocity = linear_velocity.reshape(frame_count, arm_count * 3)
    angular_velocity = angular_velocity.reshape(frame_count, arm_count * 3)

    linear_acceleration = np.zeros((frame_count,), dtype=np.float64)
    angular_acceleration = np.zeros((frame_count,), dtype=np.float64)
    if frame_count > 2:
        linear_values = np.linalg.norm(np.diff(linear_velocity[1:], axis=0) * fps, axis=1)
        angular_values = np.linalg.norm(np.diff(angular_velocity[1:], axis=0) * fps, axis=1)
        linear_acceleration[2:] = linear_values
        angular_acceleration[2:] = angular_values
        linear_acceleration[:2] = linear_values[0]
        angular_acceleration[:2] = angular_values[0]

    return PoseKinematics(
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
        linear_speed=np.linalg.norm(linear_velocity, axis=1),
        angular_speed=np.linalg.norm(angular_velocity, axis=1),
        linear_acceleration=linear_acceleration,
        angular_acceleration=angular_acceleration,
    )
