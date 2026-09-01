# ISR–VA / ISR–VF LeRobot Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a non-destructive offline converter that reads the 355-episode LeRobot 2.1 dataset and writes separate ISR–VA and ISR–VF time-compressed datasets without changing PI0.5.

**Architecture:** Pure NumPy modules recover SE(3) poses, calculate VA/VF information density, and select source indices with bounded ISR dynamic programming. A LeRobot adapter applies the same indices to every parquet signal and video stream, rewrites row metadata and statistics, and stages output before an atomic rename. The input root is always opened read-only and is rejected as an output target.

**Tech Stack:** Python 3.9+, NumPy, PyArrow, FFmpeg/FFprobe, pytest.

---

## File structure

- Create `src/isr/pose.py`: 6D rotation recovery and translational/angular kinematics.
- Create `src/isr/trajectory_acceleration.py`: VA/VF priorities and bounded ISR index selection.
- Create `src/isr/lerobot_v21.py`: LeRobot metadata, parquet, statistics, and video rewriting.
- Create `src/isr/cli/accelerate_lerobot.py`: safe two-mode command-line entry point.
- Create `tests/test_pose.py`, `tests/test_trajectory_acceleration.py`, and `tests/test_lerobot_v21.py`.
- Modify `pyproject.toml`: optional LeRobot dependencies and `isr-accelerate-lerobot` entry point.
- Modify `README.md`: document two independent output commands and invariants.

### Task 1: SE(3) pose support

**Files:** Create `src/isr/pose.py`; test `tests/test_pose.py`.

- [x] Write tests first for identity 6D rotation, non-orthogonal Gram–Schmidt input, 90-degree angular velocity, constant translation velocity, and zero constant-velocity acceleration.

```python
def test_rotation_6d_identity():
    matrix = rotation_6d_to_matrix(np.array([[1, 0, 0, 0, 1, 0]], dtype=float))
    np.testing.assert_allclose(matrix[0], np.eye(3), atol=1e-7)

def test_pose_kinematics_reports_quarter_turn():
    poses = np.stack([identity_pose(), z_quarter_turn_pose()])
    result = compute_pose_kinematics(poses, fps=2.0)
    assert result.angular_speed[1] == pytest.approx(np.pi, rel=1e-6)
```

- [x] Run `PYTHONPATH=src pytest tests/test_pose.py -q`; expect import failure for `isr.pose`.
- [x] Implement `rotation_6d_to_matrix(values)`, `rotation_log_vector(matrix)`, and `compute_pose_kinematics(eef_pose, fps)` for `[T,10]` single-arm data. Treat columns `0:3` as xyz, `3:9` as two concatenated rotation-matrix columns, and column `9` as gripper only.
- [x] Re-run the focused tests, then `PYTHONPATH=src pytest -q`; expect all tests to pass.

### Task 2: VA/VF priorities and ISR indices

**Files:** Create `src/isr/trajectory_acceleration.py`; test `tests/test_trajectory_acceleration.py`.

- [x] Write tests first for robust normalization, VA priority peaks at acceleration, VF priority peaks at force steps, gripper-change retention, first/last retention, maximum skip, deterministic selection, and denser selection in high-information regions.

```python
def test_vf_priority_marks_force_step():
    force = np.r_[np.zeros(10), np.ones(10) * 5]
    score = compute_vf_priority(straight_poses(20), force[:, None], fps=30).priority
    assert np.argmax(score) in range(10, 13)

def test_selector_never_exceeds_max_skip():
    selected = select_isr_indices(np.zeros(31), target_retention=0.2, max_skip=4)
    assert max(np.diff(selected)) <= 4
```

- [x] Run `PYTHONPATH=src pytest tests/test_trajectory_acceleration.py -q`; expect import failure.
- [x] Implement immutable `PrioritySignals`, `compute_va_priority`, `compute_vf_priority`, and `select_isr_indices`. Use clipped Q10/Q90 normalization, force debias + causal EMA + derivative deadband, and bounded dynamic programming. Tune the target information spacing by deterministic bisection to approach `round(T * target_retention)`.
- [x] Force indices `0`, `T-1`, and gripper transitions in both modes. In VF mode also force force-event peaks and configured neighboring frames. Reject non-finite signals and invalid FPS/retention/max-skip.
- [x] Run focused and full tests; expect all tests to pass.

### Task 3: Non-destructive LeRobot parquet conversion

**Files:** Create `src/isr/lerobot_v21.py`; test `tests/test_lerobot_v21.py`.

- [x] Write a tiny two-episode LeRobot fixture using PyArrow. Test that all non-index columns equal `source.take(selected)`, actions are sampled at the same source rows as states, timestamps become `arange(M)/fps`, global indices are contiguous, and `sampling.speed_force_weight` plus its metadata contract are removed.

```python
def test_resample_table_preserves_action_state_phase(tiny_table):
    output = resample_episode_table(tiny_table, np.array([0, 2, 4]), fps=30, global_start=7)
    assert output["actions.eef_pose"].to_pylist() == tiny_table.take([0, 2, 4])["actions.eef_pose"].to_pylist()
    assert output["index"].to_pylist() == [7, 8, 9]
```

- [x] Run with `/home/jiongwei/miniconda3/envs/openpi/bin/python -m pytest tests/test_lerobot_v21.py -q`; expect import failure.
- [x] Implement input/output path guards, metadata loading, episode table resampling, numeric episode statistics, `episodes.jsonl`, `episodes_stats.jsonl`, and `info.json` rewriting. Preserve task metadata and feature schemas; add `source_frame_index` and a `trajectory_acceleration` audit block.
- [x] Write each output under a sibling staging directory. Always refuse an existing output; never delete or rename the input root.
- [x] Run focused and full tests under the OpenPI environment.

### Task 4: Exact video frame selection

**Files:** Modify `src/isr/lerobot_v21.py`; extend `tests/test_lerobot_v21.py`.

- [x] Generate a six-frame synthetic H.264 video in a temporary directory and write a failing test expecting source indices `[0,2,5]` to produce exactly three 30 FPS frames.
- [x] Implement `rewrite_video(source, destination, selected_indices, fps)` using FFmpeg's `select` filter and `setpts=N/(fps*TB)`. Probe source and destination frame counts with FFprobe and fail if they disagree with parquet/source lengths.
- [x] Re-run focused tests and verify both camera keys are discovered from `info.json`, not hard-coded.

### Task 5: Dataset converter and CLI

**Files:** Create `src/isr/cli/accelerate_lerobot.py`; modify `pyproject.toml` and `README.md`; extend tests.

- [x] Write parser and safety tests for `--mode va|vf`, `--input`, `--output`, `--target-retention`, and `--max-skip`. Verify equal resolved input/output paths are rejected.
- [x] Implement `convert_dataset(...)`: calibrate VF bias/noise from each episode's configured contact-free prefix, calculate per-episode selections, rewrite parquet/video/statistics, and emit `selection_manifest.jsonl` containing source indices and achieved speedup.
- [x] Add optional dependencies:

```toml
lerobot = ["pandas>=2", "pyarrow>=14"]
```

and script:

```toml
isr-accelerate-lerobot = "isr.cli.accelerate_lerobot:main"
```

- [x] Document that each mode needs a separate output directory and separate PI0.5 normalization statistics.
- [x] Run CLI help and all tests.

### Task 6: Smoke conversion and full conversion

**Files:** No source edits unless a failing test exposes a defect.

- [x] Run one real episode through VA and VF conversion into distinct temporary outputs.
- [x] Audit output row/video counts, contiguous indices, 30 FPS timestamps, feature schema, source mapping, selected-index monotonicity, and unchanged input checksums.
- [x] Run the full source dataset twice, without `--overwrite`, into new sibling directories ending `_isr_va` and `_isr_vf`.
- [x] Document that PI0.5 normalization statistics must be recomputed for each output dataset outside this converter; do not modify model code or enable prioritized sampling.
- [x] Run final tests, available static checks, `git diff --check`, and dataset integrity audit.

## Self-review

- Input mutation is prohibited by path guards, staging, and checksum verification.
- VA and VF share all reconstruction behavior; only priority calculation differs.
- State, action, tactile, joint, and video signals use identical selected source indices.
- The 10D EEF schema and two-column 6D rotation convention are explicit.
- Stale speed-force sampling weights and metadata cannot leak into accelerated outputs.
- Tests cover algorithms, schema reconstruction, video alignment, and destructive-path rejection.
