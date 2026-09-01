# Dual-Arm EEF Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept single-arm `[T, 10]` and dual-arm `[T, 20]` EEF pose arrays while preserving motion and gripper events from either arm, and document exact video/action alignment.

**Architecture:** Interpret EEF pose as `[T, arm_count, 10]`, calculate each arm's SE(3) differences independently, then flatten per-arm velocity vectors before computing a combined norm. Pass all gripper columns to the existing change detector so either arm can force frames. Dataset rewriting remains unchanged because it already applies one selected row vector to every Parquet column and video stream.

**Tech Stack:** Python 3.9+, NumPy, PyArrow, pytest, Markdown.

---

### Task 1: Dual-arm pose kinematics and priority signals

**Files:**
- Modify: `tests/test_pose.py`
- Modify: `tests/test_trajectory_acceleration.py`
- Modify: `src/isr/pose.py`
- Modify: `src/isr/trajectory_acceleration.py`

- [x] **Step 1: Write failing dual-arm tests**

Add tests that concatenate two valid 10-D arm poses, require `compute_pose_kinematics` and VA/VF priority generation to return finite `[T]` scores, and require a `[T, 2]` gripper array to preserve changes from either column.

- [x] **Step 2: Verify RED**

Run `PYTHONPATH=src python -m pytest tests/test_pose.py tests/test_trajectory_acceleration.py -q`. Expect rejection of `[T, 20]` by `compute_pose_kinematics` and rejection of the 2-D gripper input.

- [x] **Step 3: Implement minimal multi-arm handling**

Reshape finite EEF input to `[T, arm_count, 10]`, with a positive dimension divisible by 10. Compute per-arm positions and rotations, flatten velocities to `[T, 3 * arm_count]`, and retain the existing scalar speed/acceleration API. Allow `find_gripper_change_indices` to accept `[T]` or `[T, arm_count]` and return the union of both sides of every column transition.

- [x] **Step 4: Verify GREEN**

Re-run the focused tests and require all to pass.

### Task 2: Dataset selection integration and documentation

**Files:**
- Modify: `tests/test_lerobot_v21.py`
- Modify: `src/isr/lerobot_v21.py`
- Modify: `docs/isr_va_vf_project_guide.md`

- [x] **Step 1: Write a failing integration test**

Build a 20-D episode table in which only the second arm's gripper changes. Call `_episode_selection` and assert both transition-side indices appear in `forced_source_indices`.

- [x] **Step 2: Verify RED**

Run the new test and expect the current `poses[:, 9]` extraction to miss the second arm or the 20-D pose validation to fail.

- [x] **Step 3: Implement dataset integration**

Pass `poses[:, 9::10]` to the gripper detector. Do not modify `resample_episode_table`: its existing `table.take(selected)` is the required action/state alignment policy.

- [x] **Step 4: Document alignment**

Explain with source/output index equations that output video frame `k`, every output Parquet signal row `k`, and Action row `k` all come from the same original row `selected[k]`. State that timestamps are compressed to `k/fps`, while `source_frame_index` and the selection manifest retain provenance; no interpolation, shift, state-to-action regeneration, or PI0.5 change occurs.

- [x] **Step 5: Verify complete behavior**

Run the full suite in the default and OpenPI environments, compile the sources, run `git diff --check`, and inspect the final diff.

## Self-review

- Single-arm 10-D behavior and public result shapes remain compatible.
- Dual-arm movement from either arm contributes to VA/VF speed.
- Gripper transitions from either arm become forced frames.
- Action/video alignment uses the identical source-index vector and remains offline/read-only with respect to the input dataset.
