# Faster LeRobot Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ISR LeRobot conversion visibly progress and substantially faster on multi-camera datasets while allowing noisy gripper changes to reach the requested retention ratio.

**Architecture:** Keep episode metadata and Parquet processing sequential, but encode the independent video streams for each episode through a bounded thread pool that launches FFmpeg subprocesses concurrently. Thread a configurable gripper-change tolerance into selection, record it in output metadata, and expose deterministic progress callbacks so the CLI can print elapsed time and ETA to stderr without corrupting its JSON stdout.

**Tech Stack:** Python 3.9+, concurrent.futures, FFmpeg, PyArrow, pytest.

---

### Task 1: CLI configuration and progress formatting

**Files:** Modify `tests/test_accelerate_lerobot_cli.py`; modify `src/isr/cli/accelerate_lerobot.py`.

- [x] Add failing parser tests requiring defaults `gripper_change_tolerance == 1e-4` and `video_workers == 4`, explicit overrides, and rejection of nonpositive values.
- [x] Add a failing pure-function test for a stable progress message containing completed/total episodes, percentage, elapsed time, and ETA.
- [x] Run the CLI tests and confirm the new expectations fail.
- [x] Implement positive argparse types, the two options, and progress formatting. Send callback output to stderr with flushing; keep the final conversion summary as JSON on stdout.
- [x] Re-run CLI tests and require them to pass.

### Task 2: Gripper tolerance through selection

**Files:** Modify `tests/test_lerobot_v21.py`; modify `src/isr/lerobot_v21.py`.

- [x] Add a failing selection test containing sub-threshold gripper noise and a real transition; require only the real transition sides in `forced_source_indices` when tolerance is `1e-4`.
- [x] Thread `gripper_change_tolerance` through `_episode_selection`, `convert_dataset`, and `rewrite_info`; validate it is finite and nonnegative and record it under `trajectory_acceleration`.
- [x] Re-run focused selection and metadata tests.

### Task 3: Bounded parallel video encoding

**Files:** Modify `tests/test_lerobot_v21.py`; modify `src/isr/lerobot_v21.py`.

- [x] Add failing tests that `rewrite_video` validates an expected source-frame count and that `convert_dataset` rejects nonpositive worker counts.
- [x] Add a conversion test with a patched video writer that records two video tasks overlapping when `video_workers=2`, and confirm the progress callback receives `(1, 1)`.
- [x] Implement `ThreadPoolExecutor(max_workers=video_workers)` for the video keys within each episode. Move source-frame-count validation into `rewrite_video` so each source video is probed once rather than twice.
- [x] Run the complete LeRobot conversion test module.

### Task 4: Documentation and verification

**Files:** Modify `README.md`, `docs/isr_va_vf_project_guide.md`.

- [x] Document both new options, stderr progress, why `1e-4` is appropriate for the measured dual-arm gripper noise, and a complete command for `insert_pcb_0831_100episodes`.
- [x] Run default and OpenPI full suites, compile sources, run `git diff --check`, and smoke-test CLI help.
- [x] Benchmark one representative episode or video and report measured evidence without starting a full dataset conversion.

## Self-review

- The input dataset remains read-only and an existing output is still rejected.
- Parallel tasks write distinct video destinations.
- Exceptions still remove the staging directory.
- Progress never contaminates machine-readable JSON stdout.
- The new default reaches approximately 50% retention on the audited insert-PCB dataset without discarding genuine gripper transitions above the configured threshold.
