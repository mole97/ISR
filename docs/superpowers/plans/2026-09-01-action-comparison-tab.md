# Accelerated Action Comparison Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second viewer tab that plots original, ISR–VA, and ISR–VF action values for the loaded episode on compressed or source-aligned time axes.

**Architecture:** A new GUI-independent module reads action columns directly from the three LeRobot 2.1 Parquet files and produces validated NumPy series with dimension names and source-index mappings. The existing Tkinter window gains a `ttk.Notebook`; its video controls remain in the first tab, while a Matplotlib canvas in the second tab selects an action key, one dimension, and the time-axis interpretation. Action loading is lazy so the video tab remains usable when PyArrow is unavailable.

**Tech Stack:** Python 3.9+, NumPy, PyArrow, Matplotlib, Tkinter, pytest.

---

## File structure

- Create `src/isr/action_compare.py`: Parquet path discovery, action loading, names, and plot-axis construction.
- Create `tests/test_action_compare.py`: synthetic LeRobot metadata/Parquet tests.
- Modify `src/isr/cli/compare_videos.py`: Notebook layout and embedded Action plot tab.
- Modify `pyproject.toml`, `requirements.txt`: make PyArrow available to the viewer.
- Modify `README_viewer.md`, `README.md`: explain the Action tab and both time axes.

### Task 1: Resolve and load action columns

**Files:** Create `src/isr/action_compare.py`; create `tests/test_action_compare.py`.

- [x] Write a PyArrow-backed tiny three-dataset fixture containing `actions.eef_pose`, `actions.joint_position`, `meta/info.json`, and VA/VF selection manifests. Test common action key discovery and chunk-aware Parquet path resolution.
- [x] Run `PYTHONPATH=src /home/jiongwei/miniconda3/envs/openpi/bin/python -m pytest tests/test_action_compare.py -q`; expect import failure for `isr.action_compare`.
- [x] Implement immutable `ActionComparison`, `common_action_keys`, `resolve_episode_data_path`, and `load_action_comparison`. Read only `actions.*` columns; require finite `[T,D]` arrays and verify accelerated row counts match their selection manifests.
- [x] Re-run the focused tests; expect them to pass.

### Task 2: Dimension labels and action time axes

**Files:** Modify `src/isr/action_compare.py`; extend `tests/test_action_compare.py`.

- [x] Write tests that recover `x…gripper` from the nested LeRobot `names` schema, fall back to `dim_0…dim_N`, and reject dimension-count mismatches.
- [x] Write tests for `action_plot_series`: `compressed` uses `arange(T)/fps` independently for each dataset, while `source` uses recorded VA/VF source indices divided by FPS.
- [x] Run the new tests and confirm failure for missing behavior.
- [x] Implement `action_dimension_names` and immutable `ActionPlotSeries`; validate action dimension and axis mode.
- [x] Re-run the complete action test module.

### Task 3: Add the Action comparison tab

**Files:** Modify `src/isr/cli/compare_videos.py`; extend `tests/test_compare_videos_cli.py` only for new pure helpers if introduced.

- [x] Replace the root-level video layout with a `ttk.Notebook` containing `视频对比` and `Action 对比` frames; preserve all existing video controls and shortcuts.
- [x] Build Action controls for action key, dimension, and axis mode (`加速后时间`/`原始源时间`). Embed a Matplotlib `FigureCanvasTkAgg` below them and add a summary label for original/VA/VF frames, durations, and speedup.
- [x] Load actions after a video episode succeeds and whenever the action key changes. Redraw one selected dimension with distinct original, VA, and VF colors, a grid, legend, axis labels, and source-point markers in source-time mode.
- [x] Catch missing PyArrow or malformed action data inside the Action tab and show an actionable error without breaking video playback.
- [x] Run the parser, episode comparison, and action comparison tests; smoke-import the CLI without `$DISPLAY`.

### Task 4: Dependencies and documentation

**Files:** Modify `pyproject.toml`, `requirements.txt`, `README_viewer.md`, `README.md`.

- [x] Add `pyarrow>=14` to the `viewer` optional dependency while keeping it in the LeRobot conversion and root requirements.
- [x] Document that `加速后时间` shows shorter VA/VF duration and `原始源时间` places retained actions back at their original timestamps.
- [x] Add a troubleshooting note for installing PyArrow when the Action tab reports a missing dependency.

### Task 5: Verification

**Files:** No production changes unless verification exposes a defect.

- [x] Run the full default test suite; action tests may skip only when PyArrow is absent.
- [x] Run the full OpenPI-environment suite with PyArrow and require all action tests to pass.
- [x] Run `python -m compileall -q src tests`, `git diff --check`, and `isr-compare-videos --help`.
- [x] Audit real episode 0: load both action keys and verify original 945 rows versus 472 VA/VF rows, correct names, and finite plot series in both time modes.

## Self-review

- The Action tab reads datasets without modifying them.
- Curves use the stored action targets, not regenerated state values.
- Compressed time exposes actual training-sequence duration; source time exposes sampling locations.
- Video playback remains available when action dependencies or data are missing.
- The first commit remains an isolated working checkpoint before this feature.
