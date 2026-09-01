# Episode Video Comparison Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a desktop window that accepts a dropped LeRobot episode and compares the original, ISR–VA, and ISR–VF videos side by side.

**Architecture:** A GUI-independent module resolves dataset roots, episode numbers, camera paths, selection manifests, and synchronized frame indices. A Tkinter/OpenCV front end owns playback and rendering, with optional `tkinterdnd2` integration for native file dropping and a file chooser fallback. Source-aligned mode maps each original frame to the nearest retained VA/VF frame; native-time mode advances all three 30 FPS videos by output frame index to expose the actual acceleration.

**Tech Stack:** Python 3.9+, NumPy, Tkinter, tkinterdnd2, OpenCV, Pillow, pytest.

---

## File structure

- Create `src/isr/episode_compare.py`: pure path discovery, manifest loading, and frame mapping.
- Create `src/isr/cli/compare_videos.py`: drag-and-drop Tkinter window and OpenCV playback.
- Create `tests/test_episode_compare.py`: path, metadata, and synchronization tests.
- Create `tests/test_compare_videos_cli.py`: parser tests without importing GUI dependencies.
- Modify `pyproject.toml`: viewer dependencies and `isr-compare-videos` entry point.
- Modify `README.md`: launch command, drag targets, camera and playback controls.

### Task 1: Discover episode and dataset triplet

**Files:** Create `src/isr/episode_compare.py`; test `tests/test_episode_compare.py`.

- [x] Write tests that drop a dataset root, Parquet, or MP4 path and expect the enclosing `meta/info.json` root plus an `episode_000123` index. Test sibling derivation for roots ending `_isr_va` and `_isr_vf`, and reject paths without an episode index when no explicit episode is supplied.
- [x] Run `PYTHONPATH=src pytest tests/test_episode_compare.py -q`; expect import failure for `isr.episode_compare`.
- [x] Implement `extract_episode_index`, `find_dataset_root`, `derive_dataset_triplet`, and immutable `DatasetTriplet`. Validate that all three roots contain `meta/info.json` and never write to them.
- [x] Run the focused tests; expect them to pass.

### Task 2: Resolve cameras and selection mappings

**Files:** Modify `src/isr/episode_compare.py`; extend `tests/test_episode_compare.py`.

- [x] Write tests using three tiny metadata trees. Assert that only common video features are returned, the LeRobot `video_path` template resolves the requested episode/chunk, and VA/VF manifest rows return strictly increasing source indices.
- [x] Run the focused tests and confirm the new assertions fail because the functions are absent.
- [x] Implement `common_video_keys`, `resolve_video_path`, and `load_source_indices`. Read `meta/info.json` and `meta/selection_manifest.jsonl` with standard JSON only so the viewer does not require PyArrow.
- [x] Re-run the focused tests; expect them to pass.

### Task 3: Define playback synchronization

**Files:** Modify `src/isr/episode_compare.py`; extend `tests/test_episode_compare.py`.

- [x] Write tests for `nearest_retained_frame` and `map_playback_frames`. In `source` mode, original frame 5 must map to the nearest retained output frame in each ablation; in `native` mode, tick 5 must map to frame 5 in each video, clamped after shorter outputs end.
- [x] Run the focused tests and verify behavioral failures.
- [x] Implement deterministic nearest-index mapping with left-side tie breaking and a `PlaybackFrames` result containing video indices and represented source indices.
- [x] Re-run all pure-logic tests.

### Task 4: Build the comparison window

**Files:** Create `src/isr/cli/compare_videos.py`; test `tests/test_compare_videos_cli.py`.

- [x] Write parser tests for optional `--original-root`, `--va-root`, `--vf-root`, `--episode`, and `--camera` arguments. Importing the parser must not open a window or require a display.
- [x] Run the parser tests and verify import failure.
- [x] Implement a lazily imported Tkinter GUI with three labeled aspect-preserving panes, Open button, episode spinner, camera selector, source/native mode selector, play/pause, previous/next frame, and timeline. Decode with `cv2.VideoCapture`, convert BGR to `PIL.ImageTk`, and release captures on episode change or close.
- [x] If `tkinterdnd2` is installed, bind `DND_FILES` to the whole window and accept a dataset directory, episode Parquet, or episode MP4. Otherwise keep the Open button functional and show that drag-and-drop is unavailable.
- [x] In source mode drive the slider over original source frames and use the selection manifests for nearest VA/VF frames. In native mode advance each video at its own frame index at the dataset FPS, freezing a stream after its last frame.
- [x] Run parser and logic tests; smoke-import the CLI without `$DISPLAY`.

### Task 5: Package, document, and verify

**Files:** Modify `pyproject.toml`, `README.md`; no new production modules.

- [x] Add `isr-compare-videos = "isr.cli.compare_videos:main"` and a `viewer` extra containing `opencv-python`, `Pillow`, and `tkinterdnd2`.
- [x] Document `pip install -e '.[viewer]'` and a launch example using the three generated dataset roots. Explain both playback modes and supported drop targets.
- [x] Run `PYTHONPATH=src pytest -q`, `python -m compileall -q src tests`, `git diff --check`, and CLI `--help`.
- [x] When a graphical display is available, open episode 0 and verify camera switching, dragging, seeking, and both playback modes against the real original/VA/VF datasets.

## Self-review

- The viewer is read-only and does not alter any dataset.
- A single camera key is shown across all three datasets.
- Source-frame alignment uses the converter's recorded mappings, not proportional guesses.
- Native playback visibly preserves the two accelerated datasets' shorter duration.
- GUI-only dependencies are imported after argument parsing so headless tests remain usable.
