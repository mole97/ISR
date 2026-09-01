"""Desktop viewer for original, ISR--VA, and ISR--VF episode videos."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Sequence


DEFAULT_CAMERA = "observation.images.third_view"


def scaled_font_size(font_size: int, scale: float) -> int:
    """Scale a Tk point/pixel font size while preserving its unit sign."""
    if font_size == 0 or scale <= 0:
        raise ValueError("font size and UI scale must be positive")
    magnitude = max(1, round(abs(font_size) * scale))
    return magnitude if font_size > 0 else -magnitude


def _positive_ui_scale(value: str) -> float:
    try:
        scale = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("UI scale must be a number") from error
    if scale <= 0:
        raise argparse.ArgumentTypeError("UI scale must be positive")
    return scale


def fit_image_size(
    source: tuple[int, int],
    target: tuple[int, int],
) -> tuple[int, int]:
    """Fit an image inside a box while preserving aspect ratio, including upscale."""
    source_width, source_height = source
    target_width, target_height = target
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("source and target dimensions must be positive")
    scale = min(target_width / source_width, target_height / source_height)
    return (
        max(1, round(source_width * scale)),
        max(1, round(source_height * scale)),
    )


def handle_drop_data(
    data: str,
    *,
    splitlist: Callable[[str], Sequence[str]],
    load_path: Callable[[Path], None],
) -> str:
    """Load the first dropped filesystem path and acknowledge a copy action."""
    paths = splitlist(data)
    if paths:
        load_path(Path(paths[0]))
    return "copy"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="isr-compare-videos",
        description="Compare original, ISR-VA, and ISR-VF episode videos.",
    )
    parser.add_argument("--original-root", type=Path, help="Original LeRobot dataset root.")
    parser.add_argument("--va-root", type=Path, help="ISR-VA LeRobot dataset root.")
    parser.add_argument("--vf-root", type=Path, help="ISR-VF LeRobot dataset root.")
    parser.add_argument("--episode", type=int, default=0, help="Initial episode index.")
    parser.add_argument(
        "--camera",
        default=DEFAULT_CAMERA,
        help=f"Initial LeRobot video feature key (default: {DEFAULT_CAMERA}).",
    )
    parser.add_argument(
        "--ui-scale",
        type=_positive_ui_scale,
        default=2.4,
        help="Font scale for window controls and labels (default: 2.4).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    run_viewer(arguments)
    return 0


def run_viewer(arguments: argparse.Namespace) -> None:
    """Launch the graphical viewer after command-line parsing."""
    try:
        import cv2
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure
        from PIL import Image, ImageTk
        import tkinter as tk
        from tkinter import filedialog, font as tkfont, messagebox, ttk
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Video viewer dependencies are missing; install with: "
            "pip install -e '.[viewer]'"
        ) from error

    from isr.episode_compare import (
        DatasetTriplet,
        common_video_keys,
        derive_dataset_triplet,
        extract_episode_index,
        load_source_indices,
        map_playback_frames,
        resolve_video_path,
    )

    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
    except ModuleNotFoundError:
        DND_FILES = None
        TkinterDnD = None

    class ComparisonWindow:
        def __init__(self) -> None:
            self.root = TkinterDnD.Tk() if TkinterDnD is not None else tk.Tk()
            self.root.title("ISR Episode Video Comparison")
            self.root.geometry("1280x650")
            self.root.minsize(960, 560)
            self._configure_fonts()
            self.triplet: DatasetTriplet | None = None
            self.captures: dict[str, object] = {}
            self.frame_counts: dict[str, int] = {}
            self.source_indices: dict[str, object] = {}
            self.photos: dict[str, object] = {}
            self.last_frame_indices: dict[str, int] = {}
            self.playing = False
            self.after_id: str | None = None
            self.resize_after_id: str | None = None
            self.fps = 30.0
            self.source_frame_count = 1
            self.tick = 0
            self.seeking = False
            self.action_comparison = None

            self.episode_var = tk.IntVar(value=arguments.episode)
            self.camera_var = tk.StringVar(value=arguments.camera)
            self.mode_var = tk.StringVar(value="源帧对齐")
            self.timeline_var = tk.DoubleVar(value=0.0)
            self.status_var = tk.StringVar(value="拖入 episode Parquet、MP4 或数据集目录")
            self.action_key_var = tk.StringVar(value="actions.eef_pose")
            self.action_dimension_var = tk.StringVar(value="")
            self.action_axis_var = tk.StringVar(value="加速后时间")
            self.action_summary_var = tk.StringVar(value="请先加载 Episode")
            self.action_status_var = tk.StringVar(value="")
            self.frame_text_vars = {
                name: tk.StringVar(value="未加载") for name in ("original", "va", "vf")
            }
            self.image_labels: dict[str, object] = {}

            self._build_layout()
            self._bind_events()
            self.root.protocol("WM_DELETE_WINDOW", self.close)
            self.root.after_idle(self._maximize_window)
            self._load_initial_roots()

        def _configure_fonts(self) -> None:
            font_names = (
                "TkDefaultFont",
                "TkTextFont",
                "TkMenuFont",
                "TkHeadingFont",
                "TkCaptionFont",
                "TkSmallCaptionFont",
                "TkTooltipFont",
            )
            for name in font_names:
                try:
                    font = tkfont.nametofont(name, root=self.root)
                    font.configure(
                        size=scaled_font_size(int(font.cget("size")), arguments.ui_scale)
                    )
                except tk.TclError:
                    continue
            style = ttk.Style(self.root)
            style.configure(".", font=tkfont.nametofont("TkDefaultFont", root=self.root))
            style.configure(
                "TLabelframe.Label",
                font=tkfont.nametofont("TkHeadingFont", root=self.root),
            )

        def _build_layout(self) -> None:
            self.notebook = ttk.Notebook(self.root)
            self.notebook.pack(fill="both", expand=True)
            self.video_tab = ttk.Frame(self.notebook)
            self.action_tab = ttk.Frame(self.notebook)
            self.notebook.add(self.video_tab, text="视频对比")
            self.notebook.add(self.action_tab, text="Action 对比")

            header = ttk.Frame(self.video_tab, padding=(12, 10))
            header.pack(fill="x")
            ttk.Button(header, text="打开 Episode", command=self.open_episode).pack(side="left")
            ttk.Label(header, text="Episode").pack(side="left", padx=(14, 4))
            self.episode_spin = ttk.Spinbox(
                header,
                from_=0,
                to=999999,
                width=8,
                textvariable=self.episode_var,
                command=self.load_current_episode,
            )
            self.episode_spin.pack(side="left")
            ttk.Button(header, text="加载", command=self.load_current_episode).pack(
                side="left", padx=(4, 12)
            )
            ttk.Label(header, text="相机").pack(side="left", padx=(0, 4))
            self.camera_box = ttk.Combobox(
                header,
                width=40,
                textvariable=self.camera_var,
                state="readonly",
            )
            self.camera_box.pack(side="left", fill="x", expand=True)
            self.camera_box.bind("<<ComboboxSelected>>", self._on_camera_change)
            ttk.Label(header, text="播放模式").pack(side="left", padx=(12, 4))
            self.mode_box = ttk.Combobox(
                header,
                width=10,
                textvariable=self.mode_var,
                values=("源帧对齐", "真实时间"),
                state="readonly",
            )
            self.mode_box.pack(side="left")
            self.mode_box.bind("<<ComboboxSelected>>", self._on_mode_change)

            panes = ttk.Frame(self.video_tab, padding=(8, 0))
            panes.pack(fill="both", expand=True)
            titles = {
                "original": "原始数据",
                "va": "ISR–VA（速度–加速度）",
                "vf": "ISR–VF（速度–力）",
            }
            for column, name in enumerate(("original", "va", "vf")):
                pane = ttk.LabelFrame(panes, text=titles[name], padding=6)
                pane.grid(row=0, column=column, padx=4, sticky="nsew")
                panes.columnconfigure(column, weight=1, uniform="video")
                panes.rowconfigure(0, weight=1)
                image_label = tk.Label(
                    pane,
                    text="拖入 episode 以加载",
                    background="#111111",
                    foreground="#dddddd",
                    anchor="center",
                )
                image_label.pack(fill="both", expand=True)
                self.image_labels[name] = image_label
                ttk.Label(
                    pane,
                    textvariable=self.frame_text_vars[name],
                    anchor="center",
                ).pack(fill="x", pady=(5, 0))

            controls = ttk.Frame(self.video_tab, padding=(12, 8))
            controls.pack(fill="x")
            ttk.Button(controls, text="◀", width=4, command=lambda: self.step(-1)).pack(
                side="left"
            )
            self.play_button = ttk.Button(
                controls,
                text="播放",
                width=7,
                command=self.toggle_play,
            )
            self.play_button.pack(side="left", padx=4)
            ttk.Button(controls, text="▶", width=4, command=lambda: self.step(1)).pack(
                side="left"
            )
            self.timeline = ttk.Scale(
                controls,
                from_=0,
                to=0,
                variable=self.timeline_var,
                command=self._on_seek,
            )
            self.timeline.pack(side="left", fill="x", expand=True, padx=(12, 0))
            ttk.Label(
                self.video_tab,
                textvariable=self.status_var,
                padding=(12, 0, 12, 8),
            ).pack(fill="x")
            self._build_action_tab()

        def _build_action_tab(self) -> None:
            controls = ttk.Frame(self.action_tab, padding=(12, 10))
            controls.pack(fill="x")
            ttk.Label(controls, text="Action 字段").pack(side="left")
            self.action_key_box = ttk.Combobox(
                controls,
                width=28,
                textvariable=self.action_key_var,
                state="readonly",
            )
            self.action_key_box.pack(side="left", padx=(6, 18))
            self.action_key_box.bind("<<ComboboxSelected>>", self._on_action_key_change)
            ttk.Label(controls, text="维度").pack(side="left")
            self.action_dimension_box = ttk.Combobox(
                controls,
                width=18,
                textvariable=self.action_dimension_var,
                state="readonly",
            )
            self.action_dimension_box.pack(side="left", padx=(6, 18))
            self.action_dimension_box.bind(
                "<<ComboboxSelected>>",
                self._on_action_plot_change,
            )
            ttk.Label(controls, text="横轴").pack(side="left")
            self.action_axis_box = ttk.Combobox(
                controls,
                width=14,
                textvariable=self.action_axis_var,
                values=("加速后时间", "原始源时间"),
                state="readonly",
            )
            self.action_axis_box.pack(side="left", padx=(6, 18))
            self.action_axis_box.bind("<<ComboboxSelected>>", self._on_action_plot_change)
            ttk.Button(controls, text="重新加载", command=self._load_action_data).pack(
                side="left"
            )

            ttk.Label(
                self.action_tab,
                textvariable=self.action_summary_var,
                padding=(12, 0, 12, 6),
            ).pack(fill="x")
            figure_font_size = max(12, round(7 * arguments.ui_scale))
            self.action_figure_font_size = figure_font_size
            self.action_figure = Figure(figsize=(12, 7), dpi=100, constrained_layout=True)
            self.action_axes = self.action_figure.add_subplot(111)
            self.action_axes.text(
                0.5,
                0.5,
                "加载 Episode 后显示 Action 曲线",
                ha="center",
                va="center",
                transform=self.action_axes.transAxes,
                fontsize=figure_font_size,
            )
            self.action_canvas = FigureCanvasTkAgg(
                self.action_figure,
                master=self.action_tab,
            )
            self.action_canvas.draw()
            self.action_canvas.get_tk_widget().pack(fill="both", expand=True)
            self.action_toolbar = NavigationToolbar2Tk(
                self.action_canvas,
                self.action_tab,
                pack_toolbar=False,
            )
            self.action_toolbar.update()
            self.action_toolbar.pack(fill="x")
            ttk.Label(
                self.action_tab,
                textvariable=self.action_status_var,
                padding=(12, 4, 12, 8),
            ).pack(fill="x")

        def _bind_events(self) -> None:
            self.root.bind("<space>", lambda _event: self.toggle_play())
            self.root.bind("<Left>", lambda _event: self.step(-1))
            self.root.bind("<Right>", lambda _event: self.step(1))
            self.root.bind("<Configure>", self._on_resize)
            self.episode_spin.bind("<Return>", lambda _event: self.load_current_episode())
            if DND_FILES is not None:
                self.root.drop_target_register(DND_FILES)
                self.root.dnd_bind("<<Drop>>", self._on_drop)
                self.status_var.set(
                    "可拖入 episode Parquet、MP4 或数据集目录；"
                    "也可点击“打开 Episode”"
                )
            else:
                self.status_var.set(
                    "未安装 tkinterdnd2，拖放不可用；请点击“打开 Episode”"
                )

        def _maximize_window(self) -> None:
            try:
                self.root.attributes("-zoomed", True)
            except tk.TclError:
                try:
                    self.root.state("zoomed")
                except tk.TclError:
                    pass

        def _on_resize(self, event: object) -> None:
            if event.widget is not self.root or not self.captures:
                return
            if self.resize_after_id is not None:
                self.root.after_cancel(self.resize_after_id)
            self.resize_after_id = self.root.after(120, self._render_after_resize)

        def _render_after_resize(self) -> None:
            self.resize_after_id = None
            self.last_frame_indices.clear()
            self.render_current()

        def _load_initial_roots(self) -> None:
            provided = (
                arguments.original_root,
                arguments.va_root,
                arguments.vf_root,
            )
            if not any(provided):
                return
            if not all(provided):
                raise SystemExit(
                    "--original-root, --va-root, and --vf-root must be provided together"
                )
            try:
                self.triplet = derive_dataset_triplet(
                    arguments.original_root,
                    original_root=arguments.original_root,
                    va_root=arguments.va_root,
                    vf_root=arguments.vf_root,
                )
                self._configure_cameras(preferred=arguments.camera)
                self.root.after(0, self.load_current_episode)
            except Exception as error:
                self.root.after(0, lambda captured=error: self._show_error(captured))

        def _configure_cameras(self, *, preferred: str | None = None) -> None:
            if self.triplet is None:
                return
            cameras = common_video_keys(self.triplet)
            self.camera_box.configure(values=cameras)
            selected = preferred if preferred in cameras else cameras[0]
            self.camera_var.set(selected)

        def _show_action_message(self, message: str) -> None:
            self.action_axes.clear()
            self.action_axes.text(
                0.5,
                0.5,
                message,
                ha="center",
                va="center",
                wrap=True,
                transform=self.action_axes.transAxes,
                fontsize=self.action_figure_font_size,
            )
            self.action_axes.set_axis_off()
            self.action_canvas.draw_idle()

        def _load_action_data(self) -> None:
            if self.triplet is None:
                self.action_summary_var.set("请先加载 Episode")
                return
            try:
                from isr.action_compare import common_action_keys, load_action_comparison

                keys = common_action_keys(self.triplet)
                self.action_key_box.configure(values=keys)
                action_key = self.action_key_var.get()
                if action_key not in keys:
                    action_key = keys[0]
                    self.action_key_var.set(action_key)
                episode = int(self.episode_var.get())
                comparison = load_action_comparison(
                    self.triplet,
                    episode_index=episode,
                    action_key=action_key,
                )
                self.action_comparison = comparison
                dimensions = [
                    f"{index}: {name}"
                    for index, name in enumerate(comparison.dimension_names)
                ]
                self.action_dimension_box.configure(values=dimensions)
                if self.action_dimension_var.get() not in dimensions:
                    self.action_dimension_var.set(dimensions[0])
                source_frames = comparison.original.shape[0]
                va_frames = comparison.va.shape[0]
                vf_frames = comparison.vf.shape[0]
                self.action_summary_var.set(
                    f"Episode {episode:06d} · {action_key}  |  "
                    f"原始 {source_frames} 帧 / {source_frames / comparison.fps:.2f}s  |  "
                    f"VA {va_frames} 帧 / {va_frames / comparison.fps:.2f}s / "
                    f"{source_frames / va_frames:.2f}×  |  "
                    f"VF {vf_frames} 帧 / {vf_frames / comparison.fps:.2f}s / "
                    f"{source_frames / vf_frames:.2f}×"
                )
                self.action_status_var.set("")
                self._redraw_action_plot()
            except ModuleNotFoundError as error:
                self.action_comparison = None
                message = (
                    "Action 页面需要 PyArrow。请执行：\n"
                    "python -m pip install 'pyarrow>=14'"
                )
                self.action_summary_var.set("Action 数据依赖尚未安装")
                self.action_status_var.set(str(error))
                self._show_action_message(message)
            except Exception as error:
                self.action_comparison = None
                self.action_summary_var.set("Action 数据加载失败")
                self.action_status_var.set(str(error))
                self._show_action_message(str(error))

        def _on_action_key_change(self, _event: object | None = None) -> None:
            self._load_action_data()

        def _on_action_plot_change(self, _event: object | None = None) -> None:
            self._redraw_action_plot()

        def _redraw_action_plot(self) -> None:
            if self.action_comparison is None:
                return
            try:
                from isr.action_compare import action_plot_series

                dimension_text = self.action_dimension_var.get()
                dimension = int(dimension_text.split(":", 1)[0])
                axis_mode = (
                    "compressed"
                    if self.action_axis_var.get() == "加速后时间"
                    else "source"
                )
                plot = action_plot_series(
                    self.action_comparison,
                    dimension=dimension,
                    axis_mode=axis_mode,
                )
                axes = self.action_axes
                axes.clear()
                axes.set_axis_on()
                marker = "o" if axis_mode == "source" else None
                marker_size = 2.5 if marker else None
                axes.plot(
                    plot.original_time,
                    plot.original_values,
                    color="#4169E1",
                    linewidth=1.5,
                    alpha=0.75,
                    label="原始",
                )
                axes.plot(
                    plot.va_time,
                    plot.va_values,
                    color="#D62728",
                    linewidth=1.8,
                    marker=marker,
                    markersize=marker_size,
                    label="ISR–VA",
                )
                axes.plot(
                    plot.vf_time,
                    plot.vf_values,
                    color="#2CA02C",
                    linewidth=1.8,
                    marker=marker,
                    markersize=marker_size,
                    label="ISR–VF",
                )
                x_label = (
                    "加速后序列时间 (s)"
                    if axis_mode == "compressed"
                    else "原始源时间 (s)"
                )
                axes.set_xlabel(x_label, fontsize=self.action_figure_font_size)
                axes.set_ylabel(plot.dimension_name, fontsize=self.action_figure_font_size)
                axes.set_title(
                    f"{self.action_comparison.action_key} · {plot.dimension_name}",
                    fontsize=self.action_figure_font_size + 2,
                )
                axes.tick_params(labelsize=max(10, self.action_figure_font_size - 2))
                axes.grid(True, alpha=0.25)
                axes.legend(fontsize=max(10, self.action_figure_font_size - 2))
                self.action_status_var.set(
                    "加速后时间：比较压缩后的序列时长"
                    if axis_mode == "compressed"
                    else (
                        "原始源时间：按 selection manifest "
                        "显示保留 Action 的原始位置"
                    )
                )
                self.action_canvas.draw_idle()
            except Exception as error:
                self.action_status_var.set(str(error))
                self._show_action_message(str(error))

        def open_episode(self) -> None:
            selected = filedialog.askopenfilename(
                title="选择 episode Parquet 或 MP4",
                filetypes=(
                    ("LeRobot episode", "*.parquet *.mp4"),
                    ("Parquet", "*.parquet"),
                    ("MP4", "*.mp4"),
                    ("All files", "*"),
                ),
            )
            if selected:
                self.load_drop_path(Path(selected))

        def _on_drop(self, event: object) -> str:
            return handle_drop_data(
                event.data,
                splitlist=self.root.tk.splitlist,
                load_path=self.load_drop_path,
            )

        def load_drop_path(self, path: Path) -> None:
            try:
                self.pause()
                self.triplet = derive_dataset_triplet(path)
                try:
                    self.episode_var.set(extract_episode_index(path))
                except ValueError:
                    pass
                self._configure_cameras(preferred=self.camera_var.get())
                self.load_current_episode()
            except Exception as error:
                self._show_error(error)

        def _on_camera_change(self, _event: object | None = None) -> None:
            if self.triplet is not None:
                self.load_current_episode()

        def _on_mode_change(self, _event: object | None = None) -> None:
            self.pause()
            self.tick = int(round(self.timeline_var.get()))
            self.render_current()

        def load_current_episode(self) -> None:
            if self.triplet is None:
                self.status_var.set(
                    "请先拖入 episode，或通过命令行指定三组数据集"
                )
                return
            try:
                self.pause()
                episode = int(self.episode_var.get())
                camera = self.camera_var.get()
                paths = {
                    "original": resolve_video_path(
                        self.triplet.original,
                        episode_index=episode,
                        video_key=camera,
                    ),
                    "va": resolve_video_path(
                        self.triplet.va,
                        episode_index=episode,
                        video_key=camera,
                    ),
                    "vf": resolve_video_path(
                        self.triplet.vf,
                        episode_index=episode,
                        video_key=camera,
                    ),
                }
                va_indices = load_source_indices(self.triplet.va, episode_index=episode)
                vf_indices = load_source_indices(self.triplet.vf, episode_index=episode)
                self._release_captures()
                for name, path in paths.items():
                    capture = cv2.VideoCapture(str(path))
                    if not capture.isOpened():
                        raise ValueError(f"无法打开视频：{path}")
                    self.captures[name] = capture
                    self.frame_counts[name] = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                self.source_frame_count = self.frame_counts["original"]
                self.fps = float(self.captures["original"].get(cv2.CAP_PROP_FPS)) or 30.0
                if self.frame_counts["va"] != len(va_indices):
                    raise ValueError("VA 视频帧数与 selection manifest 不一致")
                if self.frame_counts["vf"] != len(vf_indices):
                    raise ValueError("VF 视频帧数与 selection manifest 不一致")
                if va_indices[-1] >= self.source_frame_count:
                    raise ValueError("VA source_frame_index 超出原视频范围")
                if vf_indices[-1] >= self.source_frame_count:
                    raise ValueError("VF source_frame_index 超出原视频范围")
                self.source_indices = {"va": va_indices, "vf": vf_indices}
                self.tick = 0
                self.timeline.configure(to=max(0, self.source_frame_count - 1))
                self.timeline_var.set(0)
                self.last_frame_indices.clear()
                self.status_var.set(
                    f"Episode {episode:06d} · {camera} · "
                    f"原始 {self.source_frame_count} / VA {len(va_indices)} / "
                    f"VF {len(vf_indices)} 帧"
                )
                self.root.update_idletasks()
                self.render_current()
                self._load_action_data()
            except Exception as error:
                self._release_captures()
                self._show_error(error)

        def _playback_mode(self) -> str:
            return "source" if self.mode_var.get() == "源帧对齐" else "native"

        def _read_frame(self, name: str, frame_index: int):
            capture = self.captures[name]
            previous = self.last_frame_indices.get(name)
            if previous == frame_index and name in self.photos:
                return None
            if previous is None or frame_index != previous + 1:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                raise ValueError(f"读取{name}视频第 {frame_index} 帧失败")
            self.last_frame_indices[name] = frame_index
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            width = max(320, self.image_labels[name].winfo_width() - 8)
            height = max(320, self.image_labels[name].winfo_height() - 8)
            display_size = fit_image_size(image.size, (width, height))
            image = image.resize(display_size, Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(image=image)

        def render_current(self) -> None:
            if not self.captures or not self.source_indices:
                return
            try:
                frames = map_playback_frames(
                    self._playback_mode(),
                    tick=self.tick,
                    source_frame_count=self.source_frame_count,
                    va_source_indices=self.source_indices["va"],
                    vf_source_indices=self.source_indices["vf"],
                )
                frame_indices = {
                    "original": frames.original_index,
                    "va": frames.va_index,
                    "vf": frames.vf_index,
                }
                source_frames = {
                    "original": frames.original_source,
                    "va": frames.va_source,
                    "vf": frames.vf_source,
                }
                for name in ("original", "va", "vf"):
                    photo = self._read_frame(name, frame_indices[name])
                    if photo is not None:
                        self.photos[name] = photo
                        self.image_labels[name].configure(image=photo, text="")
                    self.frame_text_vars[name].set(
                        f"视频帧 {frame_indices[name]}/{self.frame_counts[name] - 1}  ·  "
                        f"源帧 {source_frames[name]}"
                    )
                self.seeking = True
                self.timeline_var.set(self.tick)
                self.seeking = False
            except Exception as error:
                self.pause()
                self._show_error(error)

        def _on_seek(self, value: str) -> None:
            if self.seeking or not self.captures:
                return
            self.pause()
            self.tick = max(0, min(int(round(float(value))), self.source_frame_count - 1))
            self.render_current()

        def step(self, delta: int) -> None:
            if not self.captures:
                return
            self.pause()
            self.tick = max(0, min(self.tick + delta, self.source_frame_count - 1))
            self.render_current()

        def toggle_play(self) -> None:
            if not self.captures:
                return
            if self.playing:
                self.pause()
            else:
                if self.tick >= self.source_frame_count - 1:
                    self.tick = 0
                    self.last_frame_indices.clear()
                self.playing = True
                self.play_button.configure(text="暂停")
                self._schedule_next_frame()

        def _schedule_next_frame(self) -> None:
            if not self.playing:
                return
            if self.tick >= self.source_frame_count - 1:
                self.pause()
                return
            self.tick += 1
            self.render_current()
            delay_ms = max(1, round(1000.0 / self.fps))
            self.after_id = self.root.after(delay_ms, self._schedule_next_frame)

        def pause(self) -> None:
            self.playing = False
            self.play_button.configure(text="播放")
            if self.after_id is not None:
                self.root.after_cancel(self.after_id)
                self.after_id = None

        def _release_captures(self) -> None:
            for capture in self.captures.values():
                capture.release()
            self.captures.clear()
            self.frame_counts.clear()
            self.source_indices.clear()
            self.last_frame_indices.clear()
            self.photos.clear()

        def _show_error(self, error: Exception) -> None:
            self.status_var.set(str(error))
            messagebox.showerror("无法加载 Episode", str(error), parent=self.root)

        def close(self) -> None:
            self.pause()
            if self.resize_after_id is not None:
                self.root.after_cancel(self.resize_after_id)
                self.resize_after_id = None
            self._release_captures()
            self.root.destroy()

        def run(self) -> None:
            self.root.mainloop()

    ComparisonWindow().run()


if __name__ == "__main__":
    raise SystemExit(main())
