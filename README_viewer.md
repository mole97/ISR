# ISR Episode Video Comparison Viewer

这是一个只读桌面窗口，用于并排比较同一个 LeRobot episode 的原始、ISR–VA 和 ISR–VF 视频。程序不会修改任何数据集。

## 安装

在仓库根目录执行：

```bash
cd /home/jiongwei/projects/ISR
pip install -r requirements.txt
pip install -e .
```

Tkinter 由操作系统提供。如果启动时报 `No module named tkinter`，Ubuntu/Debian 可安装：

```bash
sudo apt install python3-tk
```

## 启动

打开空窗口，然后拖入任一 episode 的 `.parquet`、`.mp4`，或拖入数据集目录：

```bash
isr-compare-videos
```

程序会按照以下同级目录命名自动寻找三份数据：

```text
source_lerobot_v21/
source_lerobot_v21_isr_va/
source_lerobot_v21_isr_vf/
```

也可以显式指定当前数据集：

```bash
isr-compare-videos \
  --original-root /home/jiongwei/dataset/force/plug_the_cord_into_outlet_0814_0820_short_355eposides \
  --va-root /home/jiongwei/dataset/force/plug_the_cord_into_outlet_0814_0820_short_355eposides_isr_va \
  --vf-root /home/jiongwei/dataset/force/plug_the_cord_into_outlet_0814_0820_short_355eposides_isr_vf \
  --episode 0 \
  --camera observation.images.third_view
```

默认界面字体为系统 Tk 字体的 2.4 倍。如果仍然偏小，可以提高界面缩放：

```bash
isr-compare-videos --ui-scale 3.0
```

使用 `--ui-scale 1.0` 可恢复 Tk 默认字号。该参数只影响文字和控件，不改变视频内容。

如果暂时不安装项目，也可从源码运行：

```bash
PYTHONPATH=src python -m isr.cli.compare_videos
```

## 窗口操作

窗口包含两个子页面：

- `视频对比`：并排播放原始、ISR–VA 和 ISR–VF 视频。
- `Action 对比`：从三份 episode Parquet 直接读取真实 `actions.*`，绘制选定维度的三条对比曲线。

Action 页面支持：

- `Action 字段`：切换 `actions.eef_pose` 或 `actions.joint_position`。
- `维度`：选择 `x/y/z/r1…gripper` 或 `j1…gripper`。
- `加速后时间`：每份数据使用自己的 `arange(T)/fps`，VA/VF 曲线横轴更短，可直接观察约 2× 的序列加速。
- `原始源时间`：用 `selection_manifest.jsonl` 把 VA/VF Action 放回原始时间位置，适合检查哪些动作点被保留。
- Matplotlib 工具栏：支持缩放、平移和保存曲线图片。

- `相机`：同步切换三路 `third_view` 或 `left_wrist_view`。
- `源帧对齐`：原视频使用当前源帧，VA/VF 根据 `selection_manifest.jsonl` 显示最近的保留帧，适合检查选帧差异。
- `真实时间`：三路都按各自视频帧序号以 30 FPS 前进；VA/VF 结束后停在末帧，可直接观察约 2× 的时间压缩。
- 空格：播放或暂停。
- 左右方向键：前进或后退一帧。
- 时间轴：跳转到指定帧。
- Episode 输入框：输入编号后按回车或点击“加载”。

每个窗格底部同时显示当前视频帧和它对应的原始源帧，便于判断 VA/VF 是否保留了关键接触阶段。

窗口默认最大化。输入视频为 224×224 时，画面会保持宽高比并放大到各自面板的可用区域；调整窗口尺寸后会自动重新缩放。该操作只改变显示尺寸，不会修改或重新编码源视频。

## 常见问题

- 拖放不可用：确认安装了 `tkinterdnd2`；仍可使用“打开 Episode”按钮。
- 找不到 VA/VF 数据：检查三个目录是否为同级目录并使用 `_isr_va`、`_isr_vf` 后缀，或通过命令行显式指定三条路径。
- 无法打开图形窗口：需要本地图形桌面或正确设置的 `$DISPLAY`；纯 SSH 终端默认无法显示 Tk 窗口。
- 视频帧数不一致：确认正在使用转换完整结束后的数据集，并检查 `meta/selection_manifest.jsonl` 是否存在。
- Action 页面提示缺少 PyArrow：执行 `python -m pip install 'pyarrow>=14'`，然后重新启动窗口。
