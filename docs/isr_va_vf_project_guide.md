# ISR–VA/VF 离线轨迹加速项目指南

本文说明项目用途、核心技术实现，以及从环境安装、LeRobot 2.1 数据转换到可视化检查的完整流程。所有转换均为离线操作，不修改原数据集，也不需要修改 PI0.5 模型结构。

## 1. 项目内容

本项目为机器人模仿学习数据提供两种轨迹抽帧加速消融版本：

- **ISR–VA（Velocity–Acceleration）**：根据末端执行器的速度和加速度分配采样密度，减少低运动阶段的冗余帧。
- **ISR–VF（Velocity–Force）**：根据末端执行器速度和触觉力事件分配采样密度，重点保留接触、插入和受力突变阶段。

输入为带力信号的 LeRobot 2.1 数据集。转换器分别生成新的 VA、VF 数据集，并同步处理 Parquet 中的 state、action、force、索引等字段和所有视频流。输出数据仍保持 LeRobot 2.1 目录结构，可直接用于重新计算 PI0.5 normalization statistics 和后续训练。

仓库中的主要模块如下：

```text
src/isr/pose.py                    # EEF 位姿与 SE(3) 运动学
src/isr/trajectory_acceleration.py # VA/VF 信息分数与选帧算法
src/isr/lerobot_v21.py             # LeRobot 2.1 数据集重写
src/isr/action_compare.py           # Action 曲线读取与时间轴构造
src/isr/cli/accelerate_lerobot.py   # isr-accelerate-lerobot 命令
src/isr/cli/compare_videos.py       # isr-compare-videos 可视化窗口
tests/                              # 单元测试
```

## 2. 技术细节

### 2.1 EEF pose 与运动学

每只机械臂的 `observation.state.eef_pose` 占 10 维：

```text
[x, y, z, r1, r2, r3, r4, r5, r6, gripper]
```

单臂输入形状为 `[T, 10]`，双臂输入形状为 `[T, 20]`，更多机械臂也可以按每臂 10 维继续拼接。例如双臂布局为：

```text
[left_x, ..., left_gripper, right_x, ..., right_gripper]
```

程序根据最后一维除以 10 自动推断机械臂数量，无需增加 CLI 参数。其中每臂的 6 维旋转量都是旋转矩阵的前两列，程序分别通过 Gram–Schmidt 正交化恢复完整旋转矩阵。每只手独立计算平移速度和相对旋转角速度，再将所有手臂的速度向量拼接后取联合 L2 范数；因此任意一只手运动或两只手同时运动都会提高采样优先级。线加速度和角加速度由联合速度差分得到。所有量均使用数据集 `meta/info.json` 中的 FPS。

### 2.2 VA 与 VF 优先级

每个信号先使用 10%/90% 分位数稳健归一化到 `[0, 1]`。

- VA 分数：`0.5 × speed_score + 0.5 × acceleration_score`。速度和加速度分数内部都等权组合线性量与角运动量。
- VF 分数：`0.5 × speed_score + 0.5 × force_event_score`。程序读取所有 `observation.state.*tactile*` 字段的第 4 个分量（索引 3），以前 `--free-contact-seconds` 秒估计零点；随后执行 EMA 滤波、时间差分和 3σ 噪声抑制，并取各传感器中的最大力事件响应。

VF 中位于力事件 90% 分位数以上的局部峰值及其相邻帧会被强制保留。两种模式都会强制保留首尾帧，以及每只夹爪变化前后的两帧；双臂情况下，第 9、19 维中任意一只夹爪变化都会触发保留，避免开合动作被抽掉。

### 2.3 选帧与加速语义

`--target-retention 0.5` 表示期望保留约 50% 的帧，而非保证精确保留一半。强制帧和 `--max-skip` 会影响最终数量；`--max-skip 4` 保证相邻输出帧的源索引差不超过 4，即中间最多丢弃 3 帧。选帧器通过有界动态规划寻找信息间隔较均匀的路径，并通过二分搜索接近期望帧数。

选中索引后，转换器会：

1. 对 Parquet 的所有列使用相同源行索引，包括 state、action、gripper 和 force；Action 只取原数据中的对应行，不使用 state 重新生成。
2. 对所有相机视频使用相同索引抽帧，并由 FFmpeg 按原 FPS 重新编码。
3. 将 `timestamp`、episode 内 `frame_index` 和全局 `index` 重建为连续值。
4. 添加 `source_frame_index`，并在 `meta/selection_manifest.jsonl` 记录每个 episode 的完整源索引映射。
5. 更新 episode 长度、数值统计和 `meta/info.json` 中的加速信息。

因此，加速并不是简单地重新排列轨迹：例如 945 帧抽为 472 帧后，输出仍以相同 FPS 播放和训练，序列时长约减半，从而得到约 2 倍时间压缩。该流程只改变离线数据，不改变 PI0.5 网络结构；VA 与 VF 数据集应分别重新计算 normalization statistics。

### 2.4 视频抽帧后的 Action 对齐

视频和 Action 不是分别抽帧后再做二次匹配。每个 episode 只计算一次严格递增的源索引序列：

```text
S = [s0, s1, ..., s(K-1)]
```

随后对所有模态同时使用这一个序列。对任意输出帧 `k`：

```text
output_video[camera][k] = source_video[camera][S[k]]
output_action[k]        = source_action[S[k]]
output_state[k]         = source_state[S[k]]
output_force[k]         = source_force[S[k]]
```

因此，输出视频第 `k` 帧、Action 第 `k` 行、state 第 `k` 行和 force 第 `k` 行始终来自同一个原始时刻 `S[k]`。原数据中 state 与 Action 已有的相位关系会原样保留，不会发生以下操作：

- 不对 Action 插值或平滑；
- 不将 Action 向前或向后平移一帧；
- 不用抽帧后的 state 重新生成或覆盖 Action；
- 不为不同相机、state 和 Action 使用不同的选帧索引。

压缩后，`timestamp[k]` 被重建为 `k / fps`，`frame_index[k]` 重建为 `k`，所以 K 帧序列会按原 FPS 更快结束。与此同时，`source_frame_index[k]` 和 `meta/selection_manifest.jsonl` 保存 `S[k]`，可以随时追溯到原始时间。也就是说，“加速后时间”描述训练/播放的新时间轴，“原始源时间”只用于检查选中了原轨迹中的哪些时刻。

当 PI0.5 数据加载器按连续输出行构造 action chunk 时，压缩后的 chunk 对应：

```text
[source_action[S[k]], source_action[S[k+1]], ..., source_action[S[k+H-1]]]
```

相邻项在原轨迹中可能相隔多个源帧，但在新数据集中仍相隔 `1/fps` 秒；一个长度相同的 action chunk 因而覆盖更长的原始运动进度，这正是离线数据加速生效的位置。当前实现不需要在 PI0.5 内增加额外的 state/action 对齐代码。

### 2.5 可视化含义

窗口包含两个子页面：

- **视频对比**：并排显示原始、VA 和 VF。`源帧对齐`按源索引比较保留内容；`真实时间`按三个视频各自的新帧序号播放，用于观察压缩后的执行速度。
- **Action 对比**：直接读取三个 Parquet 中的 `actions.eef_pose` 或 `actions.joint_position`。`加速后时间`使用各序列自己的 `arange(T)/fps`，显示真实压缩时长；`原始源时间`使用 selection manifest 将 VA/VF Action 放回原始时间位置，显示保留了哪些动作点。

Action 页面不限制为 10 维；双臂 `actions.eef_pose` 为 20 维时，会按照 `meta/info.json` 中的 feature names 展示维度，缺少 names 时则显示 `dim_0` 至 `dim_19`。

## 3. 如何启动

以下命令均在仓库根目录 `<your-path>/ISR` 执行。

### 3.1 安装系统与 Python 依赖

需要 Python 3.9+、Tkinter、FFmpeg 和 FFprobe。Ubuntu/Debian 可执行：

```bash
sudo apt update
sudo apt install python3-venv python3-tk ffmpeg
```

创建独立环境并安装项目：

```bash
cd /home/jiongwei/projects/ISR
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

确认命令可用：

```bash
isr-accelerate-lerobot --help
isr-compare-videos --help
ffmpeg -version
ffprobe -version
```

### 3.2 转换数据集

原始数据需要为lerobot2.1：

```text
/home/jiongwei/dataset/force/plug_the_cord_into_outlet_0814_0820_short_355eposides
```

如果下面两个 `_isr_va`、`_isr_vf` 输出目录已经存在，说明对应转换已经完成，应跳过本节并直接启动可视化。转换器不会覆盖已有输出，也不要为了重跑而删除原始数据集。

生成 VA 数据集：

```bash
isr-accelerate-lerobot \
  --input /home/jiongwei/dataset/force/plug_the_cord_into_outlet_0814_0820_short_355eposides \
  --output /home/jiongwei/dataset/force/plug_the_cord_into_outlet_0814_0820_short_355eposides_isr_va \
  --mode va \
  --target-retention 0.5 \
  --max-skip 4
```

生成 VF 数据集：

```bash
isr-accelerate-lerobot \
  --input /home/jiongwei/dataset/force/plug_the_cord_into_outlet_0814_0820_short_355eposides \
  --output /home/jiongwei/dataset/force/plug_the_cord_into_outlet_0814_0820_short_355eposides_isr_vf \
  --mode vf \
  --target-retention 0.5 \
  --max-skip 4 \
  --free-contact-seconds 1.0
```

输出目录必须尚不存在，且不能位于输入目录内部。程序先写入临时 staging 目录，全部 episode、视频和元数据成功后才将其重命名为最终输出；失败时不会修改输入数据。转换会重新编码全部视频，因此耗时主要取决于 episode 数量和视频流数量。

转换完成后可检查：

```bash
python -m pytest -q
head -n 1 /home/jiongwei/dataset/force/plug_the_cord_into_outlet_0814_0820_short_355eposides_isr_va/meta/selection_manifest.jsonl
head -n 1 /home/jiongwei/dataset/force/plug_the_cord_into_outlet_0814_0820_short_355eposides_isr_vf/meta/selection_manifest.jsonl
```

### 3.3 启动可视化窗口

直接指定三份数据和初始 episode：

```bash
isr-compare-videos \
  --original-root /home/jiongwei/dataset/force/plug_the_cord_into_outlet_0814_0820_short_355eposides \
  --va-root /home/jiongwei/dataset/force/plug_the_cord_into_outlet_0814_0820_short_355eposides_isr_va \
  --vf-root /home/jiongwei/dataset/force/plug_the_cord_into_outlet_0814_0820_short_355eposides_isr_vf \
  --episode 0 \
  --camera observation.images.third_view \
  --ui-scale 2.4
```

也可以先启动空窗口，再拖入任一数据集目录、episode Parquet 或 MP4：

```bash
isr-compare-videos
```

窗口会根据同级目录的 `_isr_va` 和 `_isr_vf` 后缀自动寻找另外两份数据。视频页面可切换相机、播放模式和 episode；Action 页面可切换字段、维度与时间轴，并使用 Matplotlib 工具栏进行缩放、平移或保存图片。

若命令未找到，请确认虚拟环境已激活并重新执行 `python -m pip install -e .`。若无法打开窗口，检查当前会话是否具有图形桌面和有效的 `DISPLAY`；纯 SSH 会话通常需要 X11 转发或远程桌面。
