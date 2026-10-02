# 分析代码逐行导读

本文对应以下三个脚本：

- `project/analyze_diversity.py`
- `project/analyze_direction.py`
- `project/analyze_length.py`

三个脚本的核心代码几乎相同，区别主要是三个配置变量：

```python
EXPERIMENT_NAME = "exp_diversity_v1"
PROMPTS_FILE = ROOT / "project" / "prompts_diversity.txt"
```

```python
EXPERIMENT_NAME = "exp_direction_v1"
PROMPTS_FILE = ROOT / "project" / "prompts_direction.txt"
```

```python
EXPERIMENT_NAME = "exp_length_v1"
PROMPTS_FILE = ROOT / "project" / "prompts_length.txt"
```

运行方式：

```bat
conda activate motion-study
cd /d E:\motion-research\text-to-motion
python -m project.analyze_direction
python -m project.analyze_length
```

## 一、文件头和导入

对应原文件第 1--35 行。

```python
from __future__ import annotations
```

允许在类型标注中直接使用尚未定义或较新的类型写法，例如
`dict[str, str]`。它主要影响类型标注，不改变本实验的数学计算。

```python
import csv
import re
from itertools import combinations
from pathlib import Path
```

- `csv`：把实验结果写成 Excel 可以打开的 CSV 文件。
- `re`：用正则表达式解析文件名中的样本编号。
- `combinations`：枚举样本两两组合。例如 5 个样本会得到
  `C(5,2)=10` 个样本对。
- `Path`：跨 Windows/Linux 组织路径，避免手写反斜杠导致错误。

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

`Agg` 是无窗口后端。脚本在命令行运行时不需要弹出图形窗口，而是直接
把图保存为 PNG。这对远程服务器和批量实验很重要。

```python
import numpy as np
import torch
from scripts.motion_process import recover_from_ric
```

- `numpy`：读取 `.npy`，处理数组、速度、距离和统计量。
- `torch`：调用官方的 `recover_from_ric()`。
- `recover_from_ric`：把官方 263 维动作表示恢复成
  `22 个关节 × 3 个坐标`。

## 二、路径配置

对应原文件第 38--65 行。

```python
ROOT = Path(__file__).resolve().parents[1]
```

逐步含义：

1. `__file__` 是当前脚本路径。
2. `.resolve()` 转成绝对路径。
3. `.parents[1]` 向上两级，得到项目根目录。

因此无论从哪个当前目录启动，只要脚本位置不变，`ROOT` 都会指向：

```text
E:\motion-research\text-to-motion
```

```python
EXPERIMENT_NAME = "exp_direction_v1"
```

指定要分析哪一次推理实验。它必须和生成动作时的 `--ext` 参数一致。
例如生成时使用：

```bat
python gen_motion_script.py ... --ext study_v1
```

分析时就必须把 `EXPERIMENT_NAME` 写成 `study_v1`，否则脚本会去错误目录
寻找 `.npy` 文件。

```python
JOINTS_DIR = (
    ROOT
    / "eval_results"
    / "t2m"
    / "Comp_v6_KLD01"
    / EXPERIMENT_NAME
    / "joints"
)
```

`Path / "目录名"` 会逐级拼接路径。该目录保存官方生成器输出的
263 维 `.npy` 文件。这里的 `joints` 名称容易误导：在修正后的流程中，
文件内容实际上仍是归一化的 motion representation，不是最终的
`[T, 22, 3]` 关节坐标。

```python
PROMPTS_FILE = ROOT / "project" / "prompts_direction.txt"
```

读取实验输入文本。第 0 行对应 `C000`，第 1 行对应 `C001`，依此类推。
因此输入文件行数必须和生成动作时使用的文本行数一致。

```python
RESULTS_DIR = ROOT / "project" / "results" / EXPERIMENT_NAME
FIGURES_DIR = ROOT / "project" / "figures" / EXPERIMENT_NAME
```

分别保存表格结果和图片结果。把实验名放进路径可以避免不同实验互相覆盖。

```python
FPS = 20.0
NUM_JOINTS = 22
RESAMPLE_LENGTH = 60
```

- 官方动画帧率为 20 FPS，因此 `duration = frames / 20`。
- HumanML3D 使用 22 个关节。
- 不同样本长度不同，比较多条动作前先线性重采样到 60 个时间点。

```python
META_DIR = ROOT / "checkpoints" / "t2m" / "Comp_v6_KLD01" / "meta"
MEAN_PATH = META_DIR / "mean.npy"
STD_PATH = META_DIR / "std.npy"
```

读取官方训练集统计量。生成器保存的是：

```text
normalized = (original - mean) / std
```

恢复坐标前必须执行：

```text
original = normalized * std + mean
```

如果跳过这一步，263 维数据会被错误地当成真实运动特征，轨迹和多样性
数值就没有物理意义。

## 三、读取文本和解析样本编号

对应原文件第 69--91 行。

```python
def read_prompts(path: Path) -> dict[str, str]:
```

定义函数：输入文本文件路径，返回例如：

```python
{"C000": "a person walks forward",
 "C001": "a person walks backward"}
```

```python
if not path.exists():
    raise FileNotFoundError(...)
```

在真正计算前检查文件是否存在。主动报错比后面出现难懂的空结果更容易排查。

```python
lines = [line.strip() for line in path.read_text(...).splitlines()]
lines = [line for line in lines if line]
```

- `read_text` 读取整个文件。
- `splitlines` 按行拆分。
- `strip` 去掉首尾空格。
- 第二行删除空行。

```python
for i, text in enumerate(lines):
    prompts[f"C{i:03d}"] = text
```

`enumerate` 同时提供编号和文本。`f"C{i:03d}"` 把编号格式化为三位：
`0 -> C000`，`1 -> C001`。这个编号必须和官方生成器创建的目录一致。

```python
match = re.search(r"gen_motion_(\d+)_L\d+\.npy$", filename)
```

从文件名中解析样本编号。例如：

```text
gen_motion_04_L196.npy -> 4
```

其中：

- `\d+` 表示一个或多个数字；
- `L\d+` 匹配帧数；
- `$` 要求匹配到文件名结尾。

## 四、时间序列重采样

对应原文件第 94--126 行。

```python
def resample_sequence(sequence, target_length):
```

输入可以是：

```text
[T, 2]       根节点 XZ 轨迹
[T, 22, 3]   全身关节序列
```

输出的第一个维度统一变成 `target_length`。

```python
if sequence.ndim < 2:
    raise ValueError(...)
```

时间序列至少要有“时间维”和“特征维”。一维数组无法判断每个元素对应
哪个时间点，因此直接拒绝。

```python
src_length = sequence.shape[0]
```

取原始帧数 `T`。

```python
if src_length == target_length:
    return sequence.copy()
```

已经是目标长度时不插值，返回副本，避免修改原数组。

```python
if src_length < 2:
    return np.repeat(sequence, target_length, axis=0)
```

只有一帧时没有时间间隔，无法做线性插值，只能重复这一帧。

```python
old_t = np.linspace(0.0, 1.0, src_length)
new_t = np.linspace(0.0, 1.0, target_length)
```

把原时间轴和新时间轴都归一化到 `[0,1]`。这样比较的是相同动作进度，
而不是相同绝对帧号。

```python
flat = sequence.reshape(src_length, -1)
```

把除时间维以外的维度摊平：

```text
[T, 22, 3] -> [T, 66]
```

这样可以对每个坐标独立插值。

```python
out = np.empty((target_length, flat.shape[1]), dtype=np.float32)
```

预先分配输出数组，避免循环过程中反复扩容。

```python
for dim in range(flat.shape[1]):
    out[:, dim] = np.interp(new_t, old_t, flat[:, dim])
```

对每个坐标维度做一维线性插值。这里不是改变模型生成的原始动作，
只是为了让不同长度动作可以进行成对比较。

```python
return out.reshape((target_length,) + sequence.shape[1:])
```

把展平后的结果恢复成原来的特征结构。

## 五、从 263 维表示恢复关节坐标

对应原文件第 129--174 行。

```python
data = np.load(motion_path)
```

读取官方生成器保存的 NumPy 文件。预期形状是：

```text
[1, T, 263]
```

其中 `1` 是 batch 维，`T` 是帧数，`263` 是 HumanML3D motion feature
维度。

```python
if data.ndim != 3:
if data.shape[0] != 1:
if data.shape[2] != 263:
```

三组检查分别确认：

1. 数组是三维；
2. 当前文件包含一个样本；
3. 最后一维是官方预期的 263。

这属于“数据契约检查”：如果输入不符合假设，立刻报错，而不是输出
看似正常但实际错误的图。

```python
data = data * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
```

执行反标准化。`reshape(1,1,263)` 使 `mean/std` 能沿 batch 和时间维
广播到每一帧。

```python
tensor = torch.from_numpy(data).float()
```

把 NumPy 数组转为 PyTorch 浮点 Tensor，供官方 `recover_from_ric` 使用。

```python
with torch.no_grad():
    joints = recover_from_ric(tensor, NUM_JOINTS)
```

`no_grad` 表示这里只做推理，不需要保存反向传播计算图。
`recover_from_ric` 根据根节点旋转、根节点位移和相对关节坐标恢复：

```text
[1, T, 263] -> [1, T, 22, 3]
```

```python
joints = joints.detach().cpu().numpy()
```

脱离计算图，移回 CPU，再转换为 NumPy，方便使用 NumPy 和 Matplotlib。

```python
if joints.ndim == 4 and joints.shape[0] == 1:
    joints = joints[0]
```

去掉 batch 维：

```text
[1, T, 22, 3] -> [T, 22, 3]
```

后续每一帧的第 0 个关节就是根节点。

## 六、单个动作的统计量

对应原文件第 177--231 行。

```python
frames = joints.shape[0]
duration = frames / FPS
```

帧数除以 20 得到秒数。例如 196 帧对应 9.8 秒。

```python
root = joints[:, 0, :]
root_xz = root[:, [0, 2]]
```

取根节点在所有帧的三维坐标，再只保留水平面 X/Z。这里不把 Y 高度
纳入行走方向轨迹。

```python
root_steps = np.diff(root_xz, axis=0)
```

计算相邻帧根节点的位移向量：

```text
[T, 2] -> [T-1, 2]
```

```python
root_step_lengths = np.linalg.norm(root_steps, axis=1)
root_path_length = root_step_lengths.sum()
```

每一步取欧氏长度，再累加，得到实际走过的路径长度。动作绕弯时，
路径长度会大于起点到终点的直线距离。

```python
root_displacement = np.linalg.norm(root_xz[-1] - root_xz[0])
```

计算起点到终点的净位移。

```python
velocity = np.diff(joints, axis=0) * FPS
speed = np.linalg.norm(velocity, axis=-1)
```

用有限差分近似速度：

```text
v_t ≈ (p_t - p_{t-1}) / Δt
Δt = 1 / FPS
```

所以乘以 `FPS`。`speed` 对每个关节取三维速度的模长。

```python
acceleration = np.diff(velocity, axis=0) * FPS
```

再对速度做一次有限差分，得到加速度近似：

```text
a_t ≈ (v_t - v_{t-1}) / Δt
```

这里的 `mean_joint_acceleration` 是本项目的探索性平滑/剧烈程度指标，
不是论文官方指标，也不是经过真实单位标定的生物力学测量。

```python
straightness = root_displacement / root_path_length
```

直线运动接近 1；绕路、转弯或轨迹抖动时通常更低。路径长度为零时
返回 0，避免除零。

## 七、用于多样性比较的表示

对应原文件第 234--270 行。

```python
return joints - joints[:, 0:1, :]
```

减去每一帧根节点位置，把人体平移到根节点为原点。这一步去除了“人在
房间中走到哪里”的影响，保留身体姿态变化。

```python
root_xz = joints[:, 0, [0, 2]]
root_xz = root_xz - root_xz[0]
```

把根轨迹平移到初始点为 `(0,0)`，让不同样本可以比较形状而不是绝对位置。

```python
return resample_sequence(..., RESAMPLE_LENGTH)
```

把不同帧数的动作统一到 60 个时间点。

```python
diff = a - b
return float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))
```

计算两个序列的 RMS 距离：

1. 相减得到差异；
2. 平方；
3. 对最后坐标维求和；
4. 对所有时间和元素求平均；
5. 开平方。

`pose_diversity` 比较根中心化的身体姿态；
`trajectory_diversity` 比较根节点 XZ 轨迹。

## 八、主函数：加载数据

对应原文件第 273--350 行。

```python
if not MEAN_PATH.exists() or not STD_PATH.exists():
    raise FileNotFoundError(...)
```

确认反标准化所需文件存在。

```python
mean = np.load(MEAN_PATH).astype(np.float32)
std = np.load(STD_PATH).astype(np.float32)
```

加载并统一数据类型。

```python
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
```

递归创建输出目录；目录已经存在时不报错。

```python
prompts = read_prompts(PROMPTS_FILE)
```

获得文本编号到文本内容的映射。

```python
motions: dict[str, dict[int, dict]] = {}
sample_rows = []
```

- `motions` 保存恢复后的动作，用于后续两两比较和画轨迹。
- `sample_rows` 保存每个样本的一行统计结果，用于写 CSV。

```python
for text_id, prompt in prompts.items():
```

逐个处理文本条件，例如 `C000` 的 forward。

```python
files = sorted(
    text_dir.glob("*.npy"),
    key=lambda p: parse_sample_id(p.name),
)
```

找到该文本对应的所有动作文件，并按样本编号排序。

```python
for motion_path in files:
    sample_id = parse_sample_id(motion_path.name)
    joints = recover_motion_to_joints(motion_path, mean, std)
    metrics = compute_sample_metrics(joints)
```

对每个样本完成三步：

1. 得到样本编号；
2. 263 维表示反标准化并恢复关节坐标；
3. 计算帧数、时长、路径、速度、加速度。

```python
motions[text_id][sample_id] = {
    "path": motion_path,
    "joints": joints,
    "pose": normalized_pose(joints),
    "trajectory": normalized_root_trajectory(joints),
}
```

同时保存三种数据：

- 原文件路径；
- 原始恢复关节；
- 用于姿态多样性比较的归一化姿态；
- 用于轨迹多样性比较的归一化根轨迹。

```python
row = {..., **metrics}
sample_rows.append(row)
```

把文本编号、样本编号、提示词、文件路径和统计指标合并成一行。
`**metrics` 是字典展开语法。

## 九、写出逐样本 CSV

对应原文件第 353--379 行。

```python
sample_csv = RESULTS_DIR / "sample_metrics.csv"
```

指定输出文件。

```python
with sample_csv.open("w", newline="", encoding="utf-8-sig") as f:
```

以 UTF-8 with BOM 写文件。这样 Windows Excel 打开中文时更不容易乱码。

```python
writer = csv.DictWriter(f, fieldnames=sample_fields)
writer.writeheader()
writer.writerows(sample_rows)
```

使用字典字段名写表头，再写入所有样本行。

## 十、两两多样性和汇总

对应原文件第 382--459 行。

```python
for sample_a, sample_b in combinations(sample_ids, 2):
```

5 个动作生成 10 对比较，不比较样本与自身，也不重复比较 A-B 和 B-A。

```python
pose_d = rms_distance(...)
traj_d = rms_distance(...)
```

分别计算身体姿态和根轨迹的 RMS 距离。

```python
pairwise_rows.append({...})
```

保存每一对样本的详细结果，便于之后寻找异常样本。

```python
duration_mean_sec = float(np.mean(durations))
duration_std_sec = float(np.std(durations))
```

对同一句文本的多个采样计算平均时长和标准差。标准差大，说明长度采样
具有更强随机性；标准差小，说明长度更稳定。

```python
mean_pairwise_pose_diversity = float(np.mean(pose_distances))
```

把一个文本下的所有样本对压缩为一个平均值，方便不同文本之间画柱状图。

注意：这些指标是本项目自定义的探索性分析，不是论文中的官方
`FID / R-Precision / Matching Score`。

## 十一、保存 CSV 和绘图

对应原文件第 462--530 行。

脚本会写出：

```text
sample_metrics.csv
pairwise_diversity.csv
diversity_summary.csv
```

三者区别：

- `sample_metrics.csv`：每个动作样本一行；
- `pairwise_diversity.csv`：同一文本内每两个样本一行；
- `diversity_summary.csv`：每个文本条件一行汇总。

绘图部分的共同模式是：

```python
plt.figure(...)
plt.boxplot(...)  # 或 plt.bar / plt.plot
plt.xlabel(...)
plt.ylabel(...)
plt.title(...)
plt.tight_layout()
plt.savefig(...)
plt.close()
```

- `figure` 创建画布；
- `boxplot` 展示时长分布；
- `bar` 展示不同文本条件的平均多样性；
- `plot` 展示每个样本的根节点轨迹；
- `tight_layout` 防止标签被裁剪；
- `savefig` 保存 PNG；
- `close` 释放画布，避免批量绘图时内存不断增长。

## 十二、为什么 direction 和 length 只改两行配置

分析算法并没有改变，改变的是实验条件：

```text
同一分析器
        +
不同输入文本
        +
不同输出目录
        =
不同实验
```

因此：

- direction 实验回答“文本中的方向词是否影响根节点轨迹”；
- length 实验回答“文本复杂度和动作内容是否影响生成长度”；
- diversity 实验回答“同一句文本多次采样是否产生不同动作”。

这三组是基于官方推理结果设计的探索性实验，不应表述为论文已经报告的
新官方指标。

## 十三、推荐的阅读顺序

1. 先读 `project/analyze_direction.py` 的配置和 `main()`；
2. 再读 `recover_motion_to_joints()`，理解反标准化和 RIC 恢复；
3. 再读 `compute_sample_metrics()`，自己手算一个两帧速度；
4. 再读 `normalized_pose()` 和 `normalized_root_trajectory()`；
5. 最后读 CSV 和绘图部分；
6. 回到 `gen_motion_script.py`，理解 `.npy` 文件是怎样产生的。

建议每读完一个函数，都在命令行检查一次形状：

```python
print(data.shape)
print(joints.shape)
print(joints[0, 0])
```

真正需要掌握的不是把 550 行背下来，而是能够回答：

```text
数据从哪里来？
每一步的形状是什么？
为什么要反标准化？
为什么要恢复关节？
指标比较的对象是什么？
结果能支持什么结论，不能支持什么结论？
```
