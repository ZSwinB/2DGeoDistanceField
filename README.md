# 2DGeoDistanceField
## 2D几何->距离场快速仿真平台

本项目面向无线通信中**传播数据生成**与**快速仿真**的需求，主要针对以下类型的数据：
- 2D
- 传播时间（Time of Arrival, ToA）
- 距离场（distance map）
- 主传播路径相关信息

这些数据通常用于：

- 神经网络训练（如无线电地图、传播预测）
- 数据集构建
- 定位任务

---

---

### 1. 大规模数据需求

随着神经网络在无线通信中的应用越来越多，很多任务需要：

- 大规模数据
- 高分辨率空间采样
- 快速生成

然而传统仿真方法：

- 计算精度高
- 但速度较慢
- 难以支持大规模数据生成

---

### 2. 本项目的做法

本系统不再追求完整的信道信息，而是采用一种更高效的方式：

```
只计算每个点的传播时间
```

这样可以：

- 大幅提升速度
- 支持大规模数据生成
- 满足大多数数据驱动任务需求

---

### 3. 主路径近似（重要）

在很多应用中：

- 最早到达的路径
- 或最强的传播路径

往往已经包含了主要信息

因此本系统在设计时：

- 重点计算主要传播路径
- 并通过引入一个简单的“偏置（bias）”机制：

```
对绕射等较弱路径进行一定程度的削弱
```

从而实现：

```
更接近“最强到达路径”的结果
```

---

补充说明：

```
当 DIFF_BIAS = 0 时，系统会退化为标准的最早到达时间（ToA）计算
```

---

### 4. 总结

本项目的核心目标是：

- 用更简单的计算
- 换取更高的速度
- 支持大规模数据生成

适用于：

- 数据集生成
- 神经网络训练
- 快速仿真场景

---

系统将问题拆分为三部分：

- **Geometry（几何）**：静态信息  
- **Precompute（预计算）**：一次生成，可重复复用  
- **Simulation（仿真）**：针对具体发射机执行  

### 核心优势

- 将复杂几何关系提前处理为中间表示  
- 避免重复几何建模  
- 多发射机共享预计算结果  
- 提升大规模仿真效率
- 仅仿真时间，加快仿真速度  

---

# 当前项目定位

本项目不追求完整传统射线追踪结果，而聚焦于**传播时间大规模数据集的计算**。

### 适用场景

- 大规模快速仿真  
- 时延 / 距离估计  
- 数据集生成  
- 神经网络训练数据生产  
- 多发射机场景评估  

在多数工程和数据驱动任务中，传播时间信息是很多任务。


## 系统能力（Capabilities）

---

### ✅ 多场景叠加（Multi-Scene Composition）

系统支持多个几何场景的叠加，例如：

```
建筑 + 车辆
建筑 + 障碍物
多来源数据融合
```

通过在 `GEO_ROOT` 中配置多个路径：

```python
GEO_ROOT = [
    r"building",
    r"vehicle"
]
```

系统会自动完成融合：

```
logical OR 合并 → 单一仿真场景
```

---

### 🔌 模块化与热插拔（Modular & Scalable）

系统采用分阶段流水线设计：

```
输入 → 几何 → 预计算 → 转换 → 仿真
```

特点：

- 每个阶段**独立运行**
- 每个阶段**都有输出结果**
- 可单独执行某一阶段
- 可替换任意输入或中间数据

支持：

```
灵活组合 / 分模块处理 / 可复用
```

---

### ⚙️ 可定制传播模型（Bias Control）

系统支持通过参数调节传播行为：

```python
DIFF_BIAS
FALLBACK_BIAS
```

可实现：

- 标准 ToA（最早到达时间）：
  ```
  DIFF_BIAS = 0
  ```

- 主路径近似（DPM）：
  ```
  DIFF_BIAS > 0
  ```

👉 可在不同任务之间灵活切换：

```
ToA ↔ DPM
```

---

### ⚡ 高性能仿真（Fast Simulation）

在典型设置下：

#### ① 中间数据生成

```
约 1 分钟 / 场景
```

特点：

- 结果可保存（落盘）
- 同一场景可重复使用（无需重复计算）

---

#### ② 仿真阶段

测试条件：

```
256 × 256 网格
单发射机 → 全空间接收机
```

性能：

```
约 10 秒完成
```

---



## 项目目录结构

```text
project/
│
├── config.py                # 全局配置（路径 / 开关 / 并行）
├── README.md                # 项目说明文档
├── run_all.py               # 一键运行（预处理 + 仿真）
├── run_intermdata.py        # 仅生成中间数据（预计算阶段）
│
├── pipeline/
│   ├── stage1_input/
│   │   └── load_geo.py      # 加载场景数据（geometry / antenna / gain）
│   │
│   ├── stage2_wall/
│   │   ├── run.py           # 从 geometry 构建墙段（当前默认实现）
│   │   └── degreewall.py    # 历史版本方法（保留，非默认）
│   │
│   ├── stage3_middata/
│   │   ├── build_pvw.py     # PVW：点到可见墙
│   │   ├── build_pvp.py     # PVP：点到可见端点
│   │   └── build_wwv.py     # WWV：墙到墙可见性
│   │
│   ├── stage4_convert/
│   │   └── run.py           # 中间数据转换，适配底层高性能计算
│   │
│   └── stage5_sim/
│       ├── solver.py        # 单发射机传播计算
│       └── batch_run.py     # 多发射机批量仿真调度
│
└── utils/
    └── ...                  # 工具脚本（可视化 / 检查 / 数据处理等）


```
# 项目整体结构

本项目采用分阶段流水线结构：

```
输入 → 几何处理 → 预计算 → 数据转换 → 仿真
```

---

# 使用说明

## 1. 你需要准备什么数据

本系统需要三类输入数据，对应 config 中的三个目录：

- 几何数据（GEO_ROOT）
- 天线数据（ANT_ROOT）
- 增益数据（GAIN_ROOT）

---

## 2. 数据格式与命名规则

### 2.1 几何数据（必须）

首先，你需要在项目中找到配置文件：

```
pipeline/config.py
```

在该文件中找到如下字段：

```python
GEO_ROOT = [
    r"你的几何数据路径"
]
```

👉 你需要把你的几何数据路径填到这里。

---

### 数据文件放置方式

在你填写的 `GEO_ROOT` 路径中，放入几何数据文件，例如：

```
0.npy
1.npy
...
```

或：

```
0.png
1.png
...
```

其中：

- 文件名 `0 / 1 / ...` 表示 **场景编号（scene_id）**
- 系统会根据 `config.IDX` 读取对应编号的数据

---

### 数据格式要求

- 数据必须是 2D（H, W）
- 0 表示 free（可通行区域）
- 非 0 表示障碍（墙体/建筑/物体）

---

### 多源几何融合（重要特性）

你可以在 `GEO_ROOT` 中配置多个路径：

```python
GEO_ROOT = [
    r"建筑路径",
    r"车辆路径（或者另外一个建筑）"
]
```

系统会自动执行：

```
geo = logical_or(geo1, geo2, ...)
```

👉 即多个几何会合并成一个场景

适用于：

- 城市 + 车辆
- 建筑 + 动态物体
- 多数据源融合

---

### 关于输入图像（重要）

如果你的输入不是标准二值图，建议先进行预处理：

推荐方法：

- Canny 边缘检测（推荐）
- 阈值二值化（threshold）
- 简单去噪（可选）

例如流程：

```
原图 → Canny → 二值图 → 保存为 npy/png
```

原因：

- 本系统依赖清晰的几何边界
- 非二值图会导致：
  - 墙体提取错误
  - 可见性计算异常

---

### 推荐实践

✔ 输入应为干净的二值图  
✔ 边界清晰、连续  
✔ 避免噪点和断裂  

---
```

### 2.2 天线数据（可选）

首先，在配置文件中找到：

```
pipeline/config.py
```

对应字段：

```python
ANT_ROOT = r"你的天线数据路径"
```

👉 将你的天线数据路径填写在这里。

---

### 数据文件命名

在该路径下放置：

```
{scene_id}_{tx_id}.npy
```

例如：

```
0_0.npy
0_1.npy
```

含义：

- `scene_id`：场景编号（对应几何数据）
- `tx_id`：发射机编号

---

### 数据作用（重要说明）

👉 天线数据 **仅用于仿真阶段（Stage 5）**

如果你只需要：

- 墙体数据（Stage 2）
- 中间数据（PVW / PVP / WWV）

那么：

```
不需要提供天线数据
```

---

### 如果你没有天线数据（推荐做法）

可以使用项目提供的工具自动生成：

```
utils/antenna.py
```

该工具可以：

- 在几何场景中自动生成天线位置
- 保证生成在 **合法 free 区域（geo == 0）**
- 支持：
  - 指定生成数量
  - 指定生成区域（子区域限制）

👉 推荐流程：

```
geo → 自动生成天线 → 保存为 {scene_id}_{tx_id}.npy
```


### 2.3 增益数据（可选）

放在 `GAIN_ROOT` 中：

```
{scene_id}_{tx_id}.npy
```
只是为了仿真加速，也就是说，明确不可达的区域，就不算了。不加也可以
---



## 3. 修改配置（pipeline/config.py）

首先打开配置文件：

```
pipeline/config.py
```

该文件是整个系统的**统一控制中心**，包括：

- 数据路径
- 运行阶段
- 并行方式
- 场景选择
- 算法参数

---

### 3.1 输入 / 输出路径

```python
GEO_ROOT = [...]
ANT_ROOT = ...
GAIN_ROOT = ...
OUTPUT_ROOT = ...
```

说明：

- `GEO_ROOT`：几何数据路径（支持多个，会自动融合）
- `ANT_ROOT`：天线数据路径（仅仿真需要）
- `GAIN_ROOT`：增益数据路径（可选）
- `OUTPUT_ROOT`：所有输出的根目录

---

### 3.2 阶段开关（控制流水线）

```python
GEN_WALL = True
GEN_PVW  = True
GEN_WWV  = True
GEN_PVP  = True
GEN_CONVERT = True
```

说明：

- `GEN_WALL`：生成墙体（Stage 2）
- `GEN_PVW`：生成点→墙可见性
- `GEN_WWV`：生成墙→墙可见性
- `GEN_PVP`：生成点→端点可见性
- `GEN_CONVERT`：生成计算用数据结构（Stage 4）

👉 用法示例：

- 只要墙：
  ```python
  GEN_WALL = True
  其他全 False
  ```

- 只生成中间数据：
  ```python
  GEN_WALL = True
  GEN_PVW = GEN_WWV = GEN_PVP = True
  GEN_CONVERT = False
  ```

---

### 3.3 场景 / 发射机选择

```python
IDX_LIST = [0]
TX_LIST  = [0,1,2]
```

说明：

- `IDX_LIST`：要处理的场景编号（对应 0.npy / 1.npy）
- `TX_LIST`：每个场景的发射机编号

👉 系统会自动组合：

```
(scene, tx) → 仿真任务
```

---

### 3.4 多进程配置（重要）

```python
USE_MULTIPROCESS = False
NUM_WORKERS = None

USE_INNER_MP = True
INNER_WORKERS = 8
```

说明：

#### 外层并行（scene级）
- `USE_MULTIPROCESS`
- 多个场景同时跑

#### 内层并行（单场景）
- `USE_INNER_MP`
- 一个场景内部加速

---

### 使用建议

- 默认推荐：

```python
USE_MULTIPROCESS = False
USE_INNER_MP = True
```

- 场景很多时：

```python
USE_MULTIPROCESS = True
USE_INNER_MP = False
```

❗ 不建议：

```
两个同时全开 → CPU 过载（性能反而下降）
```

---

### 3.5 输出目录结构（了解即可）

```python
WALL_DIR     = "wall"
MIDDATA_DIR  = "middata"
PVW_DIR      = "middata/pvw"
WWV_DIR      = "middata/wwv"
PVP_DIR      = "middata/pvp"
CONVERT_DIR  = "convert"
SIM_DIR      = "sim"
```

👉 一般不需要修改  
👉 这些目录会自动在 `OUTPUT_ROOT` 下生成  

---

### 3.6 算法参数（重要）

```python
K = 1
DIFF_BIAS = 300.0
FALLBACK_BIAS = 600.0
```

---

### 参数说明

#### K

- 每个接收点保留的路径数量（top-K）
- `K = 1` → 最短路径

说明：

- 从理论上讲，该问题对应 **Hamilton–Jacobi 方程的粘性解**
- 粘性解本质上只保留“最优路径”

👉 因此：

```
K > 1 的物理意义有限
```

- 多出来的路径更多是：
  - 工程近似
  - 多径分析辅助

👉 当前保留 K 的原因：

- 保持接口扩展性
- 支持后续实验 / 数据驱动方法

---

#### DIFF_BIAS（绕射惩罚）

```python
DIFF_BIAS = 300.0
```

作用：

- 给绕射路径增加额外代价
- 控制路径排序时：

```
绕射路径 < 反射路径 < 直达路径（通常）
```

本质：

```
heuristic penalty（非严格物理量）
```

---

#### FALLBACK_BIAS（兜底上界）

```python
FALLBACK_BIAS = 600.0
```

作用：

- 用于初始化路径的“上界候选”
- 保证算法始终有一个可比较的初始解

例如：

```
distance ≈ 欧氏距离 + FALLBACK_BIAS
```

本质：

- 算法上的 **baseline / upper bound**
- 不代表真实传播路径

---

### 参数关系（重要）

```
FALLBACK_BIAS > DIFF_BIAS
```

保证：

- 绕射路径仍然优于“兜底路径”
- 剪枝逻辑稳定

---

### 调整建议

- 如果绕射路径过多：

```
增大 DIFF_BIAS
```

- 如果剪枝不充分 / 性能较慢：

```
减小 FALLBACK_BIAS（适度）
```


---

### 3.7 典型最小配置

```python
GEO_ROOT = [r"你的数据路径"]
ANT_ROOT = r"你的天线路径"

IDX_LIST = [0]
TX_LIST  = [0]

GEN_WALL = True
GEN_PVW = GEN_WWV = GEN_PVP = True
GEN_CONVERT = True

USE_INNER_MP = True
```

---

## 4. 运行流程

推荐统一使用入口文件：

```
run_all.py
```

打开该文件：

```python
def main():
    run_precompute()
    run_simulation()
```

---

### 控制运行内容（通过注释）

- 只生成中间数据：

```python
def main():
    run_precompute()
    # run_simulation()
```

- 只运行仿真：

```python
def main():
    # run_precompute()
    run_simulation()
```

- 完整流程（默认）：

```python
def main():
    run_precompute()
    run_simulation()
```

---

运行：

```bash
python run_all.py
```

---

## 5. 输出结果在哪里 & 数据格式

所有结果统一保存在：

```
outputs/ （您可以再config.py 里面自定义）
```

---

### 5.1 中间数据（.npy）

```
outputs/middata/
    pvw/{idx}.npy
    wwv/{idx}.npy
    pvp/{idx}.npy
```

说明：

- `.npy`：numpy 原生格式
- 可直接用：

```python
np.load(path, allow_pickle=True)
```

---

### 5.2 转换数据（.npy）

```
outputs/convert/{idx}/
```

主要文件：

- `PVW_mask.npy` → (H, W, N_wall) bool
- `PVP_flat.npy` → (total,) int32
- `PVP_start.npy` → (H, W) int32
- `PVP_len.npy` → (H, W) int16
- `corner_x.npy` → (N_corner,) float32
- `corner_y.npy` → (N_corner,) float32
- `walls_nb.npy` → (N_wall, 4) float64

👉 这些是仿真阶段直接使用的数据结构

---

### 5.3 最终仿真结果（.npz）

```
outputs/sim/{idx}_{tx_id}.npz
```

这是一个压缩包（numpy zip），内部字段：

```
dist_map
```

---

### dist_map 含义

```
dist_map.shape = (H, W, K)
```

- H, W：空间网格
- K：路径数量（top-K）

含义：

```
dist_map[i, j, k]
= 第 (i,j) 点到发射机的第 k 条最短传播路径长度
```

包含路径类型：

- 直达（LOS）
- 反射（1阶 / 2阶）
- 绕射

---

### 如何读取结果

```python
data = np.load("outputs/sim/0_0.npz")
dist_map = data["dist_map"]
```

---

### 总结

- 中间数据 → `.npy`
- 仿真结果 → `.npz`
- 核心字段 → `dist_map`

---

```




## Project Overview

**2DGeoDistanceField** is a fast simulation platform for generating large-scale wireless propagation datasets in 2D environments, focusing on propagation time (Time of Arrival, ToA), distance fields, and dominant path information. It is designed for data-driven applications such as neural network training (e.g., radio map prediction), dataset construction, and localization tasks, where high-resolution sampling and massive data generation are required. Instead of computing full channel information as in traditional ray tracing (which is accurate but slow), this system only models propagation time, significantly improving efficiency while retaining the most relevant information for many applications.

The core idea is to approximate dominant propagation behavior by prioritizing the earliest or strongest paths. A simple bias mechanism is introduced to suppress weaker paths such as diffraction, enabling the system to approximate dominant path (DPM) behavior; when the bias is set to zero, it naturally falls back to standard ToA computation. The system is structured as a pipeline—Geometry → Precompute → Simulation—where expensive geometric relationships are computed once and reused across multiple transmitters, greatly improving scalability.

Key capabilities include multi-scene composition (e.g., combining buildings and vehicles via logical fusion), modular and plug-and-play pipeline design (each stage can run independently with reusable outputs), configurable propagation behavior (switching between ToA and DPM via bias parameters), and high performance. In practice, intermediate data for a scene can be generated in about one minute and reused, while full simulation on a 256×256 grid for a single transmitter over all receivers can be completed in around 10 seconds. This makes the system well-suited for large-scale simulation, dataset generation, and rapid evaluation scenarios.


# User Guide

## 1. What Data You Need to Prepare

This system requires three types of input data, corresponding to three directories defined in `config`:

- Geometry data (`GEO_ROOT`)
- Antenna data (`ANT_ROOT`)
- Gain data (`GAIN_ROOT`)

---

## 2. Data Format and Naming Rules

### 2.1 Geometry Data (Required)

First, locate the configuration file:

```
pipeline/config.py
```

Find the following field:

```python
GEO_ROOT = [
    r"your geometry data path"
]
```

👉 You need to set your geometry data path here.

---

### Data Placement

In the directory specified by `GEO_ROOT`, place geometry files such as:

```
0.npy
1.npy
...
```

or:

```
0.png
1.png
...
```

Where:

- File name `0 / 1 / ...` represents the **scene ID**
- The system will load data based on `config.IDX`

---

### Data Format Requirements

- Must be 2D array (H, W)
- 0 = free (walkable space)
- non-zero = obstacle (walls / buildings / objects)

---

### Multi-source Geometry Fusion (Important Feature)

You can configure multiple paths in `GEO_ROOT`:

```python
GEO_ROOT = [
    r"building path",
    r"vehicle path (or another geometry)"
]
```

The system will automatically perform:

```
geo = logical_or(geo1, geo2, ...)
```

👉 All geometries will be merged into a single scene

Applicable scenarios:

- city + vehicles
- buildings + dynamic objects
- multi-source fusion

---

### About Input Images (Important)

If your input is not binary, preprocessing is recommended:

Suggested methods:

- Canny edge detection (recommended)
- thresholding
- simple denoising (optional)

Example workflow:

```
raw image → Canny → binary → save as npy/png
```

Reason:

- The system relies on clear geometry boundaries
- Non-binary input may cause:
  - incorrect wall extraction
  - unstable visibility computation

---

### Best Practice

✔ Use clean binary input  
✔ Ensure continuous boundaries  
✔ Avoid noise and broken edges  

---

### 2.2 Antenna Data (Optional)

Locate in config:

```
pipeline/config.py
```

Field:

```python
ANT_ROOT = r"your antenna data path"
```

👉 Set your antenna data path here.

---

### File Naming

```
{scene_id}_{tx_id}.npy
```

Example:

```
0_0.npy
0_1.npy
```

Meaning:

- `scene_id`: scene index
- `tx_id`: transmitter index

---

### Purpose (Important)

👉 Antenna data is **only required for simulation (Stage 5)**

If you only need:

- wall extraction
- intermediate data (PVW / PVP / WWV)

Then:

```
antenna data is not required
```

---

### If You Don’t Have Antenna Data (Recommended)

Use the provided utility:

```
utils/antenna.py
```

This tool can:

- randomly generate antenna positions
- ensure positions are in valid free space (`geo == 0`)
- support:
  - number of antennas
  - region constraints

Recommended workflow:

```
geo → generate antennas → save as {scene_id}_{tx_id}.npy
```

---

### 2.3 Gain Data (Optional)

Place under `GAIN_ROOT`:

```
{scene_id}_{tx_id}.npy
```

This is only used for simulation acceleration, i.e., skipping clearly unreachable regions.  
It is optional.

---

## 3. Configuration (pipeline/config.py)

Open:

```
pipeline/config.py
```

This file is the **central control of the system**, including:

- data paths
- pipeline stages
- parallel settings
- scene selection
- algorithm parameters

---

### 3.1 Input / Output Paths

```python
GEO_ROOT = [...]
ANT_ROOT = ...
GAIN_ROOT = ...
OUTPUT_ROOT = ...
```

---

### 3.2 Pipeline Switches

```python
GEN_WALL = True
GEN_PVW  = True
GEN_WWV  = True
GEN_PVP  = True
GEN_CONVERT = True
```

---

### 3.3 Scene / Transmitter Selection

```python
IDX_LIST = [0]
TX_LIST  = [0,1,2]
```

---

### 3.4 Parallel Settings

```python
USE_MULTIPROCESS = False
NUM_WORKERS = None

USE_INNER_MP = True
INNER_WORKERS = 8
```

Recommended:

```python
USE_MULTIPROCESS = False
USE_INNER_MP = True
```

---

### 3.5 Output Structure (no need to modify)

```python
WALL_DIR     = "wall"
MIDDATA_DIR  = "middata"
PVW_DIR      = "middata/pvw"
WWV_DIR      = "middata/wwv"
PVP_DIR      = "middata/pvp"
CONVERT_DIR  = "convert"
SIM_DIR      = "sim"
```

---

### 3.6 Algorithm Parameters (Important)

```python
K = 1
DIFF_BIAS = 300.0
FALLBACK_BIAS = 600.0
```

---

### Parameter Description

#### K

- Number of paths retained per receiver (top-K)
- `K = 1` → shortest path only

Explanation:

- The underlying formulation corresponds to the **viscosity solution of the Hamilton–Jacobi equation**
- Such solutions inherently select the **optimal path**

👉 Therefore:

```
K > 1 has limited physical meaning
```

- Additional paths mainly serve:
  - engineering approximation
  - multipath analysis
  - data-driven applications

👉 K is kept for:

- interface consistency
- future extensibility

---

#### DIFF_BIAS (Diffraction Penalty)

```python
DIFF_BIAS = 300.0
```

Purpose:

- Adds extra cost to diffraction paths
- Controls path ranking:

```
diffraction < reflection < LOS (typically)
```

Nature:

```
heuristic penalty (not strictly physical)
```

---

#### FALLBACK_BIAS (Fallback Upper Bound)

```python
FALLBACK_BIAS = 600.0
```

Purpose:

- Provides an initial upper-bound candidate for path distance
- Ensures the solver always starts with a valid comparison baseline

Example:

```
distance ≈ Euclidean distance + FALLBACK_BIAS
```

Nature:

- algorithmic **baseline / upper bound**
- not a physical propagation path

---

### Parameter Relationship (Important)

```
FALLBACK_BIAS > DIFF_BIAS
```

Ensures:

- diffraction paths are still preferred over fallback candidates
- pruning logic remains stable

---

### Tuning Guidelines

- Too many diffraction paths:

```
increase DIFF_BIAS
```

- Weak pruning / slow performance:

```
slightly reduce FALLBACK_BIAS
```

---

### One-line Summary

```
K → number of retained paths (interface-level)
DIFF_BIAS → controls diffraction preference
FALLBACK_BIAS → provides initial upper bound
```

---

### 3.7 Minimal Example Config

```python
GEO_ROOT = [r"your data path"]
ANT_ROOT = r"your antenna path"

IDX_LIST = [0]
TX_LIST  = [0]

GEN_WALL = True
GEN_PVW = GEN_WWV = GEN_PVP = True
GEN_CONVERT = True

USE_INNER_MP = True
```

---

## 4. Run

Use:

```
run_all.py
```

Modify:

```python
def main():
    run_precompute()
    run_simulation()
```

---

### Control Execution

- only precompute:

```python
run_precompute()
# run_simulation()
```

- only simulation:

```python
# run_precompute()
run_simulation()
```

- full pipeline:

```python
run_precompute()
run_simulation()
```

Run:

```bash
python run_all.py
```

---

## 5. Output & Data Format

All outputs are stored in:

```
outputs/
```

---

### 5.1 Intermediate Data (.npy)

```
outputs/middata/
    pvw/{idx}.npy
    wwv/{idx}.npy
    pvp/{idx}.npy
```

Load:

```python
np.load(path, allow_pickle=True)
```

---

### 5.2 Converted Data (.npy)

```
outputs/convert/{idx}/
```

---

### 5.3 Final Simulation Output (.npz)

```
outputs/sim/{idx}_{tx_id}.npz
```

Contains:

```
dist_map
```

---

### dist_map Meaning

```
dist_map.shape = (H, W, K)
```

- H, W: grid
- K: number of paths

```
dist_map[i, j, k]
= k-th shortest path from (i,j) to TX
```

Includes:

- LOS
- reflection
- diffraction

---

### Load Result

```python
data = np.load("outputs/sim/0_0.npz")
dist_map = data["dist_map"]
```

---
```





## ⚠️ Known Issue: Reflection Point Visibility

### Summary

The current reflection implementation may generate **physically invalid reflection paths**.

This happens because the algorithm:

* Validates visibility at the **wall level**
* But does **not validate the actual reflection point on the wall**

---

### Problem Description

A reflection is currently accepted if:

* The TX image and RX form a line that intersects the wall (CCW test)
* Both TX and RX are marked as “visible” to that wall (via PVW / masks)

However, this does **not guarantee** that:

> The actual intersection point (reflection point) is visible from both TX and RX.

---

### Typical Failure Scenario

This issue commonly occurs in geometries like:

* Long walls partially inside structures
* Thin protruding wall segments
* Walls with partially occluded regions

Example:

* A wall is globally visible
* But the actual reflection point lies on a **hidden segment**
* The algorithm still accepts the reflection

---

### Root Cause

* Visibility is computed using **wall endpoints**
* Reflection happens at an **arbitrary point on the wall**
* No verification is performed at that exact point

---

### Impact

* Introduces **false reflection paths**
* Affects both:

  * First-order reflections
  * Second-order reflections

---

### Current Status

* This issue is **known and accepted** in the current version
* No fix is applied yet to preserve performance

---

### Planned Fix (Next Version)

For each reflection candidate:

1. Compute the **reflection point** (line-wall intersection)
2. Validate visibility:

   * TX → reflection point
   * RX → reflection point
3. Accept the reflection **only if both are visible**

---

### Second-Order Reflection (Future Fix)

For paths:

```text
TX → P1 → P2 → RX
```

Required visibility checks:

* TX → P1
* P1 → P2
* P2 → RX

---

### Implementation Notes

Recommended pipeline:

1. Fast pruning (existing):

   * Masks (TX_mask / RX_mask)
   * Distance pruning
   * CCW intersection
2. Compute intersection point (only for valid candidates)
3. Apply `visible_fast` checks as final validation

---

### Trade-off

| Aspect     | Current Version | Next Version      |
| ---------- | --------------- | ----------------- |
| Speed      | ✅ Fast          | ⚠ Slightly slower |
| Accuracy   | ⚠ Approximate   | ✅ Correct         |
| Complexity | ✅ Low           | ⚠ Higher          |

---

### Notes

* Issue frequency depends on geometry complexity
* More likely in dense urban layouts or irregular structures

---

If needed, this can be extended into:

* A reproducible test case
* Benchmark comparisons before/after fix



---

## 🩹 Current Mitigation (Applied in This Version)

To reduce invalid reflections caused by thin protruding structures, a preprocessing step has been introduced on wall geometry:

### Spur Removal (8-neighborhood 2-core pruning)

Before extracting wall segments, we apply an iterative pruning process:

* Treat contour pixels as a graph (8-neighborhood connectivity)
* Iteratively remove pixels with degree ≤ 1
* Continue until convergence

---

### Effect

This removes:

* Thin wall protrusions (spikes)
* Small dangling structures
* Non-physical contour artifacts

As a result:

> Many invalid reflection paths (caused by these structures) are eliminated at the geometry level.

---

### Rationale

The original issue arises because:

* Reflection validation is performed at the wall level
* But thin protrusions introduce artificial wall segments
* These segments generate physically invalid reflection points

By removing such structures:

* The wall representation becomes more physically consistent
* Reflection errors are significantly reduced without modifying core logic

---

### Trade-offs

| Aspect      | Impact                                  |
| ----------- | --------------------------------------- |
| Performance | ✅ No impact on runtime reflection stage |
| Accuracy    | ⚠ Still approximate (not fully fixed)   |
| Geometry    | ⚠ May remove very thin valid structures |

---

### Limitations

* This does **not fully solve** the reflection point visibility issue
* It only reduces the frequency of problematic cases

---

### Future Work

Planned improvement (next version):

* Explicitly compute reflection points
* Validate visibility at the reflection point (TX → P and RX → P)

---

