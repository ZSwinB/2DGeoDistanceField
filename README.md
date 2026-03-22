# 2DGeoDistanceField
输入地理信息与发射天线位置，计算自由空间内的TOA场。 支持直射、反射与绕射路径，最多两次碰撞；可扩展输出多径距离与相位信息。
# 设计思想

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

在多数工程和数据驱动任务中，传播时间信息已足够支撑后续应用。

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
# 项目整体结构

本项目采用分阶段流水线结构：

```
输入 → 几何处理 → 预计算 → 数据转换 → 仿真
```

---

# 各阶段说明

## Stage 1：输入

`stage1_input/load_geo.py` 用于加载场景相关输入数据，主要包括：

- 几何结构
- 天线信息
- 增益信息

这一阶段负责将外部数据整理为后续流水线可直接使用的内部格式。

---

## Stage 2：墙体构建

- `stage2_wall/run.py`：从几何数据中构建墙段表示，供后续可见性计算使用  
- `stage2_wall/degreewall.py`：历史版本实现（保留用于参考，不是当前默认方法）

---

## Stage 3：中间数据预计算

该阶段为系统核心，用于构建可复用的几何关系表示。

### 中间数据类型

- **PVW（Point Visible Walls）**：每个点可见的墙集合  
- **PVP（Point Visible Points）**：每个点可见的关键端点集合  
- **WWV（Wall Wall Visibility）**：墙与墙之间的可见关系  

### 目标

将复杂几何关系预处理为结构化数据，供仿真阶段直接复用。

---

## Stage 4：数据转换

`stage4_convert/run.py` 负责将中间数据转换为适合高性能计算的数据格式。

### 主要工作

- 调整数据布局  
- 转换为更适合底层执行的表示  
- 减少仿真阶段运行时开销  

该阶段本质是：**面向计算内核的数据准备**

---

## Stage 5：仿真

- `stage5_sim/solver.py`：单个发射机传播计算  
- `stage5_sim/batch_run.py`：批量仿真调度  

### 含义划分

- `solver.py`：执行具体计算  
- `batch_run.py`：负责调度多个发射机  

---

# 运行入口

## 1. 仅生成中间数据

```bash
python run_intermdata.py
```

用于执行预计算流程，生成仿真所需的中间数据。

---

## 2. 执行完整流程

```bash
python run_all.py
```

包含：

- 中间数据生成  
- 仿真计算  

---










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

