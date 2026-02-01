# 2DGeoDistanceField
输入地理信息与发射天线位置，计算自由空间内的TOA场。 支持直射、反射与绕射路径，最多两次碰撞；可扩展输出多径距离与相位信息。
## 墙段数据（Wall Segment Data）

### 输入
- **geo**：二维栅格地图  
  - `shape = (H, W)`  
  - `geo[i, j] == 1` 表示墙体  
  - `geo[i, j] == 0` 表示自由空间  

### 输出
- **文件名**：`scene_id.npy`  
- **数据类型**：`numpy.ndarray(dtype=object)`  
- **数据结构**：

    ```python
    walls = [
        ((x1, y1), (x2, y2)),
        ((x3, y3), (x4, y4)),
        ...
    ]
    ```

- 每个元素表示一条墙段  
- 墙段由两个端点 `(x, y)` 表示，坐标基于地图索引  

### Index 约定
- `index` 指 `walls` 数组中的顺序索引  
- 索引顺序即墙段在生成时的存储顺序  
- 后续模块通过该 `index` 直接引用对应墙段
## WWVmove 数据

### 输入
- **geo**：二维栅格地图  
  - `shape = (H, W)`  
  - `geo[i, j] == 1` 表示墙体  
  - `geo[i, j] == 0` 表示自由空间  

- **walls**：墙段数据  
  - 来自 `wall_segment/scene_id.npy`  
  - `walls[index] = (A, B)`  
  - `A, B` 为墙段两个端点 `(x, y)`

---

### 输出
- **文件名**：`scene_id.npy`  
- **数据类型**：`dict`

- **数据结构**：

    ```python
    WWVmove = {
        wid: [dir_A, dir_B],
        ...
    }
    ```

- `wid` 为墙段的 index  
- `dir_A` / `dir_B` 为对应端点的移动方向  
- 移动方向采用 8 邻域方向 `(dx, dy)`  
- 若端点周围不存在自由空间方向，则为 `None`

---

### Index 约定
- `wid` 与 `walls` 中的索引一致  
- `WWVmove[wid]` 与 `walls[wid]` 一一对应  
## PVW / WWV 数据

### 输入
- **geo**：二维栅格地图  
  - `shape = (H, W)`  
  - `geo[i, j] == 1` 表示墙体  
  - `geo[i, j] == 0` 表示自由空间  

- **walls**：墙段数据  
  - 来自 `wall_segment/scene_id.npy`  
  - `walls[index] = (A, B)`  
  - `A, B` 为墙段两个端点 `(x, y)`

- **WWVmove**：墙段端点的自由空间方向  
  - 来自 `WWVmove/scene_id.npy`  
  - `WWVmove[wid] = [dir_A, dir_B]`  
  - `dir_A / dir_B` 为 8 邻域方向 `(dx, dy)` 或 `None`

---

## PVW（Point Visible Walls）

### 输出
- **文件名**：`PVW/scene_id.npy`  
- **数据类型**：`numpy.ndarray(dtype=object)`  
- **形状**：`(H, W)`

- **数据结构**：

    ```python
    PVW[y, x] = [w0, w1, w2, ...]
    ```

- 仅对 `geo[y, x] == 0`（自由空间点）有定义  
- 列表中元素为**从该点可见的墙段 index**

### Index 约定
- `w` 为 `walls` 中的索引  
- `PVW[y, x]` 中的索引与 `walls[w]` 一一对应  

---

## WWV（Wall–Wall Visibility）

### 输出
- **文件名**：`WWV/scene_id.npy`  
- **数据类型**：`numpy.ndarray(dtype=bool)`  
- **形状**：`(N_walls, N_walls)`

- **数据结构**：

    ```python
    WWV[i, j] = True / False
    ```

- `WWV[i, j] == True` 表示墙段 `i` 与墙段 `j` 至少存在一个自由空间视点可见  
- 矩阵为对称矩阵，且 `WWV[i, i] == False`

### Index 约定
- `i, j` 均为 `walls` 中的索引  
- `WWV` 与 `walls` 使用同一套 index 体系  

## PVW_mask 数据

### 输入
- **PVW**：点–墙可见性列表  
  - 来自 `PVW/scene_id.npy`  
  - `PVW[y, x] = [w0, w1, ...]` 或 `None`

- **walls**：墙段数据  
  - 来自 `wall_segment/scene_id.npy`  
  - 用于确定墙段总数与索引范围

---

### 输出
- **文件名**：`PVW_mask/scene_id.npy`  
- **数据类型**：`numpy.ndarray(dtype=bool)`  
- **形状**：`(H, W, N_wall)`

- **数据结构**：

    ```python
    PVW_mask[y, x, w] = True / False
    ```

- `PVW_mask[y, x, w] == True` 表示  
  自由空间点 `(x, y)` 与墙段 `w` 可见

- 对于 `PVW[y, x] is None` 的位置，整行保持为 `False`

---

### Index 约定
- `w` 为 `walls` 中的索引  
- 第三个维度与 `walls` 的 index 一一对应  

## PVP 数据

PVP 描述的是：  
**自由空间中的一个点，与墙段端点（vertex）之间的可见关系。**  
这里的 *vertex* 指的是墙段的端点，而不是墙段本身。

---

## 原始语义（逻辑层）

对于自由空间中的一个点 `(x, y)`：  
- 先确定该点可见的墙段  
- 再判断该点是否能“看到”这些墙段的端点  
- 若点到端点之间不存在严格遮挡，则认为该端点可见  

逻辑表示为：  
PVP[y, x] = [(vx1, vy1), (vx2, vy2), ...]

---

## 实际存储（压缩表示）

为了节省存储空间，不直接存储端点坐标，而是进行 ID 化。

### 1. corner_table（端点表）

- **文件名**：`PVP/corner_table/scene_id.npy`  
- **数据类型**：`numpy.ndarray`

结构：  
corner_table[id] = (vx, vy)

说明：  
- 每一个唯一出现过的墙段端点分配一个**全局 id**  
- `(vx, vy)` 顺序固定为 `(x, y)`，不交换

---

### 2. PVP_id（点–端点 ID 关系）

- **文件名**：`PVP_id/scene_id.npy`  
- **数据类型**：`numpy.ndarray(dtype=object)`  
- **形状**：`(H, W)`

数据结构：  
PVP_id[y, x] = [id0, id1, id2, ...]

说明：  
- 列表中的 `id` 对应 `corner_table[id]`  
- 若该点没有可见端点，则为 `[]`

---

## Index 约定

- `id` 为 `corner_table` 中的索引  
- `PVP_id[y, x]` 中的每个 `id` 可通过  
  `corner_table[id]`  
  还原为端点坐标 `(vx, vy)`  
- 端点来源于 `walls[wid]` 的端点，但 `PVP_id` **不直接存储 `wid`**


---

### 数据生成顺序（仿真前置）

在开始射线 / 路径仿真之前，需要依次生成以下数据：

1. **geo**  
   - 场景的二维栅格地图

2. **wall_segment**  
   - 从 `geo` 中提取的墙段数据

3. **WWVmove**  
   - 每个墙段端点对应的自由空间方向

4. **PVW**  
   - 自由空间中每个点可见的墙段索引列表

5. **PVW_mask**（可选）  
   - `PVW` 的布尔掩码形式，用于加速计算

6. **WWV**  
   - 墙段–墙段之间的可见性关系

7. **PVP（逻辑）**  
   - 自由空间点与墙段端点之间的可见关系

8. **corner_table + PVP_id（存储）**  
   - 对 PVP 进行 ID 化后的压缩存储表示

完成以上数据生成后，即可进入后续的路径、反射、绕射与距离/相位仿真阶段。
