import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 路径
# ===============================
wall_file = r"D:\Desktop\RM\data\wall_segment\0.npy"
pvw_file  = r"D:\Desktop\RM\data\PVW\0.npy"

# ===============================
# 输入你要看的 YX
# ===============================
query_y = 187
query_x = 108

# ===============================
# 1. 读数据
# ===============================
walls_raw = np.load(wall_file, allow_pickle=True)
pvw = np.load(pvw_file, allow_pickle=True)

walls = []
for p0, p1 in walls_raw:
    walls.append((tuple(p0), tuple(p1)))

# ===============================
# 2. 从 PVW 拿 wall index
# ===============================
wall_indices = pvw[query_y, query_x]

print(f"PVW[{query_y}, {query_x}] =", wall_indices)

if wall_indices is None:
    print("这个位置是 None，没有对应的 wall segment")
    wall_indices = []

wall_indices = set(wall_indices)  # 去重，方便判断

# ===============================
# 3. 画图
# ===============================
plt.figure(figsize=(8, 8))

# 3.1 所有墙：灰色
for (x0, y0), (x1, y1) in walls:
    plt.plot([x0, x1], [y0, y1],
             color="gray", linewidth=1.0)

# 3.2 命中的墙：红色高亮
for idx in wall_indices:
    if 0 <= idx < len(walls):
        (x0, y0), (x1, y1) = walls[idx]
        plt.plot([x0, x1], [y0, y1],
                 color="red", linewidth=2.5)

# ===============================
# 4. 坐标系设置（和你之前一致）
# ===============================
plt.gca().set_aspect("equal", adjustable="box")
plt.gca().invert_yaxis()
plt.grid(True)

plt.title(f"PVW[{query_y}, {query_x}] → highlighted wall segments")
plt.show()
