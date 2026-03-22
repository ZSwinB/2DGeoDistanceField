import numpy as np
import matplotlib.pyplot as plt
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from config import Config


# ===============================
# config
# ===============================
config = Config()

SCENE_ID = 0
query_y = 1
query_x = 0

# ===============================
# debug 输出目录
# ===============================
DEBUG_DIR = os.path.join(config.OUTPUT_ROOT, "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)

# ===============================
# 路径（统一 config）
# ===============================
wall_file = os.path.join(
    config.OUTPUT_ROOT,
    config.WALL_DIR,
    f"{SCENE_ID}.npy"
)

pvw_file = os.path.join(
    config.OUTPUT_ROOT,
    config.PVW_DIR,
    f"{SCENE_ID}.npy"
)

# ===============================
# 1. 读数据
# ===============================
walls_raw = np.load(wall_file, allow_pickle=True)
pvw = np.load(pvw_file, allow_pickle=True)

walls = []
for p0, p1 in walls_raw:
    walls.append((tuple(p0), tuple(p1)))

# ===============================
# 2. 取 PVW
# ===============================
wall_indices = pvw[query_y, query_x]

print(f"PVW[{query_y}, {query_x}] =", wall_indices)

if wall_indices is None:
    print("这个位置是 None，没有对应的 wall segment")
    wall_indices = []

wall_indices = set(wall_indices)

# ===============================
# 3. 画图
# ===============================
plt.figure(figsize=(8, 8))

# 所有墙（灰）
for (x0, y0), (x1, y1) in walls:
    plt.plot([x0, x1], [y0, y1], color="gray", linewidth=1.0)

# 命中墙（红）
for idx in wall_indices:
    if 0 <= idx < len(walls):
        (x0, y0), (x1, y1) = walls[idx]
        plt.plot([x0, x1], [y0, y1], color="red", linewidth=2.5)

# 查询点（蓝）
plt.scatter([query_x], [query_y], s=60)

# ===============================
# 坐标系
# ===============================
plt.gca().set_aspect("equal", adjustable="box")
plt.gca().invert_yaxis()
plt.grid(True)

plt.title(f"PVW[{query_y}, {query_x}]")

# ===============================
# 保存 debug 图
# ===============================
out_path = os.path.join(
    DEBUG_DIR,
    f"pvw_{SCENE_ID}_{query_y}_{query_x}.png"
)

plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()

print("Saved:", out_path)