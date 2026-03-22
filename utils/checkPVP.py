import numpy as np
import matplotlib.pyplot as plt
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from config import Config
from pipeline.stage1_input.load_geo import load_geo


SCENE_ID = 0
QUERY_POINT = (1, 0)  # (y, x)

# ======================
# config
# ======================
config = Config()
config.IDX = SCENE_ID

# ======================
# load data
# ======================
geo, _, _ = load_geo(config)

walls = np.load(
    os.path.join(config.OUTPUT_ROOT, config.WALL_DIR, f"{SCENE_ID}.npy"),
    allow_pickle=True
).tolist()

PVP = np.load(
    os.path.join(config.OUTPUT_ROOT, config.PVP_DIR, f"{SCENE_ID}.npy"),
    allow_pickle=True
)

# ======================
# query
# ======================
y, x = QUERY_POINT

if geo[y, x] != 0:
    raise ValueError("Query point not free.")

corners = PVP[y, x]

# ======================
# plot
# ======================
plt.figure(figsize=(8, 8))

# 墙（灰色）
for A, B in walls:
    plt.plot([A[0], B[0]], [A[1], B[1]], linewidth=1)

# 查询点（蓝）
plt.scatter([x], [y], s=60)

# 角点（红）
if corners:
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    plt.scatter(xs, ys, s=20)

plt.gca().invert_yaxis()
plt.axis("off")

DEBUG_DIR = os.path.join(config.OUTPUT_ROOT, "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)

out_path = os.path.join(
    DEBUG_DIR,
    f"PVP_vis_{SCENE_ID}_{y}_{x}.png"
)

plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", out_path)