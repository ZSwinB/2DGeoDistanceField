import numpy as np
import matplotlib.pyplot as plt

SCENE_ID = 268
QUERY_POINT = (200, 200)  # (y, x)

geo   = np.load(f"/root/geo/{SCENE_ID}.npy")
walls = np.load(f"/root/wall_segment_DRM/{SCENE_ID}.npy", allow_pickle=True).tolist()
PVP   = np.load(f"/root/RM/data/PVP/{SCENE_ID}.npy", allow_pickle=True)

y, x = QUERY_POINT

if geo[y, x] != 0:
    raise ValueError("Query point not free.")

corners = PVP[y, x]

plt.figure(figsize=(8,8))

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

plt.savefig(f"/root/PVP_vis_{SCENE_ID}.png", dpi=300, bbox_inches="tight")
plt.close()