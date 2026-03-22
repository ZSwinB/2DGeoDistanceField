import numpy as np
import os
import matplotlib.pyplot as plt
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from config import Config
from pipeline.stage1_input.load_geo import load_geo


# ==========================
# config
# ==========================
config = Config()

scene_id = 0
frame_id = 0

config.IDX = scene_id

# ==========================
# 路径（统一）
# ==========================
T_root = os.path.join(config.OUTPUT_ROOT, "sim")

DEBUG_DIR = os.path.join(config.OUTPUT_ROOT, "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)

# ==========================
# 读取 geo（统一入口）
# ==========================
geo, _, _ = load_geo(config)

# 自由空间
free_mask = (geo == 0)

# ==========================
# 读取 distance map
# ==========================
T_path = os.path.join(T_root, f"{scene_id}_{frame_id}.npz")

if not os.path.exists(T_path):
    raise RuntimeError("distance map not found")

data = np.load(T_path)
dist_map = data["dist_map"]    # (H,W,K)

H, W, K = dist_map.shape

print("Loaded dist_map:", dist_map.shape)


# ==========================
# 遍历 K
# ==========================
for k in range(K):

    T = dist_map[:, :, k].astype(np.float32)

    # ==========================
    # 中心差分
    # ==========================
    dx = np.zeros_like(T)
    dy = np.zeros_like(T)

    dx[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / 2.0
    dy[1:-1, :] = (T[2:, :] - T[:-2, :]) / 2.0

    grad_norm = np.sqrt(dx * dx + dy * dy)

    # ==========================
    # |∇T| - 1
    # ==========================
    grad_error = grad_norm - 1.0

    grad_error_vis = grad_error.copy()
    grad_error_vis[~free_mask] = np.nan

    # ==========================
    # 可视化
    # ==========================
    plt.figure(figsize=(7, 7))

    plt.imshow(
        grad_error_vis,
        cmap="inferno",
        vmin=0,
        vmax=np.nanpercentile(grad_error_vis, 99)
    )

    plt.colorbar(label="|∇T| - 1")
    plt.title(f"scene {scene_id} frame {frame_id} k {k}")
    plt.axis("off")

    save_path = os.path.join(
        DEBUG_DIR,
        f"grad_error_{scene_id}_{frame_id}_k{k}.png"
    )

    plt.savefig(save_path, dpi=200)
    plt.close()

    print("Saved:", save_path)

print("Done.")