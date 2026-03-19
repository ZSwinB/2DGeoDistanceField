import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# ==========================
# 路径
# ==========================
T_root = "/root/RM/distance_map_DPM_DRM"
wall_root = "/root/RM/geo"
car_root = "/root/RM/cars"

save_root = "/root/time"
os.makedirs(save_root, exist_ok=True)

# ==========================
# 参数
# ==========================
scene_id = 462
frame_id = 55


# ==========================
# 读取墙
# ==========================
wall_path = os.path.join(wall_root, f"{scene_id}.npy")

if not os.path.exists(wall_path):
    raise RuntimeError("wall file not found")

wall_mask = np.load(wall_path)

# ==========================
# 读取车
# ==========================
car_path = os.path.join(car_root, f"{scene_id}.png")

if os.path.exists(car_path):
    car_img = Image.open(car_path)
    car_mask = np.array(car_img) > 0
else:
    car_mask = np.zeros_like(wall_mask)

# ==========================
# 自由空间
# ==========================
occ_mask = (wall_mask > 0) | car_mask
free_mask = ~occ_mask

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
    # 计算 |∇T| - 1
    # ==========================
    grad_error = grad_norm - 1.0

    # 只显示自由空间
    grad_error_vis = grad_error.copy()
    grad_error_vis[~free_mask] = np.nan

    # ==========================
    # 画图
    # ==========================
    plt.figure(figsize=(7,7))

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
        save_root,
        f"grad_error_scene{scene_id}_frame{frame_id}_k{k}.png"
    )

    plt.savefig(save_path, dpi=200)
    plt.close()

    print("Saved:", save_path)

print("Done.")