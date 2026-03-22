import numpy as np
import os
from scipy.ndimage import label
from multiprocessing import Pool, cpu_count
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from config import Config
from pipeline.stage1_input.load_geo import load_geo


# ==========================
# config
# ==========================
config = Config()

# ==========================
# 路径
# ==========================
T_root = os.path.join(config.OUTPUT_ROOT, "sim")
save_root = os.path.join(config.OUTPUT_ROOT, "scaffold")

os.makedirs(save_root, exist_ok=True)

# ==========================
# 参数
# ==========================
epsilon = 0.08
component_threshold = 60


# ==========================
# 单个 scene
# ==========================
def process_scene(scene_id):

    print(f"Start Scene {scene_id}")

    config.IDX = scene_id
    geo, _, _ = load_geo(config)

    free_mask = (geo == 0)

    for frame_id in range(80):

        T_path = os.path.join(T_root, f"{scene_id}_{frame_id}.npz")
        if not os.path.exists(T_path):
            continue

        data = np.load(T_path)
        dist_map = data["dist_map"]   # (H,W,K)

        H, W, K = dist_map.shape

        final_mask = np.zeros((H, W, K), dtype=np.uint8)

        for k in range(K):

            T = dist_map[:, :, k].astype(np.float32)

            dx = np.zeros_like(T)
            dy = np.zeros_like(T)

            dx[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / 2.0
            dy[1:-1, :] = (T[2:, :] - T[:-2, :]) / 2.0

            grad_norm = np.sqrt(dx * dx + dy * dy)

            structure_mask = (grad_norm > (1.0 + epsilon))

            # 只保留自由空间
            structure_mask &= free_mask

            # 连通域过滤
            labeled, num = label(structure_mask, structure=np.ones((3,3)))

            clean_mask = np.zeros_like(structure_mask)

            for i in range(1, num + 1):
                comp = (labeled == i)
                if np.sum(comp) >= component_threshold:
                    clean_mask |= comp

            final_mask[:, :, k] = clean_mask.astype(np.uint8)

        save_path = os.path.join(save_root, f"{scene_id}_{frame_id}.npy")
        np.save(save_path, final_mask)

    print(f"Finish Scene {scene_id}")


# ==========================
# 主函数
# ==========================
if __name__ == "__main__":

    num_workers = max(cpu_count() - 1, 1)
    print("Using workers:", num_workers)

    with Pool(num_workers) as pool:
        pool.map(process_scene, range(0))

    print("All scenes done.")