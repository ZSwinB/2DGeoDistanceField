import numpy as np
from PIL import Image
import os

BUILDING_ROOT = r"G:\RadioMapSeer\png\buildings_complete"
OUT_ROOT = r"G:\data_SRM\antenna"

os.makedirs(OUT_ROOT, exist_ok=True)

NUM_SCENE = 701
NEW_TX_PER_SCENE = 120
START_ID = 80

for scene_id in range(NUM_SCENE):

    print(f"\n=== Scene {scene_id} ===")

    building_path = os.path.join(BUILDING_ROOT, f"{scene_id}.png")

    if not os.path.exists(building_path):
        print("skip missing:", scene_id)
        continue

    building = np.array(Image.open(building_path).convert("L"))
    H, W = building.shape

    # -----------------------------
    # 收集边界 free space 点
    # -----------------------------
    candidates = []

    for i in range(H):
        for j in range(W):
            if building[i, j] > 128:
                continue

            if i < 50 or i > 200 or j < 50 or j > 200:
                candidates.append((i, j))

    candidates = np.array(candidates)

    if len(candidates) < NEW_TX_PER_SCENE:
        print("not enough candidates:", scene_id)
        continue

    # -----------------------------
    # 随机选24个
    # -----------------------------
    np.random.seed(scene_id)  # 每个scene固定但不同
    idx = np.random.choice(len(candidates), NEW_TX_PER_SCENE, replace=False)
    new_tx = candidates[idx]

    # -----------------------------
    # 生成 one-hot antenna npy
    # -----------------------------
    for k, (i, j) in enumerate(new_tx):

        ant = np.zeros((H, W), dtype=np.uint8)
        ant[i, j] = 1   # 👈 one-hot（不是255）

        tx_id = START_ID + k

        out_path = os.path.join(
            OUT_ROOT,
            f"{scene_id}_{tx_id}.npy"
        )

        np.save(out_path, ant)

    print(f"saved TX {START_ID}~{START_ID+NEW_TX_PER_SCENE-1}")