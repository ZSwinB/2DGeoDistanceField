import numpy as np
import os

PVW_ROOT  = r"D:\Desktop\RM\data\PVW"
WALL_ROOT= r"D:\Desktop\RM\data\wall_segment"
OUT_ROOT   = "D:\Desktop\RM\data\PVW_mask"

os.makedirs(OUT_ROOT, exist_ok=True)

for scene_id in range(1):
    pvw_path  = f"{PVW_ROOT}/{scene_id}.npy"
    wall_path = f"{WALL_ROOT}/{scene_id}.npy"
    out_path  = f"{OUT_ROOT}/{scene_id}.npy"

    if not os.path.exists(pvw_path):
        print(f"[skip] PVW missing: {scene_id}")
        continue

    PVW = np.load(pvw_path, allow_pickle=True)
    walls = np.load(wall_path, allow_pickle=True)

    H, W = PVW.shape
    N_wall = len(walls)

    PVW_mask = np.zeros((H, W, N_wall), dtype=np.bool_)

    for i in range(H):
        for j in range(W):
            lst = PVW[i, j]
            if lst is None:
                continue
            PVW_mask[i, j, lst] = True

    np.save(out_path, PVW_mask)
    print(f"[done] scene {scene_id}: {PVW_mask.shape}")

print("all scenes done")
