import numpy as np
import random

SCENE_ID = 0
N_CHECK = 20   # 随机检查 20 个点

PVP_PATH = r"D:\Desktop\RM\data\PVP\0.npy"
PVP_ID_PATH = r"D:\Desktop\RM\data\PVP_id\0.npy"
TABLE_PATH = r"D:\Desktop\RM\data\PVP\corner_table\0.npy"

PVP = np.load(PVP_PATH, allow_pickle=True)
PVP_id = np.load(PVP_ID_PATH, allow_pickle=True)
corner_table = np.load(TABLE_PATH)

H, W = PVP.shape

for _ in range(N_CHECK):
    y = random.randrange(H)
    x = random.randrange(W)

    orig = PVP[y, x]
    ids = PVP_id[y, x]

    if not orig:
        if ids != []:
            raise RuntimeError(f"Mismatch at ({y},{x}): orig None, id not empty")
        continue

    restored = [tuple(corner_table[cid]) for cid in ids]

    if sorted(orig) != sorted(restored):
        raise RuntimeError(
            f"Mismatch at ({y},{x})\n"
            f"orig={orig}\n"
            f"restored={restored}"
        )

print("✅ consistency check passed")
