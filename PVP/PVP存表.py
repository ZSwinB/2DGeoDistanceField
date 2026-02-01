import numpy as np
import os

SCENE_ID = 0

PVP_PATH   = f"D:\Desktop\RM\data\PVP\{SCENE_ID}.npy"
OUT_DIR    = "D:\Desktop\RM\data\PVP_id"
TABLE_DIR  = "D:\Desktop\RM\data\PVP\corner_table"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

PVP = np.load(PVP_PATH, allow_pickle=True)

H, W = PVP.shape

corner2id = {}
corner_table = []
next_id = 0

PVP_id = np.empty((H, W), dtype=object)

for y in range(H):
    for x in range(W):
        lst = PVP[y, x]
        if not lst:
            PVP_id[y, x] = []
            continue

        ids = []
        for (cx, cy) in lst:
            key = (cx, cy)   # ⚠️ 不交换顺序
            if key not in corner2id:
                corner2id[key] = next_id
                corner_table.append(key)
                next_id += 1
            ids.append(corner2id[key])

        PVP_id[y, x] = ids

np.save(f"{OUT_DIR}/{SCENE_ID}.npy", PVP_id)
np.save(f"{TABLE_DIR}/{SCENE_ID}.npy", np.array(corner_table, dtype=np.int16))

print("done")
print("corners:", len(corner_table))

corner_table_path = r"D:\Desktop\RM\data\PVP\corner_table\0.npy"
pvp_id_path       = r"D:\Desktop\RM\data\PVP_id\0.npy"

corner_table = np.load(corner_table_path)
PVP_id = np.load(pvp_id_path, allow_pickle=True)

print("corner_table[0]:", corner_table[0])
print("corner_table shape:", corner_table.shape)

print("PVP_id[0,0]:", PVP_id[0, 0])
print("type(PVP_id[0,0]):", type(PVP_id[0, 0]))
