import numpy as np
import os
from tqdm import tqdm

# ======================================
# 路径
# ======================================
PVP_ROOT = "/root/RM/data/PVP_id"
CORNER_ROOT = "/root/RM/data/corner_table"
OUT_ROOT = "/root/RM/data/PVP_csr"

os.makedirs(OUT_ROOT, exist_ok=True)

# ======================================
# 自动扫描 scene 列表
# ======================================
SCENE_LIST = sorted([
    int(f.split(".")[0])
    for f in os.listdir(PVP_ROOT)
    if f.endswith(".npy")
])

print("Total scenes:", len(SCENE_LIST))

# ======================================
# 主循环
# ======================================
for scene_id in tqdm(SCENE_LIST, desc="Building CSR"):

    pvp_path = f"{PVP_ROOT}/{scene_id}.npy"
    corner_path = f"{CORNER_ROOT}/{scene_id}.npy"

    if not os.path.exists(pvp_path):
        continue

    if not os.path.exists(corner_path):
        continue

    # ---------- 读取 ----------
    PVP_id = np.load(pvp_path, allow_pickle=True)
    corner_table = np.load(corner_path)

    H, W = PVP_id.shape

    # ---------- 统计总长度 ----------
    total = 0
    for y in range(H):
        for x in range(W):
            lst = PVP_id[y, x]
            if lst:
                total += len(lst)

    # ---------- 分配 CSR ----------
    PVP_flat = np.zeros(total, dtype=np.int32)
    PVP_start = np.zeros((H, W), dtype=np.int32)
    PVP_len = np.zeros((H, W), dtype=np.int16)

    idx = 0

    for y in range(H):
        for x in range(W):
            lst = PVP_id[y, x]

            PVP_start[y, x] = idx

            if lst:
                length = len(lst)
                PVP_len[y, x] = length
                PVP_flat[idx:idx+length] = lst
                idx += length
            else:
                PVP_len[y, x] = 0

    # ---------- 拆分 corner_table ----------
    corner_table = corner_table.astype(np.float32)
    corner_x = corner_table[:, 0].astype(np.float32)
    corner_y = corner_table[:, 1].astype(np.float32)

    # ---------- 创建 scene 子目录 ----------
    scene_dir = f"{OUT_ROOT}/{scene_id}"
    os.makedirs(scene_dir, exist_ok=True)

    # ---------- 保存 ----------
    np.save(f"{scene_dir}/PVP_flat.npy", PVP_flat)
    np.save(f"{scene_dir}/PVP_start.npy", PVP_start)
    np.save(f"{scene_dir}/PVP_len.npy", PVP_len)
    np.save(f"{scene_dir}/corner_x.npy", corner_x)
    np.save(f"{scene_dir}/corner_y.npy", corner_y)

print("CSR build finished.")