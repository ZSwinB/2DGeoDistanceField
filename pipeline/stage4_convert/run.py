import numpy as np
import os
from utils.io import load_npy, save_npy


def run(pvw, wwv, pvp, config):
    """
    Convert intermediate data into compute-optimized formats.

    Behavior:
        - Ignores input arguments (pvw, wwv, pvp) and loads data from disk
        - Converts:
            PVW → dense boolean mask (H, W, N_wall)
            PVP → indexed representation (corner_table + PVP_id)
            PVP_id → CSR-like structure (flat / start / len)
            walls → numeric array (walls_nb)
        - Saves all outputs under:
            OUTPUT_ROOT / convert / {idx}/

    Args:
        pvw: unused (kept for interface compatibility)
        wwv: unused (kept for interface compatibility)
        pvp: unused (kept for interface compatibility)
        config: configuration object

    Outputs:
        PVW_mask.npy   : (H, W, N_wall) bool
        PVP_flat.npy   : (total,) int32
        PVP_start.npy  : (H, W) int32
        PVP_len.npy    : (H, W) int16
        corner_x.npy   : (N_corner,) float32
        corner_y.npy   : (N_corner,) float32
        walls_nb.npy   : (N_wall, 4) float64

    Returns:
        None
    """

    idx = config.IDX

    # ========= 路径 =========

    wall_path = os.path.join(
        config.OUTPUT_ROOT, config.WALL_DIR, f"{idx}.npy"
    )

    pvw_path = os.path.join(
        config.OUTPUT_ROOT, config.PVW_DIR, f"{idx}.npy"
    )

    pvp_path = os.path.join(
        config.OUTPUT_ROOT, config.PVP_DIR, f"{idx}.npy"
    )

    scene_dir = os.path.join(
        config.OUTPUT_ROOT, "convert", str(idx)
    )
    os.makedirs(scene_dir, exist_ok=True)

    # ========= 读取 =========

    print("[Load] wall + PVW + PVP")

    walls = load_npy(wall_path)
    PVW   = load_npy(pvw_path)
    PVP   = load_npy(pvp_path)

    H, W = PVW.shape
    N_wall = len(walls)

    # =========================================================
    # ① PVW → PVW_mask
    # =========================================================

    print("[Convert] PVW → PVW_mask")

    PVW_mask = np.zeros((H, W, N_wall), dtype=np.bool_)

    for i in range(H):
        for j in range(W):
            lst = PVW[i, j]
            if lst is None:
                continue
            PVW_mask[i, j, lst] = True

    save_npy(os.path.join(scene_dir, "PVW_mask.npy"), PVW_mask)

    # =========================================================
    # ② PVP → PVP_id + corner_table
    # =========================================================

    print("[Convert] PVP → PVP_id + corner_table")

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
                key = (cx, cy)
                if key not in corner2id:
                    corner2id[key] = next_id
                    corner_table.append(key)
                    next_id += 1
                ids.append(corner2id[key])

            PVP_id[y, x] = ids

    corner_table = np.array(corner_table, dtype=np.float32)

    # =========================================================
    # ③ PVP_id → CSR
    # =========================================================

    print("[Convert] PVP_id → CSR")

    total = 0
    for y in range(H):
        for x in range(W):
            lst = PVP_id[y, x]
            if lst:
                total += len(lst)

    PVP_flat = np.zeros(total, dtype=np.int32)
    PVP_start = np.zeros((H, W), dtype=np.int32)
    PVP_len = np.zeros((H, W), dtype=np.int16)

    idx_ptr = 0

    for y in range(H):
        for x in range(W):
            lst = PVP_id[y, x]

            PVP_start[y, x] = idx_ptr

            if lst:
                length = len(lst)
                PVP_len[y, x] = length
                PVP_flat[idx_ptr:idx_ptr + length] = lst
                idx_ptr += length
            else:
                PVP_len[y, x] = 0

    # =========================================================
    # ④ corner_table → x / y
    # =========================================================

    corner_x = corner_table[:, 0].astype(np.float32)
    corner_y = corner_table[:, 1].astype(np.float32)

    # =========================================================
    # ⑤ wall → walls_nb
    # =========================================================

    print("[Convert] wall → walls_nb")

    n = len(walls)
    walls_nb = np.zeros((n, 4), dtype=np.float64)

    for i, (A, B) in enumerate(walls):
        walls_nb[i, 0] = A[0]
        walls_nb[i, 1] = A[1]
        walls_nb[i, 2] = B[0]
        walls_nb[i, 3] = B[1]

    # =========================================================
    # 保存（统一 scene 目录）
    # =========================================================

    print("[Save] convert outputs")

    save_npy(os.path.join(scene_dir, "PVP_flat.npy"), PVP_flat)
    save_npy(os.path.join(scene_dir, "PVP_start.npy"), PVP_start)
    save_npy(os.path.join(scene_dir, "PVP_len.npy"), PVP_len)
    save_npy(os.path.join(scene_dir, "corner_x.npy"), corner_x)
    save_npy(os.path.join(scene_dir, "corner_y.npy"), corner_y)
    save_npy(os.path.join(scene_dir, "walls_nb.npy"), walls_nb)

    print(f"[Done] Convert finished for scene {idx}")

    return None