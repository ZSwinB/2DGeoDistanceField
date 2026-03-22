import numpy as np
import os
from tqdm import tqdm
from numba import njit
from utils.io import save_npy, load_npy


# =======================
# 参数（保持原样）
# =======================

GRID_SIZE = 8


# =======================
# 几何
# =======================

@njit
def orient_nb(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


@njit
def segment_intersect_strict_nb(ax, ay, bx, by, cx, cy, dx, dy):
    o1 = orient_nb(ax, ay, bx, by, cx, cy)
    o2 = orient_nb(ax, ay, bx, by, dx, dy)
    o3 = orient_nb(cx, cy, dx, dy, ax, ay)
    o4 = orient_nb(cx, cy, dx, dy, bx, by)
    return (o1 * o2 < 0.0 and o3 * o4 < 0.0)


# =======================
# wall_grid
# =======================

def build_wall_grid(walls):
    grid = {}
    for idx, (A, B) in enumerate(walls):
        x0, y0 = A
        x1, y1 = B

        gx0 = int(min(x0, x1) // GRID_SIZE)
        gx1 = int(max(x0, x1) // GRID_SIZE)
        gy0 = int(min(y0, y1) // GRID_SIZE)
        gy1 = int(max(y0, y1) // GRID_SIZE)

        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                grid.setdefault((gx, gy), []).append(idx)

    return grid


# =======================
# visible
# =======================

def visible_fast(p1, p2, walls, wall_grid):
    x0, y0 = p1
    x1, y1 = p2

    gx0 = int(min(x0, x1) // GRID_SIZE)
    gx1 = int(max(x0, x1) // GRID_SIZE)
    gy0 = int(min(y0, y1) // GRID_SIZE)
    gy1 = int(max(y0, y1) // GRID_SIZE)

    checked = set()

    for gx in range(gx0, gx1 + 1):
        for gy in range(gy0, gy1 + 1):
            for idx in wall_grid.get((gx, gy), []):
                if idx in checked:
                    continue
                checked.add(idx)

                A, B = walls[idx]

                if segment_intersect_strict_nb(
                    x0, y0,
                    x1, y1,
                    A[0], A[1],
                    B[0], B[1]
                ):
                    return False

    return True


# =======================
# build PVP
# =======================

def build_PVP(geo, walls, PVW, wall_grid):
    H, W = geo.shape
    PVP = np.empty((H, W), dtype=object)
    PVP[:] = None

    free_points = np.argwhere(geo == 0)

    for y, x in free_points:
        vis_walls = PVW[y, x]
        if not vis_walls:
            PVP[y, x] = []
            continue

        P = (float(x), float(y))
        corners = []
        seen = set()

        for wid in vis_walls:
            A, B = walls[wid]

            ax, ay = float(A[0]), float(A[1])
            bx, by = float(B[0]), float(B[1])

            if (ax, ay) not in seen:
                if visible_fast(P, (ax, ay), walls, wall_grid):
                    corners.append((ax, ay))
                    seen.add((ax, ay))

            if (bx, by) not in seen:
                if visible_fast(P, (bx, by), walls, wall_grid):
                    corners.append((bx, by))
                    seen.add((bx, by))

        PVP[y, x] = corners

    return PVP


# =======================
# stage接口（单场景）
# =======================

def build_pvp(geo, wall_segment, pvw, config):
    """
    输入:
        geo
        wall_segment（不使用，从磁盘读取）
        pvw（不使用，从磁盘读取）

    输出:
        PVP（落盘）
    """

    idx = config.IDX

    wall_path = os.path.join(
        config.OUTPUT_ROOT,
        config.WALL_DIR,
        f"{idx}.npy"
    )

    pvw_path = os.path.join(
        config.OUTPUT_ROOT,
        config.PVW_DIR,
        f"{idx}.npy"
    )

    out_path = os.path.join(
        config.OUTPUT_ROOT,
        config.PVP_DIR,
        f"{idx}.npy"
    )

    print("[Load] geo + walls + PVW")

    walls = load_npy(wall_path).tolist()
    PVW   = load_npy(pvw_path)

    # ===== 计算 =====
    wall_grid = build_wall_grid(walls)

    print("[Build] PVP")

    PVP = build_PVP(geo, walls, PVW, wall_grid)

    # ===== 保存 =====
    save_npy(out_path, PVP)

    print(f"[Done] PVP saved to {out_path}")

    return None