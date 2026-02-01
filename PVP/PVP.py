import numpy as np
import math
import os
from tqdm import tqdm

# =======================
# 配置
# =======================

SCENE_ID = 0

GEO_PATH  = rf"D:\Desktop\RM\feature_store\numerical_data\geo\{SCENE_ID}.npy"
WALL_PATH = rf"D:\Desktop\RM\data\wall_segment\{SCENE_ID}.npy"
PVW_PATH  = rf"D:\Desktop\RM\data\PVW\{SCENE_ID}.npy"

PVP_DIR   = r"D:\Desktop\RM\data\PVP"
PVP_PATH  = rf"{PVP_DIR}\{SCENE_ID}.npy"

GRID_SIZE = 8        # 用于 wall_grid（不是图像大小）
EPS = 1e-9

# =======================
# 几何基础
# =======================

def orient(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def on_segment(a, b, p):
    return (
        min(a[0], b[0]) - EPS <= p[0] <= max(a[0], b[0]) + EPS and
        min(a[1], b[1]) - EPS <= p[1] <= max(a[1], b[1]) + EPS
    )

def segment_intersect_strict(a, b, c, d):
    """
    严格相交：
    - 端点接触 / 共线接触 → False
    - 真正穿过内部 → True
    """
    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    if o1 * o2 < 0 and o3 * o4 < 0:
        return True

    return False

# =======================
# wall_grid 构建
# =======================

def build_wall_grid(walls):
    """
    把每个墙段放进空间网格，加速可见性判定
    """
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
# fast visible
# =======================

def visible_fast(p1, p2, walls, wall_grid):
    """
    判断 p1 -> p2 是否被任何墙严格遮挡
    端点接触 ≠ 遮挡
    """
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
                if segment_intersect_strict(p1, p2, A, B):
                    return False

    return True

# =======================
# PVP 构建
# =======================

def build_PVP(geo, walls, PVW, wall_grid):
    H, W = geo.shape
    PVP = np.empty((H, W), dtype=object)
    PVP[:] = None

    free_points = np.argwhere(geo == 0)

    for y, x in tqdm(free_points, desc=f"Build PVP scene {SCENE_ID}"):
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
# main
# =======================

def main():
    print("[Load] geo / walls / PVW")

    geo   = np.load(GEO_PATH)
    walls = np.load(WALL_PATH, allow_pickle=True).tolist()
    PVW   = np.load(PVW_PATH, allow_pickle=True)

    print("[Build] wall_grid")
    wall_grid = build_wall_grid(walls)

    print("[Build] PVP")
    PVP = build_PVP(geo, walls, PVW, wall_grid)

    os.makedirs(PVP_DIR, exist_ok=True)
    np.save(PVP_PATH, PVP)

    print(f"[Done] PVP saved to {PVP_PATH}")

# =======================
# entry
# =======================

if __name__ == "__main__":
    main()
