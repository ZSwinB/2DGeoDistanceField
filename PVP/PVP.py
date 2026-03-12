import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from numba import njit

# =======================
# 配置
# =======================

GRID_SIZE = 8

GEO_ROOT  = "/root/geo"
WALL_ROOT = "/root/wall_segment_DRM"
PVW_ROOT  = "/root/RM/data/PVW"
PVP_ROOT  = "/root/RM/data/PVP"

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
# 单场景
# =======================

def run_scene(scene_id):

    try:
        print(f"[Start] Scene {scene_id}", flush=True)

        GEO_PATH  = f"{GEO_ROOT}/{scene_id}.npy"
        WALL_PATH = f"{WALL_ROOT}/{scene_id}.npy"
        PVW_PATH  = f"{PVW_ROOT}/{scene_id}.npy"
        PVP_PATH  = f"{PVP_ROOT}/{scene_id}.npy"

        if not os.path.exists(GEO_PATH):
            print(f"[Skip] Scene {scene_id} no geo", flush=True)
            return

        if os.path.exists(PVP_PATH):
            print(f"[Skip] Scene {scene_id} already exists", flush=True)
            return

        geo   = np.load(GEO_PATH)
        walls = np.load(WALL_PATH, allow_pickle=True).tolist()
        PVW   = np.load(PVW_PATH, allow_pickle=True)

        wall_grid = build_wall_grid(walls)
        PVP = build_PVP(geo, walls, PVW, wall_grid)

        os.makedirs(PVP_ROOT, exist_ok=True)
        np.save(PVP_PATH, PVP)

        print(f"[Done] Scene {scene_id}", flush=True)

        return scene_id

    except Exception as e:
        print(f"[Error] Scene {scene_id} -> {e}", flush=True)
        return

# =======================
# 多进程入口
# =======================

if __name__ == "__main__":

    scenes = list(range(400, 701))
    n_proc = max(1, cpu_count() - 1)

    print("Using", n_proc, "processes", flush=True)

    with Pool(n_proc) as pool:
        list(tqdm(pool.imap_unordered(run_scene, scenes),
                  total=len(scenes),
                  desc="Building PVP 0-700"))