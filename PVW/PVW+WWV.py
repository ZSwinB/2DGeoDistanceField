import numpy as np
import math
import os
from tqdm import tqdm
import multiprocessing as mp

# ================= 配置 =================

SCENE_ID = 0

GEO_PATH  = rf"D:\Desktop\RM\feature_store\numerical_data\geo\{SCENE_ID}.npy"
WALL_PATH = rf"D:\Desktop\RM\data\wall_segment\{SCENE_ID}.npy"
MOVE_PATH = rf"D:\Desktop\RM\data\WWVmove\{SCENE_ID}.npy"

PVW_OUT_DIR  = r"D:\Desktop\RM\data\PVW"
PVW_OUT_PATH = rf"{PVW_OUT_DIR}\{SCENE_ID}.npy"

WWV_OUT_DIR  = r"D:\Desktop\RM\data\WWV"
WWV_OUT_PATH = rf"{WWV_OUT_DIR}\{SCENE_ID}.npy"

GRID_SIZE = 256
EPS = 1e-9
ENDPOINT_EPS = 1e-4

USE_MULTIPROCESS = True

# ================= 几何工具 =================

def angle(P, Q):
    return math.atan2(Q[1] - P[1], Q[0] - P[0])

def ray_segment_distance(P, theta, A, B):
    px, py = P
    dx = math.cos(theta)
    dy = math.sin(theta)

    ax, ay = A
    bx, by = B

    vx = bx - ax
    vy = by - ay

    det = dx * (-vy) - dy * (-vx)
    if abs(det) >= EPS:
        t = ((ax - px) * (-vy) - (ay - py) * (-vx)) / det
        u = ((ax - px) * (-dy) + (ay - py) * (dx)) / det
        if t > EPS and 0 <= u <= 1:
            return t
    return None

# ================= 单点角度扫描（FIXED PVW） =================

def visible_walls_for_point(P, walls):
    events = []

    for i, (A, B) in enumerate(walls):
        a1 = angle(P, A)
        a2 = angle(P, B)

        l = min(a1, a2)
        r = max(a1, a2)

        # 跨断点：拆，但不标 ENTER / EXIT
        if r - l > math.pi:
            events += [
                (l, i),
                ( math.pi, i),
                (-math.pi, i),
                (r, i)
            ]
        else:
            events += [(l, i), (r, i)]

    events.sort()

    active = set()
    visible = set()

    for k in range(len(events) - 1):
        ang, idx = events[k]

        # -------- 核心修正：toggle --------
        if idx in active:
            active.remove(idx)
        else:
            active.add(idx)
        # ---------------------------------

        next_ang = events[k + 1][0]
        if next_ang - ang < 1e-6:
            continue

        theta = 0.5 * (ang + next_ang)

        best_d = None
        best_i = None

        for i2 in active:
            d = ray_segment_distance(P, theta, walls[i2][0], walls[i2][1])
            if d is not None and (best_d is None or d < best_d):
                best_d = d
                best_i = i2

        if best_i is not None:
            visible.add(best_i)

    return list(visible)

# ================= workers =================

def worker_point_task(args):
    y, x, walls = args
    P = (x, y)
    vis = visible_walls_for_point(P, walls)
    return y, x, vis


def worker_wall_task(args):
    wall_id, A, B, walls, WWVmove = args

    moveA, moveB = WWVmove[wall_id]

    vis0 = []
    vis1 = []

    if moveA is not None:
        P0 = (A[0] + ENDPOINT_EPS * moveA[0],
              A[1] + ENDPOINT_EPS * moveA[1])
        vis0 = visible_walls_for_point(P0, walls)

    if moveB is not None:
        P1 = (B[0] + ENDPOINT_EPS * moveB[0],
              B[1] + ENDPOINT_EPS * moveB[1])
        vis1 = visible_walls_for_point(P1, walls)

    vis = set(vis0) | set(vis1)
    vis.discard(wall_id)

    return wall_id, vis

# ================= 主程序 =================

def main():

    print("[Load] geo + walls + moves")

    geo   = np.load(GEO_PATH)
    walls = np.load(WALL_PATH, allow_pickle=True).tolist()

    move_arr = np.load(MOVE_PATH, allow_pickle=True)
    WWVmove = move_arr.item() if move_arr.shape == () else move_arr

    H, W = geo.shape
    assert H == GRID_SIZE and W == GRID_SIZE

    free_mask = (geo == 0)
    free_points = np.argwhere(free_mask)
    total_free = len(free_points)

    N_walls = len(walls)

    print(f"[Info] Walls: {N_walls}")
    print(f"[Info] Free points: {total_free}")

    visible_walls = np.empty((H, W), dtype=object)
    visible_walls[:] = None

    os.makedirs(PVW_OUT_DIR, exist_ok=True)
    os.makedirs(WWV_OUT_DIR, exist_ok=True)

    cpu_cnt = mp.cpu_count()
    print(f"[Mode] {'Multiprocess' if USE_MULTIPROCESS else 'Single core'} | CPU cores = {cpu_cnt}")

    # ===== PVW =====

    print("[Build] PVW")

    if not USE_MULTIPROCESS:
        for (y, x) in tqdm(free_points, total=total_free, desc="PVW"):
            vis = visible_walls_for_point((int(x), int(y)), walls)
            visible_walls[y, x] = vis
    else:
        tasks = [(int(y), int(x), walls) for (y, x) in free_points]
        with mp.Pool(cpu_cnt) as pool:
            for y, x, vis in tqdm(pool.imap_unordered(worker_point_task, tasks),
                                   total=len(tasks),
                                   desc="PVW"):
                visible_walls[y, x] = vis

    np.save(PVW_OUT_PATH, visible_walls)
    print(f"[Done] PVW saved to {PVW_OUT_PATH}")

    # ===== WWV =====

    print("[Build] WWV (bool matrix)")

    WWV = np.zeros((N_walls, N_walls), dtype=bool)

    wall_tasks = [
        (i, walls[i][0], walls[i][1], walls, WWVmove)
        for i in range(N_walls)
    ]

    if not USE_MULTIPROCESS:
        for task in tqdm(wall_tasks, desc="WWV"):
            wid, vis = worker_wall_task(task)
            for j in vis:
                WWV[wid, j] = True
                WWV[j, wid] = True
    else:
        with mp.Pool(cpu_cnt) as pool:
            for wid, vis in tqdm(pool.imap_unordered(worker_wall_task, wall_tasks),
                                  total=len(wall_tasks),
                                  desc="WWV"):
                for j in vis:
                    WWV[wid, j] = True
                    WWV[j, wid] = True

    np.save(WWV_OUT_PATH, WWV)

    print(f"[Done] WWV saved to {WWV_OUT_PATH}")
    print(f"[Info] WWV shape: {WWV.shape}, dtype={WWV.dtype}")

# ================= 入口 =================

if __name__ == "__main__":
    mp.freeze_support()
    main()
