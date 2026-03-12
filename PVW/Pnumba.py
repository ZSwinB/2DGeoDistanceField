import numpy as np
import math
import os
from tqdm import tqdm
import multiprocessing as mp

# ================= 配置 =================

GEO_ROOT   = "/root/RM/geo"
WALL_ROOT  = "/root/wall_segment_DRM"
MOVE_ROOT  = "/root/RM/WWVmove"

PVW_OUT_DIR = "/root/RM/data/PVW"
WWV_OUT_DIR = "/root/RM/data/WWV"

GRID_SIZE = 256
EPS = 1e-9
ENDPOINT_EPS = 1e-4

# ================= 几何 =================

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


# ================= 可见墙 =================

def visible_walls_for_point(P, walls):

    events = []

    for i, (A, B) in enumerate(walls):

        a1 = angle(P, A)
        a2 = angle(P, B)

        l = min(a1, a2)
        r = max(a1, a2)

        if r - l > math.pi:
            events += [(l, i), ( math.pi, i), (-math.pi, i), (r, i)]
        else:
            events += [(l, i), (r, i)]

    events.sort()

    active = set()
    visible = set()

    for k in range(len(events) - 1):

        ang, idx = events[k]

        if idx in active:
            active.remove(idx)
        else:
            active.add(idx)

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


# ================= Scene 计算 =================

def run_scene(SCENE_ID):

    GEO_PATH  = f"{GEO_ROOT}/{SCENE_ID}.npy"
    WALL_PATH = f"{WALL_ROOT}/{SCENE_ID}.npy"
    MOVE_PATH = f"{MOVE_ROOT}/{SCENE_ID}.npy"

    PVW_OUT_PATH = f"{PVW_OUT_DIR}/{SCENE_ID}.npy"
    WWV_OUT_PATH = f"{WWV_OUT_DIR}/{SCENE_ID}.npy"

    if os.path.exists(PVW_OUT_PATH) and os.path.exists(WWV_OUT_PATH):
        print(f"[Skip] Scene {SCENE_ID}")
        return

    if not (os.path.exists(GEO_PATH)
            and os.path.exists(WALL_PATH)
            and os.path.exists(MOVE_PATH)):
        print(f"[Missing] Scene {SCENE_ID}")
        return

    print(f"\n===== Scene {SCENE_ID} =====")

    geo   = np.load(GEO_PATH)
    walls = np.load(WALL_PATH, allow_pickle=True).tolist()

    move_arr = np.load(MOVE_PATH, allow_pickle=True)
    WWVmove = move_arr.item() if move_arr.shape == () else move_arr

    H, W = geo.shape

    free_points = np.argwhere(geo == 0)

    N_walls = len(walls)

    visible_walls = np.empty((H, W), dtype=object)
    visible_walls[:] = None

    os.makedirs(PVW_OUT_DIR, exist_ok=True)
    os.makedirs(WWV_OUT_DIR, exist_ok=True)

    # ================= PVW =================

    for (y, x) in free_points:

        visible_walls[y, x] = visible_walls_for_point((x, y), walls)

    np.save(PVW_OUT_PATH, visible_walls)

    print("PVW done")


    # ================= WWV =================

    WWV = np.zeros((N_walls, N_walls), dtype=bool)

    for wall_id in range(N_walls):

        A, B = walls[wall_id]

        moveA, moveB = WWVmove[wall_id]

        vis = set()

        if moveA is not None:

            P0 = (
                A[0] + ENDPOINT_EPS * moveA[0],
                A[1] + ENDPOINT_EPS * moveA[1]
            )

            vis |= set(visible_walls_for_point(P0, walls))

        if moveB is not None:

            P1 = (
                B[0] + ENDPOINT_EPS * moveB[0],
                B[1] + ENDPOINT_EPS * moveB[1]
            )

            vis |= set(visible_walls_for_point(P1, walls))

        vis.discard(wall_id)

        for j in vis:

            WWV[wall_id, j] = True
            WWV[j, wall_id] = True

    np.save(WWV_OUT_PATH, WWV)

    print("WWV done")


# ================= 主入口 =================

if __name__ == "__main__":

    mp.freeze_support()

    scene_ids = list(range(701))

    cpu_cnt = mp.cpu_count()

    print("CPU:", cpu_cnt)

    with mp.Pool(cpu_cnt) as pool:

        list(tqdm(pool.imap_unordered(run_scene, scene_ids),
                  total=len(scene_ids)))