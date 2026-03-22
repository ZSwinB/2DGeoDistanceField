import numpy as np
import math
import os
from numba import njit
from tqdm import tqdm
import multiprocessing as mp
from utils.io import save_npy, load_npy


# ================= 参数（保持原样） =================

GRID_SIZE = 256
EPS = 1e-9


# ================= 几何工具 =================

def angle(P, Q):
    return math.atan2(Q[1] - P[1], Q[0] - P[0])

@njit
def angle_xy(px, py, qx, qy):
    return math.atan2(qy - py, qx - px)

@njit
def ray_segment_distance_nb(px, py, theta, ax, ay, bx, by):

    dx = math.cos(theta)
    dy = math.sin(theta)

    vx = bx - ax
    vy = by - ay

    det = dx * (-vy) - dy * (-vx)

    if abs(det) >= 1e-9:

        t = ((ax - px) * (-vy) - (ay - py) * (-vx)) / det
        u = ((ax - px) * (-dy) + (ay - py) * (dx)) / det

        if t > 1e-9 and 0 <= u <= 1:
            return t

    return -1.0   # ❗ 用 -1 代替 None（numba 不支持 None）


# ================= 单点可见性 =================

def visible_walls_for_point_nb(P, walls):

    px, py = P

    events = []

    for i in range(walls.shape[0]):

        ax = walls[i, 0]
        ay = walls[i, 1]
        bx = walls[i, 2]
        by = walls[i, 3]

        a1 = angle_xy(px, py, ax, ay)
        a2 = angle_xy(px, py, bx, by)

        l = min(a1, a2)
        r = max(a1, a2)

        if r - l > math.pi + 1e-9:
            events.append((l, i))
            events.append(( math.pi, i))
            events.append((-math.pi, i))
            events.append((r, i))
        else:
            events.append((l, i))
            events.append((r, i))

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

        best_d = 1e18
        best_i = -1

        for i2 in active:

            ax = walls[i2, 0]
            ay = walls[i2, 1]
            bx = walls[i2, 2]
            by = walls[i2, 3]

            d = ray_segment_distance_nb(px, py, theta, ax, ay, bx, by)

            if d > 1e-9 and d < best_d:
                best_d = d
                best_i = i2

        if best_i != -1:
            visible.add(best_i)

    return list(visible)

# ================= worker =================

def worker_point_task(args):
    y, x, walls = args
    P = (x, y)
    vis = visible_walls_for_point_nb(P, walls)
    return y, x, vis


# ================= stage接口 =================

def build_pvw(geo, wall_segment, config):
    """
    输入:
        geo
        wall_segment（不使用，从磁盘读取）

    输出:
        PVW（落盘）
    """

    idx = config.IDX

    out_path = os.path.join(
        config.OUTPUT_ROOT,
        config.PVW_DIR,
        f"{idx}.npy"
    )

    print("[Load] walls")

    walls_nb = load_npy(
        os.path.join(config.OUTPUT_ROOT, "convert", str(idx), "walls_nb.npy")
    )
    H, W = geo.shape
    assert H == GRID_SIZE and W == GRID_SIZE

    free_mask = (geo == 0)
    free_points = np.argwhere(free_mask)

    total_free = len(free_points)

    print(f"[Info] Free points: {total_free}")

    # ===== PVW =====
    visible_walls = np.empty((H, W), dtype=object)
    visible_walls[:] = None

    tasks = [(int(y), int(x), walls_nb) for (y, x) in free_points]

    print("[Build] PVW")

    if config.USE_INNER_MP:

        n = config.INNER_WORKERS or mp.cpu_count()
        print(f"[PVW] inner multiprocess | workers={n}")

        with mp.Pool(n) as pool:
            for y, x, vis in tqdm(
                pool.imap_unordered(worker_point_task, tasks),
                total=len(tasks),
                desc="PVW"
            ):
                visible_walls[y, x] = vis

    else:

        print("[PVW] single process")

        for args in tqdm(tasks, total=len(tasks), desc="PVW"):
            y, x, vis = worker_point_task(args)
            visible_walls[y, x] = vis

    # ===== 保存 =====
    save_npy(out_path, visible_walls)

    print(f"[Done] PVW saved to {out_path}")

    return None