import numpy as np
import math
import os
from tqdm import tqdm
import multiprocessing as mp
from utils.io import save_npy, load_npy


# ================= 参数（保留原常量，不改逻辑） =================

GRID_SIZE = 256
EPS = 1e-9
ENDPOINT_EPS = 1e-4


# ================= 8方向（来自 WWVmove） =================

DIRECTIONS_8 = [
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1),
]


# ================= WWVmove 工具 =================

def in_bounds(x, y, W, H):
    return 0 <= x < W and 0 <= y < H


def first_free_dir(P, geo):
    H, W = geo.shape
    x0, y0 = int(round(P[0])), int(round(P[1]))

    for dx, dy in DIRECTIONS_8:
        x = x0 + dx
        y = y0 + dy
        if not in_bounds(x, y, W, H):
            continue
        if geo[y, x] == 0:
            return (dx, dy)

    return None


def build_WWVmove(geo, walls):
    """
    WWVmove[wid] = [dir_for_A, dir_for_B]
    """
    WWVmove = {}

    for wid, (A, B) in enumerate(walls):
        dir_A = first_free_dir(A, geo)
        dir_B = first_free_dir(B, geo)

        WWVmove[wid] = [dir_A, dir_B]

    return WWVmove


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


# ================= 可见性 =================

def visible_walls_for_point(P, walls):
    events = []

    for i, (A, B) in enumerate(walls):
        a1 = angle(P, A)
        a2 = angle(P, B)

        l = min(a1, a2)
        r = max(a1, a2)

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


# ================= worker =================

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


# ================= stage 接口 =================

def build_wwv(geo, wall_segment, config):
    """
    输入:
        geo
        wall_segment（不使用，从磁盘读取）

    输出:
        WWV（落盘）
    """

    idx = config.IDX

    wall_path = os.path.join(
        config.OUTPUT_ROOT,
        config.WALL_DIR,
        f"{idx}.npy"
    )

    out_path = os.path.join(
        config.OUTPUT_ROOT,
        config.WWV_DIR,
        f"{idx}.npy"
    )

    print("[Load] walls")

    walls = load_npy(wall_path).tolist()
    N_walls = len(walls)

    print(f"[Info] Walls: {N_walls}")

    # ===== WWVmove（内部计算，不落盘）=====
    WWVmove = build_WWVmove(geo, walls)

    # ===== WWV =====
    WWV = np.zeros((N_walls, N_walls), dtype=bool)

    wall_tasks = [
        (i, walls[i][0], walls[i][1], walls, WWVmove)
        for i in range(N_walls)
    ]

    print("[Build] WWV")

    with mp.Pool(mp.cpu_count()) as pool:
        for wid, vis in tqdm(pool.imap_unordered(worker_wall_task, wall_tasks),
                             total=len(wall_tasks),
                             desc="WWV"):
            for j in vis:
                WWV[wid, j] = True
                WWV[j, wid] = True

    # ===== 保存 =====
    save_npy(out_path, WWV)

    print(f"[Done] WWV saved to {out_path}")
    print(f"[Info] shape={WWV.shape}, dtype={WWV.dtype}")

    return None