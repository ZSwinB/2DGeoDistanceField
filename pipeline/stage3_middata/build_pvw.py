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
walls_global = None

def init_worker(walls):
    global walls_global
    walls_global = walls

def worker_chunk(points_chunk):
    out = []
    for y, x in points_chunk:
        mask = visible_walls_for_point_nb((x, y), walls_global)
        vis = np.where(mask)[0].tolist()
        out.append((y, x, vis))
    return out

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

@njit
def visible_walls_for_point_nb(P, walls):

    px = P[0]
    py = P[1]

    num_walls = walls.shape[0]
    max_events = num_walls * 4

    events_angle = np.empty(max_events, dtype=np.float64)
    events_idx   = np.empty(max_events, dtype=np.int32)

    cnt = 0

    # ===== 构建 events =====
    for i in range(num_walls):

        ax = walls[i, 0]
        ay = walls[i, 1]
        bx = walls[i, 2]
        by = walls[i, 3]

        a1 = math.atan2(ay - py, ax - px)
        a2 = math.atan2(by - py, bx - px)

        if a1 < a2:
            l = a1
            r = a2
        else:
            l = a2
            r = a1

        if r - l > math.pi + 1e-9:
            events_angle[cnt] = l
            events_idx[cnt] = i
            cnt += 1

            events_angle[cnt] = math.pi
            events_idx[cnt] = i
            cnt += 1

            events_angle[cnt] = -math.pi
            events_idx[cnt] = i
            cnt += 1

            events_angle[cnt] = r
            events_idx[cnt] = i
            cnt += 1
        else:
            events_angle[cnt] = l
            events_idx[cnt] = i
            cnt += 1

            events_angle[cnt] = r
            events_idx[cnt] = i
            cnt += 1

    # ===== 排序（稳定 + 二级键 idx）=====
    order = np.arange(cnt)

    for i in range(1, cnt):
        key = order[i]
        j = i - 1

        while j >= 0:
            a1 = events_angle[order[j]]
            a2 = events_angle[key]

            if a1 > a2:
                order[j + 1] = order[j]
            elif abs(a1 - a2) < 1e-12:
                if events_idx[order[j]] > events_idx[key]:
                    order[j + 1] = order[j]
                else:
                    break
            else:
                break

            j -= 1

        order[j + 1] = key

    # ===== active / visible =====
    active  = np.zeros(num_walls, dtype=np.uint8)
    visible = np.zeros(num_walls, dtype=np.uint8)

    # ===== 扫描 =====
    for k in range(cnt - 1):

        idx = events_idx[order[k]]

        # toggle
        if active[idx]:
            active[idx] = 0
        else:
            active[idx] = 1

        ang = events_angle[order[k]]
        next_ang = events_angle[order[k + 1]]

        if next_ang - ang < 1e-6:
            continue

        theta = 0.5 * (ang + next_ang)

        best_d = 1e18
        best_i = -1

        for i2 in range(num_walls):

            if active[i2]:

                ax = walls[i2, 0]
                ay = walls[i2, 1]
                bx = walls[i2, 2]
                by = walls[i2, 3]

                d = ray_segment_distance_nb(px, py, theta, ax, ay, bx, by)

                if d > 1e-9 and d < best_d:
                    best_d = d
                    best_i = i2

        if best_i != -1:
            visible[best_i] = 1

    return visible

# ================= worker =================

def worker_point_task(args):
    y, x, walls = args
    P = (x, y)
    mask = visible_walls_for_point_nb(P, walls)
    vis = np.where(mask)[0].tolist()
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

        with mp.Pool(n, initializer=init_worker, initargs=(walls_nb,)) as pool:

            # ===== chunk 切分 =====
            n_chunks = n * 4
            chunks = np.array_split(free_points, n_chunks)

            # ===== 并行 =====
            pbar = tqdm(total=len(free_points), desc="PVW")

            for results in pool.imap_unordered(worker_chunk, chunks):
                for y, x, vis in results:
                    visible_walls[y, x] = vis
                pbar.update(len(results))

            pbar.close()

    else:

        print("[PVW] single process")

        for args in tqdm(tasks, total=len(tasks), desc="PVW"):
            y, x, vis = worker_point_task(args)
            visible_walls[y, x] = vis

    # ===== 保存 =====
    save_npy(out_path, visible_walls)

    print(f"[Done] PVW saved to {out_path}")

    return None