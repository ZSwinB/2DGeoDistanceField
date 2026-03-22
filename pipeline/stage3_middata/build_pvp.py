import numpy as np
import os
from tqdm import tqdm
import multiprocessing as mp
from numba import njit
from utils.io import save_npy, load_npy


# =======================
# 参数
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


@njit
def visible_fast_nb(
    x0, y0,
    x1, y1,
    walls,
    grid_start,
    grid_count,
    grid_list,
    skip_w
):

    gx0 = int(min(x0, x1) // GRID_SIZE)
    gx1 = int(max(x0, x1) // GRID_SIZE)
    gy0 = int(min(y0, y1) // GRID_SIZE)
    gy1 = int(max(y0, y1) // GRID_SIZE)

    for gx in range(gx0, gx1 + 1):
        for gy in range(gy0, gy1 + 1):

            start = grid_start[gx, gy]
            count = grid_count[gx, gy]

            for k in range(start, start + count):

                idx = grid_list[k]

                if skip_w != -1 and idx == skip_w:
                    continue

                xA = walls[idx, 0]
                yA = walls[idx, 1]
                xB = walls[idx, 2]
                yB = walls[idx, 3]

                if segment_intersect_strict_nb(
                    x0, y0,
                    x1, y1,
                    xA, yA,
                    xB, yB
                ):
                    return False

    return True


# =======================
# grid（numba版本）
# =======================

def build_wall_grid_nb(walls, H, W):

    GX = (W + GRID_SIZE) // GRID_SIZE
    GY = (H + GRID_SIZE) // GRID_SIZE

    grid_count = np.zeros((GX, GY), dtype=np.int32)

    for idx in range(walls.shape[0]):

        x0 = walls[idx, 0]
        y0 = walls[idx, 1]
        x1 = walls[idx, 2]
        y1 = walls[idx, 3]

        gx0 = max(0, int(min(x0,x1)//GRID_SIZE))
        gx1 = min(GX-1, int(max(x0,x1)//GRID_SIZE))
        gy0 = max(0, int(min(y0,y1)//GRID_SIZE))
        gy1 = min(GY-1, int(max(y0,y1)//GRID_SIZE))

        for gx in range(gx0,gx1+1):
            for gy in range(gy0,gy1+1):
                grid_count[gx,gy]+=1

    grid_start = np.zeros((GX,GY),dtype=np.int32)

    total=0
    for gx in range(GX):
        for gy in range(GY):
            grid_start[gx,gy]=total
            total+=grid_count[gx,gy]

    grid_list = np.zeros(total,dtype=np.int32)
    cursor = grid_start.copy()

    for idx in range(walls.shape[0]):

        x0 = walls[idx, 0]
        y0 = walls[idx, 1]
        x1 = walls[idx, 2]
        y1 = walls[idx, 3]

        gx0 = max(0, int(min(x0,x1)//GRID_SIZE))
        gx1 = min(GX-1, int(max(x0,x1)//GRID_SIZE))
        gy0 = max(0, int(min(y0,y1)//GRID_SIZE))
        gy1 = min(GY-1, int(max(y0,y1)//GRID_SIZE))

        for gx in range(gx0,gx1+1):
            for gy in range(gy0,gy1+1):

                k = cursor[gx,gy]
                grid_list[k] = idx
                cursor[gx,gy] += 1

    return grid_start, grid_count, grid_list


# =======================
# 多进程 worker
# =======================

def init_worker_pvp(walls_, PVW_, grid_start_, grid_count_, grid_list_):
    global walls_global, PVW_global
    global grid_start_global, grid_count_global, grid_list_global

    walls_global = walls_
    PVW_global = PVW_
    grid_start_global = grid_start_
    grid_count_global = grid_count_
    grid_list_global = grid_list_


def worker_chunk_pvp(points_chunk):
    out = []

    for y, x in points_chunk:

        vis_walls = PVW_global[y, x]

        if not vis_walls:
            out.append((y, x, []))
            continue

        px = float(x)
        py = float(y)

        corners = []
        seen = set()

        for wid in vis_walls:

            ax = walls_global[wid, 0]
            ay = walls_global[wid, 1]
            bx = walls_global[wid, 2]
            by = walls_global[wid, 3]

            if (ax, ay) not in seen:
                if visible_fast_nb(
                    px, py,
                    ax, ay,
                    walls_global,
                    grid_start_global,
                    grid_count_global,
                    grid_list_global,
                    wid
                ):
                    corners.append((ax, ay))
                    seen.add((ax, ay))

            if (bx, by) not in seen:
                if visible_fast_nb(
                    px, py,
                    bx, by,
                    walls_global,
                    grid_start_global,
                    grid_count_global,
                    grid_list_global,
                    wid
                ):
                    corners.append((bx, by))
                    seen.add((bx, by))

        out.append((y, x, corners))

    return out


# =======================
# stage接口
# =======================

def build_pvp(geo, wall_segment, pvw, config):
    """
    输入:
        geo:
            2D ndarray (H, W)
            0 表示 free，其它表示障碍

        wall_segment:
            不使用（已废弃，保留接口兼容）
            实际从磁盘读取:
                convert/{idx}/walls_nb.npy
            格式:
                (N, 4) → [ax, ay, bx, by]

        pvw:
            不使用（已废弃，保留接口兼容）
            实际从磁盘读取:
                PVW/{idx}.npy
            格式:
                (H, W) object array
                每个元素: list[int]（可见墙索引）

        config:
            需要字段:
                IDX
                OUTPUT_ROOT
                PVW_DIR
                PVP_DIR
                USE_INNER_MP
                INNER_WORKERS

    内部构建:
        grid (numba CSR形式):
            grid_start: (GX, GY)
            grid_count: (GX, GY)
            grid_list : (total,)

    输出:
        PVP:
            (H, W) object array
            每个元素: list[(x, y)]
            表示该点可见的墙端点集合

        保存路径:
            OUTPUT_ROOT / PVP_DIR / {idx}.npy
    """
    idx = config.IDX

    wall_path = os.path.join(
        config.OUTPUT_ROOT,
        "convert",
        str(idx),
        "walls_nb.npy"
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

    print("[Load] geo + walls_nb + PVW")

    walls_nb = load_npy(wall_path)
    PVW = load_npy(pvw_path)

    H, W = geo.shape

    print("[Build] grid_nb")
    grid_start, grid_count, grid_list = build_wall_grid_nb(walls_nb, H, W)

    free_points = np.argwhere(geo == 0)

    PVP = np.empty((H, W), dtype=object)
    PVP[:] = None

    print("[Build] PVP")

    if config.USE_INNER_MP:

        n = config.INNER_WORKERS or mp.cpu_count()
        print(f"[PVP] inner multiprocess | workers={n}")

        with mp.Pool(
            n,
            initializer=init_worker_pvp,
            initargs=(
                walls_nb,
                PVW,
                grid_start,
                grid_count,
                grid_list
            )
        ) as pool:

            n_chunks = n * 4
            chunks = np.array_split(free_points, n_chunks)

            pbar = tqdm(total=len(free_points), desc="PVP")

            for results in pool.imap_unordered(worker_chunk_pvp, chunks):
                for y, x, corners in results:
                    PVP[y, x] = corners
                pbar.update(len(results))

            pbar.close()

    else:

        print("[PVP] single process")

        for y, x in tqdm(free_points, desc="PVP"):

            vis_walls = PVW[y, x]

            if not vis_walls:
                PVP[y, x] = []
                continue

            px = float(x)
            py = float(y)

            corners = []
            seen = set()

            for wid in vis_walls:

                ax = walls_nb[wid, 0]
                ay = walls_nb[wid, 1]
                bx = walls_nb[wid, 2]
                by = walls_nb[wid, 3]

                if (ax, ay) not in seen:
                    if visible_fast_nb(
                        px, py,
                        ax, ay,
                        walls_nb,
                        grid_start,
                        grid_count,
                        grid_list,
                        wid
                    ):
                        corners.append((ax, ay))
                        seen.add((ax, ay))

                if (bx, by) not in seen:
                    if visible_fast_nb(
                        px, py,
                        bx, by,
                        walls_nb,
                        grid_start,
                        grid_count,
                        grid_list,
                        wid
                    ):
                        corners.append((bx, by))
                        seen.add((bx, by))

            PVP[y, x] = corners

    save_npy(out_path, PVP)

    print(f"[Done] PVP saved to {out_path}")