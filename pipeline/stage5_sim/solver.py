import numpy as np
from PIL import Image
import math
import os
import time
from collections import defaultdict
import random
import sys
from numba import njit
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT not in sys.path:
    sys.path.append(ROOT)
from config import Config
from pipeline.stage1_input.load_geo import load_geo
import sys

# =====================================================
# 参数
# =====================================================
MAX_DIST = 1000.0
config = Config()

K = config.K #在config里面改
DIFF_BIAS = config.DIFF_BIAS
FALLBACK_BIAS = config.FALLBACK_BIAS
# ===== 输入 =====

# ===== 中间数据 =====
MID_ROOT = "outputs/middata"

# ===== convert =====
CONVERT_ROOT = "outputs/convert"

# ===== 输出 =====
OUT_DIR = "outputs/sim"

GRID_SIZE = 8

# ===== CLI =====
IDX = int(sys.argv[1])
TX_ID = int(sys.argv[2])

NUM_RX = 5000


# =====================================================
# 基础几何（不改）
# =====================================================
def build_wall_grid(walls):
    """
    Build grid index (dict-based) for wall segments.

    Behavior:
        - Divides space into GRID_SIZE cells
        - Maps each grid cell to wall indices overlapping it

    Returns:
        dict:
            {(gx, gy): [wall_idx, ...]}
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


def reflect_point(p, wall):

    p = np.array(p, float)
    a = np.array(wall[0], float)
    b = np.array(wall[1], float)
    v = b - a
    v = v / np.linalg.norm(v)
    proj = a + v * np.dot(p - a, v)

    return (2 * proj - p).tolist()


def orient(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def on_segment(a, b, c, eps=1e-9):
    return (
        min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps and
        min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps
    )


def segment_intersect_strict(a, b, c, d):
    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return (o1 * o2 < 0 and o3 * o4 < 0)


def segment_intersect2(a, b, c, d, eps=1):

    a = np.array(a, float)
    b = np.array(b, float)
    c = np.array(c, float)
    d = np.array(d, float)

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    if o1 * o2 < 0 and o3 * o4 < 0:
        return True

    if abs(o1) < eps and on_segment(a, b, c, eps): return True
    if abs(o2) < eps and on_segment(a, b, d, eps): return True
    if abs(o3) < eps and on_segment(c, d, a, eps): return True
    if abs(o4) < eps and on_segment(c, d, b, eps): return True

    return False



def visible_fast(p1, p2, walls, wall_grid, skip_w=None):
    """
    Fast visibility check using grid acceleration.

    Behavior:
        - Traverses relevant grid cells
        - Tests segment intersection against candidate walls

    Returns:
        bool: True if visible, False if blocked
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
                if skip_w is not None and idx == skip_w:
                    continue
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


# =====================================================
# numba
# =====================================================


def build_wall_grid_nb(walls, H, W):
    """
    Build numba-compatible grid (CSR-like).

    Returns:
        grid_start, grid_count, grid_list
    """

    GX = int(np.ceil((W + 1) / GRID_SIZE))
    GY = int(np.ceil((H + 1) / GRID_SIZE))

    grid_count = np.zeros((GX, GY), dtype=np.int32)

    # 第一遍：统计数量
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


    # 前缀和
    grid_start = np.zeros((GX,GY),dtype=np.int32)

    total=0
    for gx in range(GX):
        for gy in range(GY):
            grid_start[gx,gy]=total
            total+=grid_count[gx,gy]


    grid_list = np.zeros(total,dtype=np.int32)

    cursor = grid_start.copy()

    # 第二遍：填墙
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

                k=cursor[gx,gy]
                grid_list[k]=idx
                cursor[gx,gy]+=1


    return grid_start, grid_count, grid_list


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
    """
    Numba-accelerated visibility check.

    Returns:
        bool: True if visible, False otherwise
    """
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

@njit
def segment_intersect_strict_nb(
    ax, ay,
    bx, by,
    cx, cy,
    dx, dy
):
    o1 = orient_nb(ax, ay, bx, by, cx, cy)
    o2 = orient_nb(ax, ay, bx, by, dx, dy)
    o3 = orient_nb(cx, cy, dx, dy, ax, ay)
    o4 = orient_nb(cx, cy, dx, dy, bx, by)

    return (o1 * o2 < 0.0 and o3 * o4 < 0.0)
@njit
def orient_nb(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)



@njit
def on_segment_nb(ax, ay, bx, by, cx, cy, eps):
    return (
        min(ax, bx) - eps <= cx <= max(ax, bx) + eps and
        min(ay, by) - eps <= cy <= max(ay, by) + eps
    )


@njit
def segment_intersect2_nb(
    ax, ay,
    bx, by,
    cx, cy,
    dx, dy,
    eps
):
    o1 = orient_nb(ax, ay, bx, by, cx, cy)
    o2 = orient_nb(ax, ay, bx, by, dx, dy)
    o3 = orient_nb(cx, cy, dx, dy, ax, ay)
    o4 = orient_nb(cx, cy, dx, dy, bx, by)

    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return True

    if abs(o1) < eps and on_segment_nb(ax, ay, bx, by, cx, cy, eps):
        return True
    if abs(o2) < eps and on_segment_nb(ax, ay, bx, by, dx, dy, eps):
        return True
    if abs(o3) < eps and on_segment_nb(cx, cy, dx, dy, ax, ay, eps):
        return True
    if abs(o4) < eps and on_segment_nb(cx, cy, dx, dy, bx, by, eps):
        return True

    return False


@njit
def dist2d_nb(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx*dx + dy*dy)

@njit
def valid_first_reflection_nb(
    tx_x, tx_y,
    rx_x, rx_y,
    ax, ay,
    bx, by
):
    return segment_intersect2_nb(
        tx_x, tx_y,
        rx_x, rx_y,
        ax, ay,
        bx, by,
        1e-9
    )


@njit
def second_reflection_dist_nb(
    tx_x, tx_y,
    rx_x, rx_y,
    w1x1, w1y1, w1x2, w1y2,
    w2x1, w2y1, w2x2, w2y2,
    best,
    eps
):
    """
    Compute second-order reflection path distance with pruning.

    Returns:
        float: updated best distance
    """
    # ---------- TX 侧两次反射 ----------
    tx1x, tx1y = reflect_point_nb(tx_x, tx_y, w1x1, w1y1, w1x2, w1y2)
    tx2x, tx2y = reflect_point_nb(tx1x, tx1y, w2x1, w2y1, w2x2, w2y2)

    # ---------- RX 侧两次反射 ----------
    rx1x, rx1y = reflect_point_nb(rx_x, rx_y, w2x1, w2y1, w2x2, w2y2)
    rx2x, rx2y = reflect_point_nb(rx1x, rx1y, w1x1, w1y1, w1x2, w1y2)


    # ---------- 距离（真实路径） ----------
    d = dist2d_nb(tx2x, tx2y, rx_x, rx_y)
    if d >= best:
        return best   # 剪枝

    # ---------- 几何可行性 ----------
    if not segment_intersect2_nb(
        tx2x, tx2y,
        rx_x, rx_y,
        w2x1, w2y1, w2x2, w2y2,
        eps
    ):
        return best

    if not segment_intersect2_nb(
        rx2x, rx2y,
        tx_x, tx_y,
        w1x1, w1y1, w1x2, w1y2,
        eps
    ):
        return best

    return d


@njit
def reflect_point_nb(px, py, ax, ay, bx, by):
    """
    Reflect a point across a wall segment.

    Returns:
        tuple[float, float]: reflected point
    """
    # wall direction
    vx = bx - ax
    vy = by - ay

    # normalize v
    norm = math.sqrt(vx*vx + vy*vy)
    vx /= norm
    vy /= norm

    # vector from a to p
    apx = px - ax
    apy = py - ay

    # projection length
    t = apx * vx + apy * vy

    # projection point
    projx = ax + vx * t
    projy = ay + vy * t

    # reflection
    rx = 2.0 * projx - px
    ry = 2.0 * projy - py

    return rx, ry


@njit
def intersect_wall_return_point_nb(
    px, py,   # P：镜像点
    qx, qy,   # Q：corner
    ax, ay,   # A：墙端点
    bx, by,   # B：墙端点
    eps=1e-9
):
    # PQ = P + t * d
    dx = qx - px
    dy = qy - py

    # AB = A + u * e
    ex = bx - ax
    ey = by - ay

    denom = dx * ey - dy * ex
    if abs(denom) < eps:
        return False, 0.0, 0.0   # 平行或共线，不算穿墙

    t = ((ax - px) * ey - (ay - py) * ex) / denom
    u = ((ax - px) * dy - (ay - py) * dx) / denom

    # u ∈ [0,1] → 交点在墙段内
    # t > 0     → 在 P → Q 的射线方向
    if u < -eps or u > 1.0 + eps or t < eps or t > 1.0 + eps:
        return False, 0.0, 0.0

    rx = px + t * dx
    ry = py + t * dy
    return True, rx, ry



@njit
def diffraction_2nd_mirror_nb(
    tx_x, tx_y,
    rx_x, rx_y,
    TX_mask, RX_mask,
    TX_wall_order,
    walls_nb,
    grid_start, grid_count, grid_list,
    PVP_start, PVP_len, PVP_flat,
    corner_x, corner_y,
    best
):
    """
    Compute diffraction combined with reflection (mirror method).

    Returns:
        np.ndarray: updated best distances
    """

    K = best.shape[0]

    # ------------------------------------------------
    # 进入 reference space
    # ------------------------------------------------
    for k in range(K):
        best[k] -= DIFF_BIAS

    start_tx = PVP_start[tx_y, tx_x]
    len_tx   = PVP_len[tx_y, tx_x]

    start_rx = PVP_start[rx_y, rx_x]
    len_rx   = PVP_len[rx_y, rx_x]

    if len_tx == 0 and len_rx == 0:

        for k in range(K):
            best[k] += DIFF_BIAS

        return best

    # =================================================
    # 情况1：TX镜像 + 绕射
    # =================================================
    for wi in range(len(TX_wall_order)):

        w = TX_wall_order[wi]

        if not TX_mask[w]:
            continue

        ax = walls_nb[w,0]
        ay = walls_nb[w,1]
        bx = walls_nb[w,2]
        by = walls_nb[w,3]

        tx_img_x, tx_img_y = reflect_point_nb(tx_x, tx_y, ax, ay, bx, by)

        dx = tx_img_x - rx_x
        dy = tx_img_y - rx_y
        if math.sqrt(dx*dx + dy*dy) >= best[K-1]:
            continue

        for k in range(len_rx):

            cid = PVP_flat[start_rx + k]

            cx = corner_x[cid]
            cy = corner_y[cid]

            ok, rxp, ryp = intersect_wall_return_point_nb(
                tx_img_x, tx_img_y,
                cx, cy,
                ax, ay, bx, by
            )

            if not ok:
                continue

            if not visible_fast_nb(
                rxp, ryp,
                cx, cy,
                walls_nb,
                grid_start,
                grid_count,
                grid_list,
                w
            ):
                continue

            dx1 = tx_x - rxp
            dy1 = tx_y - ryp

            dx2 = rxp - cx
            dy2 = ryp - cy

            dx3 = cx - rx_x
            dy3 = cy - rx_y

            d = (
                math.sqrt(dx1*dx1 + dy1*dy1) +
                math.sqrt(dx2*dx2 + dy2*dy2) +
                math.sqrt(dx3*dx3 + dy3*dy3)
            )

            # -----------------------
            # top-K 插入
            # -----------------------
            if d < best[K-1]:

                pos = K - 1
                best[pos] = d

                while pos > 0 and best[pos] < best[pos-1]:

                    tmp = best[pos]
                    best[pos] = best[pos-1]
                    best[pos-1] = tmp

                    pos -= 1


    # =================================================
    # 情况2：RX镜像 + 绕射
    # =================================================
    for w in range(len(RX_mask)):

        if not RX_mask[w]:
            continue

        ax = walls_nb[w,0]
        ay = walls_nb[w,1]
        bx = walls_nb[w,2]
        by = walls_nb[w,3]

        rx_img_x, rx_img_y = reflect_point_nb(rx_x, rx_y, ax, ay, bx, by)

        dx = rx_img_x - tx_x
        dy = rx_img_y - tx_y
        if math.sqrt(dx*dx + dy*dy) >= best[K-1]:
            continue

        for k in range(len_tx):

            cid = PVP_flat[start_tx + k]

            cx = corner_x[cid]
            cy = corner_y[cid]

            ok, rxp, ryp = intersect_wall_return_point_nb(
                rx_img_x, rx_img_y,
                cx, cy,
                ax, ay, bx, by
            )

            if not ok:
                continue

            if not visible_fast_nb(
                rxp, ryp,
                cx, cy,
                walls_nb,
                grid_start,
                grid_count,
                grid_list,
                w
            ):
                continue

            dx1 = tx_x - cx
            dy1 = tx_y - cy

            dx2 = cx - rxp
            dy2 = cy - ryp

            dx3 = rxp - rx_x
            dy3 = ryp - rx_y

            d = (
                math.sqrt(dx1*dx1 + dy1*dy1) +
                math.sqrt(dx2*dx2 + dy2*dy2) +
                math.sqrt(dx3*dx3 + dy3*dy3)
            )

            if d < best[K-1]:

                pos = K - 1
                best[pos] = d

                while pos > 0 and best[pos] < best[pos-1]:

                    tmp = best[pos]
                    best[pos] = best[pos-1]
                    best[pos-1] = tmp

                    pos -= 1


    # ------------------------------------------------
    # 恢复 penalty
    # ------------------------------------------------
    for k in range(K):
        best[k] += DIFF_BIAS

    return best


@njit
def diffraction_toa_nb(
    tx_x, tx_y,
    rx_x, rx_y,
    PVP_start, PVP_len, PVP_flat,
    corner_x, corner_y,
    best
):
    """
    Compute diffraction path (corner-based) with top-K update.

    Returns:
        np.ndarray: updated best distances
    """

    K = best.shape[0]

    # ------------------------------------------------
    # 进入函数：统一减 penalty（参考空间）
    # ------------------------------------------------
    for k in range(K):
        best[k] -= DIFF_BIAS

    start_tx = PVP_start[tx_y, tx_x]
    len_tx   = PVP_len[tx_y, tx_x]

    start_rx = PVP_start[rx_y, rx_x]
    len_rx   = PVP_len[rx_y, rx_x]

    if len_tx == 0 or len_rx == 0:

        # 恢复 penalty
        for k in range(K):
            best[k] += DIFF_BIAS

        return best

    # ------------------------------------------------
    # candidate 生成 + top-K 插入
    # ------------------------------------------------
    for i in range(len_tx):

        cid = PVP_flat[start_tx + i]

        for j in range(len_rx):

            if cid != PVP_flat[start_rx + j]:
                continue

            cx = corner_x[cid]
            cy = corner_y[cid]

            dx1 = tx_x - cx
            dy1 = tx_y - cy

            dx2 = cx - rx_x
            dy2 = cy - rx_y

            d = math.sqrt(dx1*dx1 + dy1*dy1) + math.sqrt(dx2*dx2 + dy2*dy2)

            # 剪枝
            if d >= best[K-1]:
                continue

            # insert_topk
            pos = K - 1
            best[pos] = d

            while pos > 0 and best[pos] < best[pos-1]:

                tmp = best[pos]
                best[pos] = best[pos-1]
                best[pos-1] = tmp

                pos -= 1

    # ------------------------------------------------
    # 退出函数：恢复 penalty
    # ------------------------------------------------
    for k in range(K):
        best[k] += DIFF_BIAS

    return best


@njit
def merge_topk(best, cand):

    K = len(best)
    M = len(cand)

    tmp = np.empty(K)

    i = 0
    j = 0
    k = 0

    while k < K:

        if j >= M or (i < K and best[i] <= cand[j]):
            tmp[k] = best[i]
            i += 1
        else:
            tmp[k] = cand[j]
            j += 1

        k += 1

    for t in range(K):
        best[t] = tmp[t]


@njit
def insert_topk(best, d):

    K = best.shape[0]

    if d >= best[K-1]:
        return

    i = K-1
    best[i] = d

    while i > 0 and best[i] < best[i-1]:

        tmp = best[i]
        best[i] = best[i-1]
        best[i-1] = tmp

        i -= 1



@njit
def solver_rx_loop(
    rx_list,
    tx_x, tx_y,
    tx_i, tx_j,
    dist_map,
    PVW_mask,
    TX_mask,
    TX_wall_order,
    walls_nb,
    grid_start, grid_count, grid_list,
    PVP_start, PVP_len, PVP_flat,
    corner_x, corner_y,
    WWV,
    tx_img,
    K,
    MAX_DIST
):
    """
    Core per-receiver solving loop (fully numba-accelerated).

    High-level:
        For each receiver (rx), compute top-K shortest propagation paths
        from transmitter (tx) under multiple mechanisms:
            - Direct path (LOS)
            - First-order reflection
            - Second-order reflection
            - Diffraction (corner-based)
            - Diffraction + reflection (hybrid)

    ------------------------------------------------
    Per-receiver workflow:
    ------------------------------------------------

    1. Initialization:
        - Initialize `best` array of size K with MAX_DIST
        - Insert a large penalized Euclidean distance (baseline upper bound)

    2. Line-of-Sight (LOS):
        - Check visibility using grid-accelerated intersection test
        - If visible, insert direct Euclidean distance into top-K

    3. Reflection (mirror method):
        For each wall w1 visible from TX:
            - Compute mirrored TX position (tx_img[w1])
            - Lower-bound pruning: skip if distance ≥ current worst best[K-1]

        3.1 First-order reflection:
            - If RX can see wall w1 (RX_mask[w1])
            - Validate reflection geometry (intersection on wall)
            - Insert distance if valid

        3.2 Second-order reflection:
            - Candidate walls w2 = RX_mask ∩ WWV[w1]
            - For each w2:
                - Reflect TX again across w2
                - Lower-bound pruning
                - Validate double-reflection geometry
                - Insert distance if valid

    4. Diffraction (corner-based):
        - Triggered only if (euclidean + penalty) < current worst
        - Uses shared visible corners (PVP) between TX and RX
        - Computes path: TX → corner → RX
        - Updates top-K

    5. Diffraction + reflection:
        - Also gated by same pruning condition
        - Two symmetric cases:
            (a) TX mirrored → diffraction → RX
            (b) RX mirrored → diffraction → TX
        - Uses wall visibility + corner visibility + grid occlusion checks
        - Updates top-K

    6. Write-back:
        - Store final sorted top-K distances into dist_map[i, j, :]

    ------------------------------------------------
    Key optimizations:
    ------------------------------------------------
    - Top-K maintenance via insertion sort (small K)
    - Strong lower-bound pruning before expensive geometry checks
    - Grid-based spatial acceleration for visibility
    - PVW_mask reduces candidate walls
    - WWV prunes second-order reflection pairs
    - PVP CSR structure enables fast corner lookup
    - All heavy loops run inside numba (no Python overhead)

    ------------------------------------------------
    Data dependencies:
    ------------------------------------------------
    - PVW_mask:
        Per-point visible walls (prunes reflection candidates)

    - WWV:
        Wall-to-wall visibility (prunes second reflection pairs)

    - PVP (CSR):
        Corner visibility for diffraction

    - grid_*:
        Spatial index for fast intersection tests

    ------------------------------------------------
    Returns:
        None (results written in-place to dist_map)
    """

    for n in range(rx_list.shape[0]):

        i = rx_list[n,0]
        j = rx_list[n,1]

        rx_x = j
        rx_y = i

        if i == tx_i and j == tx_j:
            for k in range(K):
                dist_map[i, j, k] = 0.0
            continue

        best = np.full(K, MAX_DIST)

        dx = tx_x - rx_x
        dy = tx_y - rx_y
        euclid = (dx*dx + dy*dy) ** 0.5

        insert_topk(best, euclid + FALLBACK_BIAS)

        # LOS
        if visible_fast_nb(
            tx_x, tx_y,
            rx_x, rx_y,
            walls_nb,
            grid_start,
            grid_count,
            grid_list,
            -1
        ):

            d = euclid
            insert_topk(best, d)

        RX_mask = PVW_mask[i,j]

        # ------------------------------------------------
        # reflection
        # ------------------------------------------------

        for w1 in TX_wall_order:

            if not TX_mask[w1]:
                continue

            tx1 = tx_img[w1]

            dx = tx1[0] - rx_x
            dy = tx1[1] - rx_y
            d = (dx*dx + dy*dy) ** 0.5

            if d < best[-1]:

                if RX_mask[w1]:

                    if valid_first_reflection_nb(
                        tx1[0], tx1[1],
                        rx_x, rx_y,
                        walls_nb[w1,0], walls_nb[w1,1],
                        walls_nb[w1,2], walls_nb[w1,3]
                    ):
                        insert_topk(best,d)

            valid_w2 = RX_mask & WWV[w1]

            if not np.any(valid_w2):
                continue

            for w2 in np.nonzero(valid_w2)[0]:

                tx2 = reflect_point_nb(
                    tx1[0], tx1[1],
                    walls_nb[w2,0], walls_nb[w2,1],
                    walls_nb[w2,2], walls_nb[w2,3]
                )

                dx = tx2[0] - rx_x
                dy = tx2[1] - rx_y
                d_lb = (dx*dx + dy*dy) ** 0.5

                if d_lb >= best[-1]:
                    continue

                d = second_reflection_dist_nb(
                    tx_x, tx_y,
                    rx_x, rx_y,
                    walls_nb[w1,0], walls_nb[w1,1],
                    walls_nb[w1,2], walls_nb[w1,3],
                    walls_nb[w2,0], walls_nb[w2,1],
                    walls_nb[w2,2], walls_nb[w2,3],
                    best[-1],
                    1e-9
                )

                insert_topk(best,d)

        # ------------------------------------------------
        # diffraction
        # ------------------------------------------------

        if euclid + DIFF_BIAS < best[-1]:

            best = diffraction_toa_nb(
                tx_x, tx_y,
                rx_x, rx_y,
                PVP_start, PVP_len, PVP_flat,
                corner_x, corner_y,
                best
            )

        # ------------------------------------------------
        # diffraction + reflection
        # ------------------------------------------------

        if euclid + DIFF_BIAS < best[-1]:

            best = diffraction_2nd_mirror_nb(
                tx_x, tx_y,
                rx_x, rx_y,
                TX_mask, RX_mask,
                TX_wall_order,
                walls_nb,
                grid_start, grid_count, grid_list,
                PVP_start, PVP_len, PVP_flat,
                corner_x, corner_y,
                best
            )

        for k in range(K):
            dist_map[i, j, k] = best[k]
# =====================================================
# 一阶 / 二阶反射、绕射（原样）
# =====================================================
def valid_first_reflection(tx_img, rx, wall):

    if not segment_intersect2(tx_img,rx,wall[0],wall[1]):

        return False


    return True

def valid_second_reflection(tx, rx, w1, w2, walls):


    # ---------- 判定 A：W2 上是否存在反射点 ----------
    tx1 = reflect_point(tx, walls[w1])
    tx2 = reflect_point(tx1, walls[w2])

    if not segment_intersect2(tx2, rx, walls[w2][0], walls[w2][1]):

        return False

    # ---------- 判定 B：W1 上是否存在反射点 ----------
    rx1 = reflect_point(rx, walls[w2])
    rx2 = reflect_point(rx1, walls[w1])

    if not segment_intersect2(rx2, tx, walls[w1][0], walls[w1][1]):

        return False


    return True


def diffraction_toa(tx, rx, PVP_id, corner_table, best_dist):


    ids_tx = PVP_id[tx[1], tx[0]]
    ids_rx = PVP_id[rx[1], rx[0]]

    if not ids_tx or not ids_rx:

        return best_dist

    for cid in set(ids_tx) & set(ids_rx):
        cx, cy = corner_table[cid]

        d = math.dist(tx, (cx, cy)) + math.dist((cx, cy), rx)

        if d < best_dist:
            best_dist = d


    return best_dist


def diffraction_2nd_toa(tx, rx, PVP_id, corner_table, walls, wall_grid, best_dist):


    ids_tx = PVP_id[tx[1], tx[0]]
    ids_rx = PVP_id[rx[1], rx[0]]

    if not ids_tx or not ids_rx:

        return best_dist

    for cid1 in ids_tx:
        c1 = corner_table[cid1]

        d1 = math.dist(tx, c1)

        if d1 >= best_dist:
            continue

        for cid2 in ids_rx:
            c2 = corner_table[cid2]
            if not visible_fast(c1, c2, walls, wall_grid):
                continue

            d = d1 + math.dist(c1, c2) + math.dist(c2, rx)

            if d < best_dist:
                best_dist = d


    return best_dist

def diffraction_2nd_mirror(
    tx, rx,
    TX_mask, RX_mask,
    TX_wall_order,
    walls,        # 原结构
    walls_nb,     # 新 numba结构
    grid_start,
    grid_count,
    grid_list,
    PVP_id,
    corner_table,
    best_dist
):
    tx_x, tx_y = tx
    rx_x, rx_y = rx

    ids_tx = PVP_id[tx_y, tx_x]
    ids_rx = PVP_id[rx_y, rx_x]

    if not ids_tx and not ids_rx:
        return best_dist

    ids_tx_set = set(ids_tx) if ids_tx else set()
    ids_rx_set = set(ids_rx) if ids_rx else set()

    # =================================================
    # 情况 1：TX 镜像 + 绕射
    # =================================================
    for w in TX_wall_order:
        if not TX_mask[w]:
            continue
        ax, ay = walls[w][0]
        bx, by = walls[w][1]

        # TX 镜像
        tx_img_x, tx_img_y = reflect_point_nb(
            tx_x, tx_y,
            ax, ay, bx, by
        )

        # 下界剪枝（镜像点到 RX）
        if math.dist((tx_img_x, tx_img_y), (rx_x, rx_y)) >= best_dist:
            continue

        for cid in ids_rx_set:
            cx, cy = corner_table[cid]

            # 穿墙 + 反射点
            ok, rxp, ryp = intersect_wall_return_point_nb(
                tx_img_x, tx_img_y,
                cx, cy,
                ax, ay, bx, by,
                1e-9
            )
            if not ok:
                continue

            # 真实遮挡（跳过本墙）
            if not visible_fast_nb(
                    rxp, ryp,
                    cx, cy,
                    walls_nb,
                    grid_start,
                    grid_count,
                    grid_list,
                    w
            ):
                continue

            d = (
                math.dist((tx_x, tx_y), (rxp, ryp)) +
                math.dist((rxp, ryp), (cx, cy)) +
                math.dist((cx, cy), (rx_x, rx_y))
            )

            if d < best_dist:
                best_dist = d
    
    # =================================================
    # 情况 2：RX 镜像 + 绕射（对称）
    # =================================================
    for w in np.nonzero(RX_mask)[0]:
        ax, ay = walls[w][0]
        bx, by = walls[w][1]

        # RX 镜像
        rx_img_x, rx_img_y = reflect_point_nb(
            rx_x, rx_y,
            ax, ay, bx, by
        )

        # 下界剪枝（镜像点到 TX）
        if math.dist((rx_img_x, rx_img_y), (tx_x, tx_y)) >= best_dist:
            continue

        for cid in ids_tx_set:
            cx, cy = corner_table[cid]

            # 穿墙 + 反射点
            ok, rxp, ryp = intersect_wall_return_point_nb(
                rx_img_x, rx_img_y,
                cx, cy,
                ax, ay, bx, by,
                1e-9
            )
            if not ok:
                continue

            # 真实遮挡（跳过本墙）
            if not visible_fast_nb(
                    rxp, ryp,
                    cx, cy,
                    walls_nb,
                    grid_start,
                    grid_count,
                    grid_list,
                    w
            ):
                continue

            d = (
                math.dist((tx_x, tx_y), (cx, cy)) +
                math.dist((cx, cy), (rxp, ryp)) +
                math.dist((rxp, ryp), (rx_x, rx_y))
            )

            if d < best_dist:
                best_dist = d
    
    return best_dist
# ================= 主流程 =================

def main():
    """
    Entry point for simulation.

    Behavior:
        - Loads geometry and precomputed data
        - Builds acceleration structures
        - Runs solver over all receiver points
        - Saves compressed result

    CLI:
        python solver.py IDX TX_ID

    Output:
        outputs/sim/{IDX}_{TX_ID}.npz
    """

    MAIN_START_TIME = time.perf_counter()


    # =====================================================
    # 读取（只改这里）
    # =====================================================
    config.IDX = IDX   
    geo, ant, gain = load_geo(config, tx_id=TX_ID)

    # ===== convert =====
    scene_dir = f"{CONVERT_ROOT}/{IDX}"

    PVW_mask = np.load(f"{scene_dir}/PVW_mask.npy")
    walls_nb = np.load(f"{scene_dir}/walls_nb.npy")

    PVP_flat  = np.load(f"{scene_dir}/PVP_flat.npy")
    PVP_start = np.load(f"{scene_dir}/PVP_start.npy")
    PVP_len   = np.load(f"{scene_dir}/PVP_len.npy")

    corner_x = np.load(f"{scene_dir}/corner_x.npy")
    corner_y = np.load(f"{scene_dir}/corner_y.npy")

    # ===== middata =====
    WWV = np.load(f"{MID_ROOT}/wwv/{IDX}.npy")

    # =====================================================
    # （下面逻辑完全不动）
    # =====================================================

    H, W = geo.shape

    grid_start, grid_count, grid_list = build_wall_grid_nb(walls_nb,H,W)

    dist_map = np.full((H, W, K), MAX_DIST, dtype=np.float32)

    tx_i, tx_j = np.argwhere(ant == 1)[0]
    tx = (tx_j, tx_i)

    TX_mask = PVW_mask[tx_i, tx_j]

    TX_wall_order = []
    for w in np.nonzero(TX_mask)[0]:
        x0,y0,x1,y1 = walls_nb[w]
        mx = 0.5*(x0+x1)
        my = 0.5*(y0+y1)
        dist2 = (tx[0] - mx)**2 + (tx[1] - my)**2
        TX_wall_order.append((dist2, w))

    TX_wall_order.sort()
    TX_wall_order = np.array([w for _, w in TX_wall_order], dtype=np.int32)

    searchable = (geo == 0)
    rx_list = np.argwhere(searchable).astype(np.int32)

    tx_img = []
    for i in range(walls_nb.shape[0]):
        tx_img.append(
            reflect_point_nb(
                tx[0], tx[1],
                walls_nb[i,0], walls_nb[i,1],
                walls_nb[i,2], walls_nb[i,3]
            )
        )
    tx_img = np.array(tx_img, dtype=np.float32)

    solver_rx_loop(
        rx_list,
        tx[0], tx[1],
        tx_i, tx_j,
        dist_map,
        PVW_mask,
        TX_mask,
        TX_wall_order,
        walls_nb,
        grid_start, grid_count, grid_list,
        PVP_start, PVP_len, PVP_flat,
        corner_x, corner_y,
        WWV,
        tx_img,
        K,
        MAX_DIST
    )

    os.makedirs(OUT_DIR, exist_ok=True)

    out_path = f"{OUT_DIR}/{IDX}_{TX_ID}.npz"

    np.savez_compressed(out_path, dist_map=dist_map.astype(np.float16))

    print(f"[Done] {IDX}_{TX_ID} | time={time.perf_counter()-MAIN_START_TIME:.2f}s")


if __name__ == "__main__":
    main()