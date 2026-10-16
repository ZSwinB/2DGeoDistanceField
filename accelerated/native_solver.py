import math
import numpy as np
from numba import njit
from collections import defaultdict
from .cpu_visibility import visible_between_points_nb
GRID_SIZE=8
DIFF_BIAS=300.
FALLBACK_BIAS=600.
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
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

def on_segment(a, b, c, eps=1e-09):
    return min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps and min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps

def segment_intersect_strict(a, b, c, d):
    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return o1 * o2 < 0 and o3 * o4 < 0

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
    if abs(o1) < eps and on_segment(a, b, c, eps):
        return True
    if abs(o2) < eps and on_segment(a, b, d, eps):
        return True
    if abs(o3) < eps and on_segment(c, d, a, eps):
        return True
    if abs(o4) < eps and on_segment(c, d, b, eps):
        return True
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
                if segment_intersect_strict_nb(x0, y0, x1, y1, A[0], A[1], B[0], B[1]):
                    return False
    return True

def build_wall_grid_nb(walls, H, W):
    """
    Build numba-compatible grid (CSR-like).

    Returns:
        grid_start, grid_count, grid_list
    """
    GX = int(np.ceil((W + 1) / GRID_SIZE))
    GY = int(np.ceil((H + 1) / GRID_SIZE))
    grid_count = np.zeros((GX, GY), dtype=np.int32)
    for idx in range(walls.shape[0]):
        x0 = walls[idx, 0]
        y0 = walls[idx, 1]
        x1 = walls[idx, 2]
        y1 = walls[idx, 3]
        gx0 = max(0, int(min(x0, x1) // GRID_SIZE))
        gx1 = min(GX - 1, int(max(x0, x1) // GRID_SIZE))
        gy0 = max(0, int(min(y0, y1) // GRID_SIZE))
        gy1 = min(GY - 1, int(max(y0, y1) // GRID_SIZE))
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                grid_count[gx, gy] += 1
    grid_start = np.zeros((GX, GY), dtype=np.int32)
    total = 0
    for gx in range(GX):
        for gy in range(GY):
            grid_start[gx, gy] = total
            total += grid_count[gx, gy]
    grid_list = np.zeros(total, dtype=np.int32)
    cursor = grid_start.copy()
    for idx in range(walls.shape[0]):
        x0 = walls[idx, 0]
        y0 = walls[idx, 1]
        x1 = walls[idx, 2]
        y1 = walls[idx, 3]
        gx0 = max(0, int(min(x0, x1) // GRID_SIZE))
        gx1 = min(GX - 1, int(max(x0, x1) // GRID_SIZE))
        gy0 = max(0, int(min(y0, y1) // GRID_SIZE))
        gy1 = min(GY - 1, int(max(y0, y1) // GRID_SIZE))
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                k = cursor[gx, gy]
                grid_list[k] = idx
                cursor[gx, gy] += 1
    return (grid_start, grid_count, grid_list)

@njit(cache=True)
def visible_fast_nb(x0, y0, x1, y1, walls, grid_start, grid_count, grid_list, skip_w):
    return visible_between_points_nb(x0, y0, x1, y1, walls, grid_start, grid_count, grid_list, skip_w, GRID_SIZE)

@njit(cache=True)
def segment_intersect_strict_nb(ax, ay, bx, by, cx, cy, dx, dy):
    o1 = orient_nb(ax, ay, bx, by, cx, cy)
    o2 = orient_nb(ax, ay, bx, by, dx, dy)
    o3 = orient_nb(cx, cy, dx, dy, ax, ay)
    o4 = orient_nb(cx, cy, dx, dy, bx, by)
    return o1 * o2 < 0.0 and o3 * o4 < 0.0

@njit(cache=True)
def orient_nb(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

@njit(cache=True)
def on_segment_nb(ax, ay, bx, by, cx, cy, eps):
    return min(ax, bx) - eps <= cx <= max(ax, bx) + eps and min(ay, by) - eps <= cy <= max(ay, by) + eps

@njit(cache=True)
def segment_intersect2_nb(ax, ay, bx, by, cx, cy, dx, dy, eps):
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

@njit(cache=True)
def dist2d_nb(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx * dx + dy * dy)

@njit(cache=True)
def valid_first_reflection_nb(tx_x, tx_y, rx_x, rx_y, ax, ay, bx, by):
    return segment_intersect2_nb(tx_x, tx_y, rx_x, rx_y, ax, ay, bx, by, 1e-09)

@njit(cache=True)
def first_reflection_visible_nb(tx_x, tx_y, rx_x, rx_y, img_x, img_y, w, walls, gs, gc, gl):
    ax, ay, bx, by = walls[w]
    ok, px, py = intersect_wall_return_point_nb(img_x, img_y, rx_x, rx_y, ax, ay, bx, by)
    if not ok:
        return False
    return visible_fast_nb(tx_x, tx_y, px, py, walls, gs, gc, gl, w) and visible_fast_nb(px, py, rx_x, rx_y, walls, gs, gc, gl, w)

@njit(cache=True)
def second_reflection_dist_nb(tx_x, tx_y, rx_x, rx_y, w1x1, w1y1, w1x2, w1y2, w2x1, w2y1, w2x2, w2y2, best, eps, w1, w2, walls, gs, gc, gl):
    tx1x, tx1y = reflect_point_nb(tx_x, tx_y, w1x1, w1y1, w1x2, w1y2)
    tx2x, tx2y = reflect_point_nb(tx1x, tx1y, w2x1, w2y1, w2x2, w2y2)
    distance = dist2d_nb(tx2x, tx2y, rx_x, rx_y)
    if distance >= best:
        return best
    ok2, p2x, p2y = intersect_wall_return_point_nb(tx2x, tx2y, rx_x, rx_y, w2x1, w2y1, w2x2, w2y2, eps)
    if not ok2:
        return best
    ok1, p1x, p1y = intersect_wall_return_point_nb(tx1x, tx1y, p2x, p2y, w1x1, w1y1, w1x2, w1y2, eps)
    if not ok1:
        return best
    if orient_nb(w1x1, w1y1, w1x2, w1y2, tx_x, tx_y) * orient_nb(w1x1, w1y1, w1x2, w1y2, p2x, p2y) < -eps:
        return best
    if orient_nb(w2x1, w2y1, w2x2, w2y2, p1x, p1y) * orient_nb(w2x1, w2y1, w2x2, w2y2, rx_x, rx_y) < -eps:
        return best
    if not visible_fast_nb(tx_x, tx_y, p1x, p1y, walls, gs, gc, gl, w1):
        return best
    if not visible_fast_nb(p1x, p1y, p2x, p2y, walls, gs, gc, gl, -1):
        return best
    if not visible_fast_nb(p2x, p2y, rx_x, rx_y, walls, gs, gc, gl, w2):
        return best
    return distance

@njit(cache=True)
def reflect_point_nb(px, py, ax, ay, bx, by):
    """
    Reflect a point across a wall segment.

    Returns:
        tuple[float, float]: reflected point
    """
    vx = bx - ax
    vy = by - ay
    norm = math.sqrt(vx * vx + vy * vy)
    vx /= norm
    vy /= norm
    apx = px - ax
    apy = py - ay
    t = apx * vx + apy * vy
    projx = ax + vx * t
    projy = ay + vy * t
    rx = 2.0 * projx - px
    ry = 2.0 * projy - py
    return (rx, ry)

@njit(cache=True)
def intersect_wall_return_point_nb(px, py, qx, qy, ax, ay, bx, by, eps=1e-09):
    dx = qx - px
    dy = qy - py
    ex = bx - ax
    ey = by - ay
    denom = dx * ey - dy * ex
    if abs(denom) < eps:
        return (False, 0.0, 0.0)
    t = ((ax - px) * ey - (ay - py) * ex) / denom
    u = ((ax - px) * dy - (ay - py) * dx) / denom
    if u < -eps or u > 1.0 + eps or t < eps or (t > 1.0 + eps):
        return (False, 0.0, 0.0)
    rx = px + t * dx
    ry = py + t * dy
    return (True, rx, ry)

@njit(cache=True)
def diffraction_2nd_mirror_nb(tx_x, tx_y, rx_x, rx_y, rx_i, rx_j, TX_mask, RX_mask, TX_wall_order, walls_nb, grid_start, grid_count, grid_list, PVP_start, PVP_len, PVP_flat, TX_corner_ids, corner_x, corner_y, best):
    """
    Compute diffraction combined with reflection (mirror method).

    Returns:
        np.ndarray: updated best distances
    """
    K = best.shape[0]
    for k in range(K):
        best[k] -= DIFF_BIAS
    len_tx = len(TX_corner_ids)
    start_rx = PVP_start[rx_i, rx_j]
    len_rx = PVP_len[rx_i, rx_j]
    if len_tx == 0 and len_rx == 0:
        for k in range(K):
            best[k] += DIFF_BIAS
        return best
    for wi in range(len(TX_wall_order)):
        w = TX_wall_order[wi]
        if not TX_mask[w]:
            continue
        ax = walls_nb[w, 0]
        ay = walls_nb[w, 1]
        bx = walls_nb[w, 2]
        by = walls_nb[w, 3]
        tx_img_x, tx_img_y = reflect_point_nb(tx_x, tx_y, ax, ay, bx, by)
        dx = tx_img_x - rx_x
        dy = tx_img_y - rx_y
        if math.sqrt(dx * dx + dy * dy) >= best[K - 1]:
            continue
        for k in range(len_rx):
            cid = PVP_flat[start_rx + k]
            cx = corner_x[cid]
            cy = corner_y[cid]
            ok, rxp, ryp = intersect_wall_return_point_nb(tx_img_x, tx_img_y, cx, cy, ax, ay, bx, by)
            if not ok:
                continue
            if not visible_fast_nb(tx_x, tx_y, rxp, ryp, walls_nb, grid_start, grid_count, grid_list, w):
                continue
            if not visible_fast_nb(rxp, ryp, cx, cy, walls_nb, grid_start, grid_count, grid_list, w):
                continue
            dx1 = tx_x - rxp
            dy1 = tx_y - ryp
            dx2 = rxp - cx
            dy2 = ryp - cy
            dx3 = cx - rx_x
            dy3 = cy - rx_y
            d = math.sqrt(dx1 * dx1 + dy1 * dy1) + math.sqrt(dx2 * dx2 + dy2 * dy2) + math.sqrt(dx3 * dx3 + dy3 * dy3)
            if d < best[K - 1]:
                pos = K - 1
                best[pos] = d
                while pos > 0 and best[pos] < best[pos - 1]:
                    tmp = best[pos]
                    best[pos] = best[pos - 1]
                    best[pos - 1] = tmp
                    pos -= 1
    for w in range(len(RX_mask)):
        if not RX_mask[w]:
            continue
        ax = walls_nb[w, 0]
        ay = walls_nb[w, 1]
        bx = walls_nb[w, 2]
        by = walls_nb[w, 3]
        rx_img_x, rx_img_y = reflect_point_nb(rx_x, rx_y, ax, ay, bx, by)
        dx = rx_img_x - tx_x
        dy = rx_img_y - tx_y
        if math.sqrt(dx * dx + dy * dy) >= best[K - 1]:
            continue
        for k in range(len_tx):
            cid = TX_corner_ids[k]
            cx = corner_x[cid]
            cy = corner_y[cid]
            ok, rxp, ryp = intersect_wall_return_point_nb(rx_img_x, rx_img_y, cx, cy, ax, ay, bx, by)
            if not ok:
                continue
            if not visible_fast_nb(rxp, ryp, rx_x, rx_y, walls_nb, grid_start, grid_count, grid_list, w):
                continue
            if not visible_fast_nb(rxp, ryp, cx, cy, walls_nb, grid_start, grid_count, grid_list, w):
                continue
            dx1 = tx_x - cx
            dy1 = tx_y - cy
            dx2 = cx - rxp
            dy2 = cy - ryp
            dx3 = rxp - rx_x
            dy3 = ryp - rx_y
            d = math.sqrt(dx1 * dx1 + dy1 * dy1) + math.sqrt(dx2 * dx2 + dy2 * dy2) + math.sqrt(dx3 * dx3 + dy3 * dy3)
            if d < best[K - 1]:
                pos = K - 1
                best[pos] = d
                while pos > 0 and best[pos] < best[pos - 1]:
                    tmp = best[pos]
                    best[pos] = best[pos - 1]
                    best[pos - 1] = tmp
                    pos -= 1
    for k in range(K):
        best[k] += DIFF_BIAS
    return best

@njit(cache=True)
def diffraction_toa_nb(tx_x, tx_y, rx_x, rx_y, rx_i, rx_j, PVP_start, PVP_len, PVP_flat, TX_corner_ids, corner_x, corner_y, best):
    """
    Compute diffraction path (corner-based) with top-K update.

    Returns:
        np.ndarray: updated best distances
    """
    K = best.shape[0]
    for k in range(K):
        best[k] -= DIFF_BIAS
    len_tx = len(TX_corner_ids)
    start_rx = PVP_start[rx_i, rx_j]
    len_rx = PVP_len[rx_i, rx_j]
    if len_tx == 0 or len_rx == 0:
        for k in range(K):
            best[k] += DIFF_BIAS
        return best
    for i in range(len_tx):
        cid = TX_corner_ids[i]
        for j in range(len_rx):
            if cid != PVP_flat[start_rx + j]:
                continue
            cx = corner_x[cid]
            cy = corner_y[cid]
            dx1 = tx_x - cx
            dy1 = tx_y - cy
            dx2 = cx - rx_x
            dy2 = cy - rx_y
            d = math.sqrt(dx1 * dx1 + dy1 * dy1) + math.sqrt(dx2 * dx2 + dy2 * dy2)
            if d >= best[K - 1]:
                continue
            pos = K - 1
            best[pos] = d
            while pos > 0 and best[pos] < best[pos - 1]:
                tmp = best[pos]
                best[pos] = best[pos - 1]
                best[pos - 1] = tmp
                pos -= 1
    for k in range(K):
        best[k] += DIFF_BIAS
    return best

@njit(cache=True)
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

@njit(cache=True)
def insert_topk(best, d):
    K = best.shape[0]
    if d >= best[K - 1]:
        return
    i = K - 1
    best[i] = d
    while i > 0 and best[i] < best[i - 1]:
        tmp = best[i]
        best[i] = best[i - 1]
        best[i - 1] = tmp
        i -= 1

@njit(nogil=True, cache=True)
def solver_rx_loop(rx_list, tx_x, tx_y, tx_i, tx_j, dist_map, PVW_mask, TX_mask, TX_wall_order, walls_nb, grid_start, grid_count, grid_list, PVP_start, PVP_len, PVP_flat, TX_corner_ids, corner_x, corner_y, WWV, tx_img, K, MAX_DIST, sample_scale, y_origin):
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
        i = rx_list[n, 0]
        j = rx_list[n, 1]
        rx_x = j / sample_scale
        rx_y = (i + y_origin) / sample_scale
        if i == tx_i and j == tx_j:
            for k in range(K):
                dist_map[i, j, k] = 0.0
            continue
        best = np.full(K, MAX_DIST)
        dx = tx_x - rx_x
        dy = tx_y - rx_y
        euclid = (dx * dx + dy * dy) ** 0.5
        insert_topk(best, euclid + FALLBACK_BIAS)
        if visible_fast_nb(tx_x, tx_y, rx_x, rx_y, walls_nb, grid_start, grid_count, grid_list, -1):
            d = euclid
            insert_topk(best, d)
        RX_mask = PVW_mask[i, j]
        for w1 in TX_wall_order:
            if not TX_mask[w1]:
                continue
            tx1 = tx_img[w1]
            dx = tx1[0] - rx_x
            dy = tx1[1] - rx_y
            d = (dx * dx + dy * dy) ** 0.5
            if d < best[-1]:
                if RX_mask[w1]:
                    if first_reflection_visible_nb(tx_x, tx_y, rx_x, rx_y, tx1[0], tx1[1], w1, walls_nb, grid_start, grid_count, grid_list):
                        insert_topk(best, d)
            valid_w2 = RX_mask & WWV[w1]
            if not np.any(valid_w2):
                continue
            for w2 in np.nonzero(valid_w2)[0]:
                tx2 = reflect_point_nb(tx1[0], tx1[1], walls_nb[w2, 0], walls_nb[w2, 1], walls_nb[w2, 2], walls_nb[w2, 3])
                dx = tx2[0] - rx_x
                dy = tx2[1] - rx_y
                d_lb = (dx * dx + dy * dy) ** 0.5
                if d_lb >= best[-1]:
                    continue
                d = second_reflection_dist_nb(tx_x, tx_y, rx_x, rx_y, walls_nb[w1, 0], walls_nb[w1, 1], walls_nb[w1, 2], walls_nb[w1, 3], walls_nb[w2, 0], walls_nb[w2, 1], walls_nb[w2, 2], walls_nb[w2, 3], best[-1], 1e-09, w1, w2, walls_nb, grid_start, grid_count, grid_list)
                insert_topk(best, d)
        if euclid + DIFF_BIAS < best[-1]:
            best = diffraction_toa_nb(tx_x, tx_y, rx_x, rx_y, i, j, PVP_start, PVP_len, PVP_flat, TX_corner_ids, corner_x, corner_y, best)
        if euclid + DIFF_BIAS < best[-1]:
            best = diffraction_2nd_mirror_nb(tx_x, tx_y, rx_x, rx_y, i, j, TX_mask, RX_mask, TX_wall_order, walls_nb, grid_start, grid_count, grid_list, PVP_start, PVP_len, PVP_flat, TX_corner_ids, corner_x, corner_y, best)
        for k in range(K):
            dist_map[i, j, k] = best[k]

def valid_first_reflection(tx_img, rx, wall):
    if not segment_intersect2(tx_img, rx, wall[0], wall[1]):
        return False
    return True

def valid_second_reflection(tx, rx, w1, w2, walls):
    tx1 = reflect_point(tx, walls[w1])
    tx2 = reflect_point(tx1, walls[w2])
    if not segment_intersect2(tx2, rx, walls[w2][0], walls[w2][1]):
        return False
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

def diffraction_2nd_mirror(tx, rx, TX_mask, RX_mask, TX_wall_order, walls, walls_nb, grid_start, grid_count, grid_list, PVP_id, corner_table, best_dist):
    tx_x, tx_y = tx
    rx_x, rx_y = rx
    ids_tx = PVP_id[tx_y, tx_x]
    ids_rx = PVP_id[rx_y, rx_x]
    if not ids_tx and (not ids_rx):
        return best_dist
    ids_tx_set = set(ids_tx) if ids_tx else set()
    ids_rx_set = set(ids_rx) if ids_rx else set()
    for w in TX_wall_order:
        if not TX_mask[w]:
            continue
        ax, ay = walls[w][0]
        bx, by = walls[w][1]
        tx_img_x, tx_img_y = reflect_point_nb(tx_x, tx_y, ax, ay, bx, by)
        if math.dist((tx_img_x, tx_img_y), (rx_x, rx_y)) >= best_dist:
            continue
        for cid in ids_rx_set:
            cx, cy = corner_table[cid]
            ok, rxp, ryp = intersect_wall_return_point_nb(tx_img_x, tx_img_y, cx, cy, ax, ay, bx, by, 1e-09)
            if not ok:
                continue
            if not visible_fast_nb(rxp, ryp, cx, cy, walls_nb, grid_start, grid_count, grid_list, w):
                continue
            d = math.dist((tx_x, tx_y), (rxp, ryp)) + math.dist((rxp, ryp), (cx, cy)) + math.dist((cx, cy), (rx_x, rx_y))
            if d < best_dist:
                best_dist = d
    for w in np.nonzero(RX_mask)[0]:
        ax, ay = walls[w][0]
        bx, by = walls[w][1]
        rx_img_x, rx_img_y = reflect_point_nb(rx_x, rx_y, ax, ay, bx, by)
        if math.dist((rx_img_x, rx_img_y), (tx_x, tx_y)) >= best_dist:
            continue
        for cid in ids_tx_set:
            cx, cy = corner_table[cid]
            ok, rxp, ryp = intersect_wall_return_point_nb(rx_img_x, rx_img_y, cx, cy, ax, ay, bx, by, 1e-09)
            if not ok:
                continue
            if not visible_fast_nb(rxp, ryp, cx, cy, walls_nb, grid_start, grid_count, grid_list, w):
                continue
            d = math.dist((tx_x, tx_y), (cx, cy)) + math.dist((cx, cy), (rxp, ryp)) + math.dist((rxp, ryp), (rx_x, rx_y))
            if d < best_dist:
                best_dist = d
    return best_dist