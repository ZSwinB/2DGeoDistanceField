import math
import numpy as np
from numba import njit,cuda,float64
GRID_SIZE=8
DIFF_BIAS=300.0
FALLBACK_BIAS=600.0
@cuda.jit(device=True,inline=True)
def gpu_min(a,b):
    return b if b<a else a
@cuda.jit(device=True,inline=True)
def gpu_max(a,b):
    return b if b>a else a
@cuda.jit(device=True,inline=True)
def packed_has(row,index):
    return ((row[index//8]>>(7-index%8))&1)!=0
"""Wall visibility with proper crossings and through-vertex crossings.

A ray may touch a boundary tangentially or end at a diffraction corner. A wall
vertex strictly inside the ray blocks it when incident edges lie on both sides.
"""
import math
from numba import njit

@cuda.jit(device=True, inline=True, fastmath=False)
def _orient(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

@cuda.jit(device=True, inline=True, fastmath=False)
def _vertex_crosses_ray(x0, y0, x1, y1, vx, vy, walls, starts, counts, indices, skip_w, cell_size, tol):
    dx, dy = (x1 - x0, y1 - y0)
    squared = dx * dx + dy * dy
    projection = (vx - x0) * dx + (vy - y0) * dy
    if projection <= tol or projection >= squared - tol:
        return False
    gx = int(vx // cell_size)
    gy = int(vy // cell_size)
    if gx < 0 or gy < 0 or gx >= starts.shape[0] or (gy >= starts.shape[1]):
        return False
    positive = False
    negative = False
    begin = starts[gx, gy]
    end = begin + counts[gx, gy]
    for k in range(begin, end):
        idx = indices[k]
        if idx == skip_w:
            continue
        ax, ay, bx, by = walls[idx]
        side = 0.0
        if abs(ax - vx) <= 1e-09 and abs(ay - vy) <= 1e-09:
            side = _orient(x0, y0, x1, y1, bx, by)
        elif abs(bx - vx) <= 1e-09 and abs(by - vy) <= 1e-09:
            side = _orient(x0, y0, x1, y1, ax, ay)
        else:
            continue
        if side > tol:
            positive = True
        if side < -tol:
            negative = True
        if positive and negative:
            return True
    return False

@cuda.jit(device=True, inline=True, fastmath=False)
def visible_between_points_nb(x0, y0, x1, y1, walls, starts, counts, indices, skip_w, cell_size):
    dx, dy = (x1 - x0, y1 - y0)
    squared = dx * dx + dy * dy
    if squared == 0:
        return True
    tol = 1e-09 * gpu_max(1.0, math.sqrt(squared))
    gx0 = gpu_max(0, int(gpu_min(x0, x1) // cell_size))
    gx1 = gpu_min(starts.shape[0] - 1, int(gpu_max(x0, x1) // cell_size))
    gy0 = gpu_max(0, int(gpu_min(y0, y1) // cell_size))
    gy1 = gpu_min(starts.shape[1] - 1, int(gpu_max(y0, y1) // cell_size))
    for gx in range(gx0, gx1 + 1):
        for gy in range(gy0, gy1 + 1):
            begin = starts[gx, gy]
            end = begin + counts[gx, gy]
            for k in range(begin, end):
                idx = indices[k]
                if idx == skip_w:
                    continue
                ax, ay, bx, by = walls[idx]
                o1 = _orient(x0, y0, x1, y1, ax, ay)
                o2 = _orient(x0, y0, x1, y1, bx, by)
                o3 = _orient(ax, ay, bx, by, x0, y0)
                o4 = _orient(ax, ay, bx, by, x1, y1)
                wall_tol = 1e-09 * gpu_max(1.0, math.hypot(bx - ax, by - ay))
                ray_opposite = o1 > tol and o2 < -tol or (o1 < -tol and o2 > tol)
                wall_opposite = o3 > wall_tol and o4 < -wall_tol or (o3 < -wall_tol and o4 > wall_tol)
                if ray_opposite and wall_opposite:
                    return False
                if abs(o1) <= tol and _vertex_crosses_ray(x0, y0, x1, y1, ax, ay, walls, starts, counts, indices, skip_w, cell_size, tol):
                    return False
                if abs(o2) <= tol and _vertex_crosses_ray(x0, y0, x1, y1, bx, by, walls, starts, counts, indices, skip_w, cell_size, tol):
                    return False
    return True
@cuda.jit(device=True, fastmath=False)
def visible_fast_nb(x0, y0, x1, y1, walls, grid_start, grid_count, grid_list, skip_w):
    return visible_between_points_nb(x0, y0, x1, y1, walls, grid_start, grid_count, grid_list, skip_w, GRID_SIZE)

@cuda.jit(device=True, fastmath=False)
def segment_intersect_strict_nb(ax, ay, bx, by, cx, cy, dx, dy):
    o1 = orient_nb(ax, ay, bx, by, cx, cy)
    o2 = orient_nb(ax, ay, bx, by, dx, dy)
    o3 = orient_nb(cx, cy, dx, dy, ax, ay)
    o4 = orient_nb(cx, cy, dx, dy, bx, by)
    return o1 * o2 < 0.0 and o3 * o4 < 0.0

@cuda.jit(device=True, fastmath=False)
def orient_nb(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

@cuda.jit(device=True, fastmath=False)
def on_segment_nb(ax, ay, bx, by, cx, cy, eps):
    return gpu_min(ax, bx) - eps <= cx <= gpu_max(ax, bx) + eps and gpu_min(ay, by) - eps <= cy <= gpu_max(ay, by) + eps

@cuda.jit(device=True, fastmath=False)
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

@cuda.jit(device=True, fastmath=False)
def dist2d_nb(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx * dx + dy * dy)

@cuda.jit(device=True, fastmath=False)
def valid_first_reflection_nb(tx_x, tx_y, rx_x, rx_y, ax, ay, bx, by):
    return segment_intersect2_nb(tx_x, tx_y, rx_x, rx_y, ax, ay, bx, by, 1e-09)

@cuda.jit(device=True, fastmath=False)
def first_reflection_visible_nb(tx_x, tx_y, rx_x, rx_y, img_x, img_y, w, walls, gs, gc, gl):
    ax, ay, bx, by = walls[w]
    ok, px, py = intersect_wall_return_point_nb(img_x, img_y, rx_x, rx_y, ax, ay, bx, by, 1e-09)
    if not ok:
        return False
    return visible_fast_nb(tx_x, tx_y, px, py, walls, gs, gc, gl, w) and visible_fast_nb(px, py, rx_x, rx_y, walls, gs, gc, gl, w)

@cuda.jit(device=True, fastmath=False)
def second_reflection_dist_nb(tx_x, tx_y, rx_x, rx_y, w1x1, w1y1, w1x2, w1y2, w2x1, w2y1, w2x2, w2y2, best, eps, w1, w2, walls, gs, gc, gl, tx_images, source_state):
    tx1x, tx1y = (tx_images[w1, 0], tx_images[w1, 1])
    tx2x, tx2y = (source_state[6][w1, w2, 0], source_state[6][w1, w2, 1])
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

@cuda.jit(device=True, fastmath=False)
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

@cuda.jit(device=True, fastmath=False)
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

@cuda.jit(device=True, fastmath=False)
def diffraction_2nd_mirror_nb(tx_x, tx_y, rx_x, rx_y, rx_i, rx_j, TX_mask, RX_mask, TX_wall_order, walls_nb, grid_start, grid_count, grid_list, PVP_start, PVP_len, PVP_flat, TX_corner_ids, corner_x, corner_y, best, source_state, tx_images):
    """
    Compute diffraction combined with reflection (mirror method).

    Returns:
        np.ndarray: updated best distances
    """
    K = best.shape[0]
    for k in range(K):
        best[k] -= DIFF_BIAS
    len_tx = len(TX_corner_ids)
    start_rx = 0
    len_rx = len(PVP_len)
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
        tx_img_x, tx_img_y = (tx_images[w, 0], tx_images[w, 1])
        dx = tx_img_x - rx_x
        dy = tx_img_y - rx_y
        if math.sqrt(dx * dx + dy * dy) >= best[K - 1]:
            continue
        for k in range(len_rx):
            cid = PVP_len[k]
            if not packed_has(PVP_start[rx_i, rx_j], cid):
                continue
            cx = corner_x[cid]
            cy = corner_y[cid]
            if not source_state[0][w, cid]:
                continue
            rxp = source_state[1][w, cid]
            ryp = source_state[2][w, cid]
            dx1 = tx_x - rxp
            dy1 = tx_y - ryp
            dx2 = rxp - cx
            dy2 = ryp - cy
            dx3 = cx - rx_x
            dy3 = cy - rx_y
            d = source_state[3][w, cid] + math.sqrt(dx3 * dx3 + dy3 * dy3)
            if d < best[K - 1]:
                pos = K - 1
                best[pos] = d
                while pos > 0 and best[pos] < best[pos - 1]:
                    tmp = best[pos]
                    best[pos] = best[pos - 1]
                    best[pos - 1] = tmp
                    pos -= 1
    for w in range(len(walls_nb)):
        if not packed_has(RX_mask, w):
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
            bound_dx = rx_x - source_state[7][w, cid, 0]
            bound_dy = rx_y - source_state[7][w, cid, 1]
            lower = source_state[8][cid] + math.sqrt(bound_dx * bound_dx + bound_dy * bound_dy)
            if lower > best[K - 1] + 1e-08:
                continue
            cx = corner_x[cid]
            cy = corner_y[cid]
            ok, rxp, ryp = intersect_wall_return_point_nb(rx_img_x, rx_img_y, cx, cy, ax, ay, bx, by, 1e-09)
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

@cuda.jit(device=True, fastmath=False)
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
    start_rx = 0
    len_rx = len(PVP_len)
    if len_tx == 0 or len_rx == 0:
        for k in range(K):
            best[k] += DIFF_BIAS
        return best
    for i in range(len_tx):
        cid = TX_corner_ids[i]
        if not packed_has(PVP_start[rx_i, rx_j], cid):
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

@cuda.jit(device=True, fastmath=False)
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

@cuda.jit(fastmath=False)
def solver_rx_loop(rx_list, tx_x, tx_y, tx_i, tx_j, dist_map, PVW_mask, TX_mask, TX_wall_order, walls_nb, grid_start, grid_count, grid_list, PVP_start, PVP_len, PVP_flat, TX_corner_ids, corner_x, corner_y, WWV, tx_img, K, MAX_DIST, sample_scale, y_origin, source_state):
    n = cuda.grid(1)
    if n >= rx_list.shape[0]:
        return
    i = rx_list[n, 0]
    j = rx_list[n, 1]
    rx_x = j / sample_scale
    rx_y = (i + y_origin) / sample_scale
    if i == tx_i and j == tx_j:
        for k in range(K):
            dist_map[i, j, k] = 0.0
        return
    best = cuda.local.array(1, float64)
    best[0] = MAX_DIST
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
            if packed_has(RX_mask, w1):
                if first_reflection_visible_nb(tx_x, tx_y, rx_x, rx_y, tx1[0], tx1[1], w1, walls_nb, grid_start, grid_count, grid_list):
                    insert_topk(best, d)
        for q2 in range(source_state[5][w1]):
            w2 = source_state[4][w1, q2]
            if not packed_has(RX_mask, w2):
                continue
            tx2 = source_state[6][w1, w2]
            dx = tx2[0] - rx_x
            dy = tx2[1] - rx_y
            d_lb = (dx * dx + dy * dy) ** 0.5
            if d_lb >= best[-1]:
                continue
            d = second_reflection_dist_nb(tx_x, tx_y, rx_x, rx_y, walls_nb[w1, 0], walls_nb[w1, 1], walls_nb[w1, 2], walls_nb[w1, 3], walls_nb[w2, 0], walls_nb[w2, 1], walls_nb[w2, 2], walls_nb[w2, 3], best[-1], 1e-09, w1, w2, walls_nb, grid_start, grid_count, grid_list, tx_img, source_state)
            insert_topk(best, d)
    if euclid + DIFF_BIAS < best[-1]:
        best = diffraction_toa_nb(tx_x, tx_y, rx_x, rx_y, i, j, PVP_start, PVP_len, PVP_flat, TX_corner_ids, corner_x, corner_y, best)
    if euclid + DIFF_BIAS < best[-1]:
        best = diffraction_2nd_mirror_nb(tx_x, tx_y, rx_x, rx_y, i, j, TX_mask, RX_mask, TX_wall_order, walls_nb, grid_start, grid_count, grid_list, PVP_start, PVP_len, PVP_flat, TX_corner_ids, corner_x, corner_y, best, source_state, tx_img)
    for k in range(K):
        dist_map[i, j, k] = best[k]