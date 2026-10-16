from .r3_sort import repair_near_angles
import math
import numpy as np
from numba import njit

@njit(cache=True)
def angle_xy(px, py, qx, qy):
    """
    Compute angle between two points using raw coordinates.

    Returns:
        float: angle in radians (range [-pi, pi])
    """
    return math.atan2(qy - py, qx - px)

@njit(cache=True)
def ray_segment_distance_nb(px, py, theta, ax, ay, bx, by):
    """
    Compute intersection distance between a ray and a segment.

    Behavior:
        - Ray starts at (px, py) with direction theta
        - Segment defined by A(ax,ay) → B(bx,by)
        - Returns first valid intersection distance

    Returns:
        float:
            distance t if intersection exists (t > 0),
            -1.0 if no valid intersection
    """
    dx = math.cos(theta)
    dy = math.sin(theta)
    vx = bx - ax
    vy = by - ay
    det = dx * -vy - dy * -vx
    if abs(det) >= 1e-09:
        t = ((ax - px) * -vy - (ay - py) * -vx) / det
        u = ((ax - px) * -dy + (ay - py) * dx) / det
        if t > 1e-09 and 0 <= u <= 1:
            return t
    return -1.0

@njit(cache=True)
def visible_walls_for_point_nb(P, walls):
    """
    Compute visible walls from a single point using angular sweep.

    Behavior:
        - Builds angular events for all wall endpoints
        - Performs sweep-line over angle domain
        - Tracks active walls and finds nearest intersection
        - Marks visible walls

    Args:
        P (tuple[float, float]): query point (x, y)
        walls (np.ndarray): wall segments (N,4)

    Returns:
        np.ndarray:
            binary mask (N,), 1 means visible, 0 means not visible
    """
    px = P[0]
    py = P[1]
    num_walls = walls.shape[0]
    max_events = num_walls * 4
    events_angle = np.empty(max_events, dtype=np.float64)
    events_idx = np.empty(max_events, dtype=np.int32)
    cnt = 0
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
        if r - l > math.pi + 1e-09:
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
    active = np.zeros(num_walls, dtype=np.uint8)
    visible = np.zeros(num_walls, dtype=np.uint8)
    for k in range(cnt - 1):
        idx = events_idx[order[k]]
        if active[idx]:
            active[idx] = 0
        else:
            active[idx] = 1
        ang = events_angle[order[k]]
        next_ang = events_angle[order[k + 1]]
        if next_ang - ang < 1e-06:
            continue
        theta = 0.5 * (ang + next_ang)
        best_d = 1e+18
        best_i = -1
        for i2 in range(num_walls):
            if active[i2]:
                ax = walls[i2, 0]
                ay = walls[i2, 1]
                bx = walls[i2, 2]
                by = walls[i2, 3]
                d = ray_segment_distance_nb(px, py, theta, ax, ay, bx, by)
                if d > 1e-09 and d < best_d:
                    best_d = d
                    best_i = i2
        if best_i != -1:
            visible[best_i] = 1
    return visible

@njit(cache=True)
def ray_direction_nb(px, py, dx, dy, ax, ay, bx, by):
    """
    Compute intersection distance between a ray and a segment.

    Behavior:
        - Ray starts at (px, py) with direction theta
        - Segment defined by A(ax,ay) → B(bx,by)
        - Returns first valid intersection distance

    Returns:
        float:
            distance t if intersection exists (t > 0),
            -1.0 if no valid intersection
    """
    vx = bx - ax
    vy = by - ay
    det = dx * -vy - dy * -vx
    if abs(det) >= 1e-09:
        t = ((ax - px) * -vy - (ay - py) * -vx) / det
        u = ((ax - px) * -dy + (ay - py) * dx) / det
        if t > 1e-09 and 0 <= u <= 1:
            return t
    return -1.0

@njit(cache=True)
def wall_visibility_direction(P, walls):
    """
    Compute visible walls from a single point using angular sweep.

    Behavior:
        - Builds angular events for all wall endpoints
        - Performs sweep-line over angle domain
        - Tracks active walls and finds nearest intersection
        - Marks visible walls

    Args:
        P (tuple[float, float]): query point (x, y)
        walls (np.ndarray): wall segments (N,4)

    Returns:
        np.ndarray:
            binary mask (N,), 1 means visible, 0 means not visible
    """
    px = P[0]
    py = P[1]
    num_walls = walls.shape[0]
    max_events = num_walls * 4
    events_angle = np.empty(max_events, dtype=np.float64)
    events_idx = np.empty(max_events, dtype=np.int32)
    cnt = 0
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
        if r - l > math.pi + 1e-09:
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
    order = np.argsort(events_angle[:cnt], kind='mergesort')
    close_unequal = False
    for q in range(1, cnt):
        delta = events_angle[order[q]] - events_angle[order[q - 1]]
        if 0.0 < delta < 1e-12:
            close_unequal = True
            break
    if close_unequal:
        order = repair_near_angles(events_angle, events_idx, order, cnt)
    active = np.zeros(num_walls, dtype=np.uint8)
    active_ids = np.empty(num_walls, dtype=np.int32)
    positions = np.full(num_walls, -1, dtype=np.int32)
    active_count = 0
    visible = np.zeros(num_walls, dtype=np.uint8)
    for k in range(cnt - 1):
        idx = events_idx[order[k]]
        if active[idx]:
            active[idx] = 0
            slot = positions[idx]
            active_count -= 1
            swapped = active_ids[active_count]
            active_ids[slot] = swapped
            positions[swapped] = slot
            positions[idx] = -1
        else:
            active[idx] = 1
            positions[idx] = active_count
            active_ids[active_count] = idx
            active_count += 1
        ang = events_angle[order[k]]
        next_ang = events_angle[order[k + 1]]
        if next_ang - ang < 1e-06:
            continue
        theta = 0.5 * (ang + next_ang)
        ray_dx = math.cos(theta)
        ray_dy = math.sin(theta)
        best_d = 1e+18
        best_i = -1
        for slot in range(active_count):
            i2 = active_ids[slot]
            if True:
                ax = walls[i2, 0]
                ay = walls[i2, 1]
                bx = walls[i2, 2]
                by = walls[i2, 3]
                d = ray_direction_nb(px, py, ray_dx, ray_dy, ax, ay, bx, by)
                if d > 1e-09 and (d < best_d or (d == best_d and (best_i < 0 or i2 < best_i))):
                    best_d = d
                    best_i = i2
        if best_i != -1:
            visible[best_i] = 1
    return visible

@njit(cache=True)
def ray_direction_nb(px, py, dx, dy, ax, ay, bx, by):
    """
    Compute intersection distance between a ray and a segment.

    Behavior:
        - Ray starts at (px, py) with direction theta
        - Segment defined by A(ax,ay) → B(bx,by)
        - Returns first valid intersection distance

    Returns:
        float:
            distance t if intersection exists (t > 0),
            -1.0 if no valid intersection
    """
    vx = bx - ax
    vy = by - ay
    det = dx * -vy - dy * -vx
    if abs(det) >= 1e-09:
        t = ((ax - px) * -vy - (ay - py) * -vx) / det
        u = ((ax - px) * -dy + (ay - py) * dx) / det
        if t > 1e-09 and 0 <= u <= 1:
            return t
    return -1.0

@njit(cache=True)
def wall_visibility_incremental(P, walls, unsafe):
    """
    Compute visible walls from a single point using angular sweep.

    Behavior:
        - Builds angular events for all wall endpoints
        - Performs sweep-line over angle domain
        - Tracks active walls and finds nearest intersection
        - Marks visible walls

    Args:
        P (tuple[float, float]): query point (x, y)
        walls (np.ndarray): wall segments (N,4)

    Returns:
        np.ndarray:
            binary mask (N,), 1 means visible, 0 means not visible
    """
    px = P[0]
    py = P[1]
    num_walls = walls.shape[0]
    max_events = num_walls * 4
    events_angle = np.empty(max_events, dtype=np.float64)
    events_idx = np.empty(max_events, dtype=np.int32)
    cnt = 0
    for i in range(num_walls):
        ax = walls[i, 0]
        ay = walls[i, 1]
        bx = walls[i, 2]
        by = walls[i, 3]
        ux = bx - ax
        uy = by - ay
        length2 = ux * ux + uy * uy
        cross = ux * (py - ay) - uy * (px - ax)
        dot = (px - ax) * ux + (py - ay) * uy
        if cross * cross <= 1e-12 * length2 and -1e-06 <= dot <= length2 + 1e-06:
            return wall_visibility_direction(P, walls)
        if (ax - px) ** 2 + (ay - py) ** 2 < 1e-12 or (bx - px) ** 2 + (by - py) ** 2 < 1e-12:
            return wall_visibility_direction(P, walls)
        a1 = math.atan2(ay - py, ax - px)
        a2 = math.atan2(by - py, bx - px)
        if a1 < a2:
            l = a1
            r = a2
        else:
            l = a2
            r = a1
        if r - l > math.pi + 1e-09:
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
    order = np.argsort(events_angle[:cnt], kind='mergesort')
    close_unequal = False
    for q in range(1, cnt):
        delta = events_angle[order[q]] - events_angle[order[q - 1]]
        if 0.0 < delta < 1e-12:
            close_unequal = True
            break
    if close_unequal:
        order = repair_near_angles(events_angle, events_idx, order, cnt)
    active = np.zeros(num_walls, dtype=np.uint8)
    active_ids = np.empty(num_walls, dtype=np.int32)
    positions = np.full(num_walls, -1, dtype=np.int32)
    active_count = 0
    current_best = -1
    added_ids = np.empty(max_events, dtype=np.int32)
    added_count = 0
    dirty = True
    visible = np.zeros(num_walls, dtype=np.uint8)
    for k in range(cnt - 1):
        idx = events_idx[order[k]]
        if active[idx]:
            if idx == current_best:
                dirty = True
            active[idx] = 0
            slot = positions[idx]
            active_count -= 1
            swapped = active_ids[active_count]
            active_ids[slot] = swapped
            positions[swapped] = slot
            positions[idx] = -1
        else:
            active[idx] = 1
            positions[idx] = active_count
            active_ids[active_count] = idx
            active_count += 1
            added_ids[added_count] = idx
            added_count += 1
        ang = events_angle[order[k]]
        next_ang = events_angle[order[k + 1]]
        if next_ang - ang < 1e-06:
            continue
        theta = 0.5 * (ang + next_ang)
        ray_dx = math.cos(theta)
        ray_dy = math.sin(theta)
        recompute = dirty or current_best < 0
        if not recompute:
            recompute = unsafe[current_best] or not active[current_best]
        if not recompute:
            best_i = current_best
            ax, ay, bx, by = walls[best_i]
            best_d = ray_direction_nb(px, py, ray_dx, ray_dy, ax, ay, bx, by)
            if best_d <= 1e-09:
                recompute = True
            else:
                for q in range(added_count):
                    i2 = added_ids[q]
                    if not active[i2]:
                        continue
                    ax, ay, bx, by = walls[i2]
                    d = ray_direction_nb(px, py, ray_dx, ray_dy, ax, ay, bx, by)
                    if d > 1e-09 and (d < best_d or (d == best_d and i2 < best_i)):
                        best_d = d
                        best_i = i2
        if recompute:
            best_d = 1e+18
            best_i = -1
            for slot in range(active_count):
                i2 = active_ids[slot]
                if True:
                    ax = walls[i2, 0]
                    ay = walls[i2, 1]
                    bx = walls[i2, 2]
                    by = walls[i2, 3]
                    d = ray_direction_nb(px, py, ray_dx, ray_dy, ax, ay, bx, by)
                    if d > 1e-09 and (d < best_d or (d == best_d and (best_i < 0 or i2 < best_i))):
                        best_d = d
                        best_i = i2
        current_best = best_i
        dirty = False
        added_count = 0
        if best_i != -1:
            visible[best_i] = 1
    return visible