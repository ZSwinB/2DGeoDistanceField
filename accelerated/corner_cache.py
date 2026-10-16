"""Experimental continuous-angle corner visibility cache with exact degenerate fallback."""
import math
import numpy as np
from numba import njit
from .cpu_visibility import visible_between_points_nb

@njit(nogil=True, cache=True)
def wall_intersections(walls):
    points = np.empty((len(walls) * len(walls) // 2, 2), np.float64)
    n = 0
    for i in range(len(walls)):
        ax, ay, bx, by = walls[i]
        ux = bx - ax
        uy = by - ay
        for j in range(i):
            cx, cy, dx, dy = walls[j]
            vx = dx - cx
            vy = dy - cy
            den = ux * vy - uy * vx
            if abs(den) < 1e-12:
                continue
            t = ((cx - ax) * vy - (cy - ay) * vx) / den
            u = ((cx - ax) * uy - (cy - ay) * ux) / den
            if 1e-12 < t < 1 - 1e-12 and 1e-12 < u < 1 - 1e-12:
                points[n, 0] = ax + t * ux
                points[n, 1] = ay + t * uy
                n += 1
    return points[:n]

@njit(nogil=True, cache=True)
def make_tables(corners, walls, critical_points):
    maxn = len(critical_points) + 2
    angles = np.empty((len(corners), maxn), np.float64)
    nearest = np.full((len(corners), maxn), -1, np.int32)
    counts = np.zeros(len(corners), np.int32)
    for c in range(len(corners)):
        ox, oy = corners[c]
        raw = np.empty(maxn, np.float64)
        n = 2
        raw[0] = -math.pi
        raw[1] = math.pi
        for k in range(len(critical_points)):
            x, y = critical_points[k]
            if abs(x - ox) + abs(y - oy) < 1e-12:
                continue
            raw[n] = math.atan2(y - oy, x - ox)
            n += 1
        ordered = np.sort(raw[:n])
        count = 0
        for k in range(n):
            if count == 0 or ordered[k] != angles[c, count - 1]:
                angles[c, count] = ordered[k]
                count += 1
        counts[c] = count
        for k in range(count - 1):
            theta = (angles[c, k] + angles[c, k + 1]) * 0.5
            rx = math.cos(theta)
            ry = math.sin(theta)
            best = 1e+100
            wid = -1
            for w in range(len(walls)):
                ax, ay, bx, by = walls[w]
                vx = bx - ax
                vy = by - ay
                orient = vx * (oy - ay) - vy * (ox - ax)
                dot = (ox - ax) * vx + (oy - ay) * vy
                len2 = vx * vx + vy * vy
                if abs(orient) <= 1e-09 * max(1.0, math.sqrt(len2)) and -1e-09 <= dot <= len2 + 1e-09:
                    continue
                den = rx * vy - ry * vx
                if abs(den) < 1e-15:
                    continue
                t = ((ax - ox) * vy - (ay - oy) * vx) / den
                u = ((ax - ox) * ry - (ay - oy) * rx) / den
                if t > 0 and 0 <= u <= 1 and (t < best):
                    best = t
                    wid = w
            nearest[c, k] = wid
    return (angles, nearest, counts)

@njit(nogil=True, cache=True)
def cached_visible(x, y, c, corners, walls, gs, gc, gl, angles, nearest, counts):
    if x < 0.0 or y < 0.0 or x >= gs.shape[0] * 8 or (y >= gs.shape[1] * 8):
        return visible_between_points_nb(x, y, corners[c, 0], corners[c, 1], walls, gs, gc, gl, -1, 8)
    ox, oy = corners[c]
    dx = x - ox
    dy = y - oy
    squared = dx * dx + dy * dy
    if squared == 0:
        return True
    theta = math.atan2(dy, dx)
    lo = 0
    hi = counts[c]
    while lo < hi:
        m = (lo + hi) // 2
        if angles[c, m] <= theta:
            lo = m + 1
        else:
            hi = m
    k = lo - 1
    if k < 0 or k + 1 >= counts[c] or theta - angles[c, k] < 1e-08 or (angles[c, k + 1] - theta < 1e-08):
        return visible_between_points_nb(x, y, ox, oy, walls, gs, gc, gl, -1, 8)
    w = nearest[c, k]
    if w < 0:
        return True
    ax, ay, bx, by = walls[w]
    vx = bx - ax
    vy = by - ay
    den = dx * vy - dy * vx
    if den == 0:
        return visible_between_points_nb(x, y, ox, oy, walls, gs, gc, gl, -1, 8)
    t = ((ax - ox) * vy - (ay - oy) * vx) / den
    raytol = 1e-09 * max(1.0, math.sqrt(squared))
    walltol = 1e-09 * max(1.0, math.hypot(vx, vy))
    o1 = dx * (ay - oy) - dy * (ax - ox)
    o2 = dx * (by - oy) - dy * (bx - ox)
    o3 = vx * (oy - ay) - vy * (ox - ax)
    o4 = vx * (y - ay) - vy * (x - ax)
    if (o1 > raytol and o2 < -raytol or (o1 < -raytol and o2 > raytol)) and (o3 > walltol and o4 < -walltol or (o3 < -walltol and o4 > walltol)):
        return False
    if t > 1 + 1e-08:
        return True
    return visible_between_points_nb(x, y, ox, oy, walls, gs, gc, gl, -1, 8)

@njit(nogil=True, cache=True)
def batch_cached(points, corners, walls, gs, gc, gl, angles, nearest, counts, scale):
    out = np.zeros((len(points), len(corners)), np.bool_)
    for k in range(len(points)):
        y, x = points[k]
        for c in range(len(corners)):
            out[k, c] = cached_visible(x / scale, y / scale, c, corners, walls, gs, gc, gl, angles, nearest, counts)
    return out

def prepare(corners, walls):
    intersections = wall_intersections(walls)
    critical = np.unique(np.concatenate([corners, intersections]), axis=0)
    return (make_tables(corners, walls, critical), len(intersections))