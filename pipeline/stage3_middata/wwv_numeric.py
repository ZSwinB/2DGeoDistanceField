"""Compiled WWV sweep, preserving the Python sweep's events and tolerances."""
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def visible_walls_numeric(px, py, walls, eps):
    events = []
    for i in range(len(walls)):
        ax, ay, bx, by = walls[i]
        a = math.atan2(ay-py, ax-px)
        b = math.atan2(by-py, bx-px)
        left = min(a, b)
        right = max(a, b)
        if right-left > math.pi:
            events.extend([(left, i), (math.pi, i), (-math.pi, i), (right, i)])
        else:
            events.extend([(left, i), (right, i)])
    events.sort()
    active = np.empty(len(walls), np.int32)
    positions = np.full(len(walls), -1, np.int32)
    count = 0
    visible = np.zeros(len(walls), np.bool_)
    tied = False
    for k in range(len(events)-1):
        angle, idx = events[k]
        position = positions[idx]
        if position >= 0:
            count -= 1
            moved = active[count]
            active[position] = moved
            positions[moved] = position
            positions[idx] = -1
        else:
            active[count] = idx
            positions[idx] = count
            count += 1
        next_angle = events[k+1][0]
        if next_angle-angle < 1e-6:
            continue
        theta = .5*(angle+next_angle)
        dx, dy = math.cos(theta), math.sin(theta)
        best, winner = math.inf, -1
        for ai in range(count):
            i = active[ai]
            ax, ay, bx, by = walls[i]
            vx, vy = bx-ax, by-ay
            det = dx*(-vy)-dy*(-vx)
            if abs(det) >= eps:
                t = ((ax-px)*(-vy)-(ay-py)*(-vx))/det
                u = ((ax-px)*(-dy)+(ay-py)*dx)/det
                if t > eps and 0 <= u <= 1:
                    if t < best:
                        best, winner = t, i
                    elif t == best:
                        tied = True
        if winner >= 0:
            visible[winner] = True
    return visible, tied


def compute_wwv(walls, movements, reference_visibility, workers=1,
                eps=1e-9, endpoint_eps=1e-4):
    numeric = np.asarray(walls, dtype=np.float64).reshape(-1, 4)
    result = np.zeros((len(numeric), len(numeric)), dtype=np.bool_)
    if not len(numeric):
        return result
    # Compile before worker threads enter the dispatcher.
    visible_walls_numeric(0., 0., numeric, float(eps))

    def one_wall(wid):
        visible = np.zeros(len(numeric), dtype=np.bool_)
        for endpoint in range(2):
            move = movements[wid][endpoint]
            if move is None:
                continue
            a = numeric[wid, endpoint*2:endpoint*2+2]
            point = (a[0]+endpoint_eps*move[0], a[1]+endpoint_eps*move[1])
            mask, tied = visible_walls_numeric(*point, numeric, float(eps))
            if tied:
                # Python set iteration is part of the historical tie behavior.
                mask[:] = False
                mask[reference_visibility(point, walls)] = True
            visible |= mask
        visible[wid] = False
        return wid, np.flatnonzero(visible)

    def collect(rows):
        for wid, ids in rows:
            result[wid, ids] = True
            result[ids, wid] = True

    if workers <= 1:
        collect(map(one_wall, range(len(numeric))))
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            collect(pool.map(one_wall, range(len(numeric))))
    return result
