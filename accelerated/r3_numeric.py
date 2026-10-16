import math
import numpy as np
from numba import njit
from .r3_pvw import wall_visibility_incremental
@njit(nogil=True, cache=True)
def raster(y0, rows, width, scale, verts, starts, bounds):
    g = np.zeros((rows, width), np.uint8)
    for i in range(rows):
        y = (i + y0) / scale
        for j in range(width):
            x = j / scale
            for r in range(len(starts) - 1):
                if x < bounds[r, 0] or x > bounds[r, 2] or y < bounds[r, 1] or (y > bounds[r, 3]):
                    continue
                inside = False
                boundary = False
                for k in range(starts[r], starts[r + 1]):
                    a = verts[k]
                    b = verts[starts[r] if k + 1 == starts[r + 1] else k + 1]
                    dx = b[0] - a[0]
                    dy = b[1] - a[1]
                    cross = dx * (y - a[1]) - dy * (x - a[0])
                    dot = (x - a[0]) * dx + (y - a[1]) * dy
                    length2 = dx * dx + dy * dy
                    if abs(cross) <= 1e-09 * np.sqrt(length2) and dot >= -1e-09 and (dot <= length2 + 1e-09):
                        boundary = True
                    if dy != 0 and (a[1] > y) != (b[1] > y) and (x < a[0] + (y - a[1]) * dx / dy):
                        inside = not inside
                if inside or boundary:
                    g[i, j] = 1
                    break
    return g
@njit(nogil=True, cache=True)
def unsafe_walls(walls):
    unsafe = np.zeros(len(walls), np.bool_)
    for i in range(len(walls)):
        ax, ay, bx, by = walls[i]
        ux = bx - ax
        uy = by - ay
        for j in range(i):
            cx, cy, dx, dy = walls[j]
            vx = dx - cx
            vy = dy - cy
            den = ux * vy - uy * vx
            if abs(den) > 1e-12:
                t = ((cx - ax) * vy - (cy - ay) * vx) / den
                u = ((cx - ax) * uy - (cy - ay) * ux) / den
                if 1e-12 < t < 1 - 1e-12 and 1e-12 < u < 1 - 1e-12:
                    unsafe[i] = True
                    unsafe[j] = True
            elif abs(ux * (cy - ay) - uy * (cx - ax)) < 1e-09:
                t0 = ((cx - ax) * ux + (cy - ay) * uy) / (ux * ux + uy * uy)
                t1 = ((dx - ax) * ux + (dy - ay) * uy) / (ux * ux + uy * uy)
                if min(1.0, max(t0, t1)) - max(0.0, min(t0, t1)) > 1e-12:
                    unsafe[i] = True
                    unsafe[j] = True
    return unsafe
@njit(cache=True,nogil=True)
def pvw_tile(points,y0,rows,width,scale,walls,unsafe):
    out=np.zeros((rows,width,(len(walls)+7)//8),np.uint8)
    for k in range(len(points)):
        i,j=points[k]
        vis=wall_visibility_incremental((j/scale,(i+y0)/scale),walls,unsafe)
        for c in range(len(walls)):
            if vis[c]:out[i,j,c//8]|=np.uint8(1<<(7-c%8))
    return out
