import math
from numba import cuda
from .gpu_solver_packed_generated import visible_between_points_nb,gpu_min,gpu_max
@cuda.jit(device=True, inline=True, fastmath=False)
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
    raytol = 1e-09 * gpu_max(1.0, math.sqrt(squared))
    walltol = 1e-09 * gpu_max(1.0, math.hypot(vx, vy))
    o1 = dx * (ay - oy) - dy * (ax - ox)
    o2 = dx * (by - oy) - dy * (bx - ox)
    o3 = vx * (oy - ay) - vy * (ox - ax)
    o4 = vx * (y - ay) - vy * (x - ax)
    if (o1 > raytol and o2 < -raytol or (o1 < -raytol and o2 > raytol)) and (o3 > walltol and o4 < -walltol or (o3 < -walltol and o4 > walltol)):
        return False
    if t > 1 + 1e-08:
        return True
    return visible_between_points_nb(x, y, ox, oy, walls, gs, gc, gl, -1, 8)
@cuda.jit(fastmath=False)
def packed_pvp(geo,scale,corners,walls,gs,gc,gl,angles,nearest,counts,out):
    k=cuda.grid(1);nb=(len(corners)+7)//8
    if k>=out.size:return
    p=k//nb;byte=k%nb;y=p//geo.shape[1];x=p%geo.shape[1];bits=0
    if geo[y,x]==0:
        for j in range(8):
            c=byte*8+j
            if c<len(corners) and cached_visible(x/scale,y/scale,c,corners,walls,gs,gc,gl,angles,nearest,counts):bits|=1<<(7-j)
    out[y,x,byte]=bits

@cuda.jit
def fill_field(out):
    k=cuda.grid(1)
    if k<out.shape[0]*out.shape[1]:out[k//out.shape[1],k%out.shape[1],0]=1000.
