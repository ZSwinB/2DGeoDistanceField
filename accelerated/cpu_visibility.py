"""Wall visibility with proper crossings and through-vertex crossings.

A ray may touch a boundary tangentially or end at a diffraction corner. A wall
vertex strictly inside the ray blocks it when incident edges lie on both sides.
"""
import math
from numba import njit


@njit
def _orient(ax, ay, bx, by, cx, cy):
    return (bx-ax)*(cy-ay)-(by-ay)*(cx-ax)


@njit
def _vertex_crosses_ray(x0,y0,x1,y1,vx,vy,walls,starts,counts,indices,skip_w,cell_size,tol):
    dx,dy=x1-x0,y1-y0
    squared=dx*dx+dy*dy
    projection=(vx-x0)*dx+(vy-y0)*dy
    # Endpoints of the query may lie on walls for reflection/diffraction.
    if projection <= tol or projection >= squared-tol:
        return False
    gx=int(vx//cell_size);gy=int(vy//cell_size)
    if gx<0 or gy<0 or gx>=starts.shape[0] or gy>=starts.shape[1]:
        return False
    positive=False;negative=False
    begin=starts[gx,gy];end=begin+counts[gx,gy]
    for k in range(begin,end):
        idx=indices[k]
        if idx==skip_w:continue
        ax,ay,bx,by=walls[idx]
        side=0.0
        if abs(ax-vx)<=1e-9 and abs(ay-vy)<=1e-9:
            side=_orient(x0,y0,x1,y1,bx,by)
        elif abs(bx-vx)<=1e-9 and abs(by-vy)<=1e-9:
            side=_orient(x0,y0,x1,y1,ax,ay)
        else:continue
        if side>tol:positive=True
        if side<-tol:negative=True
        if positive and negative:return True
    return False


@njit
def visible_between_points_nb(x0,y0,x1,y1,walls,starts,counts,indices,skip_w,cell_size):
    dx,dy=x1-x0,y1-y0
    squared=dx*dx+dy*dy
    if squared==0:return True
    tol=1e-9*max(1.0,math.sqrt(squared))
    gx0=max(0,int(min(x0,x1)//cell_size));gx1=min(starts.shape[0]-1,int(max(x0,x1)//cell_size))
    gy0=max(0,int(min(y0,y1)//cell_size));gy1=min(starts.shape[1]-1,int(max(y0,y1)//cell_size))
    for gx in range(gx0,gx1+1):
        for gy in range(gy0,gy1+1):
            begin=starts[gx,gy];end=begin+counts[gx,gy]
            for k in range(begin,end):
                idx=indices[k]
                if idx==skip_w:continue
                ax,ay,bx,by=walls[idx]
                o1=_orient(x0,y0,x1,y1,ax,ay);o2=_orient(x0,y0,x1,y1,bx,by)
                o3=_orient(ax,ay,bx,by,x0,y0);o4=_orient(ax,ay,bx,by,x1,y1)
                wall_tol=1e-9*max(1.0,math.hypot(bx-ax,by-ay))
                ray_opposite=(o1>tol and o2<-tol) or (o1<-tol and o2>tol)
                wall_opposite=(o3>wall_tol and o4<-wall_tol) or (o3<-wall_tol and o4>wall_tol)
                if ray_opposite and wall_opposite:
                    return False
                if abs(o1)<=tol and _vertex_crosses_ray(x0,y0,x1,y1,ax,ay,walls,starts,counts,indices,skip_w,cell_size,tol):
                    return False
                if abs(o2)<=tol and _vertex_crosses_ray(x0,y0,x1,y1,bx,by,walls,starts,counts,indices,skip_w,cell_size,tol):
                    return False
    return True
