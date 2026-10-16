"""Numeric replacements for Python WWV and source setup; scientific rules unchanged."""
import math
import numpy as np
from numba import njit


@njit(nogil=True)
def wwv_point(px, py, walls):
    # Tuple sorting preserves the original (angle, wall id) event ordering.
    events = []
    for i in range(len(walls)):
        ax, ay, bx, by = walls[i]
        a = math.atan2(ay-py, ax-px)
        b = math.atan2(by-py, bx-px)
        l = min(a, b); r = max(a, b)
        if r-l > math.pi:
            events.extend([(l, i), (math.pi, i), (-math.pi, i), (r, i)])
        else:
            events.extend([(l, i), (r, i)])
    events.sort()
    active=np.empty(len(walls),np.int32)
    positions=np.full(len(walls),-1,np.int32)
    active_count=0
    visible = np.zeros(len(walls), np.bool_)
    tied = False
    for k in range(len(events)-1):
        ang, idx = events[k]
        position=positions[idx]
        if position>=0:
            active_count-=1
            moved=active[active_count];active[position]=moved;positions[moved]=position
            positions[idx]=-1
        else:
            active[active_count]=idx;positions[idx]=active_count;active_count+=1
        next_ang = events[k+1][0]
        if next_ang-ang < 1e-6: continue
        theta = .5*(ang+next_ang)
        dx = math.cos(theta); dy = math.sin(theta)
        best = math.inf; winner = -1
        for ai in range(active_count):
            i=active[ai]
            ax, ay, bx, by = walls[i]; vx = bx-ax; vy = by-ay
            det = dx*(-vy)-dy*(-vx)
            if abs(det) >= 1e-9:
                t = ((ax-px)*(-vy)-(ay-py)*(-vx))/det
                u = ((ax-px)*(-dy)+(ay-py)*dx)/det
                if t > 1e-9 and 0 <= u <= 1:
                    if t < best: best=t; winner=i
                    elif t == best: tied=True
        if winner >= 0: visible[winner]=True
    # Python and Numba integer sets may visit ties in a different order.
    # Caller uses the original Python sweep for any such point.
    return visible, tied


@njit(nogil=True)
def source_prefixes(tx, walls, corners, order, imgs, gs, gc, gl,
                    intersect, visible):
    shape=(len(walls),len(corners))
    ok=np.zeros(shape,np.bool_); px=np.zeros(shape); py=np.zeros(shape); length=np.zeros(shape)
    for wi in range(len(order)):
        w=order[wi]
        for cid in range(len(corners)):
            cx,cy=corners[cid]
            yes,x,y=intersect(imgs[w,0],imgs[w,1],cx,cy,walls[w,0],walls[w,1],walls[w,2],walls[w,3])
            if not yes:continue
            if not visible(tx[0],tx[1],x,y,walls,gs,gc,gl,w):continue
            if not visible(x,y,cx,cy,walls,gs,gc,gl,w):continue
            ok[w,cid]=True;px[w,cid]=x;py[w,cid]=y
            dx1=tx[0]-x;dy1=tx[1]-y;dx2=x-cx;dy2=y-cy
            length[w,cid]=math.sqrt(dx1*dx1+dy1*dy1)+math.sqrt(dx2*dx2+dy2*dy2)
    return ok,px,py,length


@njit(nogil=True)
def source_mirrors(walls,corners,order,txids,imgs,wwv,reflect):
    nw=len(walls);targets=np.zeros((nw,nw),np.int32);counts=np.zeros(nw,np.int32)
    double=np.zeros((nw,nw,2),np.float64)
    for ii in range(len(order)):
        w1=order[ii]
        for w2 in range(nw):
            if wwv[w1,w2]:
                targets[w1,counts[w1]]=w2;counts[w1]+=1
                double[w1,w2]=reflect(imgs[w1,0],imgs[w1,1],walls[w2,0],walls[w2,1],walls[w2,2],walls[w2,3])
    cornerimgs=np.zeros((nw,len(corners),2),np.float64)
    for w in range(nw):
        for i in range(len(txids)):
            c=txids[i]
            cornerimgs[w,c]=reflect(corners[c,0],corners[c,1],walls[w,0],walls[w,1],walls[w,2],walls[w,3])
    return targets,counts,double,cornerimgs


@njit(nogil=True)
def packed_to_csr(packed, points, ids, y0, stride):
    """Read only eligible bits; avoid a dense RX-by-all-corners temporary."""
    rows=packed.shape[0]//stride
    width=packed.shape[1]//stride
    ps=np.zeros((rows,width),np.int32);pl=np.zeros((rows,width),np.int16)
    lens=np.zeros(len(points),np.int32)
    for n in range(len(points)):
        i,j=points[n]
        for k in range(len(ids)):
            c=ids[k]
            lens[n]+=(packed[i*stride,j*stride,c//8]>>(7-c%8))&1
    total=np.sum(lens)
    flat=np.empty(total,np.int32);pos=0
    for n in range(len(points)):
        i,j=points[n];ps[i,j]=pos;pl[i,j]=lens[n]
        for k in range(len(ids)):
            c=ids[k]
            if (packed[i*stride,j*stride,c//8]>>(7-c%8))&1:
                flat[pos]=c;pos+=1
    return ps,pl,flat
