import math
EPS=1e-9
ENDPOINT_EPS=1e-4
DIRECTIONS_8=[(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
def in_bounds(x, y, W, H):
    """
    Check whether a coordinate is within image bounds.

    Returns:
        bool: True if inside [0,W) x [0,H), otherwise False
    """
    return 0 <= x < W and 0 <= y < H

def first_free_dir(P, geo):
    """
    Find the first free direction (8-neighborhood) from a point.

    Behavior:
        - Rounds P to nearest grid coordinate
        - Searches 8 directions for a free cell (geo == 0)

    Args:
        P (tuple[float, float]): point (x, y)
        geo (np.ndarray): geometry map

    Returns:
        tuple[int, int] | None:
            direction (dx, dy) if found, otherwise None
    """
    H, W = geo.shape
    x0, y0 = (int(round(P[0])), int(round(P[1])))
    for dx, dy in DIRECTIONS_8:
        x = x0 + dx
        y = y0 + dy
        if not in_bounds(x, y, W, H):
            continue
        if geo[y, x] == 0:
            return (dx, dy)
    return None

def build_WWVmove(geo, walls):
    """
    Build movement directions for each wall endpoint.

    Behavior:
        - For each wall, computes a valid offset direction
          from both endpoints into free space

    Args:
        geo (np.ndarray): geometry map
        walls: list of wall segments [(A,B), ...]

    Returns:
        dict:
            {wall_id: [dir_A, dir_B]}
            each dir is (dx, dy) or None
    """
    WWVmove = {}
    for wid, (A, B) in enumerate(walls):
        dir_A = first_free_dir(A, geo)
        dir_B = first_free_dir(B, geo)
        WWVmove[wid] = [dir_A, dir_B]
    return WWVmove

def angle(P, Q):
    """
    Compute angle from point P to Q.

    Returns:
        float: angle in radians [-pi, pi]
    """
    return math.atan2(Q[1] - P[1], Q[0] - P[0])

def ray_segment_distance(P, theta, A, B):
    """
    Compute intersection distance between a ray and a segment.

    Behavior:
        - Ray starts at P with direction theta
        - Segment defined by A → B
        - Returns closest valid intersection

    Returns:
        float | None:
            distance if intersection exists, otherwise None
    """
    px, py = P
    dx = math.cos(theta)
    dy = math.sin(theta)
    ax, ay = A
    bx, by = B
    vx = bx - ax
    vy = by - ay
    det = dx * -vy - dy * -vx
    if abs(det) >= EPS:
        t = ((ax - px) * -vy - (ay - py) * -vx) / det
        u = ((ax - px) * -dy + (ay - py) * dx) / det
        if t > EPS and 0 <= u <= 1:
            return t
    return None

def visible_walls_for_point(P, walls):
    """
    Compute visible walls from a point using angular sweep.

    Behavior:
        - Builds angular events from wall endpoints
        - Performs sweep-line over angle domain
        - Tracks active walls and selects nearest intersection

    Args:
        P (tuple[float, float]): query point
        walls: list of wall segments

    Returns:
        list[int]: indices of visible walls
    """
    events = []
    for i, (A, B) in enumerate(walls):
        a1 = angle(P, A)
        a2 = angle(P, B)
        l = min(a1, a2)
        r = max(a1, a2)
        if r - l > math.pi:
            events += [(l, i), (math.pi, i), (-math.pi, i), (r, i)]
        else:
            events += [(l, i), (r, i)]
    events.sort()
    active = set()
    visible = set()
    for k in range(len(events) - 1):
        ang, idx = events[k]
        if idx in active:
            active.remove(idx)
        else:
            active.add(idx)
        next_ang = events[k + 1][0]
        if next_ang - ang < 1e-06:
            continue
        theta = 0.5 * (ang + next_ang)
        best_d = None
        best_i = None
        for i2 in active:
            d = ray_segment_distance(P, theta, walls[i2][0], walls[i2][1])
            if d is not None and (best_d is None or d < best_d):
                best_d = d
                best_i = i2
        if best_i is not None:
            visible.add(best_i)
    return list(visible)