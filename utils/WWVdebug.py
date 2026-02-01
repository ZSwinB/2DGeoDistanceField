import numpy as np
import math
import matplotlib.pyplot as plt

# ================= 路径 =================

WALL_PATH   = r"D:\Desktop\RM\data\wall_segment\0.npy"
MOVE_PATH   = r"D:\Desktop\RM\data\WWVmove\0.npy"
TARGET_WALL_ID = 0

# ================= 参数 =================

EPS = 1e-4

# ================= 几何函数（原样） =================

def angle(P, Q):
    return math.atan2(Q[1] - P[1], Q[0] - P[0])

def ray_segment_distance(P, theta, A, B):
    px, py = P
    dx = math.cos(theta)
    dy = math.sin(theta)

    ax, ay = A
    bx, by = B

    vx = bx - ax
    vy = by - ay

    det = dx * (-vy) - dy * (-vx)
    if abs(det) >= EPS:
        t = ((ax - px) * (-vy) - (ay - py) * (-vx)) / det
        u = ((ax - px) * (-dy) + (ay - py) * (dx)) / det
        if t > EPS and 0 <= u <= 1:
            return t
    return None


# ================= PVW（修正后的版本） =================

def visible_walls_for_point(P, walls):
    events = []

    for i, (A, B) in enumerate(walls):
        a1 = angle(P, A)
        a2 = angle(P, B)
        l, r = min(a1, a2), max(a1, a2)

        # 跨断点：拆，但不标 ENTER / EXIT
        if r - l > math.pi:
            events += [
                (l, i),
                ( math.pi, i),
                (-math.pi, i),
                (r, i)
            ]
        else:
            events += [(l, i), (r, i)]

    events.sort()

    active = set()
    visible = set()

    for k in range(len(events) - 1):
        ang, idx = events[k]

        # ---------- 核心修改：toggle ----------
        if idx in active:
            active.remove(idx)
        else:
            active.add(idx)
        # -------------------------------------

        next_ang = events[k + 1][0]
        if next_ang - ang < 1e-6:
            continue

        theta = 0.5 * (ang + next_ang)
        best_d, best_i = None, None

        for i2 in active:
            d = ray_segment_distance(P, theta, walls[i2][0], walls[i2][1])
            if d is not None and (best_d is None or d < best_d):
                best_d, best_i = d, i2

        if best_i is not None:
            visible.add(best_i)

    return list(visible)


# ================= Debug 主逻辑（不变） =================

def debug_single_wall(walls, moves, wall_id):
    A, B = walls[wall_id]
    moveA, moveB = moves[wall_id]

    # 用 WWVmove 给出的自由方向
    P0 = (A[0] + EPS * moveA[0], A[1] + EPS * moveA[1]) if moveA is not None else A
    P1 = (B[0] + EPS * moveB[0], B[1] + EPS * moveB[1]) if moveB is not None else B

    vis0 = visible_walls_for_point(P0, walls)
    vis1 = visible_walls_for_point(P1, walls)

    vis = set(vis0) | set(vis1)
    vis.discard(wall_id)

    print(f"[Debug] wall {wall_id}")
    print("  moveA:", moveA, " P0:", P0, "sees:", vis0)
    print("  moveB:", moveB, " P1:", P1, "sees:", vis1)
    print("  union:", vis)

    # ================= 可视化 =================

    plt.figure(figsize=(6, 6))

    # 所有墙（灰）
    for (C, D) in walls:
        plt.plot([C[0], D[0]], [C[1], D[1]],
                 color='lightgray', linewidth=1)

    # 可见墙（蓝）
    for j in vis:
        C, D = walls[j]
        plt.plot([C[0], D[0]], [C[1], D[1]],
                 color='blue', linewidth=2)

    # 目标墙（红）
    plt.plot([A[0], B[0]], [A[1], B[1]],
             color='red', linewidth=3)

    # 采样点（黑）
    plt.scatter([P0[0], P1[0]], [P0[1], P1[1]],
                color='black', s=40, zorder=5)

    plt.axis('equal')
    plt.title(f"WWV Debug – wall {wall_id}")
    plt.show()


# ================= 入口 =================

if __name__ == "__main__":
    walls = np.load(WALL_PATH, allow_pickle=True).tolist()

    move_arr = np.load(MOVE_PATH, allow_pickle=True)
    moves = move_arr.item() if move_arr.shape == () else move_arr

    debug_single_wall(walls, moves, TARGET_WALL_ID)
