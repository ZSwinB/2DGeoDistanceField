import numpy as np
import math
import matplotlib.pyplot as plt

# ================= 路径 =================

WALL_PATH = r"D:\Desktop\RM\data\wall_segment\0.npy"
WWV_PATH  = r"D:\Desktop\RM\data\WWV\0.npy"
MOVE_PATH = r"D:\Desktop\RM\data\WWVmove\0.npy"

TARGET_WALL_ID = 40
GRID_SIZE = 256

EPS = 1e-9
ENDPOINT_EPS = 1e-4

# ================= geometry =================

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

# ================= angle scan with ray record (FIXED PVW) =================

def visible_walls_with_hit_rays(P, walls):
    events = []

    for i, (A, B) in enumerate(walls):
        a1 = angle(P, A)
        a2 = angle(P, B)
        l = min(a1, a2)
        r = max(a1, a2)

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
    rays = {}   # wall_id -> (theta, dist)

    for k in range(len(events) - 1):
        ang, idx = events[k]

        # --------- 核心修正：toggle 逻辑 ---------
        if idx in active:
            active.remove(idx)
        else:
            active.add(idx)
        # ----------------------------------------

        next_ang = events[k + 1][0]
        if next_ang - ang < 1e-6:
            continue

        theta = 0.5 * (ang + next_ang)

        best_d = None
        best_i = None

        for i2 in active:
            d = ray_segment_distance(P, theta, walls[i2][0], walls[i2][1])
            if d is not None and (best_d is None or d < best_d):
                best_d = d
                best_i = i2

        if best_i is not None and best_i not in rays:
            rays[best_i] = (theta, best_d)

    return rays

# ================= main =================

def main():
    walls = np.load(WALL_PATH, allow_pickle=True).tolist()
    WWV   = np.load(WWV_PATH)

    move_arr = np.load(MOVE_PATH, allow_pickle=True)
    WWVmove = move_arr.item() if move_arr.shape == () else move_arr

    N = len(walls)
    assert WWV.shape == (N, N)

    A, B = walls[TARGET_WALL_ID]
    moveA, moveB = WWVmove[TARGET_WALL_ID]

    visible_ids = np.where(WWV[TARGET_WALL_ID])[0].tolist()

    print("Target wall:", TARGET_WALL_ID)
    print("Visible walls:", visible_ids)
    print("Move A:", moveA, " Move B:", moveB)

    # ===== 采样点（与 WWV 构建完全一致） =====

    P0 = (A[0] + ENDPOINT_EPS * moveA[0],
          A[1] + ENDPOINT_EPS * moveA[1]) if moveA is not None else A

    P1 = (B[0] + ENDPOINT_EPS * moveB[0],
          B[1] + ENDPOINT_EPS * moveB[1]) if moveB is not None else B

    rays0 = visible_walls_with_hit_rays(P0, walls)
    rays1 = visible_walls_with_hit_rays(P1, walls)

    # ================= 可视化 =================

    plt.figure(figsize=(9, 9))

    # 所有墙（灰）
    for W0, W1 in walls:
        plt.plot([W0[0], W1[0]], [W0[1], W1[1]],
                 color="#cccccc", linewidth=0.8)

    # WWV 可见墙（橙）
    for j in visible_ids:
        A2, B2 = walls[j]
        plt.plot([A2[0], B2[0]], [A2[1], B2[1]],
                 color="orange", linewidth=2)

    # 目标墙（红）
    plt.plot([A[0], B[0]], [A[1], B[1]],
             color="red", linewidth=3)

    # 端点（黑）
    plt.scatter([A[0], B[0]], [A[1], B[1]],
                c="black", s=50, zorder=5)

    # 采样点（紫）
    plt.scatter([P0[0], P1[0]], [P0[1], P1[1]],
                c="purple", s=60, zorder=6)

    # free-move 方向箭头
    if moveA is not None:
        plt.arrow(A[0], A[1],
                  ENDPOINT_EPS * moveA[0] * 10,
                  ENDPOINT_EPS * moveA[1] * 10,
                  color="purple", width=0.01)

    if moveB is not None:
        plt.arrow(B[0], B[1],
                  ENDPOINT_EPS * moveB[0] * 10,
                  ENDPOINT_EPS * moveB[1] * 10,
                  color="purple", width=0.01)

    # 射线（绿）
    for j in visible_ids:
        if j in rays0:
            theta, dist = rays0[j]
            x2 = P0[0] + math.cos(theta) * dist
            y2 = P0[1] + math.sin(theta) * dist
            plt.plot([P0[0], x2], [P0[1], y2],
                     color="green", linewidth=1)

        if j in rays1:
            theta, dist = rays1[j]
            x2 = P1[0] + math.cos(theta) * dist
            y2 = P1[1] + math.sin(theta) * dist
            plt.plot([P1[0], x2], [P1[1], y2],
                     color="green", linewidth=1)

    plt.title(f"WWV Debug (free-move based) – wall {TARGET_WALL_ID}")
    plt.xlim(0, GRID_SIZE)
    plt.ylim(0, GRID_SIZE)
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.grid(True)
    plt.show()

# ================= 入口 =================

if __name__ == "__main__":
    main()
