import numpy as np
import math
import matplotlib.pyplot as plt

# ================= 路径 =================
WALL_PATH = r"D:\Desktop\RM\data\wall_segment\0.npy"
LOG_PATH  = r"pvw_event_log.txt"

# ================= 固定点 =================
P = (10, 108)
EPS = 1e-9

# ================= 几何工具 =================

def angle(P, Q):
    return math.atan2(Q[1]-P[1], Q[0]-P[0])

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
        t = ((ax - px)*(-vy) - (ay - py)*(-vx)) / det
        u = ((ax - px)*(-dy) + (ay - py)*(dx)) / det
        if t > EPS and 0 <= u <= 1:
            return t
    return None


# ================= 角度扫描（修改 ENTER / EXIT 逻辑） =================

def visible_walls_with_rays(P, walls, logf):

    events = []

    for i, (A, B) in enumerate(walls):
        a1 = angle(P, A)
        a2 = angle(P, B)

        l = min(a1, a2)
        r = max(a1, a2)

        if r - l > math.pi:
            # 跨断点：拆成两段，但【不再标 ENTER / EXIT】
            events += [
                (l, i),
                ( math.pi, i),
                (-math.pi, i),
                (r, i)
            ]
        else:
            events += [(l, i), (r, i)]

    # 只按角度排序
    events.sort()

    logf.write("===== EVENT LIST (sorted, no ENTER/EXIT) =====\n")
    for ang, idx in events:
        logf.write(f"{ang:+.6f}  wall {idx:4d}\n")

    active = set()
    visible = set()
    rays = {}

    logf.write("\n===== SWEEP LOG =====\n")

    for k in range(len(events)-1):
        ang, idx = events[k]

        # ==================== MODIFIED ====================
        # toggle 逻辑：由 active 决定 ENTER / EXIT
        if idx in active:
            active.remove(idx)
            action = "EXIT "
        else:
            active.add(idx)
            action = "ENTER"
        # ==================================================

        next_ang = events[k+1][0]

        logf.write(
            f"\n[k={k:4d}] "
            f"{ang:+.6f} -> {next_ang:+.6f} | "
            f"{action} wall {idx:4d}\n"
        )
        logf.write(f"    active = {sorted(active)}\n")

        if next_ang - ang < 1e-6:
            logf.write("    skip: tiny interval\n")
            continue

        theta = 0.5 * (ang + next_ang)

        best_d = None
        best_i = None

        for i2 in active:
            d = ray_segment_distance(P, theta, walls[i2][0], walls[i2][1])
            if d is not None:
                logf.write(f"        hit cand wall {i2:4d} dist={d:.6f}\n")
                if best_d is None or d < best_d:
                    best_d = d
                    best_i = i2

        if best_i is not None:
            visible.add(best_i)
            if best_i not in rays:
                rays[best_i] = (theta, best_d)
            logf.write(f"    ==> SELECT wall {best_i} dist={best_d:.6f}\n")
        else:
            logf.write("    ==> no hit\n")

    return sorted(visible), rays


# ================= 可视化 =================

def visualize_with_all_rays(P, walls, visible_set, rays):

    plt.figure(figsize=(9,9))

    for wid, (A, B) in enumerate(walls):
        plt.plot([A[0],B[0]],[A[1],B[1]], color="#cccccc", linewidth=0.7)
        mx = 0.5 * (A[0] + B[0])
        my = 0.5 * (A[1] + B[1])
        plt.text(mx, my, str(wid), color="#888888", fontsize=5)

    for i in visible_set:
        A,B = walls[i]
        plt.plot([A[0],B[0]],[A[1],B[1]], color="red", linewidth=2)

    for i, (theta, dist) in rays.items():
        x2 = P[0] + math.cos(theta) * dist
        y2 = P[1] + math.sin(theta) * dist
        plt.plot([P[0], x2], [P[1], y2], color="green", linewidth=1)

    plt.scatter([P[0]],[P[1]], c="black", s=60, zorder=5)

    plt.xlim(0,256)
    plt.ylim(0,256)
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.grid(True)
    plt.title("Red=visible walls, Green=rays")
    plt.show()


# ================= 主程序 =================

def main():
    walls = np.load(WALL_PATH, allow_pickle=True).tolist()

    with open(LOG_PATH, "w", encoding="utf-8") as logf:
        logf.write(f"Point: {P}\n")
        logf.write(f"Total walls: {len(walls)}\n\n")

        visible, rays = visible_walls_with_rays(P, walls, logf)

    print(f"[OK] event log written to: {LOG_PATH}")

    visualize_with_all_rays(P, walls, set(visible), rays)


if __name__ == "__main__":
    main()
