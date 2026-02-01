import numpy as np
from PIL import Image
import math
import os
import time
from collections import defaultdict
import random
from numba import njit
# =====================================================
# 参数
# =====================================================
MAX_DIST = 1000.0

GEO_ROOT  = r"/root/RM/geo"
ANT_ROOT  = r"/root/RM/antenna"
GAIN_ROOT = r"/root/RM/IRT2"

WALL_ROOT = r"/root/RM/wall_segment"
WWV_ROOT  = r"/root/RM/data/WWV"
PVW_ROOT  = r"/root/RM/data/PVW"
PVP_ROOT  = r"/root/RM/data/PVP"
PVP_ID_ROOT = r"/root/RM/data/PVP_id"
CORNER_TABLE_ROOT = r"/root/RM/data/corner_table"
PVW_MASK_ROOT = r"/root/RM/data/PVW_mask"

OUT_DIR = r"/root/RM/distance_map"
LOG_PATH = r"/root/RM/debug_profile.log"

GRID_SIZE = 8
SCENE_ID = 0
TX_ID = 0

NUM_RX = 5000     # 🔴 改成 1 就只跑一个 RX

# =====================================================
# Debug 统计工具（只加，不改逻辑）
# =====================================================
T_TIME = defaultdict(float)
T_CNT  = defaultdict(int)
from collections import defaultdict

DBG_CNT = defaultdict(int)
DBG_SUM = defaultdict(int)
MAIN_START_TIME = None
MAIN_TOTAL_TIME = 0.0

def dbg_enter(name):
    T_CNT[name] += 1
    return time.perf_counter()

def dbg_exit(name, t0):
    T_TIME[name] += time.perf_counter() - t0

# =====================================================
# 基础几何（不改）
# =====================================================
def build_wall_grid(walls):
    t0 = dbg_enter("build_wall_grid")

    grid = {}
    for idx, (A, B) in enumerate(walls):
        x0, y0 = A
        x1, y1 = B

        gx0 = int(min(x0, x1) // GRID_SIZE)
        gx1 = int(max(x0, x1) // GRID_SIZE)
        gy0 = int(min(y0, y1) // GRID_SIZE)
        gy1 = int(max(y0, y1) // GRID_SIZE)

        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                grid.setdefault((gx, gy), []).append(idx)

    dbg_exit("build_wall_grid", t0)
    return grid


def reflect_point(p, wall):
    t0 = dbg_enter("reflect_point")
    p = np.array(p, float)
    a = np.array(wall[0], float)
    b = np.array(wall[1], float)
    v = b - a
    v = v / np.linalg.norm(v)
    proj = a + v * np.dot(p - a, v)
    dbg_exit("reflect_point", t0)
    return (2 * proj - p).tolist()


def orient(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def on_segment(a, b, c, eps=1e-9):
    return (
        min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps and
        min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps
    )


def segment_intersect_strict(a, b, c, d):
    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return (o1 * o2 < 0 and o3 * o4 < 0)


def segment_intersect2(a, b, c, d, eps=1):
    t0 = dbg_enter("segment_intersect2")
    a = np.array(a, float)
    b = np.array(b, float)
    c = np.array(c, float)
    d = np.array(d, float)

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    if o1 * o2 < 0 and o3 * o4 < 0:
        return True

    if abs(o1) < eps and on_segment(a, b, c, eps): return True
    if abs(o2) < eps and on_segment(a, b, d, eps): return True
    if abs(o3) < eps and on_segment(c, d, a, eps): return True
    if abs(o4) < eps and on_segment(c, d, b, eps): return True
    dbg_exit("segment_intersect2", t0)
    return False


def visible_fast(p1, p2, walls, wall_grid, skip_w=None):
    t0 = dbg_enter("visible_fast")

    x0, y0 = p1
    x1, y1 = p2

    gx0 = int(min(x0, x1) // GRID_SIZE)
    gx1 = int(max(x0, x1) // GRID_SIZE)
    gy0 = int(min(y0, y1) // GRID_SIZE)
    gy1 = int(max(y0, y1) // GRID_SIZE)

    checked = set()

    for gx in range(gx0, gx1 + 1):
        for gy in range(gy0, gy1 + 1):
            for idx in wall_grid.get((gx, gy), []):
                if skip_w is not None and idx == skip_w:
                    continue
                if idx in checked:
                    continue
                checked.add(idx)
                A, B = walls[idx]
                if segment_intersect_strict(p1, p2, A, B):
                    dbg_exit("visible_fast", t0)
                    return False

    dbg_exit("visible_fast", t0)
    return True


# =====================================================
# numba
# =====================================================
@njit
def orient_nb(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


@njit
def on_segment_nb(ax, ay, bx, by, cx, cy, eps):
    return (
        min(ax, bx) - eps <= cx <= max(ax, bx) + eps and
        min(ay, by) - eps <= cy <= max(ay, by) + eps
    )


@njit
def segment_intersect2_nb(
    ax, ay,
    bx, by,
    cx, cy,
    dx, dy,
    eps
):
    o1 = orient_nb(ax, ay, bx, by, cx, cy)
    o2 = orient_nb(ax, ay, bx, by, dx, dy)
    o3 = orient_nb(cx, cy, dx, dy, ax, ay)
    o4 = orient_nb(cx, cy, dx, dy, bx, by)

    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return True

    if abs(o1) < eps and on_segment_nb(ax, ay, bx, by, cx, cy, eps):
        return True
    if abs(o2) < eps and on_segment_nb(ax, ay, bx, by, dx, dy, eps):
        return True
    if abs(o3) < eps and on_segment_nb(cx, cy, dx, dy, ax, ay, eps):
        return True
    if abs(o4) < eps and on_segment_nb(cx, cy, dx, dy, bx, by, eps):
        return True

    return False


@njit
def dist2d_nb(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx*dx + dy*dy)


def valid_first_reflection_nb(tx_img, rx, wall):
    return segment_intersect2_nb(
        tx_img[0], tx_img[1],
        rx[0], rx[1],
        wall[0][0], wall[0][1],
        wall[1][0], wall[1][1],
        1e-9
    )


@njit
def second_reflection_dist_nb(
    tx_x, tx_y,
    rx_x, rx_y,
    w1x1, w1y1, w1x2, w1y2,
    w2x1, w2y1, w2x2, w2y2,
    best,
    eps
):
    # ---------- TX 侧两次反射 ----------
    tx1x, tx1y = reflect_point_nb(tx_x, tx_y, w1x1, w1y1, w1x2, w1y2)
    tx2x, tx2y = reflect_point_nb(tx1x, tx1y, w2x1, w2y1, w2x2, w2y2)

    # ---------- RX 侧两次反射 ----------
    rx1x, rx1y = reflect_point_nb(rx_x, rx_y, w2x1, w2y1, w2x2, w2y2)
    rx2x, rx2y = reflect_point_nb(rx1x, rx1y, w1x1, w1y1, w1x2, w1y2)


    # ---------- 距离（真实路径） ----------
    d = dist2d_nb(tx2x, tx2y, rx_x, rx_y)
    if d >= best:
        return best   # 剪枝

    # ---------- 几何可行性 ----------
    if not segment_intersect2_nb(
        tx2x, tx2y,
        rx_x, rx_y,
        w2x1, w2y1, w2x2, w2y2,
        eps
    ):
        return best

    if not segment_intersect2_nb(
        rx2x, rx2y,
        tx_x, tx_y,
        w1x1, w1y1, w1x2, w1y2,
        eps
    ):
        return best

    return d


@njit
def reflect_point_nb(px, py, ax, ay, bx, by):
    # wall direction
    vx = bx - ax
    vy = by - ay

    # normalize v
    norm = math.sqrt(vx*vx + vy*vy)
    vx /= norm
    vy /= norm

    # vector from a to p
    apx = px - ax
    apy = py - ay

    # projection length
    t = apx * vx + apy * vy

    # projection point
    projx = ax + vx * t
    projy = ay + vy * t

    # reflection
    rx = 2.0 * projx - px
    ry = 2.0 * projy - py

    return rx, ry


@njit
def intersect_wall_return_point_nb(
    px, py,   # P：镜像点
    qx, qy,   # Q：corner
    ax, ay,   # A：墙端点
    bx, by,   # B：墙端点
    eps=1e-9
):
    # PQ = P + t * d
    dx = qx - px
    dy = qy - py

    # AB = A + u * e
    ex = bx - ax
    ey = by - ay

    denom = dx * ey - dy * ex
    if abs(denom) < eps:
        return False, 0.0, 0.0   # 平行或共线，不算穿墙

    t = ((ax - px) * ey - (ay - py) * ex) / denom
    u = ((ax - px) * dy - (ay - py) * dx) / denom

    # u ∈ [0,1] → 交点在墙段内
    # t > 0     → 在 P → Q 的射线方向
    if u < -eps or u > 1.0 + eps or t < eps or t > 1.0 + eps:
        return False, 0.0, 0.0

    rx = px + t * dx
    ry = py + t * dy
    return True, rx, ry
# =====================================================
# 一阶 / 二阶反射、绕射（原样）
# =====================================================
def valid_first_reflection(tx_img, rx, wall):
    t0 = dbg_enter("valid_first_reflection")
    if not segment_intersect2(tx_img,rx,wall[0],wall[1]):
        dbg_exit("valid_first_reflection", t0)
        return False

    dbg_exit("valid_first_reflection", t0)
    return True

def valid_second_reflection(tx, rx, w1, w2, walls):
    t0 = dbg_enter("valid_second_reflection")

    # ---------- 判定 A：W2 上是否存在反射点 ----------
    tx1 = reflect_point(tx, walls[w1])
    tx2 = reflect_point(tx1, walls[w2])

    if not segment_intersect2(tx2, rx, walls[w2][0], walls[w2][1]):
        dbg_exit("valid_second_reflection", t0)
        return False

    # ---------- 判定 B：W1 上是否存在反射点 ----------
    rx1 = reflect_point(rx, walls[w2])
    rx2 = reflect_point(rx1, walls[w1])

    if not segment_intersect2(rx2, tx, walls[w1][0], walls[w1][1]):
        dbg_exit("valid_second_reflection", t0)
        return False

    dbg_exit("valid_second_reflection", t0)
    return True


def diffraction_toa(tx, rx, PVP_id, corner_table, best_dist):
    t0 = dbg_enter("diffraction_toa")

    ids_tx = PVP_id[tx[1], tx[0]]
    ids_rx = PVP_id[rx[1], rx[0]]

    if not ids_tx or not ids_rx:
        dbg_exit("diffraction_toa", t0)
        return best_dist

    for cid in set(ids_tx) & set(ids_rx):
        cx, cy = corner_table[cid]
        t0 = dbg_enter("dist")
        d = math.dist(tx, (cx, cy)) + math.dist((cx, cy), rx)
        dbg_exit("dist", t0)
        if d < best_dist:
            best_dist = d

    dbg_exit("diffraction_toa", t0)
    return best_dist


def diffraction_2nd_toa(tx, rx, PVP_id, corner_table, walls, wall_grid, best_dist):
    t0 = dbg_enter("diffraction_2nd_toa")

    ids_tx = PVP_id[tx[1], tx[0]]
    ids_rx = PVP_id[rx[1], rx[0]]

    if not ids_tx or not ids_rx:
        dbg_exit("diffraction_2nd_toa", t0)
        return best_dist

    for cid1 in ids_tx:
        c1 = corner_table[cid1]
        t0 = dbg_enter("dist")
        d1 = math.dist(tx, c1)
        dbg_exit("dist", t0)
        if d1 >= best_dist:
            continue

        for cid2 in ids_rx:
            c2 = corner_table[cid2]
            if not visible_fast(c1, c2, walls, wall_grid):
                continue
            t0 = dbg_enter("dist")
            d = d1 + math.dist(c1, c2) + math.dist(c2, rx)
            dbg_exit("dist", t0)
            if d < best_dist:
                best_dist = d

    dbg_exit("diffraction_2nd_toa", t0)
    return best_dist

def diffraction_2nd_mirror(
    tx, rx,
    TX_mask, RX_mask,
    TX_wall_order, 
    walls,
    wall_grid,
    PVP_id,
    corner_table,
    best_dist
):
    tx_x, tx_y = tx
    rx_x, rx_y = rx

    ids_tx = PVP_id[tx_y, tx_x]
    ids_rx = PVP_id[rx_y, rx_x]

    if not ids_tx and not ids_rx:
        return best_dist

    ids_tx_set = set(ids_tx) if ids_tx else set()
    ids_rx_set = set(ids_rx) if ids_rx else set()

    # =================================================
    # 情况 1：TX 镜像 + 绕射
    # =================================================
    for w in TX_wall_order:
        if not TX_mask[w]:
            continue
        ax, ay = walls[w][0]
        bx, by = walls[w][1]

        # TX 镜像
        tx_img_x, tx_img_y = reflect_point_nb(
            tx_x, tx_y,
            ax, ay, bx, by
        )
        t0 = dbg_enter("dist")
        # 下界剪枝（镜像点到 RX）
        if math.dist((tx_img_x, tx_img_y), (rx_x, rx_y)) >= best_dist:
            continue
        dbg_exit("dist", t0)
        for cid in ids_rx_set:
            cx, cy = corner_table[cid]

            # 穿墙 + 反射点
            ok, rxp, ryp = intersect_wall_return_point_nb(
                tx_img_x, tx_img_y,
                cx, cy,
                ax, ay, bx, by,
                1e-9
            )
            if not ok:
                continue

            # 真实遮挡（跳过本墙）
            if not visible_fast(
                (rxp, ryp),
                (cx, cy),
                walls,
                wall_grid,
                skip_w=w
            ):
                continue
            t0 = dbg_enter("dist")
            d = (
                math.dist((tx_x, tx_y), (rxp, ryp)) +
                math.dist((rxp, ryp), (cx, cy)) +
                math.dist((cx, cy), (rx_x, rx_y))
            )
            dbg_exit("dist", t0)
            if d < best_dist:
                best_dist = d
    
    # =================================================
    # 情况 2：RX 镜像 + 绕射（对称）
    # =================================================
    for w in np.nonzero(RX_mask)[0]:
        ax, ay = walls[w][0]
        bx, by = walls[w][1]

        # RX 镜像
        rx_img_x, rx_img_y = reflect_point_nb(
            rx_x, rx_y,
            ax, ay, bx, by
        )
        t0 = dbg_enter("dist")
        # 下界剪枝（镜像点到 TX）
        if math.dist((rx_img_x, rx_img_y), (tx_x, tx_y)) >= best_dist:
            continue
        dbg_exit("dist", t0)
        for cid in ids_tx_set:
            cx, cy = corner_table[cid]

            # 穿墙 + 反射点
            ok, rxp, ryp = intersect_wall_return_point_nb(
                rx_img_x, rx_img_y,
                cx, cy,
                ax, ay, bx, by,
                1e-9
            )
            if not ok:
                continue

            # 真实遮挡（跳过本墙）
            if not visible_fast(
                (rxp, ryp),
                (cx, cy),
                walls,
                wall_grid,
                skip_w=w
            ):
                continue
            t0 = dbg_enter("dist")
            d = (
                math.dist((tx_x, tx_y), (cx, cy)) +
                math.dist((cx, cy), (rxp, ryp)) +
                math.dist((rxp, ryp), (rx_x, rx_y))
            )
            dbg_exit("dist", t0)
            if d < best_dist:
                best_dist = d
    
    return best_dist




# =====================================================
# 主流程（只改 RX 数量）
# =====================================================
def main():
    global MAIN_START_TIME, MAIN_TOTAL_TIME

    MAIN_START_TIME = time.perf_counter()

    geo  = np.load(f"{GEO_ROOT}/{SCENE_ID}.npy")
    ant  = np.load(f"{ANT_ROOT}/{SCENE_ID}_{TX_ID}.npy")
    gray = np.array(Image.open(f"{GAIN_ROOT}/{SCENE_ID}_{TX_ID}.png").convert("L"))

    walls = np.load(f"{WALL_ROOT}/{SCENE_ID}.npy", allow_pickle=True).tolist()
    WWV   = np.load(f"{WWV_ROOT}/{SCENE_ID}.npy")
    PVW_mask = np.load(f"{PVW_MASK_ROOT}/{SCENE_ID}.npy")
    PVP_id = np.load(f"{PVP_ID_ROOT}/{SCENE_ID}.npy", allow_pickle=True)
    corner_table = np.load(f"{CORNER_TABLE_ROOT}/{SCENE_ID}.npy")

    wall_grid = build_wall_grid(walls)

    H, W = geo.shape
    dist_map = np.full((H, W), MAX_DIST, dtype=np.float32)

    tx_i, tx_j = np.argwhere(ant == 1)[0]
    tx = (tx_j, tx_i)

    TX_mask = PVW_mask[tx_i, tx_j]

    TX_wall_order = []
    for w in np.nonzero(TX_mask)[0]:
        wall = walls[w]
        mx = 0.5 * (wall[0][0] + wall[1][0])
        my = 0.5 * (wall[0][1] + wall[1][1])
        dist2 = (tx[0] - mx)**2 + (tx[1] - my)**2
        TX_wall_order.append((dist2, w))
    TX_wall_order.sort()
    TX_wall_order = [w for _, w in TX_wall_order]

    searchable = (geo == 0) & (gray > 0)
    rx_list = np.argwhere(searchable)

    # 🔴 随机抽 RX
    #rx_list = random.sample(list(rx_list), min(NUM_RX, len(rx_list)))

    for i, j in rx_list:
        rx = (j, i)

        if (i == tx_i and j == tx_j):
            dist_map[i, j] = 0.0
            continue
        t0 = dbg_enter("dist")
        if visible_fast(tx, rx, walls, wall_grid):
            dist_map[i, j] = math.dist(tx, rx)
            continue
        dbg_exit("dist", t0)
        RX_mask = PVW_mask[i, j]
        best = MAX_DIST
        # ---------- 一阶反射 ----------
        valid_mask = RX_mask & TX_mask
        for w in TX_wall_order:
            if not valid_mask[w]:
                continue

            wall = walls[w]

            # ---------- 外层：直接算候选距离 ----------
            tx_img = reflect_point_nb(
                tx[0], tx[1],
                wall[0][0], wall[0][1],
                wall[1][0], wall[1][1]
            )


            dx = tx_img[0] - rx[0]
            dy = tx_img[1] - rx[1]
            d = (dx * dx + dy * dy) ** 0.5

            if d >= best:
                continue    # 剪枝：不调用函数

            # ---------- 内层：只做“是否存在反射点”的判断 ----------
            if valid_first_reflection_nb(tx_img, rx, wall):
                best = d

        # ---------- 二阶反射 ----------
        for w1 in TX_wall_order:
            # 只考虑 TX 可见的墙
            if not TX_mask[w1]:
                continue

            # RXV 的规模
            rxv_cnt = int(RX_mask.sum())
            DBG_SUM["RXV_size"] += rxv_cnt
            DBG_CNT["RXV_size"] += 1

            valid_w2 = RX_mask & WWV[w1]

            # valid_w2 的规模
            v2_cnt = int(valid_w2.sum())
            DBG_SUM["valid_w2_size"] += v2_cnt
            DBG_CNT["valid_w2_size"] += 1

            if not np.any(valid_w2):
                continue            
            # 预先在 w1 层算一次（如果你还没算）
            
            tx1 = reflect_point_nb(
                tx[0], tx[1],
                walls[w1][0][0], walls[w1][0][1],
                walls[w1][1][0], walls[w1][1][1]
            )

            for w2 in np.nonzero(valid_w2)[0]:
                # ---------- 几何下界：外层剪枝 ----------
                tx2 = reflect_point_nb(
                    tx1[0], tx1[1],
                    walls[w2][0][0], walls[w2][0][1],
                    walls[w2][1][0], walls[w2][1][1]
                )
                t0 = dbg_enter("dist")
                d_lb = math.dist(tx2, rx)
                dbg_exit("dist", t0)
                if d_lb >= best:
                    continue    # ← 直接剪掉，不调用函数

                # ---------- 只有可能更优，才计数 + 调函数 ----------
                DBG_CNT["valid_second_reflection_call"] += 1
                t0 = dbg_enter("valid_second_reflection")

                d = second_reflection_dist_nb(
                    tx[0], tx[1],
                    rx[0], rx[1],
                    walls[w1][0][0], walls[w1][0][1],
                    walls[w1][1][0], walls[w1][1][1],
                    walls[w2][0][0], walls[w2][0][1],
                    walls[w2][1][0], walls[w2][1][1],
                    best,
                    1e-9
                )
                dbg_exit("valid_second_reflection", t0)
                if d < best:
                    best = d


        if best >= MAX_DIST:
            best = diffraction_toa(tx, rx, PVP_id, corner_table, best)
        '''
        if best >= MAX_DIST:
            best = diffraction_2nd_toa(tx, rx, PVP_id, corner_table, walls, wall_grid, best)
        '''
        if best >= MAX_DIST:
            t0 = dbg_enter("diffraction_2nd_mirror")
            best = diffraction_2nd_mirror(
                tx, rx,
                TX_mask, RX_mask,
                TX_wall_order, 
                walls,
                wall_grid,        # ← 新增
                PVP_id,
                corner_table,
                best
            )
            dbg_exit("diffraction_2nd_mirror", t0)
        
        dist_map[i, j] = best

        
    MAIN_TOTAL_TIME = time.perf_counter() - MAIN_START_TIME

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{SCENE_ID}_{TX_ID}.npy")
    np.save(out_path, dist_map)
    print(f"[Done] Saved to {out_path}")

# =====================================================
# 输出 debug log
# =====================================================
def dump_log():
    with open(LOG_PATH, "w") as f:
        f.write("Name,CallCount,TotalTime(s),AvgTime(us)\n")
        for k in sorted(T_CNT, key=lambda x: -T_TIME[x]):
            avg = 1e6 * T_TIME[k] / max(T_CNT[k], 1)
            f.write(f"{k},{T_CNT[k]},{T_TIME[k]:.6f},{avg:.2f}\n")

        # ---------- 主流程总时间 ----------
        f.write("\n# ===== main total time =====\n")
        f.write(f"main_total_time(s) = {MAIN_TOTAL_TIME:.6f}\n")

        # ---------- 你已有的统计 ----------
        f.write("\n# ===== valid_w2 statistics =====\n")

        if DBG_CNT["RXV_size"] > 0:
            f.write(
                f"RXV avg size = "
                f"{DBG_SUM['RXV_size'] / DBG_CNT['RXV_size']:.2f}\n"
            )

        if DBG_CNT["valid_w2_size"] > 0:
            f.write(
                f"valid_w2 avg size = "
                f"{DBG_SUM['valid_w2_size'] / DBG_CNT['valid_w2_size']:.2f}\n"
            )

        f.write(
            f"valid_second_reflection total calls = "
            f"{DBG_CNT['valid_second_reflection_call']}\n"
        )


# =====================================================
if __name__ == "__main__":
    main()
    dump_log()
