import numpy as np
import cv2
import os
from utils.io import save_npy
from collections import deque


def prune_spurs(contour):

    S = set(contour)

    nbr8 = [(dx, dy) for dx in (-1,0,1)
                     for dy in (-1,0,1)
                     if not (dx == 0 and dy == 0)]

    deg = {}
    for x, y in S:
        cnt = 0
        for dx, dy in nbr8:
            if (x+dx, y+dy) in S:
                cnt += 1
        deg[(x,y)] = cnt

    q = deque([p for p in S if deg[p] <= 1])

    while q:
        p = q.popleft()
        if p not in S:
            continue

        S.remove(p)

        x, y = p
        for dx, dy in nbr8:
            nb = (x+dx, y+dy)
            if nb in S:
                deg[nb] -= 1
                if deg[nb] <= 1:
                    q.append(nb)

    return S


def run(geo, config):
    idx = config.IDX

    save_path = os.path.join(
        config.OUTPUT_ROOT,
        config.WALL_DIR,
        f"{idx}.npy"
    )

    img = (geo > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    all_segments = []

    def direction(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dirs = {
            (1,0):0,(1,-1):1,(0,-1):2,(-1,-1):3,
            (-1,0):4,(-1,1):5,(0,1):6,(1,1):7
        }
        return dirs.get((dx,dy), None)

    # ========= 分段 =========
    for cnt in contours:

        # ✅ 关键修复1：跳过单点 / 极小 contour
        if cnt.shape[0] < 2:
            continue

        pts = cnt.squeeze()
        # ===== 去毛刺（2-core 剥离）=====
        pts_list = [tuple(p) for p in pts]

        pts_pruned = prune_spurs(pts_list)

        # ⚠ 防御：避免删空 / 太小
        if len(pts_pruned) < 2:
            continue

        # ⚠ 转回有序序列（保持原 contour 顺序）
        pts = np.array(
            [p for p in pts_list if p in pts_pruned],
            dtype=np.int32
        )

        if len(pts) < 2:
            continue

        # ✅ 关键修复2：防止 squeeze 变成 (2,)
        if pts.ndim != 2 or pts.shape[1] != 2:
            continue

        if len(pts) < 2:
            continue

        dirs = []
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i+1) % len(pts)]
            d = direction(p1, p2)

            # 可选：防御非法方向
            if d is None:
                continue

            dirs.append(d)

        if len(dirs) == 0:
            continue

        start_idx = 0

        for i in range(1, len(dirs)):
            if dirs[i] != dirs[i-1]:
                all_segments.append(
                    (tuple(pts[start_idx]), tuple(pts[i]))
                )
                start_idx = i

        all_segments.append(
            (tuple(pts[start_idx]), tuple(pts[0]))
        )

    # ========= 保存 =========
    save_npy(save_path, np.array(all_segments, dtype=object))

    print(f"[wall] idx={idx} segments={len(all_segments)}")

    # ===== walls_nb =====
    n = len(all_segments)
    walls_nb = np.zeros((n, 4), dtype=np.float32)

    for i, (A, B) in enumerate(all_segments):
        walls_nb[i, 0] = A[0]
        walls_nb[i, 1] = A[1]
        walls_nb[i, 2] = B[0]
        walls_nb[i, 3] = B[1]

    save_npy(
        os.path.join(config.OUTPUT_ROOT, "convert", str(idx), "walls_nb.npy"),
        walls_nb
    )

    return None