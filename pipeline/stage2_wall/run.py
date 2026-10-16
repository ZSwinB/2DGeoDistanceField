import numpy as np
import cv2
import os
from utils.io import save_npy
from collections import deque


def prune_spurs(contour):
    """
    Remove spurs (degree <= 1 points) from a contour using iterative pruning (2-core extraction).

    Behavior:
        - Treats contour points as an undirected graph with 8-neighborhood connectivity
        - Iteratively removes points with degree <= 1
        - Returns the remaining core points

    Args:
        contour (list[tuple[int, int]]): list of (x, y) points representing the contour

    Returns:
        set[tuple[int, int]]: pruned set of contour points (2-core)
    """
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


def split_pruned_contour(points, retained):
    """Compress surviving original 8-neighbor edges without inventing gap edges."""
    segments = []
    active_start = None
    active_end = None
    active_direction = None
    for i in range(len(points)):
        p = tuple(points[i])
        q = tuple(points[(i + 1) % len(points)])
        if p == q:
            continue
        dx, dy = q[0] - p[0], q[1] - p[1]
        valid = p in retained and q in retained and abs(dx) <= 1 and abs(dy) <= 1
        direction = (dx, dy)
        if not valid:
            if active_start is not None:
                segments.append((active_start, active_end))
            active_start = active_end = active_direction = None
            continue
        if active_start is not None and active_end == p and active_direction == direction:
            active_end = q
        else:
            if active_start is not None:
                segments.append((active_start, active_end))
            active_start, active_end, active_direction = p, q, direction
    if active_start is not None:
        segments.append((active_start, active_end))

    # Removed excursions may return to the same retained point; merge only
    # touching, forward-collinear segments, including the contour seam.
    merged = []
    for p, q in segments:
        if merged:
            a, b = merged[-1]
            v = (b[0]-a[0], b[1]-a[1])
            w = (q[0]-p[0], q[1]-p[1])
            if b == p and v[0]*w[1] == v[1]*w[0] and v[0]*w[0]+v[1]*w[1] > 0:
                merged[-1] = (a, q)
                continue
        merged.append((p, q))
    if len(merged) > 1:
        a, b = merged[-1]
        p, q = merged[0]
        v = (b[0]-a[0], b[1]-a[1])
        w = (q[0]-p[0], q[1]-p[1])
        if b == p and v[0]*w[1] == v[1]*w[0] and v[0]*w[0]+v[1]*w[1] > 0:
            merged[0] = (a, q)
            merged.pop()
    return merged


def run(geo, config):
    """
    Generate wall segments from geometry and save results.

    Behavior:
        - Extracts contours from binary geometry
        - Removes spurs using 2-core pruning
        - Splits contours into directional segments
        - Saves:
            1. raw segments (object array)
            2. numeric wall representation (walls_nb)

    Args:
        geo (np.ndarray): binary geometry map (uint8 or bool)
        config: configuration object containing paths and scene index

    Outputs:
        - {OUTPUT_ROOT}/{WALL_DIR}/{idx}.npy
        - {OUTPUT_ROOT}/convert/{idx}/walls_nb.npy

    Returns:
        None
    """
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
        approximation = float(getattr(config, "WALL_APPROX_EPS", 0.0))
        if approximation > 0:
            surviving = np.array([p for p in pts_list if p in pts_pruned], dtype=np.int32)
            polygon = cv2.approxPolyDP(surviving.reshape(-1, 1, 2), approximation, True).reshape(-1, 2)
            if len(polygon) >= 3:
                for i in range(len(polygon)):
                    p = tuple(polygon[i])
                    q = tuple(polygon[(i + 1) % len(polygon)])
                    if p != q:
                        all_segments.append((p, q))
            else:
                all_segments.extend(split_pruned_contour(pts_list, pts_pruned))
        else:
            all_segments.extend(split_pruned_contour(pts_list, pts_pruned))

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
