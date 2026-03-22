import numpy as np
import os
from collections import deque
from tqdm import tqdm


# =====================================================
# 2-core 剥离（8邻域）
# =====================================================
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


# =====================================================
# 主算法
# =====================================================
def extract_walls(geo):

    H, W = geo.shape

    # =====================================================
    # Step 1 8邻域提取 contour
    # =====================================================
    contour = set()

    for i in range(H):
        for j in range(W):
            if geo[i, j] != 1:
                continue
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if ni < 0 or ni >= H or nj < 0 or nj >= W or geo[ni, nj] == 0:
                        contour.add((j, i))
                        break

    # =====================================================
    # Step 2 去毛刺
    # =====================================================
    contour = prune_spurs(contour)

    # =====================================================
    # Step 3 删除“原始4邻域全是墙”的点
    # =====================================================
    nbr4 = [(1,0), (-1,0), (0,1), (0,-1)]
    to_remove = []

    for x, y in contour:
        all_wall = True
        for dx, dy in nbr4:
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= W or ny < 0 or ny >= H:
                all_wall = False
                break
            if geo[ny, nx] != 1:
                all_wall = False
                break
        if all_wall:
            to_remove.append((x,y))

    for p in to_remove:
        contour.remove(p)

    # =====================================================
    # Step 4 8邻域分 bucket
    # =====================================================
    visited = set()
    buckets = []

    for p in contour:
        if p in visited:
            continue

        q = deque([p])
        comp = {p}
        visited.add(p)

        while q:
            x, y = q.popleft()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nxt = (x + dx, y + dy)
                    if nxt in contour and nxt not in visited:
                        visited.add(nxt)
                        comp.add(nxt)
                        q.append(nxt)

        buckets.append(comp)

    # =====================================================
    # Step 5 无回溯轮廓追踪
    # =====================================================
    dirs = [
        (1, 0), (1, 1), (0, 1), (-1, 1),
        (-1, 0), (-1, -1), (0, -1), (1, -1),
    ]

    walls = []

    for bucket in buckets:

        S = set(bucket)
        if not S:
            continue

        while S:

            def degree_in_S(p):
                cnt = 0
                for dx, dy in dirs:
                    if (p[0] + dx, p[1] + dy) in S:
                        cnt += 1
                return cnt

            # 优先找 2-2-2 点
            cur = None

            for p in S:
                if degree_in_S(p) != 2:
                    continue

                nbrs = []
                for dx, dy in dirs:
                    nb = (p[0] + dx, p[1] + dy)
                    if nb in S:
                        nbrs.append(nb)

                if len(nbrs) != 2:
                    continue

                if degree_in_S(nbrs[0]) == 2 and degree_in_S(nbrs[1]) == 2:
                    cur = p
                    break

            if cur is None:
                cur = min(S, key=degree_in_S)

            S.remove(cur)
            path = [cur]

            while True:

                candidates = []

                for dx, dy in dirs:
                    nxt = (cur[0] + dx, cur[1] + dy)
                    if nxt in S:
                        candidates.append(nxt)

                if not candidates:
                    break

                def degree(p):
                    cnt = 0
                    for dx, dy in dirs:
                        if (p[0] + dx, p[1] + dy) in S:
                            cnt += 1
                    return cnt

                candidates.sort(key=lambda p: degree(p))
                nxt = candidates[0]

                cur = nxt
                S.remove(cur)
                path.append(cur)

            # 几何闭合补接
            if len(path) > 2:
                sx, sy = path[0]
                ex, ey = path[-1]
                if abs(sx - ex) <= 1 and abs(sy - ey) <= 1:
                    path.append(path[0])

            # =================================================
            # 最大直线段压缩
            # =================================================
            if len(path) < 2:
                continue

            line_start = path[0]
            prev_dir = None

            for i in range(1, len(path)):
                dx = path[i][0] - path[i - 1][0]
                dy = path[i][1] - path[i - 1][1]
                cur_dir = (dx, dy)

                if prev_dir is None:
                    prev_dir = cur_dir
                    continue

                if cur_dir != prev_dir:
                    walls.append((line_start, path[i - 1]))
                    line_start = path[i - 1]

                prev_dir = cur_dir

            walls.append((line_start, path[-1]))

    return walls


# =====================================================
# 入口（批量处理）
# =====================================================
if __name__ == "__main__":

    geo_root = r"/root/geo"
    save_root = r"/root/RM/wall_segment"

    os.makedirs(save_root, exist_ok=True)

    for scene_id in tqdm(range(701)):

        geo_path = os.path.join(geo_root, f"{scene_id}.npy")

        if not os.path.exists(geo_path):
            continue

        geo = np.load(geo_path)

        walls = extract_walls(geo)

        np.save(
            os.path.join(save_root, f"{scene_id}.npy"),
            np.array(walls, dtype=object)
        )