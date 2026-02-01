import numpy as np
import os
from collections import deque
from tqdm import tqdm

# =====================================================
# 8 邻域几何工具
# =====================================================
def ccw(a, b, c):
    return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])


# =====================================================
# 墙段提取（与你现在用的一致）
# =====================================================
def extract_walls(geo):
    H, W = geo.shape

    # Step 1. 墙边缘像素
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

    # Step 2. 8 连通分 bucket
    buckets = []
    visited = set()
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

    # Step 3. 轮廓追踪 → 最大直线段
    dirs = [
        (1, 0), (1, 1), (0, 1), (-1, 1),
        (-1, 0), (-1, -1), (0, -1), (1, -1),
    ]

    walls = []

    for bucket in buckets:
        # 度分析
        degree_one = []
        for (x, y) in bucket:
            deg = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    if (x + dx, y + dy) in bucket:
                        deg += 1
            if deg == 1:
                degree_one.append((x, y))

        has_deg1 = len(degree_one) > 0
        S = set(bucket)

        while S:
            # ===== 关键：确定 bucket 的真正起点 =====
            if has_deg1:
                starts = [p for p in degree_one if p in S]
                cur = starts[0] if starts else next(iter(S))
            else:
                cur = next(iter(S))

            bucket_start = cur   # ★ 新增：闭环真正起点

            S.remove(cur)
            prev_dir = None
            line_start = cur

            while True:
                found = False
                order = range(8) if prev_dir is None else [(prev_dir + k) % 8 for k in range(8)]
                for d in order:
                    dx, dy = dirs[d]
                    nxt = (cur[0] + dx, cur[1] + dy)

                    # 切角规则
                    if abs(dx) == 1 and abs(dy) == 1:
                        if (cur[0] + dx, cur[1]) in S or (cur[0], cur[1] + dy) in S:
                            continue

                    if nxt in S:
                        found = True
                        break

                if not found:
                    break

                if prev_dir is not None and d != prev_dir:
                    walls.append((line_start, cur))
                    line_start = cur

                prev_dir = d
                cur = nxt
                S.remove(cur)

            # 收尾当前直线段
            walls.append((line_start, cur))

            # ===== 修正后的闭环补边 =====
            if not has_deg1:
                if cur != bucket_start:
                    walls.append((cur, bucket_start))

            break

    return walls



# =====================================================
# 批量处理 0–700
# =====================================================
def batch_extract_walls(
    geo_root=r"D:\Desktop\RM\feature_store\numerical_data\geo",
    out_root=r"D:\Desktop\RM\data\wall_segment",
    max_scene_id=700
):
    os.makedirs(out_root, exist_ok=True)

    for scene_id in tqdm(range(max_scene_id + 1), desc="Extract wall segments"):
        geo_path = os.path.join(geo_root, f"{scene_id}.npy")
        if not os.path.exists(geo_path):
            print(f"[Skip] {scene_id}.npy 不存在")
            continue

        geo = np.load(geo_path)
        walls = extract_walls(geo)

        out_path = os.path.join(out_root, f"{scene_id}.npy")
        np.save(out_path, np.array(walls, dtype=object))


# =====================================================
# 入口
# =====================================================
if __name__ == "__main__":
    batch_extract_walls()
