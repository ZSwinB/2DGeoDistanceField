import os
import numpy as np
from PIL import Image

PNG_ROOT = r"G:\RadioMapSeer\png\cars"
WALL_ROOT = r"G:\data_SRM\wall_segment"
OUT_ROOT  = r"G:\data_DRM\wall_segment_DRM"

os.makedirs(OUT_ROOT, exist_ok=True)

# =====================================================
# 判断是否是边界像素
# =====================================================
def is_boundary(geo, i, j):

    H, W = geo.shape

    if i == 0 or geo[i-1, j] == 0:
        return True
    if i == H-1 or geo[i+1, j] == 0:
        return True
    if j == 0 or geo[i, j-1] == 0:
        return True
    if j == W-1 or geo[i, j+1] == 0:
        return True

    return False


# =====================================================
# 提取边
# =====================================================
def extract_car_edges(geo):

    H, W = geo.shape
    edges = []

    for i in range(H):
        for j in range(W):

            if geo[i, j] != 1:
                continue

            if not is_boundary(geo, i, j):
                continue

            if i == 0 or geo[i-1, j] == 0:
                edges.append(((j, i), (j+1, i)))

            if i == H-1 or geo[i+1, j] == 0:
                edges.append(((j, i+1), (j+1, i+1)))

            if j == 0 or geo[i, j-1] == 0:
                edges.append(((j, i), (j, i+1)))

            if j == W-1 or geo[i, j+1] == 0:
                edges.append(((j+1, i), (j+1, i+1)))

    return edges


# =====================================================
# 合并共线线段
# =====================================================
def merge_segments(segs):

    horiz = {}
    vert = {}

    for (x1,y1),(x2,y2) in segs:

        if y1 == y2:
            y = y1
            x1,x2 = sorted((x1,x2))
            horiz.setdefault(y, []).append((x1,x2))
        else:
            x = x1
            y1,y2 = sorted((y1,y2))
            vert.setdefault(x, []).append((y1,y2))

    merged = []

    for y, lst in horiz.items():

        lst.sort()
        s,e = lst[0]

        for ns,ne in lst[1:]:
            if ns <= e:
                e = max(e,ne)
            else:
                merged.append(((s,y),(e,y)))
                s,e = ns,ne

        merged.append(((s,y),(e,y)))

    for x, lst in vert.items():

        lst.sort()
        s,e = lst[0]

        for ns,ne in lst[1:]:
            if ns <= e:
                e = max(e,ne)
            else:
                merged.append(((x,s),(x,e)))
                s,e = ns,ne

        merged.append(((x,s),(x,e)))

    return merged


# =====================================================
# 主批处理
# =====================================================
for idx in range(701):

    print("processing", idx)

    # 读取 car png
    png_path = os.path.join(PNG_ROOT, f"{idx}.png")

    img = Image.open(png_path).convert("L")
    geo = (np.array(img) > 0).astype(np.uint8)

    # 提取 car edge
    edges = extract_car_edges(geo)
    edges = merge_segments(edges)

    # 读取原始墙
    wall_path = os.path.join(WALL_ROOT, f"{idx}.npy")
    walls = list(np.load(wall_path, allow_pickle=True))

    # 合并
    walls.extend(edges)

    # 保存
    out_path = os.path.join(OUT_ROOT, f"{idx}.npy")
    np.save(out_path, np.array(walls, dtype=object))

    print("walls:", len(walls))