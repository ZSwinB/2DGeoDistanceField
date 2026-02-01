import numpy as np
import os

# ================= 路径 =================

SCENE_ID = 0

GEO_PATH  = rf"D:\Desktop\RM\feature_store\numerical_data\geo\{SCENE_ID}.npy"
WALL_PATH = rf"D:\Desktop\RM\data\wall_segment\{SCENE_ID}.npy"

OUT_DIR  = r"D:\Desktop\RM\data\WWVmove"
OUT_PATH = rf"{OUT_DIR}\{SCENE_ID}.npy"

# ================= 8 个方向（顺序很重要） =================

DIRECTIONS_8 = [
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1),
]

# ================= 工具 =================

def in_bounds(x, y, W, H):
    return 0 <= x < W and 0 <= y < H

def first_free_dir(P, geo):
    """
    给定端点 P=(x,y)
    返回第一个能进入 free cell 的方向 (dx,dy)
    若没有，返回 None
    """
    H, W = geo.shape
    x0, y0 = int(round(P[0])), int(round(P[1]))

    for dx, dy in DIRECTIONS_8:
        x = x0 + dx
        y = y0 + dy
        if not in_bounds(x, y, W, H):
            continue
        if geo[y, x] == 0:
            return (dx, dy)

    return None

# ================= 主逻辑 =================

def build_WWVmove(geo, walls):
    """
    输出格式：
    WWVmove[wid] = [dir_for_A, dir_for_B]
    """
    WWVmove = {}

    for wid, (A, B) in enumerate(walls):
        dir_A = first_free_dir(A, geo)
        dir_B = first_free_dir(B, geo)

        WWVmove[wid] = [dir_A, dir_B]

    return WWVmove

# ================= 入口 =================

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    geo   = np.load(GEO_PATH)
    walls = np.load(WALL_PATH, allow_pickle=True).tolist()

    WWVmove = build_WWVmove(geo, walls)

    np.save(OUT_PATH, WWVmove)

    print(f"[Done] WWVmove saved to {OUT_PATH}")
    print(f"[Info] walls = {len(walls)}")
