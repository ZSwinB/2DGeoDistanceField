import numpy as np
import cv2
import os

# ========= 参数 =========
roots = [
    r"G:\data_SRM\geo",
    # r"D:\other_path\geo",
]

idx = 0  # 对应 0.npy
save_path = r"G:\RM\segments.npy"

# ========= 读取并合并 =========
img = None

for root in roots:
    path = os.path.join(root, f"{idx}.npy")
    data = np.load(path).astype(np.uint8)
    data = (data > 0).astype(np.uint8)  # 0/1

    if img is None:
        img = data
    else:
        img = np.logical_or(img, data)

# 转成 0/255
img = img.astype(np.uint8) * 255

# ========= 轮廓提取 =========
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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
    pts = cnt.squeeze()

    if len(pts) < 2:
        continue

    dirs = []
    for i in range(len(pts)):
        p1 = pts[i]
        p2 = pts[(i+1) % len(pts)]
        dirs.append(direction(p1, p2))

    start_idx = 0
    segments = []

    for i in range(1, len(dirs)):
        if dirs[i] != dirs[i-1]:
            segments.append((tuple(pts[start_idx]), tuple(pts[i])))
            start_idx = i

    segments.append((tuple(pts[start_idx]), tuple(pts[0])))

    all_segments.extend(segments)

# ========= 保存 =========
np.save(save_path, np.array(all_segments, dtype=object))

print("完成，段数:", len(all_segments))