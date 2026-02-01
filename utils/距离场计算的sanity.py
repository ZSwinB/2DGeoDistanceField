import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

MAX_DIST = 1000.0

# ========= 路径 =========
dist_path = r"D:\Desktop\RM\data\distance_map\0_0.npy"
geo_path  = r"D:\Desktop\RM\feature_store\numerical_data\geo\0.npy"
gain_path = r"D:\Desktop\RadioMapSeer\gain\IRT2\0_0.png"

# ========= 读数据 =========
dist = np.load(dist_path, allow_pickle=True)
if isinstance(dist, np.ndarray) and dist.ndim == 0:
    dist = dist.item()

geo = np.load(geo_path)
gray = np.array(Image.open(gain_path).convert("L"))

assert dist.shape == geo.shape == gray.shape, "shape 不一致"

# ========= Sanity-3 掩码 =========
# 自由空间 + 灰度>0 + 却被写成 MAX
bad_mask = (geo == 0) & (gray > 0) & (dist == MAX_DIST)

ys, xs = np.where(bad_mask)
print("Sanity-3 点数量:", len(xs))

# ========= 可视化 =========
plt.figure(figsize=(6, 6))

# 背景用灰度图，方便你对照
plt.imshow(gray, cmap="gray", origin="upper")

# Sanity-3 点：红色
plt.scatter(xs, ys, s=2, c="red")

plt.title("Sanity-3 Visualization (red = should NOT be MAX)")
plt.axis("off")
plt.tight_layout()
plt.show()
