import numpy as np
import matplotlib.pyplot as plt

PATH = r"C:\Users\ZSwinB\Documents\GitHub\2DGeoDistanceField\outputs\sim\0_0.npz"

data = np.load(PATH)
dist_map = data["dist_map"]

if dist_map.ndim == 3:
    dist_map = dist_map[:, :, 0]

# 去掉无效值
img = dist_map.copy()
img[img > 1000] = np.nan

plt.figure(figsize=(6, 6))

plt.imshow(img, cmap="viridis")   # ⭐ 换这个
plt.colorbar()

plt.title("Distance Map")
plt.axis('off')

plt.show()