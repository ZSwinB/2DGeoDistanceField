import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# -----------------------------
# 路径
# -----------------------------
SCENE_ID = 700

ANT_ROOT = r"G:\data_SRM\antenna"
BUILDING_ROOT = r"G:\RadioMapSeer\png\buildings_complete"

# -----------------------------
# 读取 building
# -----------------------------
building_path = os.path.join(BUILDING_ROOT, f"{SCENE_ID}.png")
building = np.array(Image.open(building_path).convert("L"))

H, W = building.shape

# -----------------------------
# 收集 24 个新 TX（80~103）
# -----------------------------
tx_list = []

for tx_id in range(80, 103):
    path = os.path.join(ANT_ROOT, f"{SCENE_ID}_{tx_id}.npy")

    if not os.path.exists(path):
        print("missing:", path)
        continue

    ant = np.load(path)

    coords = np.argwhere(ant == 1)

    for (i, j) in coords:
        tx_list.append((i, j))

tx_list = np.array(tx_list)

print("Loaded TX:", len(tx_list))

# -----------------------------
# 可视化
# -----------------------------
plt.figure(figsize=(8, 8))

# 建筑
plt.imshow(building, cmap="gray", origin='upper')

# TX（蓝色）
if len(tx_list) > 0:
    plt.scatter(
        tx_list[:,1],
        tx_list[:,0],
        c='blue',
        s=25,
        label='New TX (80~103)'
    )

plt.title(f"Scene {SCENE_ID} - New TX Visualization")
plt.legend()

plt.show()