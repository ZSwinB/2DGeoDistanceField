import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

ANT_ROOT = r"G:\RadioMapSeer\png\antennas"
BUILDING_PATH = r"G:\RadioMapSeer\png\buildings_complete\425.png"

# -----------------------------
# 读取 building
# -----------------------------
building = np.array(Image.open(BUILDING_PATH).convert("L"))
H, W = building.shape

print("building shape:", building.shape)

# -----------------------------
# 收集原始 TX（红色）
# -----------------------------
tx_list = []

for tx_id in range(80):
    path = os.path.join(ANT_ROOT, f"425_{tx_id}.png")

    if not os.path.exists(path):
        continue

    ant = np.array(Image.open(path).convert("L"))
    coords = np.argwhere(ant > 128)

    for (i, j) in coords:
        tx_list.append((i, j))

tx_list = np.array(tx_list)
print("TX count:", len(tx_list))

# -----------------------------
# 检查是否有 TX 在建筑内
# -----------------------------
bad_tx = []

for tx_id in range(80):
    path = os.path.join(ANT_ROOT, f"425_{tx_id}.png")

    if not os.path.exists(path):
        continue

    ant = np.array(Image.open(path).convert("L"))
    coords = np.argwhere(ant > 128)

    for (i, j) in coords:
        if building[i, j] > 128:
            print(f"[BAD] TX {tx_id} at ({i},{j}) inside building")
            bad_tx.append(tx_id)
            break

print("bad TX list:", bad_tx)

# -----------------------------
# 生成新的 30 个 TX（蓝色）
# -----------------------------
new_tx = []

for i in range(H):
    for j in range(W):

        # 边界区域
        if i < 50 or i > 200 or j < 50 or j > 200:

            # free space
            if building[i, j] <= 128:
                new_tx.append((i, j))

new_tx = np.array(new_tx)

# 均匀随机取30个（更合理）
np.random.seed(0)
idx = np.random.choice(len(new_tx), 24, replace=False)
new_tx = new_tx[idx]

print("New TX count:", len(new_tx))

# -----------------------------
# 可视化（全部在一个figure里）
# -----------------------------
plt.figure(figsize=(8, 8))

# 建筑
plt.imshow(building, cmap="gray", origin='upper')

# 原 TX（红色）
if len(tx_list) > 0:
    plt.scatter(
        tx_list[:,1],
        tx_list[:,0],
        c='red',
        s=10,
        label='Original TX'
    )

# 新 TX（蓝色）
if len(new_tx) > 0:
    plt.scatter(
        new_tx[:,1],
        new_tx[:,0],
        c='blue',
        s=20,
        label='New TX (edge)'
    )

plt.title("TX Distribution (Original + New)")
plt.legend()

plt.show()
import os
print(os.cpu_count())