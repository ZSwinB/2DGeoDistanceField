import numpy as np
import matplotlib.pyplot as plt

# ===== 1. 读取数据 =====
data = np.load(r"D:\Desktop\RM\data\distance_map\0_0.npy").astype(float)
h, w = data.shape

# ===== 2. 障碍 mask =====
obs = (data == 1000)

# ===== 3. 单边差分 + 障碍屏蔽 =====
dx = np.zeros_like(data)
dy = np.zeros_like(data)

for i in range(1, h - 1):
    for j in range(1, w - 1):

        if obs[i, j]:
            continue

        gx = 0.0
        gy = 0.0

        if not obs[i, j + 1]:
            gx += data[i, j + 1] - data[i, j]
        if not obs[i, j - 1]:
            gx += data[i, j] - data[i, j - 1]
        if not obs[i + 1, j]:
            gy += data[i + 1, j] - data[i, j]
        if not obs[i - 1, j]:
            gy += data[i, j] - data[i - 1, j]

        dx[i, j] = gx
        dy[i, j] = gy

# ===== 4. 归一化 + 取负方向 =====
mag = np.sqrt(dx**2 + dy**2)
eps = 1e-8

dx_n = -dx / (mag + eps)
dy_n = -dy / (mag + eps)

dx_n[obs] = 0
dy_n[obs] = 0

# ===== 5. 可视化（障碍涂白）=====
vis = data.copy()
vis[obs] = np.nan

cmap = plt.cm.viridis
cmap.set_bad(color='white')

# ===== 6. 画矢量场 =====
step = 5
Y, X = np.mgrid[0:h:step, 0:w:step]

plt.figure(figsize=(6, 6))
plt.imshow(vis, cmap=cmap)
plt.colorbar()

plt.quiver(
    X, Y,
    dx_n[::step, ::step],
    dy_n[::step, ::step],
    color='red',
    scale=30
)

plt.tight_layout()
plt.show()
