import numpy as np
import matplotlib.pyplot as plt

path = r"G:\RMdata\distancemap\0_0.npy"

# =========================
# 读取数据
# =========================
T = np.load(path).astype(np.float32)
if T.ndim == 3:
    T = T[:, :, 0]

H, W = T.shape

# =========================
# 1. 中心差分
# =========================
gx = np.zeros_like(T)
gy = np.zeros_like(T)

gx[:, 1:-1] = (T[:, 2:] - T[:, :-2]) * 0.5
gy[1:-1, :] = (T[2:, :] - T[:-2, :]) * 0.5

mag_center = np.sqrt(gx**2 + gy**2)

# =========================
# 2. upwind 4组合
# =========================
mag_up = np.zeros_like(T)
vx = np.zeros_like(T)
vy = np.zeros_like(T)

for i in range(1, H-1):
    for j in range(1, W-1):

        dx_f = T[i, j+1] - T[i, j]
        dx_b = T[i, j]   - T[i, j-1]

        dy_f = T[i+1, j] - T[i, j]
        dy_b = T[i, j]   - T[i-1, j]

        best_score = 1e9
        best_mag = 0.0
        bx, by = 0.0, 0.0

        for gx_ in (dx_f, dx_b):
            for gy_ in (dy_f, dy_b):

                mag = np.sqrt(gx_*gx_ + gy_*gy_)
                score = abs(mag - 1.0)

                if score < best_score:
                    best_score = score
                    best_mag = mag
                    bx, by = gx_, gy_

        mag_up[i, j] = best_mag
        vx[i, j] = bx
        vy[i, j] = by


# =========================
# 图1：模长对比
# =========================
plt.figure(figsize=(6,10))

plt.subplot(2,1,1)
plt.imshow(mag_center, cmap="viridis")
plt.colorbar()
plt.title("Central Difference |grad|")

plt.subplot(2,1,2)
plt.imshow(mag_up, cmap="viridis")
plt.colorbar()
plt.title("Upwind Combo |grad|~1")

plt.tight_layout()
plt.show()


# =========================
# 图2：方向场对比
# =========================
plt.figure(figsize=(12,6))

step = 3

# 中心差分方向
mag_c = np.sqrt(gx**2 + gy**2) + 1e-8
vx_c = gx / mag_c
vy_c = gy / mag_c

plt.subplot(1,2,1)
plt.imshow(T, cmap="gray")
plt.quiver(
    np.arange(0, W, step),
    np.arange(0, H, step),
    vx_c[::step, ::step],
    vy_c[::step, ::step],
    color='red',
    scale=150
)
plt.title("Central Difference Direction")

# upwind方向
mag_u = np.sqrt(vx**2 + vy**2) + 1e-8
vx_u = vx / mag_u
vy_u = vy / mag_u

plt.subplot(1,2,2)
plt.imshow(T, cmap="gray")
plt.quiver(
    np.arange(0, W, step),
    np.arange(0, H, step),
    vx_u[::step, ::step],
    vy_u[::step, ::step],
    color='red',
    scale=150
)
plt.title("Upwind Selected Direction")

plt.tight_layout()
plt.show()