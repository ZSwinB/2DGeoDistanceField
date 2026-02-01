import numpy as np
import matplotlib.pyplot as plt

# 路径
dist_path = r"D:\Desktop\RM\data\distance_map\0_0.npy"

# 读取
dist_map = np.load(dist_path)

# 找出值为 1000 的点（y, x）
ys, xs = np.where(dist_map == 1000)

print("Number of points with value 1000:", len(xs))

# 画图
plt.figure(figsize=(6, 6))
plt.scatter(xs, ys, s=2)

plt.gca().set_aspect("equal", adjustable="box")
plt.gca().invert_yaxis()   # 图像坐标系：左上角 (0,0)
plt.grid(True)

plt.title("Distance map points == 1000")
plt.show()
