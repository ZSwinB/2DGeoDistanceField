import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 配置
# ===============================
wall_file = r"D:\Desktop\RM\data\wall_segments.npy"
HIGHLIGHT_IDX = 317  # ← 手动指定你要看的墙 index

# ===============================
# 1. 读取并检查格式 / 维度
# ===============================
data = np.load(wall_file, allow_pickle=True)

print("=== Wall file format check ===")
print("type:", type(data))
print("dtype:", data.dtype)
print("shape:", data.shape)


# ===============================
# 2. 规范化为 [(p0, p1), ...]
# ===============================
walls = []
for seg in data:
    p0, p1 = seg
    walls.append((tuple(p0), tuple(p1)))

# ===============================
# 3. 画图：整条线段高亮
# ===============================
plt.figure(figsize=(8, 8))

# 所有墙（灰色）
for p0, p1 in walls:
    x0, y0 = p0
    x1, y1 = p1
    plt.plot([x0, x1], [y0, y1],
             color="lightgray", linewidth=1)

# 高亮指定墙（整条线段）
p0, p1 = walls[HIGHLIGHT_IDX]
x0, y0 = p0
x1, y1 = p1
plt.plot([x0, x1], [y0, y1],
         color="red", linewidth=4)

plt.gca().set_aspect("equal", adjustable="box")
plt.gca().invert_yaxis()   # image / grid 坐标系
plt.grid(True)
plt.title(f"Highlight wall segment #{HIGHLIGHT_IDX}")
plt.show()


# ===============================
# 1. 读取数据
# ===============================
data = np.load(wall_file, allow_pickle=True)

print("=== Wall file basic info ===")
print("type:", type(data))
print("dtype:", data.dtype)
print("shape:", data.shape)

# ===============================
# 2. 打印指定 index 的原始内容
# ===============================
print("\n=== Raw wall segment at index", HIGHLIGHT_IDX, "===")
raw = data[HIGHLIGHT_IDX]
print(raw)
print("type:", type(raw))

# ===============================
# 3. 如果它是两个端点，继续拆开看
# ===============================
try:
    p0, p1 = raw
    print("\n--- endpoint p0 ---")
    print(p0)
    print("type:", type(p0))
    print("shape:", np.array(p0).shape)
    print("values:", list(p0))

    print("\n--- endpoint p1 ---")
    print(p1)
    print("type:", type(p1))
    print("shape:", np.array(p1).shape)
    print("values:", list(p1))

except Exception as e:
    print("\n[!] Cannot unpack raw data into two endpoints")
    print("Error:", e)
