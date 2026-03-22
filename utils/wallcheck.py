import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# ===============================
# 引入你的 config
# ===============================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from config import Config
config = Config()
SCENE_ID =5
HIGHLIGHT_IDX = 317  # 手动指定

# ===============================
# 路径（使用 config）
# ===============================
wall_file = os.path.join(
    config.OUTPUT_ROOT,
    config.WALL_DIR,
    f"{SCENE_ID}.npy"
)

print("wall_file:", wall_file)

# ===============================
# 1. 读取并检查格式
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
# 3. 画图
# ===============================
plt.figure(figsize=(8, 8))

# 所有墙（灰色）
for p0, p1 in walls:
    x0, y0 = p0
    x1, y1 = p1
    plt.plot([x0, x1], [y0, y1],
             color="lightgray", linewidth=1)

# 高亮指定墙
p0, p1 = walls[HIGHLIGHT_IDX]
x0, y0 = p0
x1, y1 = p1

plt.plot([x0, x1], [y0, y1],
         color="red", linewidth=4)

plt.gca().set_aspect("equal", adjustable="box")
plt.gca().invert_yaxis()
plt.grid(True)
plt.title(f"Highlight wall segment #{HIGHLIGHT_IDX}")
plt.show()

# ===============================
# 4. 打印该 segment 详细信息
# ===============================
print("\n=== Raw wall segment at index", HIGHLIGHT_IDX, "===")
raw = data[HIGHLIGHT_IDX]
print(raw)
print("type:", type(raw))

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