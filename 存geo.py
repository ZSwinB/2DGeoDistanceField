import numpy as np
from PIL import Image
import os

BUILDING_ROOT = r"G:\RadioMapSeer\png\buildings_complete"
OUT_ROOT = r"G:\data_SRM\geo"

os.makedirs(OUT_ROOT, exist_ok=True)

NUM_SCENE = 701

for scene_id in range(NUM_SCENE):

    print(f"processing scene {scene_id}")

    img_path = os.path.join(BUILDING_ROOT, f"{scene_id}.png")

    if not os.path.exists(img_path):
        print("skip missing:", scene_id)
        continue

    # -----------------------------
    # 读取 PNG
    # -----------------------------
    img = np.array(Image.open(img_path).convert("L"))

    # -----------------------------
    # 转 geo（关键）
    # -----------------------------
    geo = np.zeros_like(img, dtype=np.uint8)

    # building = 1
    geo[img > 128] = 1

    # free = 0（默认）

    # -----------------------------
    # 保存
    # -----------------------------
    out_path = os.path.join(OUT_ROOT, f"{scene_id}.npy")
    np.save(out_path, geo)

print("all scenes done")