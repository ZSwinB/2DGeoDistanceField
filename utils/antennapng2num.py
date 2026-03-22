import numpy as np
from PIL import Image
import os
from multiprocessing import Pool, cpu_count

PNG_ROOT = r"G:\RadioMapSeer\png\antennas"
OUT_ROOT = r"G:\data_SRM\antenna"

os.makedirs(OUT_ROOT, exist_ok=True)

NUM_SCENE = 701

# -----------------------------
# 单个任务（一个文件）
# -----------------------------
def process_one(args):
    scene_id, tx_id = args

    in_path = os.path.join(PNG_ROOT, f"{scene_id}_{tx_id}.png")

    if not os.path.exists(in_path):
        return f"missing {scene_id}_{tx_id}"

    img = np.array(Image.open(in_path).convert("L"))

    ant = np.zeros_like(img, dtype=np.uint8)

    coords = np.argwhere(img > 128)
    for (i, j) in coords:
        ant[i, j] = 1

    out_path = os.path.join(OUT_ROOT, f"{scene_id}_{tx_id}.npy")
    np.save(out_path, ant)

    return None


# -----------------------------
# 主程序（多进程）
# -----------------------------
if __name__ == "__main__":

    tasks = []

    for scene_id in range(NUM_SCENE):
        for tx_id in range(80):
            tasks.append((scene_id, tx_id))

    print("Total tasks:", len(tasks))

    # 👉 推荐用你刚查到的核数，比如 24
    N_WORKERS = min(24, cpu_count())

    with Pool(N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(process_one, tasks)):
            if res is not None:
                print(res)

            if i % 1000 == 0:
                print(f"progress: {i}/{len(tasks)}")

    print("All done")