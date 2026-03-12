import os
import numpy as np

SRC_ROOT = "/root/wall_segment"
DST_ROOT = "/root/wall_segment_nb"

os.makedirs(DST_ROOT, exist_ok=True)

for scene_id in range(701):

    src_path = f"{SRC_ROOT}/{scene_id}.npy"

    if not os.path.exists(src_path):
        continue

    walls = np.load(src_path, allow_pickle=True)

    n = len(walls)

    walls_nb = np.zeros((n,4), dtype=np.float64)

    for i,(A,B) in enumerate(walls):

        walls_nb[i,0] = A[0]
        walls_nb[i,1] = A[1]
        walls_nb[i,2] = B[0]
        walls_nb[i,3] = B[1]

    dst_path = f"{DST_ROOT}/{scene_id}.npy"

    np.save(dst_path, walls_nb)

    print("converted scene", scene_id)