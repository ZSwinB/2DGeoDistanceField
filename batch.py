import multiprocessing as mp
import subprocess
import sys
import time   # ✅ 新增

SCRIPT_PATH = r"C:\Users\ZSwinB\Documents\GitHub\2DGeoDistanceField\solver.py"

SCENE_START = 684
SCENE_END = 700

TX_START = 80
TX_END = 103

NUM_WORKERS = max(1, mp.cpu_count() // 2)


def run_tx(args):

    scene_id, tx_id = args

    cmd = [
        sys.executable,
        SCRIPT_PATH,
        str(scene_id),
        str(tx_id)
    ]

    print(f"[START] scene={scene_id} tx={tx_id}")

    subprocess.run(cmd)

    print(f"[DONE] scene={scene_id} tx={tx_id}")


def run_scene(scene_id):

    print(f"\n========== SCENE {scene_id} ==========\n")

    tasks = [(scene_id, tx) for tx in range(TX_START, TX_END + 1)]

    with mp.Pool(NUM_WORKERS, maxtasksperchild=1) as pool:
        pool.map(run_tx, tasks)

    # ✅ 关键补丁：释放系统资源
    time.sleep(0.5)


def main():

    for scene in range(SCENE_START, SCENE_END + 1):
        run_scene(scene)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    mp.freeze_support()   # ✅ 再加一层保险

    main()