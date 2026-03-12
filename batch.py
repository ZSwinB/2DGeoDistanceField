import multiprocessing as mp
import subprocess

SCRIPT_PATH = "solver.py"

SCENE_START = 0
SCENE_END = 700

TX_START = 0
TX_END = 79

NUM_WORKERS = mp.cpu_count()


def run_tx(args):

    scene_id, tx_id = args

    cmd = [
        "python3",
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

    with mp.Pool(NUM_WORKERS) as pool:
        pool.map(run_tx, tasks)


def main():

    for scene in range(SCENE_START, SCENE_END + 1):
        run_scene(scene)


if __name__ == "__main__":
    main()