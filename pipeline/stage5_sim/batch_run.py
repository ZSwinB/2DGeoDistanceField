# pipeline/stage5_sim/batch_run.py

import multiprocessing as mp
import subprocess
import sys
import time
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT not in sys.path:
    sys.path.append(ROOT)
from config import Config


# ===== 初始化配置 =====
config = Config()

IDX_LIST = config.IDX_LIST
TX_LIST  = config.TX_LIST

NUM_WORKERS = config.NUM_WORKERS or max(1, mp.cpu_count() // 2)

# solver路径（相对当前文件）
SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__),
    "solver.py"
)


def run_tx(args):

    idx, tx_id = args

    cmd = [
        sys.executable,
        SCRIPT_PATH,
        str(idx),
        str(tx_id)
    ]

    print(f"[START] idx={idx} tx={tx_id}")

    subprocess.run(cmd)

    print(f"[DONE] idx={idx} tx={tx_id}")


def run_scene(idx):

    print(f"\n========== IDX {idx} ==========\n")

    tasks = [(idx, tx) for tx in TX_LIST]

    with mp.Pool(NUM_WORKERS, maxtasksperchild=1) as pool:
        pool.map(run_tx, tasks)

    # 释放系统资源
    time.sleep(0.5)


def main():

    print(f"[Batch] scenes={len(IDX_LIST)} | tx={len(TX_LIST)} | workers={NUM_WORKERS}")

    for idx in IDX_LIST:
        run_scene(idx)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    mp.freeze_support()
    main()