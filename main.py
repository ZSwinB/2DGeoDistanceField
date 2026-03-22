# main.py

from config import Config
import multiprocessing as mp
import platform
from pipeline.stage1_input.load_geo import load_geo
from pipeline.stage2_wall.run import run as build_wall
from pipeline.stage3_middata.build_pvw import build_pvw
from pipeline.stage3_middata.build_wwv import build_wwv
from pipeline.stage3_middata.build_pvp import build_pvp
from pipeline.stage4_convert.run import run as convert

def setup_mp():
    system = platform.system()

    if system == "Windows":
        mp.set_start_method("spawn", force=True)
    else:
        # Linux / Mac → 用默认 fork，不要动
        pass
def run_one_scene(idx):

    config = Config()
    config.IDX = idx

    print(f"\n========== IDX {idx} ==========\n")

    geo, antenna, gain = load_geo(config)

    if config.GEN_WALL:
        build_wall(geo, config)

    if config.GEN_PVW:
        build_pvw(geo, None, config)

    if config.GEN_WWV:
        build_wwv(geo, None, config)

    if config.GEN_PVP:
        build_pvp(geo, None, None, config)

    if config.GEN_CONVERT:
        convert(None, None, None, config)


def main():

    config = Config()

    idx_list = config.IDX_LIST

    # =====================
    # 单进程
    # =====================
    if not config.USE_MULTIPROCESS:

        for idx in idx_list:
            run_one_scene(idx)

    # =====================
    # 多进程（scene级）
    # =====================
    else:

        n = config.NUM_WORKERS or mp.cpu_count()

        print(f"[Mode] Multiprocess scenes | workers={n}")

        with mp.Pool(n) as pool:
            pool.map(run_one_scene, idx_list)


if __name__ == "__main__":
    setup_mp()
    mp.freeze_support()
    main()