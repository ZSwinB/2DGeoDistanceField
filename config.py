# pipeline/config.py

class Config:

    # =====================
    # 输入
    # =====================

    GEO_ROOT = [
    r"G:\data_SRM\geo",
    # "inputs/geo2",
    # ...
    ]
    ANT_ROOT = r"G:\data_SRM\antenna"
    GAIN_ROOT = r"G:\RadioMapSeer\gain\DPM"

    # =====================
    # 输出
    # =====================
    OUTPUT_ROOT = "outputs"

    # ===== stage2 =====
    WALL_DIR = "wall"

    # ===== stage3 =====
    MIDDATA_DIR = "middata"
    PVW_DIR = "middata/pvw"
    WWV_DIR = "middata/wwv"
    PVP_DIR = "middata/pvp"

    # ===== stage4 =====
    CONVERT_DIR = "convert"

    # ===== stage5 =====
    SIM_DIR = "sim"



    # ===== 多进程 =====
    USE_MULTIPROCESS = False
    NUM_WORKERS = 4   # 或 None=自动
    USE_INNER_MP = True
    INNER_WORKERS =8
    # config.py

    IDX_LIST = list(range(0,1))  # 要跑哪些scene
    TX_LIST  = list(range(0, 50))

    NUM_WORKERS = None   # None=自动
    # =====================
    # 运行控制
    # =====================

    GEN_WALL = True
    GEN_PVW  = True
    GEN_WWV  = True
    GEN_PVP  = True
    GEN_CONVERT = True

    # 仿真不在这里跑（交给 batch）