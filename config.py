# pipeline/config.py

class Config:

    # =====================
    # 输入
    # =====================

    GEO_ROOT = [
        r"G:\data_SRM\geo",
        # "inputs/geo2",
        # ...
    ]  # list[str]: geometry root directories; each path can represent a different geometry source (e.g., buildings, vehicles, obstacles);
    # all entries will be merged/combined into a single scene during preprocessing if multiple paths are provided

    ANT_ROOT = r"G:\data_SRM\antenna"  # str: antenna parameter directory (per scene / transmitter)

    GAIN_ROOT = r"G:\RadioMapSeer\gain\DPM"  # str: gain pattern directory

    # =====================
    # 输出
    # =====================

    OUTPUT_ROOT = "outputs"  # str: root directory for all generated outputs ,set as you like

    # ===== stage2 =====
    WALL_DIR = "wall"  # str: directory name for storing generated wall segments

    # ===== stage3 =====
    MIDDATA_DIR = "middata"      # str: root directory for all intermediate data
    PVW_DIR = "middata/pvw"      # str: Point Visible Walls storage path
    WWV_DIR = "middata/wwv"      # str: Wall-Wall visibility storage path
    PVP_DIR = "middata/pvp"      # str: Point Visible Points storage path
    # recommend not to change these
    # ===== stage4 =====
    CONVERT_DIR = "convert"  # str: directory for converted (compute-optimized) data
    # recommend not to change these
    # ===== stage5 =====
    SIM_DIR = "sim"  # str: directory for simulation outputs

    # =====================
    # 多进程
    # =====================

    USE_MULTIPROCESS = False  # bool: enable outer-level multiprocessing (scene-level parallelism)

    NUM_WORKERS = None  # int | None: number of outer workers; None means auto (CPU count)

    USE_INNER_MP = True  # bool: enable inner multiprocessing (within a single scene)

    INNER_WORKERS = 8  # int: number of inner workers used inside stage computations
    # note:
    # - prefer enabling INNER multiprocessing for most cases (better load balance within a scene)
    # - if the number of scenes is very large, you may enable only outer multiprocessing instead
    # - avoid enabling both outer and inner multiprocessing simultaneously, as it may cause CPU oversubscription and severe performance degradation
    # =====================
    # 任务选择
    # =====================

    IDX_LIST = list(range(0, 1))  # list[int]: scene indices to process

    TX_LIST = list(range(0, 1))  # list[int]: transmitter indices per scene

    K = 1  # int: number of top-K paths to keep per receiver (e.g., multipath count)
    DIFF_BIAS = 300.0  # float: penalty added to diffraction paths (used to control preference vs reflection/LOS)
    FALLBACK_BIAS = 600.0  # default upper-bound distance (used as initial candidate)

    # =====================
    # 运行控制
    # =====================

    GEN_WALL = True      # bool: generate wall data (Stage 2)

    GEN_PVW = True       # bool: generate PVW (Point Visible Walls)

    GEN_WWV = True       # bool: generate WWV (Wall-Wall Visibility)

    GEN_PVP = True       # bool: generate PVP (Point Visible Points)

    GEN_CONVERT = True   # bool: run data conversion (Stage 4)

    # 仿真不在这里跑（由 batch_run.py 控制）