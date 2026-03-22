# pipeline/stage1_input/load_geo.py

import numpy as np
import os
from PIL import Image


def _load_binary(base_path):
    """
    尝试读取：
        base_path.npy
        base_path.png

    返回：
        0/1 np.array 或 None
    """

    npy_path = base_path + ".npy"
    png_path = base_path + ".png"

    # 优先 npy
    if os.path.exists(npy_path):
        data = np.load(npy_path)
        return (data > 0).astype(np.uint8)

    # 再 png
    if os.path.exists(png_path):
        img = Image.open(png_path).convert("L")
        arr = np.array(img)
        return (arr == 255).astype(np.uint8)

    return None


def load_geo(config, tx_id=None):
    """
    返回：
        geo
        antenna（可为None）
        gain（可为None）
    """

    idx = config.IDX

    # =====================
    # geo（多root融合）
    # =====================
    geo = None

    for root in config.GEO_ROOT:

        base = os.path.join(root, f"{idx}")
        data = _load_binary(base)

        if data is None:
            continue

        if geo is None:
            geo = data
        else:
            geo = np.logical_or(geo, data)

    if geo is None:
        raise FileNotFoundError(f"geo not found: idx={idx}")

    geo = geo.astype(np.uint8)

    # =====================
    # antenna
    # =====================
    antenna = None
    if tx_id is not None:
        ant_base = os.path.join(config.ANT_ROOT,  f"{idx}_{tx_id}")
        antenna = _load_binary(ant_base)

    # =====================
    # gain
    # =====================
    gain = None
    if tx_id is not None:
        gain_base = os.path.join(config.GAIN_ROOT,  f"{idx}_{tx_id}")
        gain = _load_binary(gain_base)

    return geo, antenna, gain