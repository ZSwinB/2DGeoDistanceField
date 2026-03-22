import os
import numpy as np


# =====================
# 路径工具
# =====================

def ensure_dir(path):
    if path is None or path == "":
        return
    os.makedirs(path, exist_ok=True)


def join(*args):
    return os.path.join(*args)


# =====================
# 保存
# =====================

def save_npy(path, data):
    ensure_dir(os.path.dirname(path))
    np.save(path, data)


def save_npz(path, **kwargs):
    ensure_dir(os.path.dirname(path))
    np.savez_compressed(path, **kwargs)


# =====================
# 读取
# =====================

def load_npy(path, allow_pickle=True):
    return np.load(path, allow_pickle=allow_pickle)


def exists(path):
    return os.path.exists(path)


# =====================
# 扫描
# =====================

def list_npy_ids(root):
    """
    返回目录下所有 .npy 的 id（去掉后缀）
    """
    if not os.path.exists(root):
        return []

    ids = []
    for f in os.listdir(root):
        if f.endswith(".npy"):
            try:
                ids.append(int(f.split(".")[0]))
            except:
                continue
    return sorted(ids)


# =====================
# 批量构建路径
# =====================

def build_path(root, name, ext=".npy"):
    return os.path.join(root, f"{name}{ext}")