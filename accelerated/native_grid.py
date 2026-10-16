import numpy as np
GRID_SIZE=8
def build_wall_grid_nb(walls, H, W):
    """
    Build spatial grid index for walls (CSR-like structure).

    Behavior:
        - Partitions space into GRID_SIZE cells
        - Assigns each wall to overlapping grid cells
        - Produces compact indexing arrays

    Args:
        walls (np.ndarray): wall segments (N,4)
        H (int): grid height
        W (int): grid width

    Returns:
        tuple:
            grid_start (np.ndarray): start index per cell
            grid_count (np.ndarray): number of walls per cell
            grid_list (np.ndarray): flattened wall indices
    """
    GX = (W + GRID_SIZE) // GRID_SIZE
    GY = (H + GRID_SIZE) // GRID_SIZE
    grid_count = np.zeros((GX, GY), dtype=np.int32)
    for idx in range(walls.shape[0]):
        x0 = walls[idx, 0]
        y0 = walls[idx, 1]
        x1 = walls[idx, 2]
        y1 = walls[idx, 3]
        gx0 = max(0, int(min(x0, x1) // GRID_SIZE))
        gx1 = min(GX - 1, int(max(x0, x1) // GRID_SIZE))
        gy0 = max(0, int(min(y0, y1) // GRID_SIZE))
        gy1 = min(GY - 1, int(max(y0, y1) // GRID_SIZE))
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                grid_count[gx, gy] += 1
    grid_start = np.zeros((GX, GY), dtype=np.int32)
    total = 0
    for gx in range(GX):
        for gy in range(GY):
            grid_start[gx, gy] = total
            total += grid_count[gx, gy]
    grid_list = np.zeros(total, dtype=np.int32)
    cursor = grid_start.copy()
    for idx in range(walls.shape[0]):
        x0 = walls[idx, 0]
        y0 = walls[idx, 1]
        x1 = walls[idx, 2]
        y1 = walls[idx, 3]
        gx0 = max(0, int(min(x0, x1) // GRID_SIZE))
        gx1 = min(GX - 1, int(max(x0, x1) // GRID_SIZE))
        gy0 = max(0, int(min(y0, y1) // GRID_SIZE))
        gy1 = min(GY - 1, int(max(y0, y1) // GRID_SIZE))
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                k = cursor[gx, gy]
                grid_list[k] = idx
                cursor[gx, gy] += 1
    return (grid_start, grid_count, grid_list)