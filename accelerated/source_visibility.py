"""Compute source visibility from continuous coordinates independently of receiver grids."""
import numpy as np
from .r3_pvw import visible_walls_for_point_nb
from .cpu_visibility import visible_between_points_nb


def source_visibility(tx, walls, gs, gc, gl, corner_x, corner_y):
    mask = visible_walls_for_point_nb((float(tx[0]), float(tx[1])), walls).astype(bool)
    # Same wall/corner candidate rule as receiver PVP, queried at the actual source.
    corner_to_id = {(float(x), float(y)): i for i, (x, y) in enumerate(zip(corner_x, corner_y))}
    corners = list(corner_to_id)
    ids, seen = [], set()
    for wid in np.flatnonzero(mask):
        for x, y in walls[wid].reshape(2, 2):
            key = (float(x), float(y))
            if key in seen:
                continue
            if visible_between_points_nb(float(tx[0]), float(tx[1]), float(x), float(y), walls, gs, gc, gl, int(wid), 8):
                seen.add(key)
                if key not in corner_to_id:
                    corner_to_id[key] = len(corners)
                    corners.append(key)
                ids.append(corner_to_id[key])
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    return mask, np.asarray(ids, dtype=np.int32), corners[:, 0].copy(), corners[:, 1].copy()
