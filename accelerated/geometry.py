"""Official vector geometry in image coordinates, without raster contour fitting."""
import json
import numpy as np


def load_polygons(path, height=256):
    polygons = []
    for ring in json.loads(path.read_text()):
        vertices = np.asarray(ring, dtype=np.float64)
        assert vertices.ndim == 2 and vertices.shape[1] == 2
        assert np.isfinite(vertices).all()
        vertices = vertices[np.r_[True, np.any(np.diff(vertices, axis=0) != 0, axis=1)]]
        if np.array_equal(vertices[0], vertices[-1]):
            vertices = vertices[:-1]
        assert len(vertices) >= 3
        vertices[:, 1] = height - 1 - vertices[:, 1]
        polygons.append(vertices)
    return polygons


def occupied(points, polygons, tolerance=1e-9):
    """Polygon union; boundary points count as occupied. Inputs are continuous XY."""
    points = np.asarray(points, dtype=np.float64)
    x, y = points[..., 0], points[..., 1]
    result = np.zeros(x.shape, dtype=bool)
    for vertices in polygons:
        inside = np.zeros(x.shape, dtype=bool)
        boundary = inside.copy()
        for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
            dx, dy = b - a
            cross = dx * (y-a[1]) - dy * (x-a[0])
            dot = (x-a[0])*dx + (y-a[1])*dy
            length2 = dx*dx + dy*dy
            boundary |= (np.abs(cross) <= tolerance*np.sqrt(length2)) & (dot >= -tolerance) & (dot <= length2+tolerance)
            if dy != 0:
                inside ^= ((a[1] > y) != (b[1] > y)) & (x < a[0] + (y-a[1])*dx/dy)
        result |= inside | boundary
    return result


def walls_from_polygons(polygons):
    walls = np.concatenate([np.stack([p, np.roll(p, -1, axis=0)], axis=1) for p in polygons])
    assert not np.any(np.all(walls[:, 0] == walls[:, 1], axis=1))
    return walls.astype(np.float64)
