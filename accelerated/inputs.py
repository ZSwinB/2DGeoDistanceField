"""External input contracts and deterministic scene/TX selection."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np

from . import geometry


def read_index(path):
    path = Path(path)
    if path.suffix.lower() == '.npy':
        values = np.load(path, allow_pickle=False)
        if values.ndim != 2 or values.shape[1] != 2 or values.dtype.kind not in 'iu':
            raise ValueError('Index NPY must be an integer (N,2) array: [scene,tx].')
        rows = [(int(s), int(t)) for s, t in values]
    elif path.suffix.lower() == '.csv':
        with path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ['scene', 'tx']:
                raise ValueError('CSV header must be exactly: scene,tx')
            rows = [(int(row['scene']), int(row['tx'])) for row in reader]
    else:
        raise ValueError('Index must be .npy or .csv; see docs/INPUTS.md.')
    if not rows or len(set(rows)) != len(rows) or any(s < 0 or t < 0 for s, t in rows):
        raise ValueError('Index must contain unique nonnegative scene/TX pairs.')
    result = {}
    for scene, tx in rows:
        result.setdefault(scene, []).append(tx)
    return dict(sorted(result.items()))


def inspect_inputs(root, index):
    """Validate every scene in the supplied index, without CUDA or simulation."""
    root = Path(root)
    manifest = []
    for scene, txs in index.items():
        polygon = root / 'polygon' / 'buildings_complete' / f'{scene}.json'
        antenna = root / 'antenna' / f'{scene}.json'
        polys = geometry.load_polygons(polygon)
        if not polys:
            raise ValueError(f'Scene {scene}: at least one nondegenerate polygon is required.')
        walls = geometry.walls_from_polygons(polys)
        if not np.isfinite(walls).all():
            raise ValueError(f'Scene {scene}: nonfinite wall coordinates.')
        antennas = json.loads(antenna.read_text(encoding='utf-8'))
        if not isinstance(antennas, list):
            raise ValueError(f'Scene {scene}: antenna JSON must be a list indexed by TX id.')
        for tx in txs:
            if tx >= len(antennas):
                raise ValueError(f'Scene {scene}: TX {tx} not present in antenna JSON.')
            xy = np.asarray(antennas[tx], dtype=np.float64)
            if xy.shape != (2,) or not np.isfinite(xy).all():
                raise ValueError(f'Scene {scene} TX {tx}: expected finite [x,y].')
            # The published/validated source placement uses integer coordinates.
            # Fractional TX support needs explicit source-zero semantics regression.
            if not np.array_equal(xy, np.rint(xy)) or (xy < 0).any() or (xy > 255).any():
                raise ValueError(f'Scene {scene} TX {tx}: validated format requires integer XY in [0,255].')
        manifest.append({
            'scene': scene, 'txs': txs, 'wall_count': len(walls),
            'polygon_sha256': hashlib.sha256(polygon.read_bytes()).hexdigest(),
            'antenna_sha256': hashlib.sha256(antenna.read_bytes()).hexdigest(),
        })
    return manifest
