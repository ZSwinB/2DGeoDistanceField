"""Compare portable-runner outputs with external historical references (not shipped)."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accelerated.lossless_distance import DistanceReader


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference-output', type=Path, help='Historical output with completed/S.json; required for scale 10.')
    p.add_argument('--coarse-reference-root', type=Path, required=True, help='External sceneS/sources.json and native_controls.npy.')
    args = p.parse_args()
    manifest = json.loads((args.output / 'run_manifest.json').read_text())
    scale = manifest['scale']
    records = []
    for scene in manifest['scenes']:
        report = json.loads((args.output / 'completed' / f'{scene}.json').read_text())
        folder = args.coarse_reference_root / f'scene{scene}'
        sources = json.loads((folder / 'sources.json').read_text())
        controls = np.load(folder / 'native_controls.npy', mmap_mode='r', allow_pickle=False)
        reference = None
        if scale == 10:
            if args.reference_output is None:
                p.error('--reference-output is required for high-resolution comparisons.')
            reference = json.loads((args.reference_output / 'completed' / f'{scene}.json').read_text())
        for row in report['outputs']:
            field = DistanceReader(args.output / row['file']).read()
            raw_hash = hashlib.sha256(field.tobytes()).hexdigest()
            assert raw_hash == row['field_sha256']
            tx = row['tx']
            original = controls[sources['txs'].index(tx)]
            assert np.array_equal(field[::scale, ::scale], original), (scene, tx, 'full coarse grid')
            if reference is not None:
                expected = next(r for r in reference['outputs'] if r['tx'] == tx)
                assert raw_hash == expected['field_sha256'], (scene, tx, 'full fine field')
            records.append({'scene': scene, 'tx': tx, 'full_coarse_equal': True,
                            'full_fine_equal': True if reference is not None else None})
    print(json.dumps({'scale': scale, 'maps': len(records), 'all_passed': True, 'comparisons': records}, indent=2))


if __name__ == '__main__':
    main()
