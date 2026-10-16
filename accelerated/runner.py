"""Portable CLI; datasets and generated outputs must be supplied externally."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import traceback

from .inputs import inspect_inputs, read_index


def parser():
    p = argparse.ArgumentParser(description='Patched K=1 CUDA distance fields; external official polygon/TX JSON and scene/TX index.')
    p.add_argument('--input-root', type=Path, required=True, help='Root containing polygon/buildings_complete/ and antenna/.')
    p.add_argument('--index', type=Path, required=True, help='Integer (N,2) NPY or CSV with scene,tx header.')
    p.add_argument('--output', type=Path, help='New/compatible output root; required unless --validate-only.')
    p.add_argument('--scale', type=int, choices=[1, 10], default=10, help='1=256x256 (h=1m), 10=2560x2560 (h=0.1m). Geometry stays in meters.')
    p.add_argument('--workers', type=int, default=4, help='External native CPU worker count, e.g. 12 or 16.')
    p.add_argument('--control-points', type=int, default=128, help='Deterministic independent native CPU control points per map.')
    p.add_argument('--scenes', help='Optional comma-separated city IDs from the supplied index.')
    p.add_argument('--shard-index', type=int, default=0, help='Zero-based disjoint city shard index.')
    p.add_argument('--shard-count', type=int, default=1, help='Number of disjoint city shards; each worker needs its own output root.')
    p.add_argument('--reference-root', type=Path, help='Optional external sceneS/pvw_bits.npy and pvp_bits.npy regression references at the selected resolution.')
    p.add_argument('--validate-only', action='store_true', help='Check all input files and print the assignment; no CUDA context/output data.')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.workers < 1 or not 1 <= args.control_points <= 65536:
        p.error('--workers must be positive; --control-points must be in [1,65536].')
    if not 0 <= args.shard_index < args.shard_count:
        p.error('Require 0 <= --shard-index < --shard-count.')
    if args.reference_root and not args.reference_root.is_dir():
        p.error('--reference-root does not exist.')
    index = read_index(args.index)
    inputs = inspect_inputs(args.input_root, index)
    scenes = list(index)
    if args.scenes:
        selected = set(map(int, args.scenes.split(',')))
        if not selected <= set(scenes):
            p.error('--scenes contains cities absent from --index.')
        scenes = [s for s in scenes if s in selected]
    scenes = scenes[args.shard_index::args.shard_count]
    if not scenes:
        p.error('Assignment is empty.')
    plan = dict(scenes=scenes, maps=sum(len(index[s]) for s in scenes), shape=[256 * args.scale] * 2,
                scale=args.scale, workers=args.workers, input_scenes_validated=len(index))
    print('INPUTS_VALIDATED', json.dumps(plan), flush=True)
    if args.validate_only:
        return 0
    if args.output is None:
        p.error('--output is required for simulation.')
    if os.name != 'posix':
        p.error('GPU production runner requires Linux/POSIX. Input validation can run separately.')
    if args.output.resolve() == args.input_root.resolve():
        p.error('Use an output root separate from --input-root.')
    # Deliberately import CUDA and native solver only after the data-only validation path.
    import fcntl
    from numba import cuda
    from .engine import Engine, atomic_json
    if not cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use the documented Linux/CUDA environment; --validate-only needs no GPU.')
    code = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}
    config = dict(shape=plan['shape'], scale=args.scale, K=1, DIFF_BIAS=300., FALLBACK_BIAS=600., output='FP32 DMAPLS01', control_points=args.control_points)
    signature = hashlib.sha256(json.dumps(dict(code=code, inputs=inputs, config=config), sort_keys=True).encode()).hexdigest()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'run_manifest.json'
    if not manifest_path.exists() and any(out.iterdir()):
        raise ValueError('Nonempty output has no compatible run_manifest.json. Use a new output root; do not mix with the historical frozen batch.')
    lock = (out / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text(encoding='utf-8'))
            if old['signature'] != signature or old['scenes'] != scenes:
                raise ValueError('Output signature/assignment differs. Keep old data and choose a new output root.')
        manifest = dict(signature=signature, **plan, config=config, code=code, inputs=inputs,
                        input_root=str(args.input_root.resolve()), output=str(out), shard_index=args.shard_index, shard_count=args.shard_count)
        atomic_json(manifest_path, manifest)
        done, todo = [], []
        for scene in scenes:
            path = out / 'completed' / f'{scene}.json'
            if path.exists():
                report = json.loads(path.read_text(encoding='utf-8'))
                if report['signature'] != signature:
                    raise ValueError(f'City {scene} has an incompatible completion signature.')
                if len(report['outputs']) == len(index[scene]) and all((out / r['file']).exists() and (out / r['file']).stat().st_size == r['bytes'] for r in report['outputs']):
                    done.append(scene)
                    continue
            todo.append(scene)
        print('RUN_START', json.dumps(dict(already_complete=len(done), remaining=len(todo), signature=signature)), flush=True)
        if not todo:
            print('ALL_ASSIGNED_CITIES_ALREADY_COMPLETE', flush=True)
            return 0
        engine = Engine(args.input_root, index, out, scale=args.scale, workers=args.workers,
                        control_points=args.control_points, signature=signature, reference_root=args.reference_root)
        started = time.perf_counter()
        current_scene = todo[0]
        try:
            warmup = engine.warmup(todo[0])
            print('WARMUP_SECONDS', warmup, flush=True)
            future = engine.cpu_pool.submit(engine.build, todo[0])
            for i, scene in enumerate(todo):
                current_scene = scene
                a = future.result()
                if i + 1 < len(todo):
                    future = engine.cpu_pool.submit(engine.build, todo[i + 1])
                engine.consume(a)
                done.append(scene)
                del a
                failed = out / 'failed' / f'{scene}.json'
                if failed.exists():
                    failed.unlink()
                atomic_json(out / 'status.json', dict(assigned_cities=len(scenes), complete_cities=len(done),
                    complete_maps=sum(len(index[s]) for s in done), failed_scenes=[], last_scene=scene,
                    elapsed_seconds_including_warmup=time.perf_counter() - started, updated_unix=time.time()))
            atomic_json(out / 'run_summary.json', dict(complete_cities=len(done), complete_maps=sum(len(index[s]) for s in done),
                elapsed_seconds_including_warmup=time.perf_counter() - started, warmup_seconds=warmup, signature=signature))
            print('RUN_COMPLETE', len(done), flush=True)
        except BaseException:
            atomic_json(out / 'failed' / f'{current_scene}.json', dict(scene=current_scene, traceback=traceback.format_exc(), updated_unix=time.time()))
            raise
        finally:
            engine.close()
    finally:
        lock.close()
    return 0
