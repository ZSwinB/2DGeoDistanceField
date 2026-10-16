"""Portable city-resident CPU/GPU pipeline; numerical kernels retain production math."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import os
import resource
import shutil
import time

import numpy as np
from numba import cuda

from . import (corner_cache, cpu_solver as cpu, geometry, gpu_solver_packed_generated as gpu,
               native_grid, native_solver, native_wwv, pipeline_fast_cpu as source_fast,
               r3_gpu as gpu_geometry, r3_numeric as numeric, source_visibility, wwv_numeric)
from .lossless_distance import DistanceReader, write_distance


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temp, path)


class Engine:
    def __init__(self, input_root, index, output, *, scale=10, workers=4,
                 control_points=128, signature='', reference_root=None):
        from numcodecs.blosc import set_nthreads
        set_nthreads(1)
        self.input_root = Path(input_root)
        self.index = index
        self.out = Path(output)
        self.scale = scale
        self.size = 256 * scale
        self.workers = workers
        self.control_points = control_points
        self.signature = signature
        self.reference_root = Path(reference_root) if reference_root else None
        self.output_device = None
        self.io_pool = ThreadPoolExecutor(max_workers=1)
        self.cpu_pool = ThreadPoolExecutor(max_workers=1)
        self.start = time.perf_counter()

    def close(self):
        self.cpu_pool.shutdown(wait=True)
        self.io_pool.shutdown(wait=True)

    def load(self, scene):
        polygons = geometry.load_polygons(self.input_root / 'polygon/buildings_complete' / f'{scene}.json')
        walls = np.ascontiguousarray(geometry.walls_from_polygons(polygons).reshape(-1, 4))
        corners = np.ascontiguousarray(np.unique(walls.reshape(-1, 2), axis=0))
        gs, gc, gl = native_grid.build_wall_grid_nb(walls, 256, 256)
        verts = np.ascontiguousarray(np.concatenate(polygons))
        starts = np.r_[0, np.cumsum([len(p) for p in polygons])].astype(np.int32)
        bounds = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in polygons])
        coarse_geo = numeric.raster(0, 256, 256, 1., verts, starts, bounds)
        wall_pairs = walls.reshape(-1, 2, 2).tolist()
        movements = native_wwv.build_WWVmove(coarse_geo, wall_pairs)
        wwv = wwv_numeric.compute_wwv(wall_pairs, movements, native_wwv.visible_walls_for_point, workers=self.workers)
        antennas = json.loads((self.input_root / 'antenna' / f'{scene}.json').read_text(encoding='utf-8'))
        txs = self.index[scene]
        xy = {str(tx): [float(antennas[tx][0]), float(255 - antennas[tx][1])] for tx in txs}
        flags = geometry.occupied(np.array(list(xy.values())), polygons)
        sources = {'scene': scene, 'txs': txs, 'xy': xy,
                   'inside_or_on_polygon': {str(tx): bool(f) for tx, f in zip(txs, flags)}}
        folder = self.out / 'inputs' / f'scene{scene}'
        folder.mkdir(parents=True, exist_ok=True)
        for name, value in [('geo_coarse', coarse_geo), ('walls', walls), ('wwv', wwv)]:
            np.save(folder / f'{name}.npy', value)
        atomic_json(folder / 'sources.json', sources)
        return dict(scene=scene, folder=folder, walls=walls, corners=corners, gs=gs, gc=gc, gl=gl,
                    wwv=wwv, verts=verts, starts=starts, bounds=bounds, coarse_geo=coarse_geo, sources=sources)

    def source(self, a, txid):
        tx = tuple(a['sources']['xy'][str(txid)])
        w, c = a['walls'], a['corners']
        gs, gc, gl = a['gs'], a['gc'], a['gl']
        mask, ids, cx, cy = source_visibility.source_visibility(tx, w, gs, gc, gl, c[:, 0], c[:, 1])
        assert np.array_equal(cx, c[:, 0]) and np.array_equal(cy, c[:, 1])
        order = np.array(sorted(np.flatnonzero(mask), key=lambda wi: np.sum((w[wi].reshape(2, 2).mean(axis=0) - tx) ** 2)), np.int32)
        imgs = np.array([cpu.reflect_point_nb(*tx, *wall) for wall in w])
        prefix = source_fast.source_prefixes(tx, w, c, order, imgs, gs, gc, gl, cpu.intersect_wall_return_point_nb, cpu.visible_fast_nb)
        targets, counts, double, cimages = source_fast.source_mirrors(w, c, order, ids, imgs, a['wwv'], cpu.reflect_point_nb)
        state = (*prefix, targets, counts, double, cimages, np.linalg.norm(c - np.array(tx), axis=1))
        eligible = prefix[0].any(axis=0)
        eligible[ids] = True
        return dict(tx=tx, mask=mask, ids=ids, order=order, imgs=imgs, state=state, eligible=np.flatnonzero(eligible))

    def warmup(self, scene):
        started = time.perf_counter()
        a = self.load(scene)
        w = a['walls']
        unsafe = numeric.unsafe_walls(w)
        numeric.raster(0, 1, 1, float(self.scale), a['verts'], a['starts'], a['bounds'])
        numeric.pvw_tile(np.zeros((1, 2), np.int32), 0, 1, 1, float(self.scale), w, unsafe)
        tables, _ = corner_cache.prepare(a['corners'], w)
        s = self.source(a, a['sources']['txs'][0])
        fake = dict(a, geo=np.zeros((1, 1), np.uint8), points=np.zeros((1, 2), np.int32),
                    pvw=np.zeros((1, 1, (len(w) + 7) // 8), np.uint8), tables=tables)
        resident, _ = self.upload_city(fake, verify=False)
        self.solve_one(fake, s, resident)
        cuda.synchronize()
        del resident
        return time.perf_counter() - started

    def build(self, scene):
        started = time.perf_counter()
        a = self.load(scene)
        w, size, scale = a['walls'], self.size, float(self.scale)
        print('CPU_BUILD_START', scene, flush=True)
        t = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            tiles = list(pool.map(lambda y: numeric.raster(y, 16, size, scale, a['verts'], a['starts'], a['bounds']), range(0, size, 16)))
        a['geo'] = np.ascontiguousarray(np.concatenate(tiles))
        del tiles
        assert np.array_equal(a['geo'][::self.scale, ::self.scale], a['coarse_geo'])
        raster_seconds = time.perf_counter() - t
        a['points'] = np.ascontiguousarray(np.argwhere(a['geo'] == 0), np.int32)
        if not len(a['points']):
            raise ValueError(f'Scene {scene}: no free-space receivers.')
        a['pvw'] = np.zeros((size, size, (len(w) + 7) // 8), np.uint8)
        unsafe = numeric.unsafe_walls(w)
        t = time.perf_counter()
        def one(y):
            points = np.ascontiguousarray(np.argwhere(a['geo'][y:y + 16] == 0), np.int32)
            return y, numeric.pvw_tile(points, y, 16, size, scale, w, unsafe)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            jobs = [pool.submit(one, y) for y in range(0, size, 16)]
            for future in as_completed(jobs):
                y, packed = future.result()
                a['pvw'][y:y + 16] = packed
        pvw_seconds = time.perf_counter() - t
        t = time.perf_counter()
        a['tables'], _ = corner_cache.prepare(a['corners'], w)
        tables_seconds = time.perf_counter() - t
        a['prepared_sources'] = [self.source(a, tx) for tx in a['sources']['txs']]
        self.controls(a)
        a['timing'] = dict(cpu_build_seconds=time.perf_counter() - started, raster_seconds=raster_seconds,
                           pvw_seconds=pvw_seconds, angular_tables_seconds=tables_seconds)
        if self.reference_root:
            path = self.reference_root / f'scene{scene}' / 'pvw_bits.npy'
            if path.exists():
                assert np.array_equal(a['pvw'], np.load(path, mmap_mode='r')), (scene, 'PVW reference')
        return a

    def controls(self, a):
        free = np.argwhere(a['coarse_geo'] == 0)
        rng = np.random.default_rng(20261016 + a['scene'])
        sample = np.ascontiguousarray(free[rng.choice(len(free), min(self.control_points, len(free)), replace=False)], np.int32)
        w, c = a['walls'], a['corners']
        gs, gc, gl = a['gs'], a['gc'], a['gl']
        pvw = np.zeros((256, 256, len(w)), np.bool_)
        ps, pl, flat = np.zeros((256, 256), np.int64), np.zeros((256, 256), np.int32), []
        for y, x in sample:
            pvw[y, x] = source_visibility.visible_walls_for_point_nb((float(x), float(y)), w)
            ids = [i for i, corner in enumerate(c) if cpu.visible_fast_nb(float(x), float(y), float(corner[0]), float(corner[1]), w, gs, gc, gl, -1)]
            ps[y, x], pl[y, x] = len(flat), len(ids)
            flat.extend(ids)
        values = []
        for s in a['prepared_sources']:
            field = np.full((256, 256, 1), 1000, np.float32)
            tx = s['tx']
            native_solver.solver_rx_loop(sample, *tx, int(tx[1]), int(tx[0]), field, pvw, s['mask'], s['order'], w, gs, gc, gl,
                ps, pl, np.asarray(flat, np.int32), s['ids'], np.ascontiguousarray(c[:, 0]), np.ascontiguousarray(c[:, 1]), a['wwv'], s['imgs'], 1, 1000., 1., 0)
            values.append(field[sample[:, 0], sample[:, 1], 0])
        a['control_yx'], a['control_values'] = sample, np.stack(values)
        np.save(a['folder'] / 'control_yx.npy', sample)
        np.save(a['folder'] / 'control_values.npy', a['control_values'])

    def upload_city(self, a, verify=True):
        t = time.perf_counter()
        c = a['corners']
        resident = {i: cuda.to_device(v) for i, v in [(0, a['points']), (6, a['pvw']), (9, a['walls']), (10, a['gs']), (11, a['gc']), (12, a['gl']), (19, a['wwv'])]}
        dg, dc = cuda.to_device(a['geo']), cuda.to_device(c)
        dt = [cuda.to_device(v) for v in a['tables']]
        pc = cuda.device_array((*a['geo'].shape, (len(c) + 7) // 8), np.uint8)
        gpu_geometry.packed_pvp[(pc.size + 127) // 128, 128](dg, float(self.scale), dc, resident[9], resident[10], resident[11], resident[12], *dt, pc)
        resident[13] = pc
        cuda.synchronize()
        if verify and self.reference_root:
            path = self.reference_root / f'scene{a["scene"]}' / 'pvp_bits.npy'
            if path.exists():
                assert np.array_equal(pc.copy_to_host(), np.load(path, mmap_mode='r')), (a['scene'], 'PVP reference')
        return resident, time.perf_counter() - t

    def solve_one(self, a, s, resident):
        shape = (*a['geo'].shape, 1)
        if self.output_device is None or self.output_device.shape != shape:
            self.output_device = cuda.device_array(shape, np.float32)
        gpu_geometry.fill_field[(a['geo'].size + 255) // 256, 256](self.output_device)
        tx, c = s['tx'], a['corners']
        args = [a['points'], *tx, int(tx[1] * self.scale), int(tx[0] * self.scale), None, a['pvw'], s['mask'], s['order'], a['walls'], a['gs'], a['gc'], a['gl'], None,
                s['eligible'], np.zeros(1, np.int32), s['ids'], np.ascontiguousarray(c[:, 0]), np.ascontiguousarray(c[:, 1]), a['wwv'], s['imgs'], 1, 1000., float(self.scale), 0, s['state']]
        dev = []
        for i, value in enumerate(args):
            if i == 5:
                dev.append(self.output_device)
            elif i in resident:
                dev.append(resident[i])
            elif isinstance(value, np.ndarray):
                dev.append(cuda.to_device(value))
            elif isinstance(value, tuple):
                dev.append(tuple(cuda.to_device(x) for x in value))
            else:
                dev.append(value)
        gpu.solver_rx_loop[(len(a['points']) + 127) // 128, 128](*dev)
        return self.output_device.copy_to_host()[:, :, 0]

    def write_checked(self, folder, a, k, tx, field):
        assert field.dtype == np.float32 and field.shape == (self.size, self.size)
        assert np.isfinite(field).all() and (field >= 0).all() and (field <= 1000).all()
        assert (field[a['geo'] > 0] == 1000).all()
        yx = a['control_yx'] * self.scale
        assert np.array_equal(field[yx[:, 0], yx[:, 1]], a['control_values'][k]), (a['scene'], tx, 'native CPU sample mismatch')
        path = folder / f'distance_{a["scene"]}_{tx}.dmap'
        metadata = dict(scene=a['scene'], tx=tx, spacing_m=1 / self.scale, geometry_precision='float64', output_precision='float32', K=1,
                        source_xy=a['sources']['xy'][str(tx)], source_inside_or_on_polygon=a['sources']['inside_or_on_polygon'][str(tx)], run_signature=self.signature)
        write_distance(path, field, 256, metadata=metadata)
        restored = DistanceReader(path).read()
        payload = field.tobytes()
        assert restored.tobytes() == payload
        return {'file': str(path.relative_to(self.out)), 'bytes': path.stat().st_size,
                'field_sha256': hashlib.sha256(payload).hexdigest(), 'native_cpu_control_points': len(yx),
                'lossless_roundtrip_equal': True, **metadata}

    def consume(self, a):
        start = time.perf_counter()
        folder = self.out / 'distance' / f'scene{a["scene"]}'
        folder.mkdir(parents=True, exist_ok=True)
        if min(shutil.disk_usage(self.out).free, shutil.disk_usage(folder.resolve()).free) < 2 * 1024**3:
            raise OSError('Less than 2 GiB free; checkpoint retained.')
        resident, pvp_seconds = self.upload_city(a)
        rows, pending = [], []
        try:
            for k, (tx, s) in enumerate(zip(a['sources']['txs'], a['prepared_sources'])):
                field = self.solve_one(a, s, resident)
                pending.append(self.io_pool.submit(self.write_checked, folder, a, k, tx, field))
                if len(pending) >= 3:
                    rows.append(pending.pop(0).result())
            rows.extend(f.result() for f in pending)
        finally:
            # Drain outstanding writers before a failed caller can close/reuse state.
            for future in pending:
                future.result()
            del resident
            cuda.current_context().deallocations.clear()
        report = dict(scene=a['scene'], signature=self.signature, status='complete', timing=a['timing'], gpu_pvp_seconds=pvp_seconds,
                      consume_seconds=time.perf_counter() - start, outputs=rows, source_flags=a['sources']['inside_or_on_polygon'],
                      process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, completed_unix=time.time())
        atomic_json(self.out / 'completed' / f'{a["scene"]}.json', report)
        print('CITY_COMPLETE', a['scene'], len(rows), flush=True)
        return report
