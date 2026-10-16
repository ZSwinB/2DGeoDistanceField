# Historical production handoff (2026-10-16)

This is the original agent-oriented report for the completed production batch. All datasets, raw inputs, output fields, images, external evidence JSONs and archives referenced below are deliberately excluded from this repository. Historical server paths describe the prior run; they are not dependencies of the new portable entry. Use [GPU_RUNNER.md](GPU_RUNNER.md) and [INPUTS.md](INPUTS.md) for the current runnable interface. The new numeric modules are in accelerated/; numeric_provenance.json records their source relationship.

---

# 2DGeoDistanceField agent handoff — production_20261016

This is an operational/scientific handoff, not a proposal. Read this file and `simulation_config.json` before changing code or launching jobs. Paths are intentionally explicit. All numerical simulation/analysis was run on resource 4 or 6; Windows was used for orchestration, downloads, and checksum checks.

## 0. Authoritative current state

- Production is **complete**: 701 distinct cities × 16 selected TX = 11216 maps of shape `(2560,2560)`, FP32.
- Failed cities: 0. Stored map containers: **29,918,569,508 bytes = 27.863839183 GiB**.
- Source-inside/on-polygon flags: **249 city/TX groups**. These are produced and flagged, not silently moved or excluded. Do not confuse this with the earlier **78 PNG-mask-overlap cases**; they are different tests/counts.
- Final datasets are split across **three disjoint output roots**, described in section 7. Their union covers 701 cities exactly.
- Production signature: `9405c3ddf6e2c398f6985b3d204d10efd2c06b8eb0a8d2505bd9acd70035042e`.
- `runtime_snapshot/` contains the actual **18 frozen production modules**, fetched from resource 4 and checked against resource 4 and 6 runtime manifests. This is the source authority for reproducing this batch.
- `orchestration/runtime_manifest.json` hashes those files. `provenance.json` records the local repository base commit and modified worktree.
- Local repo: `D:\project\服务器\2DGeoDistanceField`, upstream `https://github.com/ZSwinB/2DGeoDistanceField.git`, base commit `9b84195221f2230215b19935614939eab1fbce0e`.
- The GPU production entry is an **independent frozen pipeline**, not the original repository `run_all.py`. Local repository changes were not pushed by this work. Repository HEAD alone cannot reproduce the modified production run.

Supporting completion evidence: `evidence/completion.json`, `evidence/final_partition_verification.json`, and three `*inventory.json` files. Old round-3 reports saying “701 production has not started” describe an earlier historical stage; section 0 supersedes that status.

## 1. Input identity, coordinates, and field semantics

### 1.1 Selection and provenance

- Input subset comes from existing cache `/root/autodl-tmp/network1_recovery/code/cache/index.npy` on resource 4. Shape is `(11216,2)`; each row identifies `(scene,tx)`.
- This is the **20% TX subset**, with 16 actual TX IDs per city, not the full original 80 TX per city. Use the index or `sources.json`; do not invent TX IDs 0–15.
- Geometry uses official `polygon/buildings_complete/{scene}.json`; TX uses official `antenna/{scene}.json`. PNG is not used to fit production walls.
- Exactly 1402 official JSONs were retrieved, with each member checked against the official ZIP CRC32, then SHA256-recorded.
- Portable archive: `inputs/official_json_and_index.tar.gz`; SHA256 `ec4a03a5dc8f5727544702058f934b5ef40e4a191d8bc32bb0fe4554de241d8f`.
- Archive includes `raw/`, `index.npy`, `raw_manifest.json`; it does not include full high-resolution fields.
- Production does not require official gain images. Official IRT2 gain is a different quantity and simulator and is not a numerical distance-map ground truth.

### 1.2 Coordinate contract

```text
official polygon / antenna point: (x, y)
production image/world coordinates: (x, 255-y), in meters
highres RX pixel (row=i,col=j): (j/10, i/10), in meters
array shape: (2560,2560); sampled coordinates 0.0 ... 255.9
pixel spacing h=0.1 m; sample_scale=10 pixels/m
walls: float64 (Nw,4) = [ax,ay,bx,by]
corners: np.unique(walls.reshape(-1,2),axis=0), float64
```

High resolution increases receiver sampling density; it does **not** multiply physical wall/TX coordinates by 10. The JSON Y flip remains `255-y`, not `2559-y`.

Polygon cleaning only removes consecutive duplicate vertices and a repeated closing vertex. No 0.75-pixel contour simplification is applied. Remaining zero-length walls are rejected.

The polygon occupancy rule counts boundary points as occupied. Coarse polygon raster `(256,256)` is still used for original WWV endpoint offset-direction selection; do not describe the entire numerical model as having no raster-dependent decisions.

### 1.3 Scientific model

This computes a minimum modeled path score, **not gain/power**, and not an unpenalized shortest obstacle-avoiding geodesic.

```text
K = 1
DIFF_BIAS = 300.0
FALLBACK_BIAS = 600.0
MAX_DIST = 1000.0
building output value = 1000.0
```

Considered families: LOS, first reflection, second reflection, diffraction, reflection/diffraction hybrids. The fallback is Euclidean TX–RX distance + 600. Geometry, mirrors, candidate lengths and comparisons use FP64. Final output arrays use FP32. CUDA uses `fastmath=False` for visibility/path functions. GPU top-K scratch is statically length **1**; K>1 is not a supported configuration merely by changing a config variable.

WWV retains original endpoint direction order `(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)`, rounded endpoint lookup in coarse occupancy, endpoint offset `1e-4 m`, angular interval rules and tie fallback. Native spatial grid cells are 8 m; original formula creates a 33×33 grid for a 256 domain.

## 2. Correctness patches deployed before/with optimization

| Fix | Production implementation / invariant |
|---|---|
| FP32→FP16 export corruption | No FP16 field export. FP32 bit patterns preserved through `.dmap` compression, full decoded field compared byte-for-byte at write. |
| Mirror/geometry rounding | FP64 wall/corner coordinates, source mirrors, second mirrors, corner mirrors and path comparisons. Output precision is a separate FP32 decision. |
| Through-vertex false visibility | `cpu_visibility.py` and translated CUDA `_vertex_crosses_ray`: a ray passing through a vertex blocks when incident edges lie on both sides; tangency and valid path endpoints handled separately. |
| Invalid reflection/hybrid paths | First reflection checks both physical legs; second reflection checks all three legs; hybrid prefix/receiver legs checked. Mirror intersection alone is not sufficient. |
| RX corner candidate omission | Complete receiver PVP across all corners, using exact-compatible angular cache with critical-case direct segment fallback; not the old wall-endpoint-only RX candidate filter. |
| PNG/polygon inconsistency | Walls and receiver occupancy derive from official polygons, same world coordinate convention. |
| TX lookup tied to receiver mask | TX visibility queried at actual continuous source coordinates, independently of whether the RX occupancy pixel is marked building. Keep the **frozen TX candidate rule** in `source_visibility.py`; complete RX PVP does not imply arbitrary replacement of TX rules. |
| Zero-length walls | Consecutive duplicate/closing vertex cleanup plus explicit rejection of remaining zero-length edges. |
| Near-angle comparator behavior | Stable angle sort plus replay in near-angle connected groups, preserving original comparator behavior near `1e-12`; do not replace with arbitrary quantization or a new tolerance. |

These fixes do not imply all model artifacts are solved. CPU/GPU equivalence is equivalence of the same model, not proof of independent physical correctness. Some isolated low regions were previously explained as narrow valid visibility wedges rather than floating-point errors.

## 3. Production execution graph and module map

```text
index + official polygon/TX JSON
  -> CPU load/clean walls + unique corners + native spatial grid
  -> CPU coarse occupancy + WWV endpoint movements + numeric WWV
  -> CPU native-threaded fine occupancy + packed PVW + corner-angle tables
  -> CPU all-16-TX source visibility, candidate ordering, prefixes and mirrors
  -> GPU city upload + packed complete PVP construction (stays on device)
  -> for each TX: refill reusable output buffer to 1000
                 upload TX state, launch per-RX CUDA solver, read FP32 field
  -> background bounded writer: native CPU sample comparison, compress,
                               decode full field, byte compare, field SHA256
  -> atomic completed/{scene}.json only after all 16 maps pass
```

CPU prepares the next city while GPU consumes the current city. Only one next-city build is prefetched. There is one IO writer thread and at most three pending map writes. This is not an all-city in-memory cache.

| Module in `runtime_snapshot/` | Role |
|---|---|
| `production_worker.py` | Official-input adapter, fresh WWV, CPU native controls, audited compression output, completion/status records. |
| `round3_batch_engine.py` | Shared engine: source setup, CPU city build, city upload/PVP, CUDA solve, bounded pools. Production subclasses it. Its original benchmark `main()` is not the production scheduler. |
| `r3_numeric.py` | Compiled CPU polygon raster, unsafe-wall detection, tiled packed PVW queries. Production raster is CPU, not the historical GPU raster prototype. |
| `r3_pvw.py`, `r3_sort.py` | Incremental wall angular visibility and exact near-angle sorting repair. Unsafe geometries retain fallback behavior. |
| `corner_cache.py` | FP64 corner critical-angle tables, nearest blockers, exact-compatible visibility queries and degenerate fallback. |
| `r3_gpu.py` | Packed complete receiver PVP kernel and output fill. |
| `gpu_solver_packed_generated.py` | CUDA visibility/path helpers, packed-bit lookup, K=1 per-RX path solver. |
| `cpu_solver.py`, `pipeline_fast_cpu.py` | CPU geometry helpers and compiled source-prefix/secondary-mirror preprocessing. |
| `cpu_visibility.py`, `source_visibility.py` | Shared patched exact-segment visibility and continuous source candidate visibility. |
| `wwv_numeric.py`, `native_wwv.py`, `native_grid.py` | Numeric WWV plus preserved reference tie/movement behavior and spatial grid builder. WWV is CPU in production. |
| `native_solver.py` | Original unoptimized sampling-only path solver extracted for independent CPU controls; no prefix/secondary-mirror optimization in this reference. |
| `geometry.py` | Official polygon load/Y flip, boundary-inclusive occupancy, wall extraction. |
| `lossless_distance.py` | Lossless FP32 container writer and full/window reader. |

`orchestration/rebalance_worker.py` is outside the frozen numeric runtime and only changes assignment/resume/storage scheduling. It imports the unchanged `Production` class. Its source hash and plan hash are included in the new output manifests.

## 4. Acceleration mechanisms, scope, and invariants

| Technique | Actual implementation | Important boundary |
|---|---|---|
| Compile slow Python | Numba native loops for WWV, PVW/raster, mirror/prefix setup. | Preserve wall ordering/tie behavior; numeric WWV reverts to Python reference on ties. |
| Parallel independent RX | CUDA thread per receiver; 128 threads/block. | K=1 only; double precision geometry, FP32 final store. |
| Parallel complete PVP | CUDA thread per packed PVP byte; 128 threads/block. | All corners considered, cache exact-compatible fallback retained. |
| Share city state | Reuse walls/grid/PVW/PVP/corner tables for the selected 16 TX. | Per-TX visibility/prefix/mirror state still computed at actual TX coordinates. |
| Prefix/geometry hoisting | Cache source-to-reflection-to-corner legs and lengths; double source mirrors and corner mirrors. | Cached legs checked for true visibility. No interpolation between receiver solutions. |
| Spatial restriction | 8 m grid for relevant wall intersection lists. | Do not just assume a nearby wall set or skip vertex crossing checks. |
| Angular cache/incremental sweep | Reuse ordered blocker-angle intervals and visible-wall sweep state. | Critical angles, degeneracies, crossing/overlapping geometry preserve fallback. |
| Lower-bound pruning | Reject path candidates that cannot beat current best. | Preserve original path-score lower bounds and biases. |
| Bit-packed visibility | PVW/PVP uint8, MSB-first: byte `id//8`, bit `7-id%8`. | This is Boolean packing, not lossy distance quantization. |
| GPU-resident PVP | Build/use PVP on device; no D2H full PVP, CPU scatter/CSR, disk write, re-upload cycle. | CPU native controls may use small dense Boolean/CSR buffers; that is not the GPU production hot path. |
| Runtime city parameters | Explicit array arguments instead of closures capturing each city's arrays. | Five-city benchmark yielded one specialization per main kernel family despite varying wall counts. |
| Persistent process/JIT cache | Warm up once per worker, reuse compiled kernels across city/TX calls. | Initial compilation is not free; do not charge it once per field. |
| Reuse output allocation | One GPU output buffer; refill all cells to 1000 each TX, 256 threads/block. | Refill is essential to prevent stale values from the previous map. |
| CPU/GPU pipeline | CPU prefetches one city while GPU solves current; one writer overlaps validated IO. | One city worker per GPU, not many independent GPU jobs competing for memory. |
| Native CPU threading | 12 workers on 4, 16 on 6, compiled loops release GIL. | `NUMBA_NUM_THREADS=1`, BLAS/OMP=1 avoid nested thread oversubscription; external native worker pool still runs 12/16 tasks. |
| Lossless compressed output | 256² chunks, reversible uint32 bit predictor, Blosc bitshuffle/Zstd3. | Primarily storage/IO reduction; do not present it as a measured path-kernel speedup. |

Do not add isolated-stage speedup factors together or multiply them to predict end-to-end speed. Some stages are overlapped, some were removed, and dataset complexity changes.

## 5. Measured performance and proper time scopes

### 5.1 Same-batch controlled benchmark: 5 cities, 80 maps

Cities `[1,5,6,21,338]`, each actual 16 TX, fresh high-resolution raster/PVW/PVP and live source state; existing WWV/grid inputs reused in **this benchmark**, unlike full production which recomputes WWV.

| Resource / mode | 5-city batch | Mean per city (16 maps) | Mean per map |
|---|---:|---:|---:|
| 4 persistent serial, NPY | 141.192 s | 28.238 s | 1.765 s |
| 4 CPU/GPU pipeline, NPY | 100.973 s | 20.195 s | 1.262 s |
| 6 pipeline + `.dmap` + full readback | 139.222 s | 27.844 s | 1.740 s |

Resource-4 same-batch pipeline reduces wall time by **28.485%**, speedup **1.3983×**. One-time warmups are separate: 15.492 s on 4, 20.967 s on 6. Resource 6 has a different output format and GPU; do not use 4-vs-6 ratios as an optimization speedup.

Historical single-city 48.812 s included its own initialization scope; do not compare it directly to the warm five-city mean of 20.195 s as if same-city/same-scope.

### 5.2 Actual completed production

| Partition | Cities/maps | Processing wall time | Scope |
|---|---|---:|---|
| 4 initial | 411 / 6576 | 4910.042 s, about 81.83 min | Production loop incl fresh inputs/WWV, source state, CPU controls, compressed writes/full readbacks; excludes prior worker warmup. |
| 4 assist | 160 / 2560 | 1886.796 s, about 31.45 min | Additional assignment after restart/warmup. |
| 6 before rebalance | 20 / 320 | 435.051 s at last complete status | Stopped and resumed from atomic city checkpoints. |
| 6 after rebalance | **110 new** / 1760 | 1850.180 s, about 30.84 min | Final status counts 130 including earlier 20; time covers the post-rebalance new work. |

4 initial amortized throughput is **0.7467 s/map**, 4 assist about **0.737 s/map**. These are batch-throughput values, not cold single-map latency. The partitions differ in complexity. User idle periods between starts/reassignment mean job wall times are not the total user-observed calendar duration.

### 5.3 Cached solve, small resolution, and analysis timings

- 256×256 cached CUDA solve+readback: 23 maps / 5 cities, min 0.00828 s, median 0.03384 s, max 0.05981 s. Source upload/setup, geometry, writing and first JIT are outside that number.
- **No equivalent complete new-production 256×256 benchmark was run.** Earlier chat's 0.1–0.3 s/map complete-pipeline number was an estimate; do not promote it to a measurement.
- 2560×2560 cached solve/upload/readback in five-city round 3: per-city TX averages 0.282–0.801 s/map. Fresh geometry/IO time is excluded from that cached stage.
- Four selected 2 m windows refined to 161² at h=0.0125 m: about 2.7–7.4 s/window on one low-priority CPU thread after initial JIT; not full-scene simulation latency.
- Four full 2560² maps: central difference 0.240–0.279 s; experimental thinning 0.032–0.047 s; reading/geometry/analysis around 1 s before plotting. These are postprocessing, not ray/path solving.
- Local refinement experiment total about 49.36 s including its first JIT, multiple resolutions, audits and plots.

Production memory: observed peak RSS ~3176 MiB (4 initial), ~2003 MiB (4 assist), ~2894 MiB (6). Full geometric inventory has max 726 walls/726 corners, each full fine packed PVW/PVP at most ~0.5554 GiB for this dataset. Do not estimate RAM from host-wide `free`; use container cgroup limits.

## 6. Environments and dependencies

Exact import locations/version probes: `environment_server4.json`, `environment_server6.json` (current collection time in provenance). Both used Python 3.12.3, NumPy 2.3.2, Numba 0.68.0, llvmlite 0.50.0, numcodecs 0.17.0. Torch 2.8.0+cu128 is present but the solver uses **Numba CUDA**, not a Torch tensor implementation.

Resource 4 analysis environment also has scipy 1.18.1 and cv2 5.0.0; those imports are absent from the recorded resource-6 production path. Do not assume analysis/plot scripts work unmodified on resource 6. Production itself does not need scipy/cv2.

```bash
# 4
env PYTHONPATH=/root/precision_storage_env:/root/autodl-tmp/city_precision_env \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  /root/miniconda3/bin/python SCRIPT.py

# 6
env PYTHONPATH=/root/precision_storage_env:/root/precision_gpu_env_v2 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  /root/miniconda3/bin/python SCRIPT.py
```

Reported hardware (record as reported, not inferred product specifications):

- 4: RTX 4090, 49140 MiB reported GPU memory; cgroup 12 CPUs, RAM 77,309,411,328 bytes (72 GiB).
- 6: RTX 4080 SUPER, 32760 MiB reported GPU memory; cgroup 16 CPUs, RAM 66,571,993,088 bytes (~62 GiB).
- Fresh `environment_server*.json` includes drivers and observed free memory; free memory is only a snapshot.
- Connection instructions/credentials remain in local `D:\project\服务器\服务器资源`; this bundle does not contain those credentials. Local SSH helper is `../remote_client.py`, dependencies `../ssh_deps/`.

## 7. Final artifact roots, manifest protocol, and reading data

Common per-server project root: `/root/precision_gpu_work/production_20261016` (`P` below).

| Server | Logical output root | Completed cities |
|---|---|---:|
| 4 | `P/output` | 411 |
| 4 | `/root/autodl-tmp/distance20_2560_assist_20261016` | 160 |
| 6 | `/root/autodl-tmp/distance20_2560_fp32_lossless_20261016` | 130 |

Within each root:

```text
manifest.json                 # assignment, signature, maps, scheduler metadata
status.json                   # latest progress; inspect timestamp/scope
distance/sceneS/distance_S_T.dmap
completed/S.json              # emitted only after all 16 maps fully pass
failed/S.json                 # failures, if any; final production has none
inputs/sceneS/                # walls, WWV, coarse geo, sources, CPU sample yx/values
inventory_audit.json          # final inventory audit for that root
```

The 4-assist root has **80 `distance/scene*` symlinks** pointing into `P/assist_spill/scene*` on its system filesystem. Other assist cities use the data filesystem. Treat the logical root as the canonical reader path. Copying/archiving the assist directory must intentionally dereference those symlinks or separately copy their targets; a default tar of just symlinks is not a self-contained backup.

Raw inputs are `P/raw/`, `P/index.npy`, `P/raw_manifest.json`; frozen code is `P/runtime/` plus `P/runtime_manifest.json`; assignment is `P/rebalance_plan.json`.

Original logs: `P/production.log`; continuation logs: `P/rebalance.log`; PIDs `P/production.pid`, `P/rebalance.pid`. Completed PID files are historical, not evidence of a live job. All production was finished at this handoff.

`.dmap` is a custom indexed **lossless** container, not `np.load`-compatible:

```python
from pathlib import Path
import sys
P = Path('/root/precision_gpu_work/production_20261016')
sys.path.insert(0, str(P / 'runtime'))
from lossless_distance import DistanceReader

reader = DistanceReader('/ABSOLUTE/OUTPUT/distance/scene1/distance_1_34.dmap')
field = reader.read()                         # float32 (2560,2560)
patch = reader.read_window(100,200,300,400)    # [100:200,300:400], half-open
tx_xy = reader.header['metadata']['source_xy']
```

Layout: magic `DMAPLS01` (8 bytes), little-endian uint64 JSON-header length, 32-byte SHA256(header), UTF8 JSON header, concatenated compressed blocks. Fixed prefix 48 bytes. Payload offsets are relative to payload start. Chunk headers store coordinates, shapes, offsets, compressed sizes and SHA256 of decoded FP32 bytes.

Compression: reinterpret FP32 as uint32, reversible two-axis modulo-2^32 differences, Blosc BITSHUFFLE + Zstd level3, 256² chunks, codec single-thread. Decoder restores bit patterns and verifies chunk hashes by default. No float-delta rounding, scaling, FP16, decimal quantization or lossy codec is involved.

Raw arrays total 273.828125 GiB, stored map containers 27.863839183 GiB (~9.83×). This excludes small sidecar JSON/input snapshots from stored-map-byte accounting. Four early benchmark cities extrapolated ~25.66 GiB; actual final total supersedes the extrapolation.

Local ready-to-review examples: `../samples_2560/` has 1/34, 5/27, 6/12, 338/7, each original `.npy`, building mask, native-size PNGs, and annotated overview. Those `.npy` are unchanged FP32 samples, not a full local copy of all production.

## 8. Completion, integrity, and resume semantics

Production signature is computed exactly as:

```python
sha256(json.dumps({
    'runtime': runtime_manifest,
    'raw_manifest': raw_manifest,
    'shape': [2560,2560],
    'K': 1,
    'output': 'FP32 lossless DMAPLS01'
}, sort_keys=True).encode()).hexdigest()
```

Changing code or the raw manifest creates a different signature. Do not bypass checks or silently reuse old completed markers under a new model. Rebalancing changed scheduling/storage only, preserving this numeric/data signature and recording the separate scheduler hash.

Production has one exclusive `worker.lock` per output root. Resume scans assigned cities and skips a completed city only when signature matches, all 16 files exist and file sizes match; **resume skip does not re-decode/re-hash every full existing field**. Full decode/byte equality happened at write. Final inventory rechecked existence/size/checksummed headers. Do not describe this as a second full final payload-hash sweep.

Map writes use temporary file + atomic replace. City completion JSON uses atomic replace. An interrupted incomplete city is recomputed; partial outputs without a completion marker are not certified complete. IO was timed with OS buffering and without per-map fsync; sudden power-loss durability is not guaranteed solely by these checks.

Rebalance history: original 4 completed 411 cities; original 6 stopped with 20 cities complete. Remaining 270 split into 160 on 4 and 110 on 6. The 6 continuation manifest now describes its final 130 cities. **Do not rerun the original `production_launch6.sh` or original `production_worker.py --server 6` assignment for this dataset:** it would reintroduce the old 290-city allocation overlapping resource-4 assist work.

For an intentional resume on unchanged current paths/version, the correct entries are:

```bash
# Original 4 partition only (411 existing cities): correct 4 environment required
/root/miniconda3/bin/python P/runtime/production_worker.py \
  --server 4 --workers 12 --output P/output

# Extra 4 assignment only (160 existing cities): correct 4 environment required
/root/miniconda3/bin/python P/rebalance_worker.py --server 4

# Final 6 assignment only (130 total, 110 were new): correct 6 environment required
/root/miniconda3/bin/python P/rebalance_worker.py --server 6
```

`P` in these examples is a placeholder to replace with its absolute value, not a defined shell variable. All jobs are complete, so no resume is currently necessary. Prefer read-only status/audit first.

From Windows workspace:

```powershell
python precision_audit/production_collect.py
python precision_audit/remote_client.py exec --server 4 --command-file precision_audit/rebalance_watch.sh
python precision_audit/remote_client.py exec --server 6 --command-file precision_audit/rebalance_watch.sh
```

The collectors understand all three roots. Older two-root scripts or historical counts may not.

## 9. Verification performed and limits

1. Patched CPU visibility/WWV/event sorting tested against preserved reference behavior (including degenerate geometries and near-angle cases in prior rounds).
2. Five-city benchmark: all 80 high-resolution fields matched at all coincident native-grid pixels against independently generated original CPU 256² controls.
3. All 80 complete high-resolution fields matched across resource-4 serial/pipeline and resource-6 pipeline raw FP32 field hashes. Scene1 all 16 also matched pre-existing complete high-resolution references; its PVW/PVP full bit arrays matched references.
4. Production: **128 deterministic free-space CPU reference points per map**, total **1,435,648** comparisons. Reference uses original sampling-only path kernel and complete RX corner segment checks, not prefix-cache optimizations. Five established cities also compared all 256² native-grid points.
5. Every production `.dmap` fully decoded and raw FP32 bytes compared to pre-compression field; buildings checked to equal 1000, finite/nonnegative/max checks performed.
6. Final inventory: all 11216 files exist, correct sizes, supported and checksum-valid headers, correct completion accounting. This is not a full post-hoc re-simulation or second complete payload-hash pass.
7. Precision study: four cities, eight 11² CPU FP64 patches, **968 points**; cast-to-FP32 output and central-difference candidate masks matched production everywhere sampled. Residual thick patches remained in FP64; max field rounding error ~3.04e-5 m.
8. Local refinement study: four 2 m windows × four spacings; all coincident samples matched production after FP32 conversion; plain-vs-instrumented CPU controls passed; canonical branch reconstruction error ~2e-13 m.

Coverage limits: not all high-resolution maps were independently solved in full on CPU, K>1 unvalidated, reference CPU shares the same model, production source flags need downstream training policy, cross-city diversity outside the present 701 not validated. Memory/disk and speed claims are for measured machines/dataset.

## 10. Switching-line extraction: separate artifact and open work

Distance fields are the delivered production product. Switch masks are derived experiments, not written back into distance maps.

```text
Promote stored FP32 to FP64 before subtraction.
gx[i,j] = (f[i,j+1]-f[i,j-1]) / (2h)
gy[i,j] = (f[i+1,j]-f[i-1,j]) / (2h)
candidate = abs(hypot(gx,gy)-1) > 0.05
h = 0.1 m for 2560; h=1 m for 256
Only center and all four axial neighbors in free space are valid.
Exclude image boundaries and TX Chebyshev radius 1 pixel at that resolution.
```

Do not use denominator 2 instead of 0.2 on high-resolution maps. A radius of 1 pixel is 0.1 m in the fine map, not the same physical exclusion as 1 coarse pixel. Do not globally enlarge the TX mask just to hide artifacts.

Narrow path domains can give thick/overlapping finite-difference responses at h=0.1 and separate under h=0.05/0.025/0.0125. In one selected 5/27 window solid3×3 centers went 42→0 at h=0.05; other junctions retained a few at finer spacings. Higher resolution does not guarantee a single-pixel contour or no gaps.

Thinning experiments:

- Early local refinement plots use a Zhang–Suen trial; only local foreground component count checked there.
- Full-size experiments use a **different sequential simple-point deletion** candidate: local foreground8/background4 topology tests, endpoints/singletons retained, directional sweeps. Four full maps preserved global component and hole counts and removed all solid3×3 centers.
- This is not proof of correct branch geometry: it preserves existing loops and gaps and cannot reliably unmerge unresolved nearby boundaries. No final training-label pipeline has been certified or deployed.
- Winner-branch/vector-residual plots use simulator metadata as an oracle, not an independent distance-only detector. Do not accidentally use those metadata as if available to a field-only downstream model.
- Near-threshold norm tests can miss directional changes even if norms remain near1; keep gap diagnosis separate from pure precision problems.

Display artifact: original overview PNGs downsampled 2560 fine pixels into much smaller axes with nearest sampling. Single-pixel red lines get omitted and look like dashed fragments. Reproduction: scene338 original thin mask has165 foreground8 components; a 512 nearest-sampled display has822. Every one of the four original masks' components survived thinning as exactly one component (per-component correspondence checked). Corrected coverage-preserving overview is display-only and can merge close lines at screen scale. Judge actual connectivity at native pixels/zoom, not by the rescaled overview.

Local reports/figures outside this bundle:

```text
../production_results/line_audit/README.md
../production_results/subpixel_branch_experiment/README.md
../samples_2560/display_audit/scene338_display_comparison.png
```

Core numerical evidence JSONs are copied into this bundle's `evidence/`.

## 11. Reproduction/extension traps and recommended next actions

- Current production code contains fixed absolute `BASE/P/ROOT`, 701/11216 assertions, warmup city1, fixed size2560/sample_scale10, and fixed original partition rules. It is a validated batch entry, not a generic CLI accepting arbitrary `--resolution` or `--K`.
- To run 256² end-to-end, intentionally parameterize raster dimensions, all RX coordinate/sample scales, source index mapping, PVP kernels, output shape/metadata, sampling controls and version signature. Changing only one `2560` literal is insufficient. A complete same-entry 256² timing remains useful future work.
- For a different dataset/resolution/version, use a new output namespace and new signature; preserve this completed batch. Update geometry/input adapters and then repeat frozen CPU and full-field regression comparisons before production.
- If moving the runtime, update hardcoded path constants and import ordering intentionally; changing source paths/constants may change file hashes/signature. Do not claim unchanged provenance after an ad hoc relocation edit.
- Do not remove `runtime_snapshot/native_*` as “slow unused code”: native production controls depend on it.
- Do not disable exact critical-angle fallback, reflection leg checks, near-angle repair, dtype checks, output refill or lossless validation to produce a misleading speed benchmark.
- Experimental jobs during production should be CPU-only low-priority, bounded small windows or read-only analysis, single native thread, `nice -n19` / `ionice -c3`; GPU experiments require actual idle resources. Ordinary native worker count remains 12/16 for production.
- A previous cleanup deleted only verified duplicate `round3/benchmark_npy/serial` and `pipeline` field arrays (~3.906 GiB); reports, formal production, and reference inputs remain. Historical scripts assuming those duplicate fields still exist must be adapted rather than treating their absence as production loss.
- Disk guard: 2 GiB free before a city; assist checks the actual symlink target filesystem as well. Final 4 disks have limited spare space; avoid expanding the whole dataset into uncompressed NPY there.
- Next scientific priority: correct branch positions/connectivity and weak-boundary/gap recovery, using local refined/native winner references to evaluate candidates. Additional blanket FP64 export is not supported as a cure for the sampled residual blocks.

## 12. Bundle use

Entry files:

```text
README.md                     # this report
simulation_config.json        # explicit machine-readable settings/state
performance.json              # measured scopes vs estimates
runtime_snapshot/*.py         # exact 18 production modules
orchestration/                # scheduling, bootstrap, plan, manifests, index, diagnostics
environment_server{4,6}.json   # real import versions/locations and cgroup/GPU reports
inputs/official_json_and_index.tar.gz
evidence/*.json               # small primary result records
provenance.json
verify_bundle.py
manifest.json                 # generated package hashes
```

Run `python verify_bundle.py` for file-integrity / JSON-consistency checks only. It does not import GPU code, run numerical simulation, require credentials, or certify scientific correctness. All production arrays remain on the listed servers; this bundle is a small handoff and source/input snapshot.
