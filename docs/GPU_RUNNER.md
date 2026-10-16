# Portable accelerated K=1 runner

This entry packages the numerical kernels used by the completed 2560² production run. The old `run_all.py` remains the historical raster/CPU entry; its `Config` values do **not** configure `python -m accelerated`.

The sixteen numerical modules in `accelerated/` retain the frozen production mathematics. Only intra-package imports were qualified; `numeric_provenance.json` records the original hashes, new hashes and non-import AST equivalence. `engine.py`, `inputs.py` and `runner.py` replace the machine-specific scheduling/input harness with explicit parameters. No absolute server directory, 701-city assertion, fixed 16-TX requirement or hardwired warmup city is needed.

## Environment

Validated machines used Linux, Python 3.12.3, NumPy 2.3.2, Numba 0.68.0, llvmlite 0.50.0, numcodecs 0.17.0, and CUDA-capable NVIDIA drivers/toolkit. The solver uses **Numba CUDA**, not Torch tensors. Torch happened to be installed on those machines but is not an application dependency.

```bash
python -m pip install -r requirements-gpu.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
python -m accelerated --help
```

CUDA driver/toolkit libraries must also be available to Numba; pip dependencies alone do not supply a functioning GPU environment. The production runner requires Linux/POSIX file locking. Data-only validation is separate.

## Run

Prepare [external inputs](INPUTS.md), use a new output root, and select the resolution:

```bash
# Fine field: 0.1 m receiver spacing, 2560x2560.
python -m accelerated --input-root /external/radiomapseer \
  --index /external/selection.csv --output /external/run-fine \
  --scale 10 --workers 12

# Coarse field: 1 m receiver spacing, 256x256.
python -m accelerated --input-root /external/radiomapseer \
  --index /external/selection.csv --output /external/run-coarse \
  --scale 1 --workers 12
```

`--workers` controls external native CPU tasks. `NUMBA_NUM_THREADS=1` prevents nested internal parallelism and does not reduce that external pool to one worker. Choose the worker count from the container quota, not host-wide logical CPU count. Production used one persistent process/GPU with 12 workers on resource4 and 16 on resource6.

Both resolutions retain FP64 geometry/mirrors/path comparisons, FP32 outputs, `K=1`, diffraction bias300, fallback bias600, building/max value1000. These path scores include penalties; they are not gain/power. K>1 is not exposed: the GPU scratch array is statically length1 and extending it requires new validation.

## Sharding and resume

```bash
# Same input/index and --scenes settings on both workers; separate output roots.
python -m accelerated --input-root /external/radiomapseer \
  --index /external/selection.csv --output /external/shard0 \
  --shard-count 2 --shard-index 0 --scale 10 --workers 12

python -m accelerated --input-root /external/radiomapseer \
  --index /external/selection.csv --output /external/shard1 \
  --shard-count 2 --shard-index 1 --scale 10 --workers 16
```

Shards distribute sorted city IDs; they never split one city's selected TX across workers. `--scenes 1,5` optionally filters assignment. Supply actual IDs from your index.

Repeat the same command to resume. A city is skipped only if its completion signature matches and all expected map files exist with matching sizes. The signature covers code hashes, input hashes/selection and numerical/control settings. Assignment changes require a new output root. Do not point this refactored runner at the old frozen production roots: run signatures and manifest protocols differ even when field values agree. An unrecognized nonempty output root is rejected.

One worker lock per output root prevents concurrent writers. Errors are recorded under `failed/` and stop this run with a nonzero exit status; completed city markers remain usable. On retry, incomplete cities are recomputed. A completion marker is written only after every selected TX has passed verification.

## Output and verification

```text
output/
  run_manifest.json
  status.json
  run_summary.json
  distance/scene<id>/distance_<scene>_<tx>.dmap
  completed/<scene>.json
  failed/<scene>.json                 # only when a run fails
  inputs/scene<id>/                   # generated audit intermediates, not raw input mutation
```

The runner builds official-polygon occupancy, native WWV/spatial grid, tiled packed PVW and all-16-or-selected-TX source state. CPU prepares one next city while the GPU constructs complete packed PVP and solves current receivers. One IO thread, at most three pending map writes; output buffer is refilled to1000 for every TX.

Every map is checked for FP32 dtype/shape, finite/nonnegative/bounded values, buildings=1000, and default128 deterministic independently solved original-CPU control points. Complete RX corner visibility is used by the CPU reference; the reference does not use optimized source-prefix state. `--control-points` changes the sample count and run signature. Each compressed map is decoded in full and compared byte-for-byte before completion. Final checks are not a claim that every high-resolution point was independently re-solved on CPU.

Optional `--reference-root /external/refs` compares `scene<id>/pvw_bits.npy` and `pvp_bits.npy` when present, at the selected resolution. Reference arrays are deliberately not shipped. For old independently generated coarse fields and historical complete high-resolution field hashes, use `tools/verify_external_regression.py --help`.

For all data writes, parent filesystem and actual target directory must each have at least2GiB free before consuming a city. Atomic replace is used for maps/JSON markers; per-map fsync is not performed. Do not confuse atomic visibility with guaranteed power-loss durability.

## Read lossless FP32 maps

```python
from accelerated.lossless_distance import DistanceReader

reader = DistanceReader('/external/run-fine/distance/scene1/distance_1_34.dmap')
field = reader.read()                    # float32 (2560,2560)
patch = reader.read_window(100,200,300,400) # half-open array coordinates
metadata = reader.header['metadata']
```

`DMAPLS01` is a custom indexed container, not NPZ and not directly `np.load` compatible. A 256² chunk predictor operates reversibly on uint32 bit patterns with modulo2^32 arithmetic, then Blosc bitshuffle/Zstd3 compresses them. Header and decoded chunk bytes have SHA256 checksums. The output has no FP16 conversion, float-delta quantization, interpolation or smoothing. Codec uses one thread.

## Numerical fixes and remaining limitations

The packaged kernels preserve: through-vertex occlusion vs tangency handling; actual propagation-leg checks for reflections/hybrids; full receiver corner candidates; continuous source visibility independent of the receiver occupancy lookup; official polygons; zero-length wall rejection; FP64 mirrors; FP32 export; near-angle comparator repair; critical-angle visibility fallbacks and `fastmath=False`.

Walls and receiver masks are vector-derived, but original WWV endpoint movement-direction selection still uses a coarse polygon raster. Source corner candidate rules also remain those of the validated model. This is CPU/GPU agreement of that model, not an independent physical correctness proof.

Switching lines are a **separate downstream product**. For fine maps, central differences divide by `2h=0.2 m`, not2; exclude building-touching stencils/image borders and source Chebyshev radius1pixel. The raw candidate criterion `abs(norm(gradient)-1)>0.05` can produce thick junctions, small loops and gaps. The experimental thinning work is documented but does not alter the stored fields and is not certified as final labels. Nearest-neighbor shrinking of a 2560-pixel overview can omit single-pixel lines and make continuous lines look dashed.
