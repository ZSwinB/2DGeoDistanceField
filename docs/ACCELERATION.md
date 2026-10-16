# Acceleration implementation and measurement scopes

Current entry: `python -m accelerated`; setup and CLI are in [GPU_RUNNER.md](GPU_RUNNER.md). Original investigation/agent handoff: [HISTORICAL_HANDOFF.md](HISTORICAL_HANDOFF.md). This repository ships code/documentation only, not the data/evidence archives mentioned in that historical report.

## Actual stage ownership

| Stage | Implementation |
|---|---|
| Official polygon loading, 1m coarse mask, spatial grid | CPU, FP64 geometry |
| WWV | Native CPU loops, external worker threads; original Python tie fallback |
| Fine polygon mask and PVW | Compiled CPU, 16-row tiles, worker pool |
| Corner angular tables and per-TX source prefixes/mirrors | Compiled CPU; reused city/source geometry |
| Complete receiver PVP | GPU directly constructs bit-packed visibility in device memory |
| K=1 path solver | GPU thread per receiver, FP64 geometry/comparisons, FP32 store |
| Compression/verification/writing | One background CPU writer, bounded pending queue |

Neither production WWV nor production polygon raster is a Torch/GPU implementation. Earlier experiments explored GPU rasterization, but the deployed raster is already short enough on compiled CPU and stays out of the producer thread's CUDA context.

## Mechanisms

1. **Numba compilation:** replace repeated Python loops/containers with native numeric loops. WWV preserves original direction/tie behavior.
2. **Reuse city preprocessing:** walls, grid, packed visibility and corner tables serve all selected TX; avoid one full city rebuild per TX.
3. **Reuse source subpaths:** visibility-valid source/reflection/corner prefixes, source mirrors, double mirrors and mirrored corners are computed once per TX instead of per receiver.
4. **Restrict geometry queries:** 8m spatial grid avoids scanning unrelated walls; angle caches and incremental sweep reuse nearest blockers. Critical angles/degeneracies retain exact fallback.
5. **Prune impossible winners:** existing path-length lower bounds reject candidates that cannot improve current best; penalties/order/tolerances remain unchanged.
6. **GPU parallelism:** many independent receiver solves; full PVP built in parallel before solving.
7. **Bit packing:** 8 Boolean visibility flags per byte, MSB first. This is lossless Boolean packing, not field-value quantization.
8. **Keep PVP on GPU:** eliminate a D2H/CPU scatter or CSR/disk/re-upload round trip from the hot path.
9. **Persistent, parameterized kernels:** city arrays are runtime arguments; no city-capturing closures and repeated shape-specific regeneration. Five-city audit had one specialization per main kernel family.
10. **Allocation reuse:** reuse GPU output buffer and refill it to1000 every TX; avoid stale values while saving allocation overhead.
11. **Pipeline stages:** prepare one next city on CPU while GPU consumes current; overlap bounded background output/validation. Worker counts use actual cgroup quotas, BLAS/OMP/Numba internal thread counts stay1.
12. **Lossless storage:** reversible uint32 bit-pattern predictor plus bitshuffle/Zstd3 reduces IO/storage. This is a storage optimization, not a measured standalone path-kernel speedup.

## Measured results, not interchangeable timing scopes

Historical five actual cities ×16 selected TX =80 maps, each2560²:

| Resource/mode | Batch wall time | Mean per city | Mean per map |
|---|---:|---:|---:|
| 4, persistent serial, NPY | 141.192s | 28.238s | 1.765s |
| 4, CPU/GPU pipeline, NPY | 100.973s | 20.195s | 1.262s |
| 6, pipeline plus lossless output/full decode checks | 139.222s | 27.844s | 1.740s |

The same-resource/same-output 4 comparison gives **28.5% wall-time reduction /1.398× speedup** from pipelining. One-time warmups were separate:15.492s and20.967s. That benchmark reused existing WWV/grid inputs; actual full production computed WWV too.

Actual completed production: 4's first411 cities/6576 maps took4910.042s in the processing loop, about0.747s/map amortized; 4's extra160 cities/2560 maps took1886.796s, about0.737s/map. Different city complexity explains why these means differ from five-city sampling. Cold single-map latency must include city setup and first JIT; it is not0.747s.

Cached GPU solve/readback for256²:23 tested maps, median0.03384s, range0.00828–0.05981s. This excludes geometry, source upload, writing and first JIT. The earlier chat estimate0.1–0.3s/map for an entire coarse pipeline was **not** a complete-pipeline measurement. Portable coarse correctness was subsequently tested (see [VALIDATION.md](VALIDATION.md)); do not substitute its cold four-single-TX test time for a16-TX city throughput benchmark.

Historical2560² cached source-solve/upload/readback averages were0.282–0.801s/map across five cities; city preprocessing/output are outside that stage.

Experimental complete2560² postprocessing: central difference0.240–0.279s; mask thinning0.032–0.047s. These are not field simulation times. Four selected2m windows refined to161² receivers at0.0125m spacing took about2.7–7.4s/window after initial compilation on one low-priority CPU thread.

## Memory, storage and resource limits

Observed production peak RSS: about3176MiB (4 initial),2003MiB (4 assist),2894MiB (6). Dataset max726 walls/corners; each full packed PVW/PVP at that maximum is about0.555GiB. One-city prefetch and at most3 pending maps bound memory; all701 cities are not cached simultaneously.

Historical resource4 quota:12 CPUs,72GiB RAM; resource6:16 CPUs, about62GiB RAM. Reported GPU names/memory were RTX4090/49140MiB and RTX4080SUPER/32760MiB. These are machine reports, not generic GPU product specifications.

11216 raw2560² FP32 arrays total273.828GiB; actual lossless containers totaled27.864GiB (~9.83×), excluding small sidecars. No field FP16 conversion or lossy predictor is used. Disk-floor check is2GiB before consuming a city; timing used OS-buffered writes without per-map fsync.

## Limits

CPU/GPU agreement checks preserve this geometric path model; they do not independently establish physical correctness. Some residual switching-line blocks come from unresolved close branch interfaces and central-difference response overlap. Thinning does not fix pre-existing gaps or certify true branch topology. Model changes, candidate/tolerance changes, K>1 or a new precision configuration require a new signature and regression testing.
