# Validation record for the repository packaging

No actual input scenes, input arrays, output maps, preview images, or heavy archives are part of this commit. Validation used external data on the authorized Linux/CUDA resources. All generated test outputs remain outside the repository.

## Numerical source provenance

Sixteen packaged numerical modules were compared with the completed production run's frozen sources. Their non-import Python ASTs are identical; only intra-package imports were made relative. Hashes are recorded in [numeric_provenance.json](numeric_provenance.json). The input/scheduling/engine wrappers were parameterized separately and tested end-to-end.

## Portable entry regression

Resource4,12 native CPU workers:

- Unit suite:5 tests passed (bitwise lossless roundtrip including IEEE edge patterns, cross-chunk window, FP64 rejection, generated input schema/geometry, duplicate and fractional-source rejection).
- CLI data-only input validation passed without producing distance fields.
- Scale1:4 complete256² maps across4 actual cities matched all pixels of previously generated independent native CPU coarse controls.
- Scale10:4 complete2560² maps across those cities matched the historical production FP32 field SHA256, plus all coincident native256² grid points.
- Repeating the scale10 command reported all4 assigned cities already complete and performed no field recomputation.

Resource6,16 native CPU workers:

- Same5-test unit suite passed.
- Scale10:2 complete2560² maps from differently sized actual city geometries matched the original production field SHA256 and complete coarse controls.

These are full-field packaging regressions, additional to the original production checks. They do not mean all11216 high-resolution fields were independently re-simulated in full on CPU.

The final Git-index source (normalized LF via `.gitattributes`, not a working-copy archive) was additionally uploaded to resource4: all5 unit tests and Python compilation passed, one complete2560² map matched the original field hash and full coarse control, and re-running skipped the completed city. Numeric provenance hashes were checked against both working-tree and staged blobs so Windows line-ending conversion cannot silently change the published source identity.

## Original production verification scope

- 701 cities ×16 selected TX =11216 completed2560² maps,0 failed cities.
- Default128 native CPU control points/map:1,435,648 point comparisons.
- Each compressed file fully decoded and compared byte-for-byte to the FP32 field before a city completion marker was written.
- Shape/dtype/finite/range/building-value assertions; final file inventory, size, supported header and header checksum checks.
- Earlier five-city80-map experiment compared complete field hashes across two machines and serial/pipeline scheduling. Scene1's16 full fields also matched earlier high-resolution references; its PVW/PVP full bit arrays matched references.
- Four-city/eight-patch FP64 CPU output audit:968 points matched production after FP32 casting; raw candidate masks were identical at the fixed0.05 threshold. Residual blocks remained under FP64.

Independent native CPU controls use the same scientific path model, so they are strong implementation checks, not an independent physical ray-tracer ground truth. Historical source-inside/on-polygon groups were retained and flagged, not silently repaired.

## Re-run checks with your own external data

```bash
python -m unittest discover -s tests -v
python -m accelerated --input-root /external/input --index /external/index.csv --validate-only
python -m accelerated --input-root /external/input --index /external/index.csv \
  --output /external/new-run --scale 10 --workers 12
python tools/verify_external_regression.py --help
```

The regression tool expects external historical completion reports for fine hashes and external `sceneS/sources.json` / `native_controls.npy` for complete coarse references. Nothing downloads or embeds those references automatically. The new portable runner intentionally uses a new version/signature namespace and rejects incompatible old output roots.
