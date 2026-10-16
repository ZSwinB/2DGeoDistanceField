# External input contract (no dataset included)

The accelerated runner takes **official vector geometry**, selected transmitter IDs, and a new output directory. It does not need building PNGs, gain images, trained weights, or precomputed PVW/PVP/WWV files. It builds the required intermediates itself.

## Required directory and selection

```text
/external/input-root/
  polygon/buildings_complete/<scene_id>.json
  antenna/<scene_id>.json
/external/selection.csv             # or selection.npy
/external/output-root/              # empty/new or compatible resumed run
```

Provide polygon and antenna files for **every scene in the supplied index**. `--scenes` and sharding select work after input validation. A filtered index can be supplied when only a small set of cities has been downloaded.

Selection CSV has exactly the header `scene,tx`, then unique nonnegative integer scene/TX pairs. Selection NPY is an integer array of shape `(N,2)` with columns `[scene,tx]`, loaded without pickle. Actual TX IDs are indices into the antenna JSON list; they are not implicitly numbered 0–15. Number of cities and selected TX per city are configurable and need not equal the historical 701×16.

Example **format description only**, not a required sample dataset:

```csv
scene,tx
<integer scene ID>,<integer TX ID>
```

## JSON schemas

Polygon JSON is a list of polygon rings. Each ring is an ordered list of `[x,y]` vertices. Rings must be finite, contain at least three vertices after consecutive duplicate/closing-point cleanup, and have no remaining zero-length edges. At least one polygon per city is required in this validated entry. There is no courtyard/hole schema extension in this adapter.

Antenna JSON is a list of `[x,y]` source positions indexed by TX ID. The validated published format has integer coordinates in `[0,255]`. Fractional coordinates are rejected instead of silently rounded; supporting them requires separate source-zero-index semantics validation.

Coordinates in both JSON types use the same original convention. The adapter transforms `(x,y)` into image/world `(x,255-y)` in meters. It keeps geometry in those meter units at both resolutions:

| `--scale` | Output shape | Receiver spacing | Receiver `(row,col)` coordinates |
|---:|---|---|---|
| 1 | 256×256 | 1 m | `(col,row)` |
| 10 | 2560×2560 | 0.1 m | `(col/10,row/10)` |

Do not resize geometry or multiply polygon/TX coordinates by 10 for the fine run. The Y transform stays `255-y`. Sources inside/on polygons are kept at their true coordinates and flagged in metadata; users must decide whether to use those samples for training.

Obtain the public inputs from the [RadioMapSeer dataset release](https://radiomapseer.github.io/index.html), or supply compatible data you are authorized to use. No downloader, credentials, input scenes, selected production index, or gain/output images are included here.

## Validate without simulation

```bash
python -m accelerated --input-root /external/input-root \
  --index /external/selection.csv --validate-only
```

This checks formats and files and prints the assignment. It does not create a CUDA context or distance fields. NumPy is required. A real simulation additionally needs the documented Linux/CUDA/Numba environment.
