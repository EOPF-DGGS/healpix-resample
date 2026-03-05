# `regrid_to_healpix.nearest` (nearest-neighbour HEALPix regridding)

`regrid_to_healpix.nearest` provides the **simplest and fastest** mapping from unstructured geographic samples
(**longitude/latitude**) to a **subset of HEALPix pixels** at a target resolution (`nside = 2**level`).

It constructs a sparse operator where each input sample contributes **only to its closest HEALPix cell**
(i.e., `Npt = 1`). The implementation is designed for **large N**, optional **batched data** `(B, N)`, and
**CUDA acceleration** using **PyTorch sparse** matrices.

> Note on geodesy: the package is intended to manage the **HEALPix authalic definition** and the Earth ellipsoid
> with **WGS84** through its geometry helper.

---

## What the class does

Given:
- sample coordinates `(lon, lat)` of shape `(N,)`
- sample values `val` of shape `(N,)` or `(B, N)`
- a HEALPix `level` (thus `nside = 2**level`)

The class:

1. Finds the **nearest HEALPix pixel** for each sample.
2. Builds a compact **subset of pixels** that actually receive contributions (size `K`).
3. Creates sparse operators:
   - **`M`** of shape `(N, K)`: maps a HEALPix field on kept pixels to sample locations
   - **`MT`** of shape `(K, N)`: maps samples back to the HEALPix subset (accumulation)

Nearest-neighbour is equivalent to a **piecewise-constant** remapping on the HEALPix grid.

---

## Constructor

```python
from regrid_to_healpix.nearest import Set

op = Set(
    lon_deg=lon,
    lat_deg=lat,
    level=level,
    device="cuda",
    dtype=torch.float32,
    nest=True,
    threshold=0.0,
    verbose=False,
)
```

### Key parameters

- **`lon_deg, lat_deg`**: sample coordinates in degrees, shape `(N,)`.
- **`level`**: HEALPix level (`nside = 2**level`).
- **`nest`**: HEALPix indexing scheme (nested if `True`).
- **`device`, `dtype`**: PyTorch device and dtype.
- **`threshold`**: if used, can prune very weakly supported pixels (often 0 for nearest).
- **`verbose`**: diagnostic prints (depending on implementation).

---

## Stored attributes (typical)

After initialization, you typically have:

- **`N`**: number of samples.
- **`K`**: number of kept HEALPix pixels touched by the samples.
- **`cell_ids`**: `(K,)` HEALPix pixel ids retained.
- **`hi`**: `(N,)` indices into `cell_ids` (nearest pixel for each sample).
- **`M`**: sparse CSR `(N, K)` operator.
- **`MT`**: sparse CSR `(K, N)` operator.

---

## Methods

### `transform(hval)`

Project a HEALPix field (on the kept pixels) back to sample locations.

- **Input**: `hval` `(K,)` or `(B, K)`
- **Output**: values at samples `(N,)` or `(B, N)`

### `fit(val)` / `apply(val)` (depending on your API)

Nearest-neighbour “fit” is usually a direct accumulation or scatter-add:
- **Input**: `val` `(N,)` or `(B, N)`
- **Output**: `hval` `(K,)` or `(B, K)`

The output corresponds to values on the **subset** of HEALPix pixels `cell_ids`.

### `get_cell_ids()`

Return the kept HEALPix pixel ids as a NumPy array `(K,)`.

---

## Typical workflow

```python
import torch
from regrid_to_healpix.nearest import Set

op = Set(lon_deg=lon, lat_deg=lat, level=level, device="cuda", dtype=torch.float32)

# Map samples to the HEALPix subset (aggregation)
hval = op.fit(val)   # or op.apply(val)

# Reconstruct back to the original sample locations
val_hat = op.transform(hval)

cell_ids = op.get_cell_ids()
```

---

## Notes and practical tips

- Nearest-neighbour is **fast** and **memory-light**, but it can be **blocky** and introduce discretization artifacts.
- It is often useful as:
  - a baseline regridding method,
  - an initialization for smoother inversions (e.g., PSF / kernel methods),
  - a quick remap for QA / visualization.
- Like the other operators, it returns a **subset** of HEALPix pixels (`cell_ids`) rather than a full-sky map.
