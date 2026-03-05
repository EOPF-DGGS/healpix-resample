# `regrid_to_healpix.psf` (PSF / multi-point HEALPix regridding)

`regrid_to_healpix.psf` provides **GPU-friendly sparse regridding** from unstructured geographic samples
(**longitude/latitude**) to a **subset of HEALPix pixels** at a target resolution (`nside = 2**level`).

In contrast to a pure nearest-neighbor operator, this class builds a **local, multi-point Gaussian kernel**
around each sample (a “PSF”-like footprint) and can **solve an inverse problem** to estimate a HEALPix field
that best explains the observed samples.

The implementation is designed for **large N** and **batched values** `(B, N)` on **CUDA** using **PyTorch sparse**
operators.

---

## What the class does

Given:
- sample coordinates `(lon, lat)` of shape `(N,)`
- sample values `val` of shape `(N,)` or `(B, N)`
- a HEALPix `level` (thus `nside = 2**level`)
- a neighbourhood size `Npt` (number of nearby HEALPix cells per sample)

The class:

1. **Selects nearby HEALPix cells** for each sample using local neighbourhood search (avoids building an `N × Npix`
   distance matrix).
2. Computes **Gaussian weights** as a function of distance (meters) with scale `sigma_m`.
3. Builds two sparse operators:
   - **`M`** of shape `(N, K)` : maps a HEALPix field on *K kept pixels* to sample points (forward model is via `MT` below)
   - **`MT`** of shape `(K, N)` : maps sample values back to the HEALPix subset (adjoint-like accumulation)

4. Provides:
   - **`resample(hval)`**: project a HEALPix field back to sample locations
   - **`fit(val)`**: estimate the HEALPix field (`hval`) from samples by solving a **damped least-squares** problem with
     **Conjugate Gradient (CG)**
   - **`fit_resample(val)`**: fit then reconstruct values at the sample points

> Note on geodesy: distances are computed in meters and the class supports the **Earth ellipsoid WGS84** and the
> **HEALPix authalic definition** through its geometry helper.

---

## Mathematical view (high level)

Let:
- `y` be the sample values `(B, N)`
- `h` be the unknown HEALPix field `(B, K)` on the kept pixels
- `M` be `(N, K)` and `MT` be `(K, N)`

A reference field is computed by weighted back-projection:
- `x_ref = y @ M`  (shape `(B, K)`)

Then the solver estimates an update `delta` by minimizing a damped normal equation:
- minimize `|| (x_ref + delta) @ MT - y ||^2 + lam * ||delta||^2`

This is solved with CG using matrix-vector products only:
- `A(v) = (v @ MT) @ M + lam * v`

Finally:
- `h = x_ref + delta`

---

## Constructor

```python
regrid_to_healpix_psf(
    lon_deg, lat_deg,
    Npt, level,
    sigma_m=None,
    threshold=0.1,
    nest=True,
    radius=6371000.0,
    ellipsoid="WGS84",
    dtype=torch.float64,
    device="cuda",
    ring_weight=None,
    ring_search_init=None,
    ring_search_max=20,
    num_threads=0,
    verbose=True,
)
```

### Key parameters

- **`lon_deg, lat_deg`**: sample coordinates in degrees, shape `(N,)`.
- **`level`**: HEALPix level (`nside = 2**level`).
- **`Npt`**: number of neighbouring HEALPix cells used per sample.
- **`sigma_m`**: Gaussian length scale in meters.
  - If `None`, a default scale based on the HEALPix pixel area is used:
    `sigma = sqrt(4*pi / (12*4**level)) * R`.
- **`threshold`**: global pruning threshold on accumulated weights; pixels with too little support are discarded.
  - This reduces `K` and keeps the operator compact.
- **`nest`**: HEALPix indexing scheme (nested if `True`).
- **`device`, `dtype`**: PyTorch device and dtype for all matrices and computations.
- **`ring_*` parameters**: control the local neighbourhood expansion strategy in the geometry helper.
- **`verbose`**: prints CG progress (depending on the helper functions).

---

## Stored attributes (after initialization)

- **`N`**: number of samples.
- **`K`**: number of kept HEALPix pixels.
- **`cell_ids`**: `(K,)` HEALPix pixel ids retained after thresholding.
- **`hi`**: `(N, Npt)` indices into `cell_ids` for each sample (the chosen neighbours).
- **`d_m`**: `(N, Npt)` distances in meters for each neighbour link.
- **`M`**: sparse CSR `(N, K)` operator.
- **`MT`**: sparse CSR `(K, N)` operator.

---

## Methods

### `fit(val, lam=0.0, max_iter=100, tol=1e-8, x0=None, return_info=False)`

Estimate a HEALPix field from samples.

- **Input**:
  - `val`: `(N,)` or `(B, N)`
  - `lam`: damping / Tikhonov regularization
  - `x0`: optional initial guess for `delta` (shape `(B, K)`)
- **Output**:
  - `hval`: `(K,)` or `(B, K)`
  - optionally CG diagnostics if `return_info=True`

### `resample(hval)`

Project HEALPix field(s) back to sample locations.

- **Input**: `hval` `(K,)` or `(B, K)`
- **Output**: reconstructed samples `(N,)` or `(B, N)`

### `fit_resample(val, ...)`

Convenience method:
1) fit `hval`, 2) return reconstructed sample values.

Returns:
- `hval` and `tilde_val` (and optionally CG info)

### `get_cell_ids()`

Return the kept HEALPix pixel ids as a NumPy array `(K,)`.

---

## Typical workflow

```python
import torch
from regrid_to_healpix.psf import regrid_to_healpix_psf

op = regrid_to_healpix_psf(
    lon_deg=lon, lat_deg=lat,
    level=level, Npt=9,
    device="cuda", dtype=torch.float32,
)

# Estimate HEALPix field on the kept pixels
hval = op.fit(val, lam=1e-3, max_iter=200, tol=1e-7)

# Reconstruct values at the original sample points
val_hat = op.resample(hval)

# Access the HEALPix pixel ids corresponding to hval
cell_ids = op.get_cell_ids()
```

---

## Notes and practical tips

- **Choose `Npt`** according to the desired smoothness / footprint:
  - small `Npt` → more local, less smooth
  - larger `Npt` → smoother but more compute
- **Tune `sigma_m`**:
  - smaller `sigma_m` → sharper PSF, more local influence
  - larger `sigma_m` → smoother field but can blur features
- **Use `lam`** to stabilize inversion when sampling is sparse/irregular:
  - `lam = 0` is pure least squares
  - `lam > 0` damps high-frequency or poorly constrained modes
- The operator only returns a **subset** of HEALPix pixels (`cell_ids`), not the full sky map.
  This is intentional for memory/performance on regional problems.

