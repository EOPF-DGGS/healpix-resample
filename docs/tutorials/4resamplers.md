---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# The four resamplers

This notebook runs all four resamplers on the same dataset so you can compare their behaviour, accuracy, and speed.

## Setup

```{code-cell} python
import numpy as np
from healpix_resample import (
    NearestResampler,
    BilinearResampler,
    PSFResampler,
    CellPointResampler,
)

# Shared dataset: a small structured grid near the origin
ndata = 128
lon_grid, lat_grid = np.meshgrid(
    0.3 * np.arange(ndata) / ndata,
    0.3 * np.arange(ndata) / ndata,
)
lon = lon_grid.ravel()
lat = lat_grid.ravel()
val = lon  # simple field: value = longitude

level = 15  # nside = 32768 — high resolution
```
### `NearestResampler` 

Each point is assigned to its single nearest HEALPix cell. Fast and simple, but can produce blocky results.

```{code-cell} python
nr_nearest = NearestResampler(lon_deg=lon, lat_deg=lat, level=level)
res_nearest = nr_nearest.resample(val)

rval_nearest = nr_nearest.invert(res_nearest.cell_data)
mse_nearest = np.mean((rval_nearest - val) ** 2)
print(f"Nearest  — output cells: {res_nearest.cell_data.shape[0]}, MSE: {mse_nearest:.2e}")
```

### `BilinearResampler` 

Uses the **4 nearest cells** with distance-based weights. Smoother than nearest, good for locally grid-like data.

```{code-cell} python
nr_bili = BilinearResampler(lon_deg=lon, lat_deg=lat, level=level)
res_bili = nr_bili.resample(val, lam=0.0)

rval_bili = nr_bili.invert(res_bili.cell_data)
mse_bili = np.mean((rval_bili - val) ** 2)
print(f"Bilinear — output cells: {res_bili.cell_data.shape[0]}, MSE: {mse_bili:.2e}")
```

### `PSFResampler`

Applies a **Gaussian PSF kernel** around each sample and solves a damped least-squares problem with Conjugate Gradient. Best reconstruction quality — especially when data is dense or the field has fine structure.

```{code-cell} python
nr_psf = PSFResampler(lon_deg=lon, lat_deg=lat, level=level, threshold=0.5, verbose=False)
res_psf = nr_psf.resample(val, lam=0.0)

rval_psf = nr_psf.invert(res_psf.cell_data)
mse_psf = np.mean((rval_psf - val) ** 2)
print(f"PSF      — output cells: {res_psf.cell_data.shape[0]}, MSE: {mse_psf:.2e}")
print(f"           CG iterations: {res_psf.cg_niters}")
```
### `CellPointResampler`

Special mode: encodes each point as a HEALPix cell ID at level 29. No interpolation — used for exact point indexing.

```{code-cell} python
nr_zuniq = CellPointResampler(lon_deg=lon, lat_deg=lat)
res_zuniq = nr_zuniq.resample(val)

rval_zuniq = nr_zuniq.invert(res_zuniq.cell_data)
max_err = np.max(np.abs(rval_zuniq - val))
print(f"Zuniq    — output cells: {res_zuniq.cell_data.shape[0]}, max error: {max_err:.2e}")
```