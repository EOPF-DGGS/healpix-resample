# `regrid_to_healpix.bilinear` (bilinear lon/lat interpolation to HEALPix)

`regrid_to_healpix.bilinear` is intended to provide a **bilinear interpolation** operator from gridded or locally
grid-like longitude/latitude data onto a HEALPix grid.

Conceptually, bilinear interpolation uses the **four surrounding grid points** around each target location and
interpolates with weights proportional to the fractional position inside the cell.

This module is useful when:
- the original data are defined on a **structured lon/lat grid** (regular or curvilinear),
- you want a result smoother than nearest-neighbour but cheaper than a wide PSF kernel.

> Note on geodesy: the package is intended to manage the **HEALPix authalic definition** and the Earth ellipsoid
> with **WGS84** through its geometry helper.

---

## What “bilinear” means here

For each target HEALPix cell center (or each sample position mapped to HEALPix), the operator:

1. Locates the surrounding lon/lat grid cell.
2. Computes fractional offsets `(fx, fy)` inside the cell.
3. Combines the four corner values with weights:
   - `(1-fx)(1-fy)`, `fx(1-fy)`, `(1-fx)fy`, `fx fy`.

This yields a **continuous** interpolation on the lon/lat grid (but still limited by the grid resolution).

---

## Expected interface (recommended)

A practical, PyTorch-friendly design is:

- `Set(...)` builds the sparse weights (4 contributions per target).
- `fit(val)` maps lon/lat grid values → HEALPix subset.
- `transform(hval)` maps HEALPix subset → target/sample locations (optional, depending on your needs).

### Suggested constructor

```python
from regrid_to_healpix.bilinear import Set

op = Set(
    lon2d=lon2d,          # (Ny, Nx) grid longitudes (deg)
    lat2d=lat2d,          # (Ny, Nx) grid latitudes  (deg)
    level=level,          # HEALPix level
    device="cuda",
    dtype=torch.float32,
    nest=True,
)
```

### Suggested inputs

- **`lon2d, lat2d`**: 2D lon/lat describing the grid (regular or curvilinear).
- **`level`**: HEALPix level (`nside = 2**level`).
- **`nest`**: HEALPix indexing scheme.
- **`device, dtype`**: torch placement and numerical type.

---

## Sparse operator structure (typical)

If you build an operator from `Ng = Ny*Nx` grid points to `K` kept HEALPix pixels:

- **`M`**: `(Ng, K)` (or `(K, Ng)` depending on your convention)
- Exactly **4 non-zeros per target** (per HEALPix pixel / per sample), for bilinear weights.

This makes bilinear interpolation:
- **much smoother than nearest**
- still **very sparse and fast**

---

## Notes and practical tips

- Bilinear assumes the field is locally well represented by **planar interpolation** in lon/lat coordinates.
  - Near the poles or across the dateline, you may need special handling (wrapping / local tangent plane).
- For curvilinear grids, bilinear interpolation requires a robust way to find the enclosing cell.
- If your sampling is highly irregular, the PSF method may be more stable than bilinear.

---

## Status

If `regrid_to_healpix.bilinear` is not yet implemented in the repository, this document describes the intended
behavior and a recommended public API. Once the class exists, you can update this file to match the actual
parameters and method names.
