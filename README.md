# regrid_to_healpix

`regrid_to_healpix` is a lightweight Python package designed to regrid
data defined on irregular longitude--latitude coordinates onto a HEALPix
grid.

The package provides GPU-accelerated operators (via PyTorch) to
construct sparse linear mappings between input geodetic coordinates and
a target HEALPix tessellation at a chosen resolution level.

## Objectives

The main goals of the package are:

-   Provide a **generic regridding framework** from (lon, lat) to
    HEALPix.
-   Support different interpolation strategies:
    -   **Nearest-neighbor mapping**
    -   **PSF / multi-point weighted interpolation**
-   Enable efficient handling of:
    -   Large numbers of input points
    -   Batched data `(B, N)`
    -   CUDA acceleration
-   Offer a reusable linear operator that can be:
    -   Applied forward (data → HEALPix)
    -   Used inside inverse problems or iterative solvers

## Design Principles

-   Modular architecture:
    -   `GEN` module: generic operator construction
    -   `nearest`: nearest-neighbor specialization
    -   `psf`: weighted multi-point interpolation
-   Sparse matrix representation for scalability
-   Torch-based implementation for CPU/GPU flexibility
-   Resolution controlled via HEALPix level parameter

## Typical Use Case

``` python
from regrid_to_healpix.regrid_to_healpix_nearest import Set

op = Set(lon_deg=lon, lat_deg=lat, level=level, device="cuda")
healpix_values = op.transform(values)
```

## Target Applications

-   Earth observation data remapping
-   Oceanographic or atmospheric gridding
-   Astronomical sky projections
-   Large-scale geospatial data harmonization
