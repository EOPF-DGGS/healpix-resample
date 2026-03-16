# Installation

This guide explains you how to install healpix-plotting on your system.

## Requirements

- Python **≥ 3.13**

## Install

### Via conda

::::{tab-set}

:::{tab-item} conda

```bash
conda install -c conda-forge healpix-resample
```

:::

:::{tab-item} mamba

```bash
mamba install -c conda-forge healpix-resample
```

:::

:::{tab-item} pixi

```bash
pixi add healpix-resample
```

:::

::::

### Via pip

::::{tab-set}

:::{tab-item} pip

```bash
pip install healpix-resample
```

:::

:::{tab-item} uv

```bash
uv add healpix-resample
```

:::

::::

## Verify

```python
import healpix_resample

print(healpix_resample.__version__)
```

## Dependencies

| Package | Role |
|---|---|
| `torch` | Sparse matrix operations, GPU support |
| `healpix-geo` | HEALPix geometry (cell lookup, neighbourhoods) |
| `numpy` | Array handling |
| `foscat` | Visualization helpers (optional, for notebooks) |