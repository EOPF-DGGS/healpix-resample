"""
zuniq.py

GPU-friendly sparse HEALPix regridding from unstructured lon/lat samples
to a subset of HEALPix pixels at a target resolution (nside = 2**level).

Core ideas:
- Compute zuniq cell_ids

This module is designed for large N and batched values (B,N) on CUDA.
"""
from typing import Generic

import math
import numpy as np
import torch

from healpix_resample.base import T_Array, ResampleResults
from healpix_resample.groupby import GroupByResampler


class ZuniqNearestResampler(GroupByResampler, Generic[T_Array]):
    def __init__(self, *args, **kwargs):
        super().__init__(level=29, *args, **kwargs)
        
