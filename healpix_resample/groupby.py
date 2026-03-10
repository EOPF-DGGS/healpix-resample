"""
groupby.py

GPU-friendly sparse HEALPix regridding from unstructured lon/lat samples
to a subset of HEALPix pixels at a target resolution (nside = 2**level).

Core ideas:
- Compute group by into cell_ids
- can apply different methods to merge information within one cell_ids 'mean','nearest','median',etc.

This module is designed for large N and batched values (B,N) on CUDA.
"""
from typing import Generic

import math
import numpy as np
import torch

from healpix_resample.base import T_Array, ResampleResults
from healpix_resample.knn import KNeighborsResampler


class GroupByResampler(KNeighborsResampler, Generic[T_Array]):
    def __init__(self,
                 reduce="mean", # Could be "sum", "prod", "mean", "amax", "amin" as for torch.scatter_reduce
                 *args, **kwargs):
                     
        assert reduce in ["sum", "prod", "mean", "amax", "amin"], \
            'reduce should be in "sum", "prod", "mean", "amax", "amin"'
            
        super().__init__(group_by=True,Npt=1, *args, **kwargs)
        
        self.reduce = reduce
        
    @torch.no_grad()
    def invert(self, hval: T_Array) -> T_Array:
        """Project HEALPix field back to the sample locations.

        Args:
            hval: (B,K) or (K,) on same device
        Returns:
            val_hat: (B,N) or (N,)
        """
        y: torch.Tensor = hval if isinstance(hval, torch.Tensor) else torch.as_tensor(hval)
        y = y.to(self.device, dtype=self.dtype)
        
        if y.ndim == 1:
            res = y.index_select(0, self.hi)          # (N,)
        else:
            res = y.index_select(1, self.hi)          # (B,N)

        if not isinstance(hval, torch.Tensor):
            res = res.cpu().numpy()
        return res
        
    @torch.no_grad()
    def resample(self, val: T_Array) -> ResampleResults[T_Array]:
        """Estimate the HEALPix field from unstructured samples.

        Args:
            val: (B,N) or (N,) values at lon/lat sample points
            lam: Tikhonov regularization strength (damping) used in CG
            max_iter, tol: CG parameters
            x0: optional initial guess for the *delta* around x_ref, shape (B,K)
            return_info: whether to return CG diagnostics

        Returns:
            hval: (B,K) or (K,)
            (optional) info: CG information dict
        """
        y = val if isinstance(val, torch.Tensor) else torch.as_tensor(val)
        y = y.to(self.device, dtype=self.dtype)
        clean_shape=False
        if y.ndim == 1:
            clean_shape=True
            y = y[None, :]

        # reference field (B,K)
        B=y.shape[0]
        hval = torch.zeros(B, self.K, device=y.device,dtype= self.dtype)

        hval.scatter_reduce_(
            1,
            self.hi.unsqueeze(0).expand(B, -1),
            y,
            reduce=self.reduce,
            include_self=False
        )
        
        cell_ids = self.cell_ids
        
        if not isinstance(val, torch.Tensor):
            hval= hval.cpu().numpy()
            cell_ids = cell_ids.cpu().numpy()
        if clean_shape:
            hval = hval[0]
        
        return ResampleResults(cell_data=hval, cell_ids=cell_ids)


class CellPointResampler(GroupByResampler):
    """Resample input lat-lon points as HEALPix "cell-points".

    The values of the input lat-lon points are resampled on HEALPix cells
    with a fixed, maximum refinement level (=29) in which the points are located.

    Level-29 cells have a tiny resolution of 0.4 milliarcseconds. In many
    applications those cells can be approximated as "points".

    If multiple input points are located within the same cell, their values
    are merged according to the `reduce` option.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(level=29, *args, **kwargs)
