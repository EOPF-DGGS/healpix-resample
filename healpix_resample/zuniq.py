"""
zuniq.py

GPU-friendly sparse HEALPix regridding from unstructured lon/lat samples
to a subset of HEALPix pixels at a target resolution (nside = 2**level).

Core ideas:
- Use regrid_to_heapix to compute zuniq cell_ids

This module is designed for large N and batched values (B,N) on CUDA.
"""
from healpix_resample.knn import KNeighborsResampler
import math
import numpy as np
import torch


class ZuniqNearestResampler(KNeighborsResampler):
    def __init__(self, *args, **kwargs):
        super().__init__(zuniq=True,Npt=1,level=29, *args, **kwargs)
        
    @torch.no_grad()
    def invert(self, hval: torch.Tensor | np.ndarray,) -> torch.Tensor:
        """Project HEALPix field back to the sample locations.

        Args:
            hval: (B,K) or (K,) on same device
        Returns:
            val_hat: (B,N) or (N,)
        """
        is_torch = isinstance(hval, torch.Tensor)

        y = hval if is_torch else torch.as_tensor(hval)
        y = y.to(self.device, dtype=self.dtype)
        
        if y.ndim == 1:
            res = y.index_select(0, self.hi)          # (N,)
        else:
            res = y.index_select(1, self.hi)          # (B,N)

        return res if is_torch else res.cpu().numpy()
        
    @torch.no_grad()
    def resample(
        self,
        val: torch.Tensor | np.ndarray,
    ):
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
        res = torch.zeros(B, self.K, device=y.device,dtype= self.dtype)

        res.scatter_reduce_(
            1,
            self.hi.unsqueeze(0).expand(B, -1),
            y,
            reduce="mean",
            include_self=False
        )
        
        if not isinstance(val, torch.Tensor):
            res=res.cpu().numpy()
        if clean_shape:
            return res[0]
        return res
