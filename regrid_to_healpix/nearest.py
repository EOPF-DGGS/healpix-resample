"""
nearest.py

GPU-friendly sparse HEALPix regridding from unstructured lon/lat samples
to a subset of HEALPix pixels at a target resolution (nside = 2**level).

Core ideas:
- Use regrid_to_heapix with npt=1.

This module is designed for large N and batched values (B,N) on CUDA.
"""
from regrid_to_healpix.knn import Set as GENSet
import math
import numpy as np
import torch


class Set(GENSet):
    def __init__(self, *args, **kwargs):
        super().__init__(Npt=1, *args, **kwargs)
        
    def comp_matrix(self):

        # Build (N,K) operator M and (K,N) operator MT.
        # We avoid numpy bincount; use torch.bincount on GPU.

        # idx: (N,Npt) row indices 0..N-1
        idx = torch.arange(self.N, device=self.device, dtype=torch.long)
        
        # -------- M : (N,K)  (normalized per column / per healpix cell)
        # norm_col[k] = sum_{i links to k} w[i,k]
        flat_hi = self.hi.reshape(-1)
        w = torch.ones((self.N,), device=self.device, dtype=self.dtype)

        norm_row = torch.bincount(flat_hi, weights=w, minlength=self.N).to(self.dtype)
        wM = w / norm_row[flat_hi]

        rowsM = idx.reshape(-1)
        colsM = flat_hi
        indicesM = torch.stack([rowsM, colsM], dim=0)
        M_coo = torch.sparse_coo_tensor(
            indicesM,
            wM,
            size=(self.N, self.K),
            device=self.device,
            dtype=self.dtype,
        ).coalesce()

        # -------- MT : (K,N) (normalized per row / per input sample)
        norm_row = torch.bincount(idx, weights=w, minlength=self.N).to(self.dtype)
        wMT = w / norm_row[idx]
        indicesMT = torch.stack([colsM, rowsM], dim=0)  # (hi, idx)
        MT_coo = torch.sparse_coo_tensor(
            indicesMT,
            wMT,
            size=(self.K, self.N),
            device=self.device,
            dtype=self.dtype,
        ).coalesce()

        # Convert to CSR for faster spMM (recommended on GPU)
        self.M  = M_coo.to_sparse_csr()
        self.MT = MT_coo.to_sparse_csr()
