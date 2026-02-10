"""
regrid_to_healpix_nearest.py

GPU-friendly sparse HEALPix regridding from unstructured lon/lat samples
to a subset of HEALPix pixels at a target resolution (nside = 2**level).

Core ideas:
- Use regrid_to_heapix with npt=1.

This module is designed for large N and batched values (B,N) on CUDA.
"""

from regrid_to_healpix import regrid_to_healpix_GEN

class regrid_to_healpix_nearest(regrid_to_healpix_GEN):
    def __init__(self, *args, **kwargs):
        # Impose que use_2D=True pour la classe scat
        super().__init__(Npt=1, *args, **kwargs)
        
    def comp_matrix(self):

        # Build (N,K) operator M and (K,N) operator MT.
        # We avoid numpy bincount; use torch.bincount on GPU.

        # idx: (N,Npt) row indices 0..N-1
        idx = torch.arange(self.N, device=self.device, dtype=torch.long)
        
        # -------- M : (N,K)  (normalized per column / per healpix cell)
        # norm_col[k] = sum_{i links to k} w[i,k]
        flat_hi = self.hi.reshape(-1)
        w = torch.ones(self.N, device=self.device, dtype=torch.dtype)


        rowsM = idx.reshape(-1)[valid]
        colsM = flat_hi
        indicesM = torch.stack([rowsM, colsM], dim=0)
        M_coo = torch.sparse_coo_tensor(
            indicesM,
            w,
            size=(self.N, self.K),
            device=self.device,
            dtype=self.dtype,
        ).coalesce()

        # -------- MT : (K,N) (normalized per row / per input sample)
        
        indicesMT = torch.stack([colsM, rowsM], dim=0)  # (hi, idx)
        MT_coo = torch.sparse_coo_tensor(
            indicesMT,
            w,
            size=(self.K, self.N),
            device=self.device,
            dtype=self.dtype,
        ).coalesce()

        # Convert to CSR for faster spMM (recommended on GPU)
        self.M  = M_coo.to_sparse_csr()
        self.MT = MT_coo.to_sparse_csr()
