"""
regrid_to_healpix_psf.py

GPU-friendly sparse HEALPix regridding from unstructured lon/lat samples
to a subset of HEALPix pixels at a target resolution (nside = 2**level).

Core ideas:
- Use HEALPix local neighbourhoods (healpix_geo.kth_neighbourhood) to avoid N×npix distance matrices.
- Build sparse operators M (samples -> grid) and MT (grid -> samples) with Gaussian weights.
- Solve a damped least-squares problem with Conjugate Gradient (CG) on normal equations.

This module is designed for large N and batched values (B,N) on CUDA.
"""

from .regrid_to_healpix_GEN import Set as GENSet
import math
import numpy as np
import torch
from typing import Tuple, Optional
from typing import Callable, Optional, Tuple, Dict

@torch.no_grad()
def conjugate_gradient(
    A_mv: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    max_iter: int = 200,
    tol: float = 1e-6,
    verbose: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Solve A x = b with Conjugate Gradient where A is SPD, using only matvec A_mv(v).
    No autograd (uses torch.no_grad).

    Returns:
        x: solution
        info: dict with residual norms history, iterations
    """
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()

    r = b - A_mv(x)          # residual
    p = r.clone()
    rs_old = torch.einsum('ik,ik->i',r,r)

    b_norm = torch.linalg.norm(b)
    if b_norm == 0:
        return x, {"residual_norms": torch.tensor([0.0], device=b.device, dtype=b.dtype),
                   "iters": torch.tensor(0, device=b.device)}

    residual_norms = [torch.sqrt(rs_old)]

    for k in range(max_iter):
        Ap = A_mv(p)
        denom = torch.einsum('ik,ik->i',p,Ap)
        if torch.max(denom.abs()) < 1e-30:
            break  # breakdown (shouldn't happen for SPD unless numerical issues)

        alpha = rs_old / denom
        x = x + torch.einsum('k,ki->ki',alpha,p)
        r = r - torch.einsum('k,ki->ki',alpha,Ap)
        rs_new = torch.einsum('ik,ik->i',r,r)

        residual_norms.append(torch.sqrt(rs_new))

        # stopping criterion: relative residual
        if torch.max(torch.sqrt(rs_new)) <= tol * b_norm:
            rs_old = rs_new
            break

        beta = rs_new / rs_old
        p = r + torch.einsum('k,ki->ki',beta,p)
        rs_old = rs_new
        if k%4==0 and verbose:
            print('Itt %d : %.4g'%(k,rs_old))

    info = {
        "residual_norms": torch.stack(residual_norms),
        "iters": torch.tensor(len(residual_norms) - 1, device=b.device),
    }
    if verbose:
        print('Final Itt %d : %.4g'%(k,rs_old))
    return x, info

@torch.no_grad()
def least_squares_cg(M,
        MT,
        y,
        x_ref,
        x0, 
        max_iter = 200,
        tol = 1e-6,
        damp = 0.0,
        verbose: bool = True,
        ):
    """
    Solve for delta in a damped least-squares problem without forming dense matrices.

    We solve:
        (MT @ M + damp*I) delta = (y - x_ref @ MT) @ M

    Shapes:
        M  : (N, K) sparse CSR
        MT : (K, N) sparse CSR
        y  : (B, N)
        x_ref : (B, K)
        delta : (B, K)
    """

    # b = M^T y
    b = (y - x_ref@MT) @ M
    def A_mv(v: torch.Tensor) -> torch.Tensor:
        # (M^T M + damp I) v
        return (v@MT) @ M + damp * v

    x, info = conjugate_gradient(A_mv=A_mv, b=b, x0=x0, max_iter=max_iter, tol=tol,verbose=verbose)
    return x, info

from .regrid_to_healpix_GEN import _sigma_level_m

class Set(GENSet):
    def __init__(self,Npt=9,sigma_m=None,threshold=0.1, *args, **kwargs):
        super().__init__(Npt=Npt,sigma_m=sigma_m,threshold=threshold, *args, **kwargs)

    def comp_matrix(self):
        # --- weights per sample->cell link
        # w = exp(-2*d^2/sigma^2)
        w = torch.exp((-2.0) * (self.d_m * self.d_m) / (self.sigma_m * self.sigma_m))

        # Build (N,K) operator M and (K,N) operator MT.
        # We avoid numpy bincount; use torch.bincount on GPU.

        # idx: (N,Npt) row indices 0..N-1
        idx = torch.arange(self.N, device=self.device, dtype=torch.long)[:, None].expand(self.N, self.Npt)

        # -------- M : (N,K)  (normalized per column / per healpix cell)
        # norm_col[k] = sum_{i links to k} w[i,k]
        flat_hi = self.hi.reshape(-1)
        flat_w = w.reshape(-1)
        valid = flat_hi >= 0
        flat_hi_v = flat_hi[valid]
        flat_w_v = flat_w[valid]

        norm_col = torch.bincount(flat_hi_v, weights=flat_w_v, minlength=self.K).to(self.dtype)
        # weight divided by column sum
        wM = flat_w_v / norm_col[flat_hi_v]

        rowsM = idx.reshape(-1)[valid]
        colsM = flat_hi_v
        indicesM = torch.stack([rowsM, colsM], dim=0)
        M_coo = torch.sparse_coo_tensor(
            indicesM,
            wM.to(self.dtype),
            size=(self.N, self.K),
            device=self.device,
            dtype=self.dtype,
        ).coalesce()

        # -------- MT : (K,N) (normalized per row / per input sample)
        # norm_row[i] = sum_{k links from i} w[i,k]
        flat_idx = idx.reshape(-1)
        flat_idx_v = flat_idx[valid]
        norm_row = torch.bincount(flat_idx_v, weights=flat_w_v, minlength=self.N).to(self.dtype)
        wMT = flat_w_v / norm_row[flat_idx_v]

        indicesMT = torch.stack([colsM, rowsM], dim=0)  # (hi, idx)
        MT_coo = torch.sparse_coo_tensor(
            indicesMT,
            wMT.to(self.dtype),
            size=(self.K, self.N),
            device=self.device,
            dtype=self.dtype,
        ).coalesce()

        # Convert to CSR for faster spMM (recommended on GPU)
        self.M  = M_coo.to_sparse_csr()
        self.MT = MT_coo.to_sparse_csr()

    @torch.no_grad()
    def transform(
        self,
        val: torch.Tensor | np.ndarray,
        *,
        lam: float = 0.0,
        max_iter: int = 100,
        tol: float = 1e-8,
        x0: Optional[torch.Tensor] = None,
        return_info: bool = False,
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
        x_ref = y @ self.M

        if x0 is None:
            x0 = torch.zeros_like(x_ref)
        else:
            x0 = x0.to(self.device, dtype=self.dtype)

        delta, info = least_squares_cg(
            M=self.M,
            MT=self.MT,
            y=y,
            x_ref=x_ref,
            x0=x0,
            max_iter=max_iter,
            tol=tol,
            damp=float(lam),
            verbose=self.verbose,
        )

        hval = delta + x_ref
        if val is not None and val.ndim == 1:
            hval = hval[0]
        if not isinstance(val, torch.Tensor):
            hval=hval.cpu().numpy()
        if return_info:
            return hval, info
        return hval
  