"""
nearest.py

GPU-friendly sparse-free HEALPix nearest-neighbour regridding.

Two modes depending on whether ``out_cell_ids`` is provided:

Forward mode  (out_cell_ids=None):
    Each source sample is assigned to its containing HEALPix cell.
    hi : (N,)  —  hi[n] = index of cell_ids that contains sample n.

    resample(val (B,N)) -> hval (B,K) : scatter-mean  (average samples per cell)
    invert  (hval(B,K)) -> val  (B,N) : index          hval[:, hi[n]]

Inverse mode  (out_cell_ids provided):
    For every requested HEALPix cell the nearest source sample is found via a
    hierarchical nested descent — O(N log N · level) and O(N+K) memory.
    hi : (K,)  —  hi[k] = index of the nearest source sample for cell k.

    resample(val (B,N)) -> hval (B,K) : index          val[:, hi[k]]
    invert  (hval(B,K)) -> val  (B,N) : scatter-mean  (average cells per sample)

No sparse matrices anywhere.
"""
from __future__ import annotations

from typing import Optional

from healpix_resample.knn import KNeighborsResampler, _lonlat_to_xyz
from healpix_resample.base import ResampleResults, T_Array
import healpix_geo
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scatter_mean(
    src: torch.Tensor,   # (B, S)  values to scatter
    idx: torch.Tensor,   # (S,)    target indices  0..T-1
    T: int,              # output size along dim-1
) -> torch.Tensor:
    """Scatter-mean: out[:, t] = mean of src[:, s] where idx[s] == t.

    Cells with no contributor stay at 0.
    Fully vectorised — no Python loop over cells or batches.
    """
    B = src.shape[0]
    out = torch.zeros(B, T, device=src.device, dtype=src.dtype)
    # expand idx to (B, S) for scatter_add_
    idx_e = idx.unsqueeze(0).expand(B, -1)           # (B, S)
    out.scatter_add_(1, idx_e, src)
    count = torch.bincount(idx, minlength=T).to(src.dtype)  # (T,)
    # avoid division by zero for empty cells
    out /= count.clamp(min=1).unsqueeze(0)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# NearestResampler
# ─────────────────────────────────────────────────────────────────────────────

class NearestResampler(KNeighborsResampler):
    """Nearest-neighbour HEALPix resampler — no sparse matrices.

    See module docstring for the two modes.
    """

    # Set before super().__init__ so the comp_matrix no-op can branch.
    _inverse_mode: bool = False
    _inverse_ready: bool = False

    def __init__(self, *args, **kwargs):
        self._inverse_mode = kwargs.get("out_cell_ids") is not None
        self._inverse_ready = False
        super().__init__(Npt=1, *args, **kwargs)

        if self._inverse_mode:
            self._setup_inverse()

    # ── comp_matrix: no-op (we never build M / MT) ────────────────────────

    def comp_matrix(self) -> None:
        """No-op: NearestResampler does not use sparse matrices."""
        pass

    # ── Inverse-mode setup ───────────────────────────────────────────────────

    def _setup_inverse(self) -> None:
        """Overwrite cell_ids / hi for the inverse mode after super().__init__."""

        # 1. cell_ids = exactly out_cell_ids, no threshold filtering.
        out_t = self.out_cell_ids
        if not isinstance(out_t, torch.Tensor):
            out_t = torch.as_tensor(out_t)
        out_t = out_t.to(device=self.device, dtype=torch.long).reshape(-1)
        self.cell_ids = out_t
        self.K = int(out_t.numel())

        # 2. xyz_cells for those cells.
        cell_np = out_t.cpu().numpy().astype(np.uint64)
        if self.nest:
            lon_c_deg, lat_c_deg = healpix_geo.nested.healpix_to_lonlat(
                cell_np, self.level, ellipsoid=self.ellipsoid
            )
        else:
            lon_c_deg, lat_c_deg = healpix_geo.ring.healpix_to_lonlat(
                cell_np, self.level, ellipsoid=self.ellipsoid
            )
        src_dtype = self.xyz_samples.dtype
        lon_c = torch.deg2rad(
            torch.as_tensor(lon_c_deg, device=self.device, dtype=src_dtype)
        )
        lat_c = torch.deg2rad(
            torch.as_tensor(lat_c_deg, device=self.device, dtype=src_dtype)
        )
        self.xyz_cells = _lonlat_to_xyz(lon_c, lat_c)   # (K, 3)

        # 3. hi[k] = index of nearest source sample for each cell k.
        self.hi = self._compute_inverse_hi()   # (K,)
        self._inverse_ready = True

    # ── Hierarchical inverse KNN ──────────────────────────────────────────────

    def _compute_inverse_hi(self) -> torch.Tensor:
        """For each output HEALPix cell find the index of its nearest source sample.

        Hierarchical algorithm exploiting the HEALPix nested structure:

          dl=0  level=L  : match each cell to sources in its own pixel.
          dl=1  level=L-1: unmatched cells look in their parent pixel (id >> 2).
          dl=2  level=L-2: parent pixel id >> 4 …
          dl=L  level=0  : 12 base pixels cover the whole sphere → all filled.

        Complexity: O(N log N · level)  vs  O(K·N) for brute force.
        Memory:     O(N + K)            vs  O(K·N) for brute force.

        The scatter-argmax is fully vectorised via a double stable-sort:
          1. sort pairs by dot-product desc  → best source first per cell
          2. stable-sort by cell index asc   → first occurrence = best source
        No Python loop over individual cells or samples.

        Returns:
            hi : (K,) long tensor — index of nearest source sample per cell.
        """
        xyz_s = self.xyz_samples                       # (N, 3)
        xyz_c = self.xyz_cells.to(xyz_s.dtype)         # (K, 3)
        K, N  = self.K, self.N
        dev   = self.device

        # Recover lon/lat of source samples from their xyz (needed by healpix_geo).
        xyz_cpu = xyz_s.cpu().numpy().astype(np.float64)
        lon_np  = np.degrees(np.arctan2(xyz_cpu[:, 1], xyz_cpu[:, 0]))
        lat_np  = np.degrees(np.arcsin(np.clip(xyz_cpu[:, 2], -1.0, 1.0)))

        out_np    = self.cell_ids.cpu().numpy().astype(np.int64)   # (K,)
        hi        = torch.full((K,), -1, dtype=torch.long, device=dev)
        remaining = torch.ones(K, dtype=torch.bool, device=dev)

        for dl in range(self.level + 1):
            if not remaining.any():
                break

            cur_level = self.level - dl

            # ── 1. Cell ID of each source at current level ─────────────────
            src_cells = healpix_geo.nested.lonlat_to_healpix(
                lon_np, lat_np, cur_level
            ).astype(np.int64)                                      # (N,)

            # ── 2. Remaining output cells mapped to current level ───────────
            rem_idx   = remaining.nonzero(as_tuple=False).squeeze(1)  # (R,)
            rem_np    = rem_idx.cpu().numpy()
            tgt_cells = (out_np[rem_np] >> (2 * dl)).astype(np.int64) # (R,)

            # ── 3. Build (src_n, tgt_k) pairs — vectorised, no Python loop ──
            #
            # Sort targets by cell ID, then binary-search each source cell to
            # find its range in the sorted target array.
            tgt_ord    = np.argsort(tgt_cells, kind="stable")
            tgt_sorted = tgt_cells[tgt_ord]   # sorted cell IDs of remaining targets
            rem_sorted = rem_np[tgt_ord]      # corresponding original k indices

            lo     = np.searchsorted(tgt_sorted, src_cells, side="left")
            hi_b   = np.searchsorted(tgt_sorted, src_cells, side="right")
            counts = (hi_b - lo).astype(np.int64)                   # (N,)

            valid_mask = counts > 0
            if not valid_mask.any():
                continue

            valid_n      = np.where(valid_mask)[0]
            valid_counts = counts[valid_n]
            total_pairs  = int(valid_counts.sum())

            # Expand source indices (np.repeat — no loop).
            src_pairs = np.repeat(valid_n, valid_counts)             # (P,)

            # Expand target indices: for source j, targets are
            # rem_sorted[lo[j] : hi_b[j]].  Flatten without a loop:
            cum = np.zeros(len(valid_n) + 1, dtype=np.int64)
            np.cumsum(valid_counts, out=cum[1:])
            local_off = (
                np.arange(total_pairs, dtype=np.int64)
                - np.repeat(cum[:-1], valid_counts)
            )
            abs_idx   = np.repeat(lo[valid_n], valid_counts) + local_off
            tgt_pairs = rem_sorted[abs_idx]                          # (P,)

            # ── 4. Scatter-argmax on GPU — double stable-sort trick ─────────
            src_t = torch.from_numpy(src_pairs.astype(np.int64)).to(dev)  # (P,)
            tgt_t = torch.from_numpy(tgt_pairs.astype(np.int64)).to(dev)  # (P,)

            # Element-wise dot product per pair (memory = 3·P floats, not K·N).
            dots = (xyz_c[tgt_t] * xyz_s[src_t]).sum(dim=1)          # (P,)

            # Step 1 — sort by dot desc: best source first within each cell.
            ord1 = torch.argsort(dots, descending=True, stable=True)
            src1 = src_t[ord1]
            tgt1 = tgt_t[ord1]

            # Step 2 — stable sort by cell asc: preserves dot order inside groups.
            ord2 = torch.argsort(tgt1, stable=True)
            src2 = src1[ord2]
            tgt2 = tgt1[ord2]

            # Step 3 — first occurrence of each cell = best dot for that cell.
            is_first      = torch.ones(len(tgt2), dtype=torch.bool, device=dev)
            is_first[1:]  = tgt2[1:] != tgt2[:-1]

            best_tgt = tgt2[is_first]
            best_src = src2[is_first]

            hi[best_tgt]        = best_src
            remaining[best_tgt] = False

            if self.verbose:
                w = len(str(K))
                print(
                    f"[NearestResampler] dl={dl} level={cur_level:>2}: "
                    f"filled {len(best_tgt):>{w}}, "
                    f"remaining {int(remaining.sum()):>{w}}/{K}"
                )

        return hi

    # ── resample ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def resample(
        self,
        val: T_Array,
        **_kwargs,   # lam / max_iter / tol ignored for nearest
    ) -> ResampleResults:
        """Project source samples onto HEALPix cells.

        Forward mode : scatter-mean — hval[:, k] = mean of val[:, n] with hi[n]==k
        Inverse mode : index        — hval[:, k] = val[:, hi[k]]
        """
        y = val if isinstance(val, torch.Tensor) else torch.as_tensor(val)
        y = y.to(self.device, dtype=self.dtype)

        squeezed = y.ndim == 1
        if squeezed:
            y = y.unsqueeze(0)   # (1, N)

        if self._inverse_mode:
            # hi : (K,)  →  direct index, no loop, no allocation beyond output
            hval = y[:, self.hi]                               # (B, K)
        else:
            # hi : (N,)  →  scatter-mean over K cells
            hval = _scatter_mean(y, self.hi, self.K)           # (B, K)

        cell_ids = self.cell_ids
        if squeezed:
            hval     = hval.squeeze(0)

        if not isinstance(val, torch.Tensor):
            hval     = hval.cpu().numpy()
            cell_ids = cell_ids.cpu().numpy()

        return ResampleResults(cell_data=hval, cell_ids=cell_ids)

    # ── invert ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def invert(self, hval: T_Array) -> T_Array:
        """Project HEALPix cells back to source sample locations.

        Forward mode : index        — val[:, n] = hval[:, hi[n]]
        Inverse mode : scatter-mean — val[:, n] = mean of hval[:, k] with hi[k]==n
        """
        y = hval if isinstance(hval, torch.Tensor) else torch.as_tensor(hval)
        y = y.to(self.device, dtype=self.dtype)

        squeezed = y.ndim == 1
        if squeezed:
            y = y.unsqueeze(0)   # (1, K)

        if self._inverse_mode:
            # hi : (K,)  →  scatter-mean back to N source samples
            res = _scatter_mean(y, self.hi, self.N)            # (B, N)
        else:
            # hi : (N,)  →  direct index into K cells
            res = y[:, self.hi]                                # (B, N)

        if squeezed:
            res = res.squeeze(0)

        if not isinstance(hval, torch.Tensor):
            res = res.cpu().numpy()

        return res
