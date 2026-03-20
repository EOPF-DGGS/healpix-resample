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

        # ── Free large geometry buffers — only needed during construction ──
        # xyz_samples (N,3) and xyz_cells (K,3) are never accessed after hi is built.
        del self.xyz_samples
        del self.xyz_cells
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        self._inverse_ready = True

    # ── Hierarchical inverse KNN ──────────────────────────────────────────────

    def _compute_inverse_hi(self, max_pairs: int = 100_000_000) -> torch.Tensor:
        """For each output HEALPix cell find the index of its nearest source sample.

        Hierarchical algorithm exploiting the HEALPix nested structure:

          dl=0  level=L  : match each cell to sources in its own pixel.
          dl=1  level=L-1: unmatched cells look in their parent pixel (id >> 2).
          …
          dl=L  level=0  : 12 base pixels cover the whole sphere → all filled.

        When all sources and targets collapse into very few (or one) parent cell,
        the number of pairs P can reach N×K which causes an OOM.  Whenever
        ``total_pairs > max_pairs`` the remaining cells are handled by
        ``_chunked_knn_fallback``, a direct chunked dot-product that uses O(R)
        memory instead of O(P=N×R).

        Args:
            max_pairs: pair-expansion budget per iteration.  When exceeded the
                       direct chunked fallback is used for the remaining cells.
                       Default 100 M  (≈ 800 MB for int64 pairs on CPU).

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
        del xyz_cpu

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

            # ── 3. Count pairs before expanding ────────────────────────────
            tgt_ord    = np.argsort(tgt_cells, kind="stable")
            tgt_sorted = tgt_cells[tgt_ord]
            rem_sorted = rem_np[tgt_ord]

            lo     = np.searchsorted(tgt_sorted, src_cells, side="left")
            hi_b   = np.searchsorted(tgt_sorted, src_cells, side="right")
            counts = (hi_b - lo).astype(np.int64)

            valid_mask = counts > 0
            if not valid_mask.any():
                continue

            valid_n      = np.where(valid_mask)[0]
            valid_counts = counts[valid_n]
            total_pairs  = int(valid_counts.sum())

            # ── Guard: too many pairs → chunked fallback for remaining cells ─
            # This happens when all sources and targets share a single coarse
            # parent (e.g. dense data in a small geographic area).
            if total_pairs > max_pairs:
                if self.verbose:
                    R = int(remaining.sum().item())
                    print(
                        f"[NearestResampler] dl={dl} level={cur_level:>2}: "
                        f"P={total_pairs:,} > max_pairs={max_pairs:,} "
                        f"→ chunked fallback for {R:,} remaining cells"
                    )
                self._chunked_knn_fallback(
                    hi, remaining, rem_idx, xyz_s, xyz_c
                )
                break

            # ── 4. Pair expansion ────────────────────────────────────────────
            src_pairs = np.repeat(valid_n, valid_counts)             # (P,)

            cum = np.zeros(len(valid_n) + 1, dtype=np.int64)
            np.cumsum(valid_counts, out=cum[1:])
            local_off = (
                np.arange(total_pairs, dtype=np.int64)
                - np.repeat(cum[:-1], valid_counts)
            )
            del cum
            abs_idx = np.repeat(lo[valid_n], valid_counts) + local_off
            del local_off
            tgt_pairs = rem_sorted[abs_idx]
            del abs_idx

            # ── 5. Scatter-argmax — scatter_reduce + winner filter ───────────
            src_t = torch.from_numpy(src_pairs.astype(np.int64)).to(dev)
            del src_pairs
            tgt_t = torch.from_numpy(tgt_pairs.astype(np.int64)).to(dev)
            del tgt_pairs

            dots = (xyz_c[tgt_t] * xyz_s[src_t]).sum(dim=1)

            max_dot = torch.full((K,), float("-inf"), device=dev, dtype=dots.dtype)
            max_dot.scatter_reduce_(0, tgt_t, dots, reduce="amax", include_self=True)

            is_winner = dots >= max_dot[tgt_t]
            del max_dot, dots
            win_tgt = tgt_t[is_winner]
            win_src = src_t[is_winner]
            del src_t, tgt_t, is_winner

            ord_w  = torch.argsort(win_tgt, stable=True)
            win_tgt = win_tgt[ord_w]
            win_src = win_src[ord_w]
            del ord_w
            is_first     = torch.ones(len(win_tgt), dtype=torch.bool, device=dev)
            is_first[1:] = win_tgt[1:] != win_tgt[:-1]
            best_tgt = win_tgt[is_first]
            best_src = win_src[is_first]
            del win_tgt, win_src, is_first

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

    # ── Chunked direct KNN fallback ───────────────────────────────────────────

    def _chunked_knn_fallback(
        self,
        hi: torch.Tensor,           # (K,) output — filled in-place
        remaining: torch.Tensor,    # (K,) bool mask — updated in-place
        rem_idx: torch.Tensor,      # (R,) indices of remaining cells in [0..K-1]
        xyz_s: torch.Tensor,        # (N, 3) source unit vectors
        xyz_c: torch.Tensor,        # (K, 3) target cell unit vectors
        mem_budget_bytes: int = 512 * 1024 * 1024,  # 512 MB
    ) -> None:
        """Direct nearest-neighbour for remaining cells via chunked dot products.

        For each remaining cell r, finds argmax_n dot(xyz_c[r], xyz_s[n]).
        Memory is bounded to ``mem_budget_bytes`` regardless of R and N.

        This is the fallback when the hierarchical pair expansion would exceed
        ``max_pairs``.  It runs in O(R × N) time but O(chunk_r × N) memory.
        """
        N   = xyz_s.shape[0]
        dev = xyz_s.device
        bpe = xyz_s.element_size()   # bytes per float element

        # chunk_r × N × bpe ≤ budget  →  chunk_r = budget // (N × bpe)
        chunk_r = max(1, mem_budget_bytes // (N * bpe))

        R       = int(rem_idx.numel())
        xyz_c_r = xyz_c[rem_idx].to(xyz_s.dtype)     # (R, 3) — only remaining

        for start in range(0, R, chunk_r):
            end   = min(start + chunk_r, R)
            # (chunk, N) dot products — the only large allocation, bounded by budget
            dots  = xyz_c_r[start:end] @ xyz_s.T     # (chunk, N)
            best  = dots.argmax(dim=1)                # (chunk,)  source indices
            del dots

            hi[rem_idx[start:end]]        = best
            remaining[rem_idx[start:end]] = False

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
