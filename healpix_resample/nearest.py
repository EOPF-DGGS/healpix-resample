"""
nearest.py

GPU-friendly sparse HEALPix regridding from unstructured lon/lat samples
to a subset of HEALPix pixels at a target resolution (nside = 2**level).

Core ideas:
- Use npt=1.

This module is designed for large N and batched values (B,N) on CUDA.

Inverse mode (when out_cell_ids is provided):
  Instead of mapping each source sample to its containing HEALPix cell
  (which leaves empty cells when HEALPix resolution > source grid resolution),
  we invert the direction: for every output HEALPix cell we find the nearest
  source sample via a chunked dot-product KNN on GPU (no empty cells possible).
"""
from healpix_resample.knn import KNeighborsResampler, _lonlat_to_xyz
import healpix_geo
import numpy as np
import torch


class NearestResampler(KNeighborsResampler):
    """Nearest-neighbour HEALPix resampler.

    When ``out_cell_ids`` is *None* (default):
        Classic forward mode — each source sample is assigned to its containing
        HEALPix cell.  Cells with no sample remain empty (suitable when the
        source grid is denser than HEALPix).

    When ``out_cell_ids`` is provided:
        **Inverse mode** — for every requested output HEALPix cell we find the
        nearest source sample using a chunked dot-product KNN executed entirely
        on the target device (GPU or CPU).  Every cell is guaranteed to receive
        a value regardless of the relative resolutions.
    """

    # Flags set before super().__init__ so comp_matrix can branch early.
    _inverse_mode: bool = False
    _inverse_ready: bool = False  # True only after _setup_inverse() completes

    def __init__(self, *args, **kwargs):
        self._inverse_mode = kwargs.get("out_cell_ids") is not None
        self._inverse_ready = False
        super().__init__(Npt=1, *args, **kwargs)

        if self._inverse_mode:
            # super() has computed xyz_samples (N,3) and set a provisional
            # cell_ids / xyz_cells from the forward pass.  We now overwrite
            # them with the correct inverse-KNN result.
            self._setup_inverse()

    # ------------------------------------------------------------------
    # Inverse-mode setup (called once, after super().__init__)
    # ------------------------------------------------------------------

    def _setup_inverse(self) -> None:
        """Fix cell_ids / xyz_cells / hi for the out_cell_ids inverse mode."""

        # 1. Enforce cell_ids = exactly out_cell_ids (no threshold filter).
        out_t = self.out_cell_ids
        if not isinstance(out_t, torch.Tensor):
            out_t = torch.as_tensor(out_t)
        out_t = out_t.to(device=self.device, dtype=torch.long).reshape(-1)

        self.cell_ids = out_t   # (K,)
        self.K = int(out_t.numel())

        # 2. Recompute xyz_cells for the full out_cell_ids.
        cell_np = out_t.cpu().numpy().astype(np.uint64)
        if self.nest:
            lon_c_deg, lat_c_deg = healpix_geo.nested.healpix_to_lonlat(
                cell_np, self.level, ellipsoid=self.ellipsoid
            )
        else:
            lon_c_deg, lat_c_deg = healpix_geo.ring.healpix_to_lonlat(
                cell_np, self.level, ellipsoid=self.ellipsoid
            )

        src_dtype = self.xyz_samples.dtype   # match source dtype (fp32/fp64)
        lon_c = torch.deg2rad(
            torch.as_tensor(lon_c_deg, device=self.device, dtype=src_dtype)
        )
        lat_c = torch.deg2rad(
            torch.as_tensor(lat_c_deg, device=self.device, dtype=src_dtype)
        )
        self.xyz_cells = _lonlat_to_xyz(lon_c, lat_c)   # (K, 3)

        # 3. Compute hi[k] = index of nearest source sample to cell k.
        self.hi = self._compute_inverse_hi()   # (K,)

        # 4. Build the sparse operators.
        self._inverse_ready = True
        self.comp_matrix()

    def _compute_inverse_hi(self) -> torch.Tensor:
        """For each output HEALPix cell find the index of its nearest source sample.

        Hierarchical algorithm exploiting the HEALPix nested structure:

        Level L   (dl=0): assign each source to its own nested cell.
                          For every out_cell_id that has ≥1 source, keep the nearest.
        Level L-1 (dl=1): remaining empty cells are mapped to their parent (cell//4).
                          Search for nearest source among all samples in the parent cell.
        ...
        Level 0          : at most 12 cells cover the whole sphere — everything is filled.

        Complexity: O(N·log N) per level  vs  O(K·N) for brute force.
        Memory:     O(N + K) — no large intermediate matrix.

        The scatter-argmax (best source per target cell) is fully vectorised via a
        double stable-sort trick, with no Python loop over individual cells.

        Returns:
            hi : (K,) long tensor — index of nearest source sample per cell.
        """
        xyz_s = self.xyz_samples                      # (N, 3)
        xyz_c = self.xyz_cells.to(xyz_s.dtype)        # (K, 3)
        K, N  = self.K, self.N
        dev   = self.device

        # ── lon/lat of source samples (needed by healpix_geo at each level) ──
        xyz_cpu = xyz_s.cpu().numpy().astype(np.float64)
        lon_np  = np.degrees(np.arctan2(xyz_cpu[:, 1], xyz_cpu[:, 0]))
        lat_np  = np.degrees(np.arcsin(np.clip(xyz_cpu[:, 2], -1.0, 1.0)))

        out_np = self.cell_ids.cpu().numpy().astype(np.int64)   # (K,) at level L

        hi        = torch.full((K,), -1, dtype=torch.long, device=dev)
        remaining = torch.ones(K, dtype=torch.bool, device=dev)  # True = not yet filled

        for dl in range(self.level + 1):
            n_rem = int(remaining.sum().item())
            if n_rem == 0:
                break

            cur_level = self.level - dl

            # ── 1. Source cells at current level (nested) ──────────────────
            src_cells = healpix_geo.nested.lonlat_to_healpix(
                lon_np, lat_np, cur_level
            ).astype(np.int64)                                   # (N,)

            # ── 2. Remaining targets mapped to current level ───────────────
            rem_idx = remaining.nonzero(as_tuple=False).squeeze(1)   # (R,)
            rem_np  = rem_idx.cpu().numpy()
            # In nested, parent at level L-dl = cell_id >> (2*dl)  (i.e. // 4^dl)
            tgt_cells = (out_np[rem_np] >> (2 * dl)).astype(np.int64)  # (R,)

            # ── 3. Join: build all (src_n, tgt_k) pairs sharing the same cell ──
            #
            # Strategy: sort targets by cell, then for each source binary-search
            # its range in the sorted target array.  The index expansion is done
            # with pure numpy without a Python loop over cells.

            tgt_ord    = np.argsort(tgt_cells, kind="stable")
            tgt_sorted = tgt_cells[tgt_ord]        # cell IDs of targets, sorted
            rem_sorted = rem_np[tgt_ord]           # original k indices, same order

            lo   = np.searchsorted(tgt_sorted, src_cells, side="left")
            hi_b = np.searchsorted(tgt_sorted, src_cells, side="right")
            counts = (hi_b - lo).astype(np.int64)  # (N,) matches per source

            valid_mask = counts > 0
            if not valid_mask.any():
                if self.verbose:
                    print(f"[NearestResampler] dl={dl} level={cur_level}: "
                          f"no source↔target matches, skipping")
                continue

            valid_n      = np.where(valid_mask)[0]   # source indices with ≥1 target
            valid_counts = counts[valid_n]
            total_pairs  = int(valid_counts.sum())

            # Expand: src_pairs[i] = source index for pair i  (np.repeat, no loop)
            src_pairs = np.repeat(valid_n, valid_counts)   # (total_pairs,)

            # tgt_pairs[i] = original k index for pair i
            # For source valid_n[j], matching targets are rem_sorted[lo[j]:hi_b[j]].
            # We build the flat index array into rem_sorted without a Python loop:
            cum = np.zeros(len(valid_n) + 1, dtype=np.int64)
            np.cumsum(valid_counts, out=cum[1:])
            local_off  = np.arange(total_pairs, dtype=np.int64) - np.repeat(cum[:-1], valid_counts)
            abs_idx    = np.repeat(lo[valid_n], valid_counts) + local_off
            tgt_pairs  = rem_sorted[abs_idx]           # (total_pairs,) original k indices

            # ── 4. Vectorised scatter-argmax on GPU ────────────────────────
            #
            # For each target k, we want:  hi[k] = argmax_{n: pair→k} dot(xyz_c[k], xyz_s[n])
            #
            # Trick: sort pairs by dot desc, then stable-sort by k asc
            # → first occurrence of each k in the final array = best dot for that k.

            src_t = torch.from_numpy(src_pairs.astype(np.int64)).to(dev)  # (M,)
            tgt_t = torch.from_numpy(tgt_pairs.astype(np.int64)).to(dev)  # (M,)

            # Element-wise dot product for each pair (no large matrix)
            dots = (xyz_c[tgt_t] * xyz_s[src_t]).sum(dim=1)               # (M,)

            # Step 1 – sort by dot descending (best first within each future group)
            ord1 = torch.argsort(dots, descending=True, stable=True)
            src1 = src_t[ord1]
            tgt1 = tgt_t[ord1]

            # Step 2 – stable sort by k ascending (preserves dot order within k)
            ord2 = torch.argsort(tgt1, stable=True)
            src2 = src1[ord2]
            tgt2 = tgt1[ord2]

            # Step 3 – first occurrence of each k = best dot for that k
            is_first        = torch.ones(len(tgt2), dtype=torch.bool, device=dev)
            is_first[1:]    = tgt2[1:] != tgt2[:-1]

            best_tgt = tgt2[is_first]   # (unique_k,)
            best_src = src2[is_first]   # (unique_k,) nearest source index

            hi[best_tgt]        = best_src
            remaining[best_tgt] = False

            if self.verbose:
                print(f"[NearestResampler] dl={dl} level={cur_level:>2}: "
                      f"filled {len(best_tgt):>{len(str(K))}}, "
                      f"remaining {int(remaining.sum()):>{len(str(K))}}/{K}")

        return hi

    # ------------------------------------------------------------------
    # comp_matrix  — dispatches to original or inverse builder
    # ------------------------------------------------------------------

    def comp_matrix(self) -> None:
        """Build sparse operators M (N,K) and MT (K,N).

        Dispatches to:
          - _comp_matrix_inverse  when inverse mode is ready, or
          - _comp_matrix_original for the classic forward mode.

        The method is called once by super().__init__() during construction.
        In inverse mode that first call is a no-op because _inverse_ready is
        still False; the real build happens at the end of _setup_inverse().
        """
        if self._inverse_mode:
            if self._inverse_ready:
                self._comp_matrix_inverse()
            # else: called from super().__init__ before _setup_inverse — skip.
            return

        self._comp_matrix_original()

    # ------------------------------------------------------------------
    # Inverse matrix builder
    # ------------------------------------------------------------------

    def _comp_matrix_inverse(self) -> None:
        """Build M / MT for the inverse KNN.

        For each output cell k, hi[k] is the index of its nearest source sample.

        M  [ hi[k], k ] = 1          shape (N, K)
            → hval = y @ M  gives  hval[:, k] = y[:, hi[k]]  (one source per cell, exact)

        MT [ k, hi[k] ] = 1 / count[hi[k]]    shape (K, N)
            where count[n] = number of HEALPix cells whose nearest source is n.
            → val_hat = hval @ MT  gives  val_hat[:, n] = mean of hval[:, k]
              over all k pointing to n  (average, not sum).
        """
        K   = self.K
        N   = self.N
        dev = self.device
        dt  = self.dtype

        cell_idx   = torch.arange(K, device=dev, dtype=torch.long)   # (K,)
        sample_idx = self.hi                                           # (K,)
        ones       = torch.ones(K, device=dev, dtype=dt)

        # M : (N, K) — one non-zero per column (one source sample per cell)
        M_coo = torch.sparse_coo_tensor(
            torch.stack([sample_idx, cell_idx]),
            ones,
            size=(N, K),
            device=dev,
            dtype=dt,
        ).coalesce()

        # MT : (K, N) — normalised by the number of cells sharing each source sample
        # count[n] = how many HEALPix cells have sample n as their nearest neighbour
        count = torch.bincount(sample_idx, minlength=N).to(dt)  # (N,)
        # weight for each (k -> hi[k]) edge: 1 / count[hi[k]]
        wMT = ones / count[sample_idx]                           # (K,)

        MT_coo = torch.sparse_coo_tensor(
            torch.stack([cell_idx, sample_idx]),
            wMT,
            size=(K, N),
            device=dev,
            dtype=dt,
        ).coalesce()

        self.M  = M_coo.to_sparse_csr()
        self.MT = MT_coo.to_sparse_csr()

    # ------------------------------------------------------------------
    # Original (forward) matrix builder — unchanged logic
    # ------------------------------------------------------------------

    def _comp_matrix_original(self) -> None:
        """Original NearestResampler comp_matrix (forward mode, no out_cell_ids)."""

        # idx: (N,) row indices 0..N-1
        idx     = torch.arange(self.N, device=self.device, dtype=torch.long)
        flat_hi = self.hi.reshape(-1)
        w       = torch.ones((self.N,), device=self.device, dtype=self.dtype)

        # M : (N, K)  normalised per HEALPix column
        norm_col = torch.bincount(flat_hi, weights=w, minlength=self.K).to(self.dtype)
        wM = w / norm_col[flat_hi]

        rowsM = idx.reshape(-1)
        colsM = flat_hi
        M_coo = torch.sparse_coo_tensor(
            torch.stack([rowsM, colsM], dim=0),
            wM,
            size=(self.N, self.K),
            device=self.device,
            dtype=self.dtype,
        ).coalesce()

        # MT : (K, N)  normalised per source sample
        norm_row = torch.bincount(idx, weights=w, minlength=self.N).to(self.dtype)
        wMT = w / norm_row[idx]
        MT_coo = torch.sparse_coo_tensor(
            torch.stack([colsM, rowsM], dim=0),
            wMT,
            size=(self.K, self.N),
            device=self.device,
            dtype=self.dtype,
        ).coalesce()

        self.M  = M_coo.to_sparse_csr()
        self.MT = MT_coo.to_sparse_csr()
