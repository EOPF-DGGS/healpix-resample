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

    @staticmethod
    def _auto_chunk_sizes(
        K: int,
        N: int,
        bytes_per_elem: int,
        device: torch.device,
        mem_fraction: float = 0.25,
        min_chunk: int = 256,
    ):
        """Return (chunk_k, chunk_n) that fit inside the available device memory.

        Strategy: reserve ``mem_fraction`` of free VRAM for the GEMM tile
        (chunk_k, chunk_n). We maximise chunk_k first (row-major sweep over K)
        then set chunk_n as large as possible.

        On CPU falls back to safe defaults (no VRAM query possible).
        """
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            free_bytes, _ = torch.cuda.mem_get_info(device)
            budget = int(free_bytes * mem_fraction)
        else:
            # CPU: use a conservative 256 MB tile
            budget = 256 * 1024 * 1024

        # tile memory = chunk_k * chunk_n * bytes_per_elem
        # Fix chunk_k = min(K, 4096) then solve for chunk_n.
        chunk_k = min(K, max(min_chunk, 4096))
        chunk_n = budget // (chunk_k * bytes_per_elem)
        chunk_n = max(min_chunk, min(N, chunk_n))

        # If even a single row doesn't fit, shrink chunk_k to compensate.
        if chunk_n < min_chunk:
            chunk_n = min_chunk
            chunk_k = max(min_chunk, budget // (chunk_n * bytes_per_elem))
            chunk_k = min(K, chunk_k)

        return int(chunk_k), int(chunk_n)

    def _compute_inverse_hi(self) -> torch.Tensor:
        """For each output HEALPix cell find the index of its nearest source sample.

        Double-chunked GEMM over (K, N) with automatic tile sizing based on
        available VRAM — no OOM regardless of K or N.

        Algorithm:
            For every cell k, argmax_n dot(xyz_cell[k], xyz_sample[n]) is
            equivalent to argmin geodesic_distance, but requires no acos().
            We maintain running (best_dot, best_idx) per cell and sweep over
            N in chunks, updating only when a better candidate is found.

        Memory per iteration: chunk_k * chunk_n * bytes_per_elem  (one tile).

        Returns:
            hi: (K,) long tensor — index of nearest source sample per cell.
        """
        xyz_s = self.xyz_samples                     # (N, 3)
        xyz_c = self.xyz_cells.to(xyz_s.dtype)       # (K, 3)
        K, N = self.K, self.N
        bpe = xyz_s.element_size()                   # 4 (fp32) or 8 (fp64)

        chunk_k, chunk_n = self._auto_chunk_sizes(K, N, bpe, self.device)

        if self.verbose:
            print(
                f"[NearestResampler] inverse KNN  K={K:,}  N={N:,}  "
                f"tile=({chunk_k}, {chunk_n})  "
                f"dtype={'fp32' if bpe==4 else 'fp64'}"
            )

        best_dot = torch.full((K,), -2.0, device=self.device, dtype=xyz_s.dtype)
        hi       = torch.zeros(K,          device=self.device, dtype=torch.long)

        for k0 in range(0, K, chunk_k):
            k1   = min(k0 + chunk_k, K)
            ck   = xyz_c[k0:k1]                      # (ck, 3)
            ck_best_dot = best_dot[k0:k1].clone()
            ck_hi       = hi[k0:k1].clone()

            for n0 in range(0, N, chunk_n):
                n1   = min(n0 + chunk_n, N)
                cn   = xyz_s[n0:n1]                  # (cn, 3)

                # (ck, cn) dot products — the only large allocation
                dots = ck @ cn.T                     # (ck, cn)

                local_max, local_idx = dots.max(dim=1)   # (ck,)

                better = local_max > ck_best_dot
                ck_best_dot = torch.where(better, local_max, ck_best_dot)
                ck_hi       = torch.where(better, local_idx + n0, ck_hi)

                # free immediately to keep peak memory = one tile
                del dots, local_max, local_idx, better

            best_dot[k0:k1] = ck_best_dot
            hi[k0:k1]       = ck_hi

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
