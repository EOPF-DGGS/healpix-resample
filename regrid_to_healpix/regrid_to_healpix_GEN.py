"""
regrid_to_healpix_GEN.py

GPU-friendly sparse HEALPix regridding from unstructured lon/lat samples
to a subset of HEALPix pixels at a target resolution (nside = 2**level).

Core ideas:
- Use HEALPix local neighbourhoods (healpix_geo.kth_neighbourhood) to avoid N×npix distance matrices.
- Build sparse operators M (samples -> grid) and MT (grid -> samples) with Gaussian weights.
- Solve a damped least-squares problem with Conjugate Gradient (CG) on normal equations.

This module is designed for large N and batched values (B,N) on CUDA.
"""

import math
import numpy as np
import torch
from typing import Tuple, Optional, Union

def _lonlat_to_xyz(lon_rad: torch.Tensor, lat_rad: torch.Tensor) -> torch.Tensor:
    clat = torch.cos(lat_rad)
    return torch.stack([clat * torch.cos(lon_rad),
                        clat * torch.sin(lon_rad),
                        torch.sin(lat_rad)], dim=-1)  # (...,3)

def _sigma_level_m(level: int, radius: float = 6371000.0) -> float:
    # sigma = sqrt(4*pi / (12*4**level)) * R
    return math.sqrt(4.0 * math.pi / (12.0 * (4.0 ** level))) * radius
    
@torch.no_grad()
def healpix_weighted_nearest(
    longitude1: torch.Tensor,          # (N,) degrés
    latitude1: torch.Tensor,           # (N,) degrés
    level: int,
    Npt: int,
    *,
    nest: bool = True,
    threshold: float = 0.1,
    radius: float = 6371000.0,
    ellipsoid: str = "WGS84",
    sigma: float = None,
    # sous-ensemble de pixels de sortie autorisés (en ids healpix au même "level")
    out_cell_ids: Optional[Union[np.ndarray, torch.Tensor]] = None,
    # voisinage utilisé pour estimer les poids (construction cell_ids)
    ring_weight: Optional[int] = None,
    # voisinage utilisé pour trouver Npt voisins parmi les pixels gardés (peut être augmenté automatiquement)
    ring_search_init: Optional[int] = None,
    ring_search_max: int = 20,
    num_threads: int = 0,
    device_for_dist: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Retourne:
      cell_ids: (K,) pixels HEALPix (au level) retenus par le seuil de poids (et éventuellement intersectés avec out_cell_ids)
      idx_k   : (N, Npt) indices dans cell_ids (0..K-1), -1 si insuffisant
      dist_k  : (N, Npt) distances (m) vers centres des pixels correspondants, inf si insuffisant

    Notes:
      - On utilise la distance géodésique via angle = acos(dot(xyz)) ; dist = R * angle.
      - healpix_geo est utilisé pour lonlat_to_healpix, kth_neighbourhood, healpix_to_lonlat.
      - La précision est très correcte pour du “pixel center matching”.
    """
    assert longitude1.ndim == latitude1.ndim == 1
    N = int(longitude1.numel())
    assert Npt >= 1

    import healpix_geo  # pip install healpix-geo

    
    # --- choix rings par défaut
    # ring minimal pour avoir >= Npt candidats dans un carré (2r+1)^2, + marge
    r_min = int(math.ceil((math.sqrt(Npt) - 1.0) / 2.0))
    if ring_weight is None:
        ring_weight = max(2, r_min + 2)  # plus large: mieux pour estimer les poids globaux
    if ring_search_init is None:
        ring_search_init = max(1, r_min + 1)

    # --- distances: on peut les faire sur GPU si dispo
    dev = device_for_dist if device_for_dist is not None else longitude1.device
    
    # --- CPU numpy pour healpix_geo
    lon_np = longitude1.detach().cpu().numpy().astype(np.float64)
    lat_np = latitude1.detach().cpu().numpy().astype(np.float64)

    if nest:
        ipix1 = healpix_geo.nested.lonlat_to_healpix(lon_np, lat_np, level, num_threads=num_threads,ellipsoid=ellipsoid)
        ipix_u, inv = np.unique(ipix1.astype(np.uint64), return_inverse=True)
        if Npt==1:
            return torch.from_numpy(ipix_u.astype(np.int64)).to(dev), torch.from_numpy(inv.astype(np.int64)).to(dev),0            
        neigh_u_w = healpix_geo.nested.kth_neighbourhood(ipix_u, level, ring_weight, num_threads=num_threads)
    else:
        ipix1 = healpix_geo.ring.lonlat_to_healpix(lon_np, lat_np, level, num_threads=num_threads,ellipsoid=ellipsoid)
        ipix_u, inv = np.unique(ipix1.astype(np.uint64), return_inverse=True)
        if Npt==1:
            return torch.from_numpy(ipix_u.astype(np.int64)).to(dev), torch.from_numpy(inv.astype(np.int64)).to(dev),0
        neigh_u_w = healpix_geo.ring.kth_neighbourhood(ipix_u, level, ring_weight, num_threads=num_threads)

    # healpix_geo.kth_neighbourhood may return -1 for "invalid" neighbours.
    # We replace invalid entries by a valid neighbour (the last valid in the row; fallback to the center pixel).
    # This keeps arrays dense (no masking) and avoids uint64 overflow issues.
    # Duplicates are later handled naturally by weight accumulation + normalization.

    neigh_u_w = neigh_u_w.astype(np.int64, copy=False)

    valid = neigh_u_w >= 0

    # position du dernier valide par ligne
    last_valid_pos = valid[:, ::-1].argmax(axis=1)
    last_valid_pos = (neigh_u_w.shape[1] - 1) - last_valid_pos
    last_valid = neigh_u_w[np.arange(neigh_u_w.shape[0]), last_valid_pos]

    # fallback si toute la ligne est invalide -> centre
    all_invalid = ~valid.any(axis=1)
    last_valid[all_invalid] = ipix_u[all_invalid].astype(np.int64, copy=False)

    # remplace les -1 (broadcast explicite)
    mask = neigh_u_w < 0
    neigh_u_w[mask] = np.broadcast_to(last_valid[:, None], neigh_u_w.shape)[mask]

    # neigh_w : (N, Kw)
    neigh_w = neigh_u_w[inv]
    Kw = neigh_w.shape[1]

    # --- centres lon/lat des pixels du voisinage (Kw*N potentiellement grand) => on unique
    neigh_w_flat = neigh_w.reshape(-1).astype(np.uint64)
    neigh_w_uniq, back = np.unique(neigh_w_flat, return_inverse=True)

    if nest:
        lon_c_deg, lat_c_deg = healpix_geo.nested.healpix_to_lonlat(neigh_w_uniq, level,ellipsoid=ellipsoid)
    else:
        lon_c_deg, lat_c_deg = healpix_geo.ring.healpix_to_lonlat(neigh_w_uniq, level,ellipsoid=ellipsoid)

    lon1 = torch.deg2rad(longitude1.to(dev))
    lat1 = torch.deg2rad(latitude1.to(dev))
    xyz1 = _lonlat_to_xyz(lon1, lat1)  # (N,3)

    lon_c = torch.deg2rad(torch.from_numpy(np.asarray(lon_c_deg)).to(dev))
    lat_c = torch.deg2rad(torch.from_numpy(np.asarray(lat_c_deg)).to(dev))
    xyz_c = _lonlat_to_xyz(lon_c, lat_c)  # (Kuniq,3)

    # Remap neighbours vers indices uniques (Kuniq)
    back_t = torch.from_numpy(back.astype(np.int64)).to(dev)         # (N*Kw,)
    back_t = back_t.view(N, Kw)                                      # (N,Kw)
    xyz_c_n = xyz_c[back_t]                                          # (N,Kw,3)

    dot = (xyz_c_n * xyz1[:, None, :]).sum(dim=-1)                   # (N,Kw)
    dot = torch.clamp(dot, -1.0, 1.0)
    ang = torch.acos(dot)
    dist = radius * ang                                              # (N,Kw)

    # --- poids et somme par pixel
    # w = exp(-2*d^2/sigma^2)
    w = torch.exp((-2.0) * (dist * dist) / (sigma * sigma))          # (N,Kw)

    # scatter_add sur pixels uniques
    # On accumule w sur chaque pixel (Kuniq)
    sums = torch.zeros((xyz_c.shape[0],), device=dev, dtype=w.dtype)
    # Thresholding stage:
    # We compute Gaussian weights from each sample to its neighbourhood pixel centers,
    # then accumulate a global weight sum per HEALPix cell. Only cells whose total
    # influence exceeds 'threshold' are kept in cell_ids_keep (size K).

    sums.scatter_add_(0, back_t.reshape(-1), w.reshape(-1))

    keep = sums >= threshold
    if not torch.any(keep):
        # aucun pixel ne passe le seuil -> on retourne vide + -1/inf
        cell_ids = torch.empty((0,), device=dev, dtype=torch.long)
        idx_k = torch.full((N, Npt), -1, device=dev, dtype=torch.long)
        dist_k = torch.full((N, Npt), float("inf"), device=dev, dtype=lon1.dtype)
        return cell_ids, idx_k, dist_k

    # cell_ids retenus (en id healpix)
    cell_ids_np = neigh_w_uniq  # np.uint64, taille Kuniq
    keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)         # indices dans [0..Kuniq-1]
    cell_ids_keep = torch.from_numpy(cell_ids_np.astype(np.int64)).to(dev)[keep_idx]  # (K,)

    # xyz des pixels retenus (K,3)
    xyz_keep = xyz_c[keep_idx]


    # --- optionnel: restreindre explicitement les pixels de sortie
    # out_cell_ids peut être une liste/array/torch tensor d'ids HEALPix (au même level).
    if out_cell_ids is not None:
        out_t = out_cell_ids if isinstance(out_cell_ids, torch.Tensor) else torch.as_tensor(out_cell_ids)
        out_t = out_t.to(device=dev, dtype=torch.long).reshape(-1)
        if out_t.numel() == 0:
            # sous-ensemble vide demandé -> rien à regriller
            cell_ids = torch.empty((0,), device=dev, dtype=torch.long)
            idx_k = torch.full((N, Npt), -1, device=dev, dtype=torch.long)
            dist_k = torch.full((N, Npt), float("inf"), device=dev, dtype=lon1.dtype)
            return cell_ids, idx_k, dist_k

        out_sorted = torch.unique(out_t)
        out_sorted, _ = torch.sort(out_sorted)

        # test d'appartenance robuste via searchsorted (évite dépendre de torch.isin)
        pos_out = torch.searchsorted(out_sorted, cell_ids_keep)
        pos_out = torch.clamp(pos_out, 0, out_sorted.numel() - 1)
        in_mask = (out_sorted[pos_out] == cell_ids_keep)

        if not torch.any(in_mask):
            cell_ids = torch.empty((0,), device=dev, dtype=torch.long)
            idx_k = torch.full((N, Npt), -1, device=dev, dtype=torch.long)
            dist_k = torch.full((N, Npt), float("inf"), device=dev, dtype=lon1.dtype)
            return cell_ids, idx_k, dist_k

        keep_idx = keep_idx[in_mask]
        cell_ids_keep = cell_ids_keep[in_mask]
        xyz_keep = xyz_c[keep_idx]

    # --- pour chaque point: trouver Npt pixels retenus les plus proches
    # Stratégie: voisinage healpix autour du pixel contenant le point, ring qui s’agrandit
    # On mappe les pixels du voisinage -> indices dans cell_ids_keep via tri+searchsorted.
    cell_sorted, order = torch.sort(cell_ids_keep)                   # (K,), (K,)
    K = int(cell_ids_keep.numel())

    idx_out = torch.full((N, Npt), -1, device=dev, dtype=torch.long)
    dist_out = torch.full((N, Npt), float("inf"), device=dev, dtype=lon1.dtype)

    r = ring_search_init
    done = torch.zeros((N,), device=dev, dtype=torch.bool)

    # On travaille côté CPU pour kth_neighbourhood, mais on ne fait ça que pour les rings nécessaires.
    # On recalcule sur pixels uniques pour limiter les appels.
    while r <= ring_search_max and not bool(torch.all(done).item()):
        if nest:
            neigh_u = healpix_geo.nested.kth_neighbourhood(ipix_u, level, r, num_threads=num_threads)
        else:
            neigh_u = healpix_geo.ring.kth_neighbourhood(ipix_u, level, r, num_threads=num_threads)

        neigh = neigh_u[inv]  # (N, Ks)
        Ks = neigh.shape[1]
        neigh_t = torch.from_numpy(neigh.astype(np.int64)).to(dev)

        # map neigh_t -> index dans cell_ids_keep
        pos = torch.searchsorted(cell_sorted, neigh_t)
        pos = torch.clamp(pos, 0, K - 1)
        hit = (cell_sorted[pos] == neigh_t)
        cand_idx_keep = torch.where(hit, order[pos], torch.full_like(pos, -1))  # indices dans cell_ids_keep

        # distances pour candidats valides
        safe = torch.clamp(cand_idx_keep, 0, K - 1)
        xyz2 = xyz_keep[safe]                                  # (N,Ks,3)

        dot2 = (xyz2 * xyz1[:, None, :]).sum(dim=-1)
        dot2 = torch.clamp(dot2, -1.0, 1.0)
        dist2 = radius * torch.acos(dot2)
        dist2 = torch.where(cand_idx_keep >= 0, dist2, torch.full_like(dist2, float("inf")))

        # topk parmi ces candidats
        k = min(Npt, Ks)
        d_k, p_k = torch.topk(dist2, k=k, dim=1, largest=False, sorted=True)
        i_k = torch.gather(cand_idx_keep, 1, p_k)

        # pour les points pas encore "done", on accepte si on a au moins Npt valides (i.e. d_k[:, Npt-1] < inf)
        valid_enough = (d_k[:, -1] < float("inf"))
        update = (~done) & valid_enough

        if torch.any(update):
            idx_out[update] = i_k[update]
            dist_out[update] = d_k[update]
            done[update] = True

        r += 1

    # Si certains points n’ont jamais trouvé Npt pixels gardés, on laisse -1/inf (ou on pourrait rendre moins que Npt)
    return cell_ids_keep, idx_out, dist_out


class Set:
    """GPU-friendly sparse HEALPix regridding via local Gaussian weights + CG deconvolution.

    This class builds two sparse operators from unstructured lon/lat samples to a subset
    of HEALPix pixels at a target resolution (nside = 2**level).

    Notation (matching your notebook):
      - N: number of samples (lon/lat)
      - K: number of kept HEALPix cells (cell_ids)
      - M: operator of shape (N, K)   (named ``M`` here)
      - MT: operator of shape (K, N)  (named ``MT`` here)

    The solver estimates ``hval`` (B,K) such that:
        M @ hval.T  matches  val (B,N)
    by solving a damped normal equation around a reference field x_ref = val @ M.
    """

    def __init__(
        self,
        lon_deg: np.ndarray | torch.Tensor,
        lat_deg: np.ndarray | torch.Tensor,
        Npt: int,
        level: int,
        *,
        nest: bool = True,
        radius: float = 6371000.0,
        ellipsoid: str = "WGS84",
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = None,
        ring_weight: Optional[int] = None,
        ring_search_init: Optional[int] = None,
        ring_search_max: int = 2,
        num_threads: int = 0,
        threshold: float = 0.1,
        sigma_m: float = None,
        verbose: bool = True,
        out_cell_ids: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> None:
        """Pre-compute sparse operators.

        Args:
            lon_deg, lat_deg: unstructured sample coordinates in degrees, shape (N,)
            Npt: number of nearest HEALPix cells used per sample
            level: HEALPix level, nside = 2**level
            sigma_m: Gaussian length scale (meters). If None, uses the HEALPix pixel scale
                     sigma = sqrt(4*pi/(12*4**level))*R.
            threshold: keep only HEALPix cells whose global weight sum >= threshold
            nest: HEALPix indexing scheme
            dtype/device: torch dtype/device for all matrices and computations
        """
        self.level = int(level)
        self.nside = 2 ** int(level)
        self.Npt = int(Npt)
        self.nest = bool(nest)
        self.radius = float(radius)
        self.ellipsoid = str(ellipsoid)
        self.dtype = dtype
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available.")

        self.device = torch.device(device)
        self.threshold = float(threshold)
        self.out_cell_ids = out_cell_ids
        # --- sigma in meters (controls the Gaussian weights used for thresholding)
        sigma = float(_sigma_level_m(level, radius=radius) if sigma_m is None else sigma_m)
        self.sigma_m = sigma

        # --- move lon/lat to torch on target device (but healpix_geo needs CPU numpy internally)
        lon_t = lon_deg if isinstance(lon_deg, torch.Tensor) else torch.as_tensor(lon_deg)
        lat_t = lat_deg if isinstance(lat_deg, torch.Tensor) else torch.as_tensor(lat_deg)
        lon_t = lon_t.to(self.device)
        lat_t = lat_t.to(self.device)
        
        if self.out_cell_ids is not None:
            self.xyz_samples = _lonlat_to_xyz(torch.deg2rad(lon_t),torch.deg2rad(lat_t))  # (N,3)
            

        # --- get kept healpix cells + per-sample nearest indices + distances
        cell_ids, hi, d = healpix_weighted_nearest(
            lon_t,
            lat_t,
            level=self.level,
            Npt=self.Npt,
            nest=self.nest,
            threshold=self.threshold,
            radius=self.radius,
            ellipsoid=self.ellipsoid,
            sigma=self.sigma_m,
            out_cell_ids=self.out_cell_ids,
            ring_weight=ring_weight,
            ring_search_init=ring_search_init,
            ring_search_max=ring_search_max,
            num_threads=num_threads,
            device_for_dist=self.device,
        )
            

        if cell_ids.numel() == 0:
            raise RuntimeError(
                "No HEALPix cell passed the threshold. "
                "Lower 'threshold' or increase neighbourhood rings."
            )

        # Store geometry outputs
        self.cell_ids = cell_ids.to(torch.long).to(self.device)   # (K,)
        self.hi = hi.to(torch.long).to(self.device)               # (N,Npt) indices into cell_ids
        if Npt>1:
            self.d_m = d.to(self.dtype).to(self.device)               # (N,Npt) meters
        self.N = int(lon_t.numel())
        self.K = int(self.cell_ids.numel())
        self.verbose = verbose

        if self.out_cell_ids is not None:
            
            # --- geometry buffers for optional fallbacks (e.g. when out_cell_ids forces empty columns)

            import healpix_geo  # pip install healpix-geo

            # unit vectors for output HEALPix cell centers (K,3)
            cell_np = self.cell_ids.detach().cpu().numpy().astype(np.uint64)
            
            if self.nest:
                lon_c_deg, lat_c_deg = healpix_geo.nested.healpix_to_lonlat(cell_np, self.level, ellipsoid=self.ellipsoid)
            else:
                lon_c_deg, lat_c_deg = healpix_geo.ring.healpix_to_lonlat(cell_np, self.level, ellipsoid=self.ellipsoid)

            lon_c = torch.deg2rad(torch.as_tensor(lon_c_deg, device=self.device, dtype=self.xyz_samples.dtype))
            lat_c = torch.deg2rad(torch.as_tensor(lat_c_deg, device=self.device, dtype=self.xyz_samples.dtype))
            
            self.xyz_cells = _lonlat_to_xyz(lon_c, lat_c)  # (K,3)
            
        self.comp_matrix()
        
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
    def invert(self, hval: torch.Tensor | np.ndarray,) -> torch.Tensor:
        """Project HEALPix field back to the sample locations.

        Args:
            hval: (B,K) or (K,) on same device
        Returns:
            val_hat: (B,N) or (N,)
        """
        y = hval if isinstance(hval, torch.Tensor) else torch.as_tensor(hval)
        y = y.to(self.device, dtype=self.dtype)
        if hval.ndim == 1:
            res = (y[None, :] @ self.MT)[0]
        else:
            res =  y @ self.MT
        
        if not isinstance(hval, torch.Tensor):
            res=res.cpu().numpy()
        
        return res

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
        res = y @ self.M
        
        if not isinstance(val, torch.Tensor):
            res=res.cpu().numpy()
        if clean_shape:
            return res[0]
        return res
        
    def get_cell_ids(self):
        return self.cell_ids.cpu().numpy()
    
  