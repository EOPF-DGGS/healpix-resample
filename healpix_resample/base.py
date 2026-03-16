from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import torch


T_Array = TypeVar("T_Array", np.ndarray, torch.Tensor)


@dataclass(frozen=True)
class ResampleResults(Generic[T_Array]):
    """Proxy to resampling results.

    Attributes
    ----------
    cell_data : numpy.ndarray or torch.Tensor
        Data values resampled on HEALPix cells
    cell_ids : numpy.ndarray or torch.Tensor
        HEALPix cell ids.
    cg_residual_norms : numpy.ndarray or torch.Tensor or None
        Conjugate gradient residual norms (if any).
    cg_niters : numpy.ndarray or torch.Tensor or None
        Conjugate gradient number of iterations (if any).
    
    """
    cell_data: T_Array
    cell_ids: T_Array
    cg_residual_norms: T_Array | None = None
    cg_niters: T_Array | None = None
