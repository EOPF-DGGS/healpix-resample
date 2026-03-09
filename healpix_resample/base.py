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
    cell_data : np.array or torch.tensor
        Data values resampled on HEALPix cells
    cell_ids : np.array or torch.tensor
        HEALPix cell ids.
    
    """
    cell_data: T_Array
    cell_ids: T_Array
