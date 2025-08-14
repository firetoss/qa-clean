from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class RecallProvider(ABC):
    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    @staticmethod
    def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        return x / norms

    def maybe_normalize(self, x: np.ndarray) -> np.ndarray:
        if self.normalize:
            return self._l2_normalize(x)
        return x

    @abstractmethod
    def build(self, xb: np.ndarray) -> None:  # xb: base vectors
        ...

    @abstractmethod
    def search(self, xq: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str, xb_dim: Optional[int] = None) -> bool:
        ...
