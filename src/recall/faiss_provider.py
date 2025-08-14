from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .base import RecallProvider


class FaissProvider(RecallProvider):
    def __init__(
        self,
        index_type: str = "flat_ip",
        nlist: int = 4096,
        nprobe: int = 16,
        hnsw_m: int = 32,
        ef_search: int = 200,
        normalize: bool = True,
        device: str = "cuda",  # cuda | cpu
    ) -> None:
        super().__init__(normalize=normalize)
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.hnsw_m = hnsw_m
        self.ef_search = ef_search
        self.device = device
        self.index = None
        self._faiss = None
        self._gpu_res = None

    def _import_faiss(self):
        if self._faiss is not None:
            return self._faiss
        try:
            if self.device == "cuda":
                import faiss
                if hasattr(faiss, "StandardGpuResources"):
                    self._faiss = faiss
                    return self._faiss
        except Exception:
            pass
        import faiss  # CPU fallback
        self._faiss = faiss
        return self._faiss

    def _to_gpu(self, index):
        faiss = self._import_faiss()
        if self.device != "cuda" or not hasattr(faiss, "StandardGpuResources"):
            return index
        try:
            self._gpu_res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(self._gpu_res, 0, index)
        except Exception:
            # fallback silently
            pass
        return index

    def build(self, xb: np.ndarray) -> None:
        faiss = self._import_faiss()
        xb = xb.astype(np.float32)
        xb = self.maybe_normalize(xb)
        d = xb.shape[1]
        if self.index_type == "flat_ip":
            index = faiss.IndexFlatIP(d)
        elif self.index_type == "ivf_flat_ip":
            quantizer = faiss.IndexFlatIP(d)
            nlist = max(1, min(self.nlist, xb.shape[0]))
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            if not index.is_trained:
                index.train(xb)
            index.nprobe = max(1, self.nprobe)
        elif self.index_type == "hnsw_ip":
            index = faiss.IndexHNSWFlat(d, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            try:
                faiss.ParameterSpace().set_index_parameter(index, "efSearch", self.ef_search)
            except Exception:
                pass
        else:
            raise ValueError(f"Unsupported index_type: {self.index_type}")

        index.add(xb)
        # move to gpu if available
        index = self._to_gpu(index)
        self.index = index

    def search(self, xq: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("Index not built or loaded")
        self._import_faiss()
        xq = xq.astype(np.float32)
        xq = self.maybe_normalize(xq)
        D, I = self.index.search(xq, topk)
        return D, I

    def save(self, path: str) -> None:
        if self.index is None:
            raise RuntimeError("Index not built")
        faiss = self._import_faiss()
        try:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        except Exception:
            cpu_index = self.index
        faiss.write_index(cpu_index, path)

    def load(self, path: str, xb_dim: Optional[int] = None) -> bool:
        faiss = self._import_faiss()
        import os
        if not os.path.exists(path):
            return False
        index = faiss.read_index(path)
        index = self._to_gpu(index)
        self.index = index
        return True
