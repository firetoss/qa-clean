"""
FAISS向量召回提供者
支持三种索引类型：flat_ip(精确)、ivf_flat_ip(大规模)、hnsw_ip(平衡)
自动处理GPU/CPU切换和索引持久化
"""
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
        """
        智能导入FAISS库，优先GPU版本，失败则降级CPU
        """
        if self._faiss is not None:
            return self._faiss
        
        try:
            # 优先尝试GPU版本
            if self.device == "cuda":
                import faiss
                if hasattr(faiss, "StandardGpuResources"):
                    self._faiss = faiss
                    return self._faiss
        except Exception:
            pass
        
        # 降级到CPU版本
        import faiss  # CPU fallback
        self._faiss = faiss
        return self._faiss

    def _to_gpu(self, index):
        """
        尝试将索引转移到GPU，失败则保持CPU版本
        """
        faiss = self._import_faiss()
        if self.device != "cuda" or not hasattr(faiss, "StandardGpuResources"):
            return index
        
        try:
            if self._gpu_res is None:
                self._gpu_res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(self._gpu_res, 0, index)
        except Exception:
            # 静默降级到CPU，避免中断流程
            pass
        return index

    def build(self, xb: np.ndarray) -> None:
        """
        构建FAISS索引
        
        Args:
            xb: 基础向量集 (N, D)，将自动转换为float32并可选归一化
        """
        faiss = self._import_faiss()
        xb = xb.astype(np.float32)
        xb = self.maybe_normalize(xb)
        d = xb.shape[1]
        
        # 根据索引类型构建不同索引
        if self.index_type == "flat_ip":
            # 精确检索，内积距离（归一化后等价余弦）
            index = faiss.IndexFlatIP(d)
            
        elif self.index_type == "ivf_flat_ip":
            # 倒排文件+精确检索，适合大规模数据
            quantizer = faiss.IndexFlatIP(d)
            nlist = max(1, min(self.nlist, xb.shape[0]))  # 确保nlist不超过数据量
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            
            # 训练聚类中心
            if not index.is_trained:
                index.train(xb)
            index.nprobe = max(1, min(self.nprobe, nlist))  # 确保nprobe不超过nlist
            
        elif self.index_type == "hnsw_ip":
            # 分层图检索，精度与速度平衡
            index = faiss.IndexHNSWFlat(d, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            try:
                # 设置搜索参数
                faiss.ParameterSpace().set_index_parameter(index, "efSearch", self.ef_search)
            except Exception:
                # 参数设置失败不影响索引构建
                pass
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}，支持：flat_ip, ivf_flat_ip, hnsw_ip")

        # 添加向量到索引
        index.add(xb)
        
        # 尝试转移到GPU（如果可用）
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
