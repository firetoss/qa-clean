"""
FAISS GPU 向量存储管理器
支持 FAISS 1.7.2 GPU 版本
"""

import numpy as np
import faiss
from typing import List, Tuple, Optional, Dict, Any
import os


class FAISSGPUStore:
    """FAISS GPU 向量存储管理器"""
    
    def __init__(self, embedding_dim: int = 1024, metric_type: str = "cosine", gpu_id: int = 0):
        """
        初始化 FAISS GPU 存储
        
        Args:
            embedding_dim: 向量维度
            metric_type: 度量类型 ("cosine" 或 "l2")
            gpu_id: GPU设备ID
        """
        self.embedding_dim = embedding_dim
        self.metric_type = metric_type
        self.gpu_id = gpu_id
        self.index = None
        self.gpu_resource = None
        self.texts = []
        self.metadata = []
        
        # 设置GPU资源
        self._setup_gpu()
        # 创建索引
        self._create_index()
    
    def _setup_gpu(self) -> None:
        """设置GPU资源"""
        try:
            # 检查GPU可用性
            ngpus = faiss.get_num_gpus()
            if ngpus == 0:
                print("⚠️  未检测到可用的GPU，将使用CPU模式")
                self.gpu_resource = None
                return
            
            if self.gpu_id >= ngpus:
                print(f"⚠️  GPU ID {self.gpu_id} 超出范围，使用 GPU 0")
                self.gpu_id = 0
            
            # 配置GPU资源
            self.gpu_resource = faiss.StandardGpuResources()
            print(f"✅ 成功初始化 GPU {self.gpu_id} 资源")
            
        except Exception as e:
            print(f"⚠️  GPU 初始化失败: {e}，将使用 CPU 模式")
            self.gpu_resource = None
    
    def _create_index(self) -> None:
        """创建 FAISS 索引"""
        if self.metric_type == "cosine":
            # 使用内积索引，向量需要归一化
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.metric_type == "l2":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            raise ValueError(f"不支持的度量类型: {self.metric_type}")
        
        # 转换为GPU索引
        if self.gpu_resource is not None:
            self.index = faiss.index_cpu_to_gpu(
                self.gpu_resource, 
                self.gpu_id, 
                self.index
            )
            print(f"✅ 索引已转移到 GPU {self.gpu_id}")
        else:
            print("✅ 使用 CPU 索引")
        
        print(f"✅ 成功创建 FAISS {self.metric_type} 索引")
    
    def insert_vectors(
        self, 
        texts: List[str], 
        embeddings_a: np.ndarray, 
        embeddings_b: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[int]:
        """
        插入向量数据
        
        Args:
            texts: 文本列表
            embeddings_a: 嵌入A向量
            embeddings_b: 嵌入B向量
            metadata: 元数据列表
            
        Returns:
            插入的ID列表
        """
        if metadata is None:
            metadata = [{} for _ in texts]
        
        # 存储文本和元数据
        start_id = len(self.texts)
        self.texts.extend(texts)
        self.metadata.extend(metadata)
        
        # 归一化向量（用于余弦相似度）
        if self.metric_type == "cosine":
            embeddings_a_norm = self._normalize_vectors(embeddings_a)
            embeddings_b_norm = self._normalize_vectors(embeddings_b)
        else:
            embeddings_a_norm = embeddings_a
            embeddings_b_norm = embeddings_b
        
        # 插入向量A
        if embeddings_a_norm.shape[0] > 0:
            self.index.add(embeddings_a_norm.astype(np.float32))
        
        # 插入向量B（如果维度相同，可以合并）
        if embeddings_b_norm.shape[0] > 0 and embeddings_b_norm.shape[1] == embeddings_a_norm.shape[1]:
            self.index.add(embeddings_b_norm.astype(np.float32))
        
        # 返回插入的ID
        inserted_ids = list(range(start_id, start_id + len(texts)))
        print(f"✅ 成功插入 {len(texts)} 个向量到 FAISS {'GPU' if self.gpu_resource else 'CPU'} 索引")
        
        return inserted_ids
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """归一化向量（用于余弦相似度）"""
        if len(vectors) == 0:
            return vectors
        
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # 避免除零
        return vectors / norms
    
    def search_similar(
        self, 
        query_vector: np.ndarray, 
        topk: int = 100, 
        use_embedding: str = "a",
        similarity_threshold: Optional[float] = None
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        搜索相似向量
        
        Args:
            query_vector: 查询向量
            topk: 返回结果数量
            use_embedding: 使用哪个嵌入 ('a' 或 'b')
            similarity_threshold: 相似度阈值
            
        Returns:
            (id, similarity_score, metadata) 列表
        """
        if self.index is None or len(self.texts) == 0:
            return []
        
        # 归一化查询向量（如果需要）
        if self.metric_type == "cosine":
            query_norm = self._normalize_vectors(query_vector.reshape(1, -1))
        else:
            query_norm = query_vector.reshape(1, -1)
        
        # 搜索
        similarities, indices = self.index.search(
            query_norm.astype(np.float32), 
            min(topk, len(self.texts))
        )
        
        # 处理结果
        results = []
        for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < 0 or idx >= len(self.texts):
                continue
            
            # 应用相似度阈值
            if similarity_threshold is not None and sim < similarity_threshold:
                continue
            
            results.append((
                int(idx),
                float(sim),
                {
                    'id': int(idx),
                    'text_content': self.texts[idx],
                    'metadata': self.metadata[idx],
                    'similarity': float(sim)
                }
            ))
        
        return results
    
    def batch_search(
        self, 
        query_vectors: np.ndarray, 
        topk: int = 100, 
        use_embedding: str = "a"
    ) -> List[List[Tuple[int, float, Dict[str, Any]]]]:
        """
        批量搜索相似向量
        
        Args:
            query_vectors: 查询向量列表
            topk: 每个查询返回结果数量
            use_embedding: 使用哪个嵌入
            
        Returns:
            每个查询的结果列表
        """
        if self.index is None or len(self.texts) == 0:
            return [[] for _ in range(len(query_vectors))]
        
        # 归一化查询向量（如果需要）
        if self.metric_type == "cosine":
            query_norms = self._normalize_vectors(query_vectors)
        else:
            query_norms = query_vectors
        
        # 批量搜索
        similarities, indices = self.index.search(
            query_norms.astype(np.float32), 
            min(topk, len(self.texts))
        )
        
        # 处理结果
        results = []
        for query_idx in range(len(query_vectors)):
            query_results = []
            for i, (sim, idx) in enumerate(zip(similarities[query_idx], indices[query_idx])):
                if idx < 0 or idx >= len(self.texts):
                    continue
                
                query_results.append((
                    int(idx),
                    float(sim),
                    {
                        'id': int(idx),
                        'text_content': self.texts[idx],
                        'metadata': self.metadata[idx],
                        'similarity': float(sim)
                    }
                ))
            
            results.append(query_results)
        
        return results
    
    def get_vector_by_id(self, vector_id: int) -> Optional[Dict[str, Any]]:
        """根据ID获取向量数据"""
        if vector_id < 0 or vector_id >= len(self.texts):
            return None
        
        return {
            'id': vector_id,
            'text_content': self.texts[vector_id],
            'metadata': self.metadata[vector_id]
        }
    
    def clear_all(self) -> None:
        """清空所有向量数据"""
        if self.index is not None:
            self.index.reset()
        self.texts.clear()
        self.metadata.clear()
        print("✅ 已清空所有 FAISS 向量数据")
    
    def close(self) -> None:
        """释放资源"""
        if self.gpu_resource is not None:
            del self.gpu_resource
            self.gpu_resource = None
        
        if self.index is not None:
            del self.index
            self.index = None
        
        print("✅ 已释放 FAISS GPU 资源")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class FAISSGPUIndex:
    """兼容 FAISS 接口的 GPU 索引包装器"""
    
    def __init__(self, vector_store: FAISSGPUStore, use_embedding: str = "a"):
        self.vector_store = vector_store
        self.use_embedding = use_embedding
        self.vectors = None
        self.ids = None
    
    def add(self, vectors: np.ndarray) -> None:
        """添加向量（兼容 FAISS 接口）"""
        # 这里需要先有文本内容，暂时存储向量
        self.vectors = vectors
        # 实际插入需要在有文本内容时调用 vector_store.insert_vectors
    
    def search(self, queries: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        """搜索（兼容 FAISS 接口）"""
        if self.vectors is None:
            raise ValueError("向量未添加，请先调用 add 方法")
        
        # 使用 FAISS GPU 搜索
        batch_results = self.vector_store.batch_search(queries, topk, self.use_embedding)
        
        # 转换为 FAISS 兼容格式
        similarities = []
        indices = []
        
        for query_results in batch_results:
            query_sims = []
            query_indices = []
            
            for result_id, similarity, _ in query_results:
                query_sims.append(similarity)
                query_indices.append(result_id)
            
            # 填充到 topk
            while len(query_sims) < topk:
                query_sims.append(0.0)
                query_indices.append(-1)
            
            similarities.append(query_sims[:topk])
            indices.append(query_indices[:topk])
        
        return np.array(similarities), np.array(indices)
    
    def reset(self) -> None:
        """重置索引"""
        if self.vector_store.index is not None:
            self.vector_store.index.reset()
        self.vectors = None
        self.ids = None
