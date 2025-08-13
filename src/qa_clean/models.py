"""
QA 数据模型和向量索引管理器
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from .vector_store import PGVectorStore, PGVectorIndex
from .faiss_store import FAISSGPUStore, FAISSGPUIndex


class DualEmbeddingModel:
    """双嵌入模型管理器"""
    
    def __init__(self, model_a_name: str, model_b_name: str, vector_store_type: str = "faiss_gpu", **kwargs):
        """
        初始化双嵌入模型
        
        Args:
            model_a_name: 模型A名称（如 bge-large-zh）
            model_b_name: 模型B名称（如 m3e-large）
            vector_store_type: 向量存储类型 (faiss_gpu, pgvector)
            **kwargs: 其他参数
        """
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name
        self.vector_store_type = vector_store_type
        
        # 初始化模型
        print(f"🔄 正在加载模型 A: {model_a_name}")
        self.model_a = SentenceTransformer(model_a_name)
        
        print(f"🔄 正在加载模型 B: {model_b_name}")
        self.model_b = SentenceTransformer(model_b_name)
        
        # 创建向量存储
        self._create_vector_store(**kwargs)
        
        print("✅ 双嵌入模型初始化完成")
    
    def _create_vector_store(self, **kwargs):
        """创建向量存储"""
        if self.vector_store_type == "pgvector":
            from .vector_store import PGVectorStore
            self.vector_store = PGVectorStore(**kwargs)
        elif self.vector_store_type == "faiss_gpu":
            self.vector_store = FAISSGPUStore(**kwargs)
        else:
            raise ValueError(f"不支持的向量存储类型: {self.vector_store_type}")
    
    def encode_texts(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        编码文本为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            (embeddings_a, embeddings_b) 元组
        """
        print(f"🔄 正在编码 {len(texts)} 个文本...")
        
        # 使用模型A编码
        embeddings_a = self.model_a.encode(texts, show_progress_bar=True)
        
        # 使用模型B编码
        embeddings_b = self.model_b.encode(texts, show_progress_bar=True)
        
        print(f"✅ 编码完成: A={embeddings_a.shape}, B={embeddings_b.shape}")
        return embeddings_a, embeddings_b
    
    def insert_texts(self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        """
        插入文本到向量存储
        
        Args:
            texts: 文本列表
            metadata: 元数据列表
            
        Returns:
            插入的ID列表
        """
        # 编码文本
        embeddings_a, embeddings_b = self.encode_texts(texts)
        
        # 插入到向量存储
        return self.vector_store.insert_vectors(texts, embeddings_a, embeddings_b, metadata)
    
    def search_similar(self, query: str, topk: int = 100, use_embedding: str = "a") -> List[Dict[str, Any]]:
        """
        搜索相似文本
        
        Args:
            query: 查询文本
            topk: 返回结果数量
            use_embedding: 使用哪个嵌入 ('a' 或 'b')
            
        Returns:
            相似文本列表
        """
        # 编码查询文本
        if use_embedding == "a":
            query_vector = self.model_a.encode([query])[0]
        else:
            query_vector = self.model_b.encode([query])[0]
        
        # 搜索相似向量
        results = self.vector_store.search_similar(query_vector, topk, use_embedding)
        
        # 转换为字典格式
        return [result[2] for result in results]
    
    def batch_search(self, queries: List[str], topk: int = 100, use_embedding: str = "a") -> List[List[Dict[str, Any]]]:
        """
        批量搜索相似文本
        
        Args:
            queries: 查询文本列表
            topk: 每个查询返回结果数量
            use_embedding: 使用哪个嵌入
            
        Returns:
            每个查询的结果列表
        """
        # 编码查询文本
        if use_embedding == "a":
            query_vectors = self.model_a.encode(queries)
        else:
            query_vectors = self.model_b.encode(queries)
        
        # 批量搜索
        batch_results = self.vector_store.batch_search(query_vectors, topk, use_embedding)
        
        # 转换为字典格式
        return [[result[2] for result in query_results] for query_results in batch_results]
    
    def get_vector_index(self, use_embedding: str = "a"):
        """获取向量索引（兼容FAISS接口）"""
        if self.vector_store_type == "pgvector":
            return PGVectorIndex(self.vector_store, use_embedding)
        elif self.vector_store_type == "faiss_gpu":
            return FAISSGPUIndex(self.vector_store, use_embedding)
        else:
            raise ValueError(f"不支持的向量存储类型: {self.vector_store_type}")
    
    def close(self):
        """释放资源"""
        if hasattr(self.vector_store, 'close'):
            self.vector_store.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PGVectorIndexManager:
    """pgvector 索引管理器（兼容 FAISS 接口）"""
    
    def __init__(self, vector_store: PGVectorStore, use_embedding: str = "a"):
        self.vector_store = vector_store
        self.use_embedding = use_embedding
    
    def add(self, vectors: np.ndarray) -> None:
        """添加向量（兼容 FAISS 接口）"""
        # 这里主要是为了兼容 FAISS 接口
        pass
    
    def search(self, queries: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        """搜索（兼容 FAISS 接口）"""
        # 转换为 FAISS 兼容格式
        batch_results = self.vector_store.batch_search(queries, topk, self.use_embedding)
        
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
        pass
