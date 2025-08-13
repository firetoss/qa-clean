"""
QA 数据处理器
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from .models import DualEmbeddingModel
from .vector_factory import create_vector_store, create_vector_index


class QAProcessor:
    """QA 数据处理器"""
    
    def __init__(self, topk: int, vector_store_type: str = "faiss_gpu", **kwargs):
        """
        初始化处理器
        
        Args:
            topk: 相似度搜索的top-k值
            vector_store_type: 向量存储类型 (faiss_gpu, pgvector)
            **kwargs: 其他参数
        """
        self.topk = topk
        self.vector_store_type = vector_store_type
        
        # 创建向量存储
        self.vector_store = create_vector_store(vector_store_type, **kwargs)
        
        # 初始化双嵌入模型
        self.embedding_model = DualEmbeddingModel(
            model_a_name="BAAI/bge-large-zh-v1.5",
            model_b_name="moka-ai/m3e-large",
            vector_store_type=vector_store_type,
            **kwargs
        )
        
        print(f"✅ QA 处理器初始化完成，使用 {vector_store_type} 存储")
    
    def process_qa_data(self, qa_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        处理QA数据
        
        Args:
            qa_data: QA数据列表
            
        Returns:
            处理结果
        """
        print(f"🔄 开始处理 {len(qa_data)} 条QA数据...")
        
        # 提取文本
        texts = [item.get('question', '') for item in qa_data]
        
        # 插入到向量存储
        inserted_ids = self.embedding_model.insert_texts(texts, qa_data)
        
        # 执行聚类
        clusters = self._perform_clustering(texts)
        
        # 生成代表问题
        representative_questions = self._generate_representative_questions(texts, clusters)
        
        # 去重结果
        dedup_results = self._deduplicate_qa_data(qa_data, clusters)
        
        return {
            'total_count': len(qa_data),
            'unique_count': len(dedup_results),
            'clusters': clusters,
            'representative_questions': representative_questions,
            'dedup_results': dedup_results
        }
    
    def _perform_clustering(self, texts: List[str], eps: float = 0.3, min_samples: int = 2) -> List[int]:
        """
        执行文本聚类
        
        Args:
            texts: 文本列表
            eps: DBSCAN的eps参数
            min_samples: DBSCAN的min_samples参数
            
        Returns:
            聚类标签列表
        """
        print("🔄 正在执行文本聚类...")
        
        # 使用模型A进行聚类
        embeddings = self.embedding_model.model_a.encode(texts)
        
        # 归一化向量
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # 执行DBSCAN聚类
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = clustering.fit_predict(embeddings_norm)
        
        # 统计聚类结果
        unique_clusters = set(cluster_labels)
        noise_count = list(cluster_labels).count(-1)
        
        print(f"✅ 聚类完成: {len(unique_clusters) - (1 if -1 in unique_clusters else 0)} 个聚类，{noise_count} 个噪声点")
        
        return cluster_labels.tolist()
    
    def _generate_representative_questions(self, texts: List[str], clusters: List[int]) -> Dict[int, str]:
        """
        为每个聚类生成代表问题
        
        Args:
            texts: 文本列表
            clusters: 聚类标签列表
            
        Returns:
            聚类ID到代表问题的映射
        """
        print("🔄 正在生成代表问题...")
        
        representative_questions = {}
        unique_clusters = set(clusters)
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # 跳过噪声点
                continue
            
            # 获取该聚类的所有文本
            cluster_texts = [texts[i] for i, label in enumerate(clusters) if label == cluster_id]
            
            if len(cluster_texts) == 1:
                # 单个文本直接作为代表
                representative_questions[cluster_id] = cluster_texts[0]
            else:
                # 多个文本选择最短的作为代表
                representative_questions[cluster_id] = min(cluster_texts, key=len)
        
        print(f"✅ 生成了 {len(representative_questions)} 个代表问题")
        return representative_questions
    
    def _deduplicate_qa_data(self, qa_data: List[Dict[str, Any]], clusters: List[int]) -> List[Dict[str, Any]]:
        """
        基于聚类结果去重QA数据
        
        Args:
            qa_data: 原始QA数据
            clusters: 聚类标签列表
            
        Returns:
            去重后的QA数据
        """
        print("🔄 正在执行数据去重...")
        
        # 按聚类分组
        cluster_groups = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(i)
        
        # 为每个聚类选择代表数据
        dedup_results = []
        for cluster_id, indices in cluster_groups.items():
            if cluster_id == -1:  # 噪声点单独保留
                for idx in indices:
                    dedup_results.append(qa_data[idx])
            else:
                # 选择第一个作为代表
                dedup_results.append(qa_data[indices[0]])
        
        print(f"✅ 去重完成: {len(qa_data)} -> {len(dedup_results)}")
        return dedup_results
    
    def search_similar_questions(self, query: str, topk: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        搜索相似问题
        
        Args:
            query: 查询问题
            topk: 返回结果数量
            
        Returns:
            相似问题列表
        """
        if topk is None:
            topk = self.topk
        
        return self.embedding_model.search_similar(query, topk)
    
    def batch_search_similar(self, queries: List[str], topk: Optional[int] = None) -> List[List[Dict[str, Any]]]:
        """
        批量搜索相似问题
        
        Args:
            queries: 查询问题列表
            topk: 每个查询返回结果数量
            
        Returns:
            每个查询的结果列表
        """
        if topk is None:
            topk = self.topk
        
        return self.embedding_model.batch_search(queries, topk)
    
    def close(self):
        """释放资源"""
        if hasattr(self.embedding_model, 'close'):
            self.embedding_model.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
