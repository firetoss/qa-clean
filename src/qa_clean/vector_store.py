"""
pgvector 向量存储管理器
"""

import numpy as np
import psycopg2
from typing import List, Tuple, Optional, Dict, Any
from psycopg2.extras import RealDictCursor


class PGVectorStore:
    """PostgreSQL + pgvector 向量存储管理器"""
    
    def __init__(self, connection_params: Dict[str, Any], table_name: str = "qa_vectors"):
        """
        初始化 pgvector 存储
        
        Args:
            connection_params: PostgreSQL 连接参数
            table_name: 向量表名
        """
        self.connection_params = connection_params
        self.table_name = table_name
        self.connection = None
        self._ensure_table_exists()
    
    def _ensure_table_exists(self) -> None:
        """确保向量表存在"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # 创建扩展（如果不存在）
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # 创建向量表
                create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    text_content TEXT NOT NULL,
                    embedding_a vector(1024),  -- bge-large-zh 维度
                    embedding_b vector(1024),  -- m3e-large 维度
                    metadata JSONB DEFAULT '{{}}'
                );
                """
                cur.execute(create_table_sql)
                
                # 创建向量索引
                cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding_a 
                ON {self.table_name} 
                USING ivfflat (embedding_a vector_cosine_ops) 
                WITH (lists = 100);
                """)
                
                cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding_b 
                ON {self.table_name} 
                USING ivfflat (embedding_b vector_cosine_ops) 
                WITH (lists = 100);
                """)
                
                conn.commit()
    
    def _get_connection(self):
        """获取数据库连接"""
        if self.connection is None or self.connection.closed:
            self.connection = psycopg2.connect(**self.connection_params)
        return self.connection
    
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
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                insert_sql = f"""
                INSERT INTO {self.table_name} (text_content, embedding_a, embedding_b, metadata)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
                """
                
                inserted_ids = []
                for text, emb_a, emb_b, meta in zip(texts, embeddings_a, embeddings_b, metadata):
                    cur.execute(insert_sql, (text, emb_a.tolist(), emb_b.tolist(), meta))
                    inserted_ids.append(cur.fetchone()[0])
                
                conn.commit()
                return inserted_ids
    
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
        embedding_col = f"embedding_{use_embedding}"
        
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if similarity_threshold is not None:
                    search_sql = f"""
                    SELECT id, text_content, metadata,
                           1 - ({embedding_col} <=> %s) as similarity
                    FROM {self.table_name}
                    WHERE 1 - ({embedding_col} <=> %s) >= %s
                    ORDER BY {embedding_col} <=> %s
                    LIMIT %s;
                    """
                    cur.execute(search_sql, (
                        query_vector.tolist(), 
                        query_vector.tolist(), 
                        similarity_threshold,
                        query_vector.tolist(), 
                        topk
                    ))
                else:
                    search_sql = f"""
                    SELECT id, text_content, metadata,
                           1 - ({embedding_col} <=> %s) as similarity
                    FROM {self.table_name}
                    ORDER BY {embedding_col} <=> %s
                    LIMIT %s;
                    """
                    cur.execute(search_sql, (
                        query_vector.tolist(), 
                        query_vector.tolist(), 
                        topk
                    ))
                
                results = []
                for row in cur.fetchall():
                    results.append((
                        row['id'],
                        float(row['similarity']),
                        dict(row)
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
        results = []
        for query_vec in query_vectors:
            query_results = self.search_similar(query_vec, topk, use_embedding)
            results.append(query_results)
        return results
    
    def get_vector_by_id(self, vector_id: int) -> Optional[Dict[str, Any]]:
        """根据ID获取向量数据"""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"""
                SELECT id, text_content, embedding_a, embedding_b, metadata
                FROM {self.table_name}
                WHERE id = %s;
                """, (vector_id,))
                
                row = cur.fetchone()
                if row:
                    return dict(row)
                return None
    
    def update_vector(
        self, 
        vector_id: int, 
        embedding_a: Optional[np.ndarray] = None,
        embedding_b: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """更新向量数据"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                update_parts = []
                params = []
                
                if embedding_a is not None:
                    update_parts.append("embedding_a = %s")
                    params.append(embedding_a.tolist())
                
                if embedding_b is not None:
                    update_parts.append("embedding_b = %s")
                    params.append(embedding_b.tolist())
                
                if metadata is not None:
                    update_parts.append("metadata = %s")
                    params.append(metadata)
                
                if not update_parts:
                    return False
                
                params.append(vector_id)
                update_sql = f"""
                UPDATE {self.table_name}
                SET {', '.join(update_parts)}
                WHERE id = %s;
                """
                
                cur.execute(update_sql, params)
                conn.commit()
                return cur.rowcount > 0
    
    def delete_vector(self, vector_id: int) -> bool:
        """删除向量数据"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self.table_name} WHERE id = %s;", (vector_id,))
                conn.commit()
                return cur.rowcount > 0
    
    def clear_all(self) -> None:
        """清空所有向量数据"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {self.table_name};")
                conn.commit()
    
    def close(self) -> None:
        """关闭数据库连接"""
        if self.connection and not self.connection.closed:
            self.connection.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PGVectorIndex:
    """兼容 FAISS 接口的 pgvector 包装器"""
    
    def __init__(self, vector_store: PGVectorStore, use_embedding: str = "a"):
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
        
        # 使用 pgvector 搜索
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
