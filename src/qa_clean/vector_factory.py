"""
向量存储工厂
支持选择 pgvector 或 FAISS GPU 实现
"""

from typing import Union, Optional
from .vector_store import PGVectorStore, PGVectorIndex
from .faiss_store import FAISSGPUStore, FAISSGPUIndex


def create_vector_store(
    store_type: str = "faiss_gpu",
    **kwargs
) -> Union[PGVectorStore, FAISSGPUStore]:
    """
    创建向量存储实例
    
    Args:
        store_type: 存储类型 ("pgvector" 或 "faiss_gpu")
        **kwargs: 其他参数
        
    Returns:
        向量存储实例
    """
    if store_type.lower() == "pgvector":
        if "connection_params" not in kwargs:
            raise ValueError("pgvector 类型需要 connection_params 参数")
        return PGVectorStore(**kwargs)
    
    elif store_type.lower() == "faiss_gpu":
        return FAISSGPUStore(**kwargs)
    
    else:
        raise ValueError(f"不支持的存储类型: {store_type}")


def create_vector_index(
    store_type: str = "faiss_gpu",
    vector_store: Optional[Union[PGVectorStore, FAISSGPUStore]] = None,
    use_embedding: str = "a",
    **kwargs
) -> Union[PGVectorIndex, FAISSGPUIndex]:
    """
    创建向量索引实例
    
    Args:
        store_type: 存储类型 ("pgvector" 或 "faiss_gpu")
        vector_store: 向量存储实例
        use_embedding: 使用哪个嵌入 ('a' 或 'b')
        **kwargs: 其他参数
        
    Returns:
        向量索引实例
    """
    if store_type.lower() == "pgvector":
        if vector_store is None:
            if "connection_params" not in kwargs:
                raise ValueError("pgvector 类型需要 connection_params 参数")
            vector_store = PGVectorStore(**kwargs)
        elif not isinstance(vector_store, PGVectorStore):
            raise TypeError("pgvector 类型需要 PGVectorStore 实例")
        return PGVectorIndex(vector_store, use_embedding)
    
    elif store_type.lower() == "faiss_gpu":
        if vector_store is None:
            vector_store = FAISSGPUStore(**kwargs)
        elif not isinstance(vector_store, FAISSGPUStore):
            raise TypeError("faiss_gpu 类型需要 FAISSGPUStore 实例")
        return FAISSGPUIndex(vector_store, use_embedding)
    
    else:
        raise ValueError(f"不支持的存储类型: {store_type}")


# 便捷函数
def get_vector_store(store_type: str = "faiss_gpu", **kwargs):
    """获取向量存储实例的便捷函数"""
    return create_vector_store(store_type, **kwargs)


def get_vector_index(store_type: str = "faiss_gpu", **kwargs):
    """获取向量索引实例的便捷函数"""
    return create_vector_index(store_type, **kwargs)
