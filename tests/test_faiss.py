#!/usr/bin/env python3
"""
测试FAISS GPU实现
"""

import numpy as np
from src.qa_clean.faiss_store import FAISSGPUStore


def test_faiss_gpu():
    """测试FAISS GPU的基本功能"""
    print("🧪 开始测试FAISS GPU...")
    
    # 创建存储实例
    store = FAISSGPUStore(embedding_dim=10, metric_type="cosine", gpu_id=0)
    
    # 准备测试数据
    texts = [
        "如何安装Python?",
        "Python怎么安装?",
        "什么是机器学习?",
        "机器学习是什么?",
        "如何学习编程?"
    ]
    
    # 生成随机向量（模拟嵌入）
    np.random.seed(42)
    embeddings_a = np.random.randn(len(texts), 10).astype(np.float32)
    embeddings_b = np.random.randn(len(texts), 10).astype(np.float32)
    
    # 插入向量
    print("📥 插入测试向量...")
    inserted_ids = store.insert_vectors(texts, embeddings_a, embeddings_b)
    print(f"✅ 插入了 {len(inserted_ids)} 个向量")
    
    # 测试搜索
    print("🔍 测试向量搜索...")
    query_vector = embeddings_a[0]  # 使用第一个向量作为查询
    
    results = store.search_similar(query_vector, topk=3, use_embedding="a")
    print(f"✅ 搜索到 {len(results)} 个相似结果")
    
    for i, (idx, similarity, metadata) in enumerate(results):
        print(f"  {i+1}. 相似度: {similarity:.3f}, 文本: {metadata['text_content']}")
    
    # 测试批量搜索
    print("🔍 测试批量搜索...")
    query_vectors = embeddings_a[:2]  # 使用前两个向量作为查询
    
    batch_results = store.batch_search(query_vectors, topk=2, use_embedding="a")
    print(f"✅ 批量搜索完成，每个查询返回 {len(batch_results[0])} 个结果")
    
    # 测试获取向量
    print("📋 测试获取向量...")
    vector_data = store.get_vector_by_id(0)
    if vector_data:
        print(f"✅ 成功获取向量: {vector_data['text_content']}")
    
    # 清空数据
    store.clear_all()
    print("✅ 数据清理完成")
    
    print("🎉 所有测试通过！")


if __name__ == "__main__":
    test_faiss_gpu()
