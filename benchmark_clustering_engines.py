#!/usr/bin/env python3
"""
三种聚类引擎性能对比脚本
测试 original, parallel, networkx 三种引擎的性能差异
"""

import time
import multiprocessing as mp
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path


def create_test_data(n_pairs: int = 10000, output_dir: str = "./outputs") -> None:
    """创建测试数据"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成随机相似对数据
    np.random.seed(42)
    pairs_data = {
        'i': np.random.randint(0, n_pairs//10, n_pairs),
        'j': np.random.randint(0, n_pairs//10, n_pairs),
        'ce_final': np.random.beta(2, 5, n_pairs)  # 偏向较低分数的分布
    }
    
    # 确保 i != j
    mask = pairs_data['i'] != pairs_data['j']
    pairs_df = pd.DataFrame({
        'i': pairs_data['i'][mask],
        'j': pairs_data['j'][mask], 
        'ce_final': pairs_data['ce_final'][mask]
    })
    
    pairs_df.to_parquet(f"{output_dir}/pair_scores.parquet")
    
    # 创建假的嵌入数据
    n_nodes = max(pairs_df['i'].max(), pairs_df['j'].max()) + 1
    emb_dim = 768
    
    for name in ['emb_a', 'emb_b', 'emb_c']:
        emb = np.random.randn(n_nodes, emb_dim).astype(np.float32)
        # 标准化
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        np.save(f"{output_dir}/{name}.npy", emb)
    
    print(f"创建测试数据完成：{len(pairs_df)} 个相似对，{n_nodes} 个节点")


def create_test_config(engine: str, output_dir: str) -> str:
    """创建测试配置文件"""
    config_content = f"""# 测试配置文件
pipeline:
  language: "zh"
  random_seed: 42
  output_dir: "{output_dir}"

data:
  input_path: "./data/raw/input.parquet"
  q_col: "question"
  a_col: "answer"

embeddings:
  models:
    a: "BAAI/bge-large-zh-v1.5"
    b: "moka-ai/m3e-large"
    c: "Alibaba-NLP/gte-large-zh"

consistency:
  cos_a: 0.875
  cos_b: 0.870
  cos_c: 0.870
  std_max: 0.04
  vote_2_of_3: true

rerank:
  thresholds:
    high: 0.83
    mid_low: 0.77

cluster:
  engine: "{engine}"
  method: "leiden"
  min_cluster_size: 2
  use_parallel: true
  n_jobs: -1
  resolution: 1.0
  
  center_constraints:
    coverage: 0.85
    mean: 0.85
    median: 0.845
    p10: 0.80
    pair_min_for_2_nodes: 0.86
  
  second_merge:
    enable: true
    ce_min: 0.81
    require_consistency_vote: true

observe:
  save_histograms: false
"""
    
    config_path = f"test_config_{engine}.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    return config_path


def test_engine(engine: str, config_path: str) -> dict:
    """测试单个引擎"""
    print(f"\n🧪 测试 {engine.upper()} 引擎...")
    
    start_time = time.time()
    success = True
    error_msg = ""
    
    try:
        from src.stages.stage4_cluster import run
        run(config_path)
        elapsed_time = time.time() - start_time
        print(f"✅ {engine.upper()} 引擎完成，耗时: {elapsed_time:.2f}秒")
    except Exception as e:
        elapsed_time = float('inf')
        success = False
        error_msg = str(e)
        print(f"❌ {engine.upper()} 引擎失败: {e}")
    
    return {
        'engine': engine,
        'success': success,
        'time': elapsed_time,
        'error': error_msg
    }


def benchmark_all_engines():
    """性能对比测试所有引擎"""
    print("🚀 三种聚类引擎性能对比测试")
    print("=" * 60)
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"使用临时目录: {temp_dir}")
        
        # 创建测试数据
        create_test_data(n_pairs=20000, output_dir=temp_dir)  # 2万个相似对
        
        print(f"CPU核心数: {mp.cpu_count()}")
        
        engines = ['original', 'parallel', 'networkx']
        results = []
        
        # 测试每个引擎
        for engine in engines:
            # 创建引擎专用的配置文件
            config_path = create_test_config(engine, temp_dir)
            
            try:
                result = test_engine(engine, config_path)
                results.append(result)
            finally:
                # 清理配置文件
                Path(config_path).unlink(missing_ok=True)
            
            # 清理输出文件，准备下一个测试
            clusters_file = Path(temp_dir) / "clusters.parquet"
            clusters_file.unlink(missing_ok=True)
    
    # 输出对比结果
    print("\n📈 性能对比结果:")
    print("=" * 60)
    
    successful_results = [r for r in results if r['success']]
    
    if len(successful_results) >= 2:
        # 按时间排序
        successful_results.sort(key=lambda x: x['time'])
        
        print("排名 | 引擎      | 耗时(秒) | 相对性能")
        print("-" * 45)
        
        base_time = successful_results[0]['time']
        for i, result in enumerate(successful_results):
            relative_perf = base_time / result['time'] if result['time'] > 0 else 1.0
            print(f"{i+1:2d}   | {result['engine']:8s} | {result['time']:7.2f} | {relative_perf:.2f}x")
        
        # 详细分析
        print("\n🔍 详细分析:")
        print("-" * 30)
        
        if len(successful_results) >= 3:
            fastest = successful_results[0]
            slowest = successful_results[-1]
            speedup = slowest['time'] / fastest['time']
            print(f"最快引擎: {fastest['engine'].upper()}")
            print(f"最慢引擎: {slowest['engine'].upper()}")
            print(f"性能差异: {speedup:.2f}x")
        
        # 推荐
        print("\n💡 推荐方案:")
        print("-" * 20)
        
        for result in successful_results:
            if result['engine'] == 'networkx':
                print(f"🥇 {result['engine'].upper()}: 功能最强，支持多种高级聚类算法")
            elif result['engine'] == 'parallel':
                print(f"⚡ {result['engine'].upper()}: 性能优化，无额外依赖")
            elif result['engine'] == 'original':
                print(f"🔧 {result['engine'].upper()}: 最小依赖，调试友好")
    
    # 显示失败的引擎
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        print("\n❌ 失败的引擎:")
        print("-" * 20)
        for result in failed_results:
            print(f"{result['engine'].upper()}: {result['error']}")
            if 'networkx' in result['error'].lower() or 'import' in result['error'].lower():
                print("  💡 提示: 可能需要安装 NetworkX 相关依赖")
                print("     pip install networkx python-igraph leidenalg")
    
    print("\n🎯 配置建议:")
    print("-" * 15)
    print("在 src/configs/config.yaml 中设置:")
    print("cluster:")
    print("  engine: \"networkx\"    # 默认推荐")
    print("  # engine: \"parallel\"   # 性能优先，无额外依赖")
    print("  # engine: \"original\"   # 最小依赖")


if __name__ == "__main__":
    benchmark_all_engines()

