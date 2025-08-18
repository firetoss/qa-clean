#!/usr/bin/env python3
"""
聚类性能对比脚本
比较串行版本和并行版本的性能差异
"""

import time
import multiprocessing as mp
import pandas as pd
import numpy as np
from src.stages.stage4_cluster_original import run as run_original
from src.stages.stage4_cluster_parallel import run as run_parallel
from src.stages.stage4_cluster_networkx import run as run_networkx
from src.stages.stage4_cluster import run as run_unified

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

def benchmark_clustering():
    """三种聚类引擎性能对比测试"""
    print("🚀 三种聚类引擎性能对比测试")
    print("=" * 60)
    
    # 创建测试数据
    create_test_data(n_pairs=50000)  # 5万个相似对
    
    config_path = "src/configs/config.yaml"
    
    print(f"CPU核心数: {mp.cpu_count()}")
    print()
    
    results = {}
    
    # 测试原始版本
    print("📊 测试Original引擎（原始单核）...")
    start_time = time.time()
    try:
        run_original(config_path)
        original_time = time.time() - start_time
        results['original'] = original_time
        print(f"✅ Original引擎完成，耗时: {original_time:.2f}秒")
    except Exception as e:
        print(f"❌ Original引擎失败: {e}")
        results['original'] = float('inf')
    
    print()
    
    # 测试并行版本
    print("⚡ 测试Parallel引擎（多核并行）...")
    start_time = time.time()
    try:
        run_parallel(config_path)
        parallel_time = time.time() - start_time
        results['parallel'] = parallel_time
        print(f"✅ Parallel引擎完成，耗时: {parallel_time:.2f}秒")
    except Exception as e:
        print(f"❌ Parallel引擎失败: {e}")
        results['parallel'] = float('inf')
    
    print()
    
    # 测试NetworkX版本
    print("🌐 测试NetworkX引擎（高级算法）...")
    start_time = time.time()
    try:
        run_networkx(config_path)
        networkx_time = time.time() - start_time
        results['networkx'] = networkx_time
        print(f"✅ NetworkX引擎完成，耗时: {networkx_time:.2f}秒")
    except Exception as e:
        print(f"❌ NetworkX引擎失败: {e}")
        results['networkx'] = float('inf')
    
    print()
    
    # 性能对比分析
    print("📈 性能对比结果:")
    print("=" * 60)
    
    valid_results = {k: v for k, v in results.items() if v != float('inf')}
    
    if len(valid_results) >= 2:
        baseline = min(valid_results.values())
        
        print(f"{'引擎':<12} {'耗时(秒)':<10} {'相对性能':<12} {'推荐场景'}")
        print("-" * 60)
        
        for engine, time_val in results.items():
            if time_val == float('inf'):
                relative = "失败"
                scenario = "依赖缺失"
            else:
                speedup = baseline / time_val
                if speedup >= 1.0:
                    relative = f"{speedup:.2f}x"
                else:
                    relative = f"1/{1/speedup:.2f}x"
                
                if engine == 'original':
                    scenario = "小数据集，最小依赖"
                elif engine == 'parallel':
                    scenario = "中大数据集，性能优先"
                else:  # networkx
                    scenario = "质量优先，生产环境"
            
            print(f"{engine:<12} {time_val if time_val != float('inf') else 'N/A':<10} {relative:<12} {scenario}")
        
        print()
        
        # 找出最佳引擎
        best_engine = min(valid_results.keys(), key=lambda k: valid_results[k])
        best_time = valid_results[best_engine]
        
        print(f"🏆 最佳性能: {best_engine} 引擎 ({best_time:.2f}秒)")
        
        if len(valid_results) >= 2:
            sorted_results = sorted(valid_results.items(), key=lambda x: x[1])
            if len(sorted_results) >= 2:
                second_best = sorted_results[1]
                speedup = second_best[1] / best_time
                print(f"📊 相比第二名提升: {speedup:.2f}x")
    
    print()
    print("💡 选择建议:")
    print("- 小数据集(<1000节点): original引擎，简单可靠")
    print("- 中等数据集(1000-10000节点): parallel引擎，性能优秀")  
    print("- 大数据集(>10000节点): networkx引擎，质量最佳")
    print("- 生产环境: networkx引擎，功能丰富")
    print()
    print("🔧 配置方法:")
    print("在 src/configs/config.yaml 中设置:")
    print('cluster:')
    print('  engine: "networkx"  # 或 "parallel" 或 "original"')

if __name__ == "__main__":
    benchmark_clustering()
