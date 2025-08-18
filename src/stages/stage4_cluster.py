"""
Stage4 聚类模块 - 统一接口

本模块提供三种聚类引擎的统一接口，根据配置自动选择合适的聚类算法：
- NetworkX引擎：高质量社区发现，支持Leiden/Louvain算法
- 并行引擎：多核优化的连通分量算法，无额外依赖
- 原始引擎：最小依赖的单核连通分量算法

引擎选择策略：
1. 配置指定引擎，若依赖缺失则自动回退
2. NetworkX引擎缺失时自动回退到并行引擎
3. 所有引擎统一输入输出接口，保证兼容性
"""

from __future__ import annotations

import argparse
from typing import Optional
import warnings

from ..utils.config import load_config

# 忽略NetworkX模块可能产生的用户警告
warnings.filterwarnings('ignore', category=UserWarning, module='networkx')


def run(cfg_path: str, input_file: str = None, n_jobs: Optional[int] = None) -> None:
    """
    Stage4 聚类统一入口函数
    
    根据配置文件中的 cluster.engine 选择不同的聚类引擎实现。
    支持自动依赖检测和回退机制，确保在依赖缺失时仍能正常工作。
    
    Args:
        cfg_path: 配置文件路径，包含聚类引擎和参数设置
        input_file: 输入数据文件路径（可选，覆盖配置文件设置）
        n_jobs: 并行进程数（可选，-1表示使用所有CPU核心）
        
    支持的引擎类型：
        - 'networkx': 使用NetworkX库的高级聚类算法
          * 支持算法：Leiden、Louvain、连通分量
          * 依赖：networkx, python-igraph, leidenalg
          * 特点：最高质量的社区发现
          
        - 'parallel': 自实现的多核并行连通分量算法
          * 支持算法：连通分量（并行优化）
          * 依赖：无额外依赖
          * 特点：高性能，内存友好
          
        - 'original': 原始单核连通分量算法
          * 支持算法：连通分量（串行）
          * 依赖：无额外依赖
          * 特点：最小依赖，调试友好
          
    异常处理：
        - ImportError: NetworkX引擎依赖缺失时自动回退到并行引擎
        - ValueError: 不支持的引擎类型
        
    输出：
        在配置指定的输出目录生成 clusters.parquet 文件
    """
    cfg = load_config(cfg_path)
    engine = cfg.get('cluster.engine', 'networkx').lower()
    
    print(f"[stage4] 聚类引擎选择: {engine}")
    
    # NetworkX引擎：高质量社区发现，支持多种算法
    if engine == 'networkx':
        try:
            from .stage4_cluster_networkx import run as run_networkx
            print("[stage4] 加载NetworkX聚类引擎")
            run_networkx(cfg_path, input_file, n_jobs)
        except ImportError as e:
            print(f"[stage4] NetworkX引擎依赖缺失: {e}")
            print("[stage4] 自动回退到并行引擎")
            from .stage4_cluster_parallel import run as run_parallel
            run_parallel(cfg_path, input_file, n_jobs)
            
    # 并行引擎：多核优化的连通分量算法
    elif engine == 'parallel':
        from .stage4_cluster_parallel import run as run_parallel
        print("[stage4] 加载并行聚类引擎")
        run_parallel(cfg_path, input_file, n_jobs)
        
    # 原始引擎：最小依赖的单核连通分量算法
    elif engine == 'original':
        from .stage4_cluster_original import run as run_original
        print("[stage4] 加载原始聚类引擎")
        run_original(cfg_path, input_file)
        
    else:
        supported_engines = ['networkx', 'parallel', 'original']
        raise ValueError(
            f"不支持的聚类引擎: {engine}. "
            f"支持的引擎: {', '.join(supported_engines)}"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stage4 聚类统一入口 - 支持NetworkX/并行/原始三种引擎',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
引擎选择说明：
  networkx  - 高质量社区发现，支持Leiden/Louvain算法（推荐）
  parallel  - 多核并行连通分量算法，无额外依赖（高性能）
  original  - 单核连通分量算法，最小依赖（兼容性）

配置示例：
  在 config.yaml 中设置:
    cluster:
      engine: "networkx"    # 或 "parallel", "original"
      method: "leiden"      # NetworkX引擎支持
      n_jobs: -1           # 使用所有CPU核心
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='src/configs/config.yaml',
        help='配置文件路径 (默认: %(default)s)'
    )
    parser.add_argument(
        '--input', '-i',
        help='输入数据文件路径（覆盖配置文件中的设置）'
    )
    parser.add_argument(
        '--n-jobs', '-j',
        type=int,
        help='并行进程数（-1表示使用所有CPU核心，0表示串行）'
    )
    
    args = parser.parse_args()
    run(args.config, args.input, args.n_jobs)