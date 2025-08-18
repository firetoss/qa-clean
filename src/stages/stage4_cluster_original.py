"""
Stage4 聚类模块 - 原始引擎实现

原始单核连通分量聚类算法，提供最小依赖和最高兼容性的聚类实现。
特点：
- 无额外依赖：仅使用标准库和基础科学计算包
- 串行实现：单线程执行，内存占用最小
- 调试友好：代码简洁清晰，易于理解和调试
- 高兼容性：适用于各种环境和小规模数据集

算法流程：
1. 基于CE分数构建无向图
2. 使用DFS/BFS查找连通分量
3. 验证簇质量和中心点约束
4. 执行二次聚合优化
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils.config import ensure_output_dir, load_config
from ..utils.io_utils import write_parquet
from ..utils.metrics import StatsRecorder


def build_graph(pairs_df: pd.DataFrame, threshold: float) -> Dict[int, List[Tuple[int, float]]]:
    """
    构建基于CE分数的无向图
    
    Args:
        pairs_df: 包含节点对和CE分数的DataFrame，必须包含列 ['i', 'j', 'ce_final']
        threshold: CE分数阈值，只有分数不低于此值的边才会被加入图中
        
    Returns:
        Dict[int, List[Tuple[int, float]]]: 邻接表表示的无向图
        键为节点ID，值为邻居节点及其边权重的列表
        
    注意：
        - 图是无向的，每条边会在两个节点的邻接表中都出现
        - 边权重即为CE分数，用于后续的中心点选择和验证
    """
    g: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for i, j, s in zip(pairs_df['i'], pairs_df['j'], pairs_df['ce_final']):
        if s >= threshold:
            g[int(i)].append((int(j), float(s)))
            g[int(j)].append((int(i), float(s)))
    return g


def connected_components(g: Dict[int, List[Tuple[int, float]]]) -> List[List[int]]:
    """
    使用深度优先搜索查找图的所有连通分量
    
    Args:
        g: 邻接表表示的无向图
        
    Returns:
        List[List[int]]: 连通分量列表，每个分量是节点ID的排序列表
        
    算法复杂度：
        时间复杂度: O(V + E)，其中V是节点数，E是边数
        空间复杂度: O(V)，用于访问标记和递归栈
        
    实现细节：
        - 使用栈实现的迭代版本DFS，避免递归深度限制
        - 对每个连通分量的节点进行排序，确保结果的确定性
    """
    vis = set()
    comps: List[List[int]] = []
    for u in g.keys():
        if u in vis:
            continue
        cur = [u]
        vis.add(u)
        q = [u]
        while q:
            x = q.pop()
            for v, _ in g.get(x, []):
                if v not in vis:
                    vis.add(v)
                    q.append(v)
                    cur.append(v)
        comps.append(sorted(cur))
    return comps


def choose_center(nodes: List[int], g: Dict[int, List[Tuple[int, float]]]) -> int:
    """
    为给定节点集合选择最佳中心点
    
    Args:
        nodes: 节点ID列表，代表一个连通分量或簇
        g: 邻接表表示的图，包含边权重信息
        
    Returns:
        int: 选中的中心点节点ID
        
    选择策略：
        选择与其他节点平均边权重最高的节点作为中心点。
        这样的中心点通常具有更好的代表性和连接性。
        
    注意：
        - 如果节点没有出边或权重为空，其得分为0
        - 对于单节点的情况，直接返回该节点
    """
    best, best_score = nodes[0], -1.0
    for u in nodes:
        ws = [w for v, w in g.get(u, []) if v in nodes and v != u]
        m = float(np.mean(ws)) if ws else 0.0
        if m > best_score:
            best_score = m
            best = u
    return best


def center_metrics(center: int, nodes: List[int], g: Dict[int, List[Tuple[int, float]]]) -> Dict[str, float]:
    """
    计算中心点的质量指标
    
    Args:
        center: 中心点节点ID
        nodes: 簇中所有节点ID列表
        g: 邻接表表示的图
        
    Returns:
        Dict[str, float]: 包含以下指标的字典
        - coverage: 覆盖率，中心点连接的节点数占簇大小的比例
        - mean: 平均权重，中心点到其他节点的平均边权重
        - median: 中位权重，边权重的中位数
        - p10: 10分位权重，最低10%的边权重
        
    用途：
        这些指标用于验证簇的质量，确保中心点具有足够的连接性和权重。
        所有指标都应满足配置文件中定义的最小阈值。
    """
    ws = [w for v, w in g.get(center, []) if v in nodes and v != center]
    if not ws:
        return {'coverage': 0.0, 'mean': 0.0, 'median': 0.0, 'p10': 0.0}
    arr = np.asarray(ws, dtype=float)
    # coverage: 中心点实际连接的节点数 / 簇中其他节点数
    coverage = float(len(ws) / max(1, len(nodes) - 1))
    return {
        'coverage': coverage,
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'p10': float(np.percentile(arr, 10)),
    }


def validate_cluster(nodes: List[int], g: Dict[int, List[Tuple[int, float]]], cons: Dict[str, float]) -> Tuple[bool, int, Dict[str, float]]:
    """
    验证簇的质量并返回中心点和指标
    
    Args:
        nodes: 簇中的节点ID列表
        g: 邻接表表示的图
        cons: 中心点约束配置，包含coverage、mean、median、p10等阈值
        
    Returns:
        Tuple[bool, int, Dict[str, float]]: 
        - bool: 是否通过验证
        - int: 选中的中心点ID
        - Dict[str, float]: 中心点质量指标
        
    验证标准：
        簇必须同时满足所有中心点约束条件才能通过验证：
        - coverage: 中心点覆盖率不低于配置阈值
        - mean: 中心点平均权重不低于配置阈值  
        - median: 中心点中位权重不低于配置阈值
        - p10: 中心点10分位权重不低于配置阈值
    """
    center = choose_center(nodes, g)
    metrics = center_metrics(center, nodes, g)
    ok = (
        metrics['coverage'] >= cons.get('coverage', 0.85)
        and metrics['mean'] >= cons.get('mean', 0.85)
        and metrics['median'] >= cons.get('median', 0.845)
        and metrics['p10'] >= cons.get('p10', 0.80)
    )
    return ok, center, metrics


def consistency_vote(i: int, j: int, emb_a: np.ndarray, emb_b: np.ndarray, emb_c: np.ndarray, std_max: float,
                     cos_a: float, cos_b: float, cos_c: float, vote_2_of_3: bool) -> bool:
    """
    三嵌入一致性投票验证
    
    Args:
        i, j: 两个节点的ID
        emb_a, emb_b, emb_c: 三个嵌入模型的向量矩阵
        std_max: 三个余弦相似度的标准差上限
        cos_a, cos_b, cos_c: 三个模型的余弦相似度阈值
        vote_2_of_3: 是否采用2/3投票制（否则要求3/3通过）
        
    Returns:
        bool: 是否通过一致性验证
        
    验证机制：
        1. 计算三个模型的余弦相似度
        2. 检查投票结果（2/3或3/3通过）
        3. 检查三个相似度的标准差是否在容忍范围内
        
    用途：
        在二次聚合阶段验证簇中心间的连接是否可靠，
        确保合并决策基于多模型的一致判断。
    """
    ca = float(np.dot(emb_a[i], emb_a[j]))
    cb = float(np.dot(emb_b[i], emb_b[j]))
    cc = float(np.dot(emb_c[i], emb_c[j]))
    votes = (1 if ca >= cos_a else 0) + (1 if cb >= cos_b else 0) + (1 if cc >= cos_c else 0)
    std = float(np.std([ca, cb, cc]))
    return ((votes >= 2) if vote_2_of_3 else (votes == 3)) and (std <= std_max)


def run(cfg_path: str, input_file: str = None) -> None:
    """
    原始引擎主函数 - 单核连通分量聚类算法
    
    Args:
        cfg_path: 配置文件路径
        input_file: 输入文件路径（未使用，保持接口一致性）
        
    算法流程：
        1. 数据加载：从pair_scores.parquet加载CE分数数据
        2. 图构建：基于高置信度阈值构建无向图
        3. 连通分量：使用DFS查找所有连通分量
        4. 簇验证：验证每个分量的中心点约束
        5. 双点保护：对不满足约束的大簇进行双点拆分
        6. 二次聚合：基于CE分数和一致性投票合并簇
        7. 结果输出：生成clusters.parquet和统计信息
        
    特点：
        - 串行实现，无并行优化
        - 最小依赖，兼容性最佳
        - 代码简洁，调试友好
        - 适合小规模数据集（<10万节点）
        
    输出文件：
        - clusters.parquet: 聚类结果
        - stage_stats.json: 统计信息
        - 可选图表: cluster_size_hist.png
    """
    cfg = load_config(cfg_path)
    out_dir = ensure_output_dir(cfg)
    stats = StatsRecorder(cfg.get('observe.stats_path', f"{out_dir}/stage_stats.json"))
    
    print("[stage4] 使用原始单核聚类算法（最小依赖版本）")

    pairs = pd.read_parquet(f"{out_dir}/stage3_ranked_pairs.parquet")
    print(f"[stage4] 加载 {len(pairs)} 个相似对")

    # 构核心图（高置信）
    high_th = float(cfg.get('rerank.thresholds.high', 0.83))
    g = build_graph(pairs, high_th)
    print(f"[stage4] 构建图完成，节点数: {len(g)}")

    # 聚类（连通分量）
    comps: List[List[int]] = connected_components(g)
    method_used = 'connected_components_original'
    print(f"[stage4] 发现 {len(comps)} 个连通分量")

    # 子簇中心点约束
    cons = cfg.get('cluster.center_constraints', {})
    min_cluster_size = int(cfg.get('cluster.min_cluster_size', 2))
    valid_clusters: List[Dict] = []
    for nodes in comps:
        if len(nodes) < min_cluster_size:
            continue
        ok, center, metrics = validate_cluster(nodes, g, cons)
        if not ok:
            # 双点簇保护
            pair_min = float(cons.get('pair_min_for_2_nodes', 0.86))
            kept_pairs = []
            for u in nodes:
                for v, w in g.get(u, []):
                    if v > u and v in nodes and w >= pair_min:
                        kept_pairs.append([u, v])
            if not kept_pairs:
                continue
            for comp in kept_pairs:
                ok2, center2, metrics2 = validate_cluster(comp, g, cons)
                if ok2:
                    valid_clusters.append({'center': center2, 'members': comp, **metrics2})
            continue
        valid_clusters.append({'center': center, 'members': nodes, **metrics})
    
    print(f"[stage4] 验证后保留 {len(valid_clusters)} 个有效簇")

    # 二次聚合
    second_cfg = cfg.get('cluster.second_merge', {})
    if second_cfg.get('enable', True) and len(valid_clusters) >= 2:
        print("[stage4] 开始二次聚合...")
        ce_min = float(second_cfg.get('ce_min', 0.81))
        require_vote = bool(second_cfg.get('require_consistency_vote', True))
        edge_map: Dict[Tuple[int, int], float] = {}
        for i, j, s in zip(pairs['i'], pairs['j'], pairs['ce_final']):
            a, b = (int(i), int(j)) if i < j else (int(j), int(i))
            if s >= ce_min:
                edge_map[(a, b)] = float(s)
        emb_a = np.load(f"{out_dir}/emb_a.npy")
        emb_b = np.load(f"{out_dir}/emb_b.npy")
        emb_c = np.load(f"{out_dir}/emb_c.npy")
        cons_cfg = cfg.get('consistency', {})
        cos_a_th = float(cons_cfg.get('cos_a', 0.875))
        cos_b_th = float(cons_cfg.get('cos_b', 0.870))
        cos_c_th = float(cons_cfg.get('cos_c', 0.870))
        std_max = float(cons_cfg.get('std_max', 0.04))
        vote_2 = bool(cons_cfg.get('vote_2_of_3', True))

        merged = True
        merge_rounds = 0
        while merged:
            merged = False
            merge_rounds += 1
            K = len(valid_clusters)
            if K <= 1:
                break
            print(f"[stage4] 二次聚合第 {merge_rounds} 轮，当前簇数: {K}")
            done = False
            for x in range(K):
                for y in range(x + 1, K):
                    cx = valid_clusters[x]['center']
                    cy = valid_clusters[y]['center']
                    a, b = (min(cx, cy), max(cx, cy))
                    ce = edge_map.get((a, b), 0.0)
                    if ce < ce_min:
                        continue
                    if require_vote and not consistency_vote(cx, cy, emb_a, emb_b, emb_c, std_max, cos_a_th, cos_b_th, cos_c_th, vote_2):
                        continue
                    new_nodes = sorted(set(valid_clusters[x]['members']) | set(valid_clusters[y]['members']))
                    ok, center_new, metrics_new = validate_cluster(new_nodes, g, cons)
                    if not ok:
                        continue
                    new_entry = {'center': center_new, 'members': new_nodes, **metrics_new}
                    keep = [valid_clusters[k] for k in range(K) if k not in (x, y)]
                    keep.append(new_entry)
                    valid_clusters = keep
                    merged = True
                    done = True
                    break
                if done:
                    break
        print(f"[stage4] 二次聚合完成，最终簇数: {len(valid_clusters)}")

    # 输出
    clusters_rows = []
    for cid, c in enumerate(valid_clusters):
        clusters_rows.append({
            'cluster_id': cid,
            'center': int(c['center']),
            'members': list(map(int, c['members'])),
            'coverage': float(c['coverage']),
            'mean': float(c['mean']),
            'median': float(c['median']),
            'p10': float(c['p10']),
        })
    clusters_df = pd.DataFrame(clusters_rows)
    write_parquet(clusters_df, f"{out_dir}/clusters.parquet")

    # 统计与图表
    stats_dict = {
        'num_clusters': int(len(valid_clusters)),
        'cluster_method': method_used,
    }
    sizes = [len(m['members']) for m in valid_clusters]
    if sizes:
        stats_dict.update({
            'size_p50': float(np.median(sizes)),
            'size_p90': float(np.percentile(sizes, 90)),
            'size_max': int(max(sizes)),
        })
        if cfg.get('observe.save_histograms', True):
            stats = StatsRecorder(cfg.get('observe.stats_path', f"{out_dir}/stage_stats.json"))
            stats.histogram_png(sizes, f"{out_dir}/figs/cluster_size_hist.png", title='Cluster size distribution')
    else:
        stats_dict.update({'size_p50': 0.0, 'size_p90': 0.0, 'size_max': 0})

    stats = StatsRecorder(cfg.get('observe.stats_path', f"{out_dir}/stage_stats.json"))
    stats.update('stage4', stats_dict)
    print(f"[stage4] 原始聚类完成，最终输出 {len(valid_clusters)} 个簇")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stage4 聚类模块 - 原始引擎（最小依赖版本）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
引擎特点：
  ✅ 最小依赖：仅需标准库和基础科学计算包
  ✅ 高兼容性：适用于各种环境和平台
  ✅ 调试友好：代码简洁清晰，易于理解
  ⚠️  性能较低：适合小规模数据集（<10万节点）
  
算法：
  - 连通分量检测（DFS实现）
  - 中心点约束验证
  - 双点簇保护机制
  - 二次聚合优化
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
    
    args = parser.parse_args()
    run(args.config, args.input)