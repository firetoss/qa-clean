"""
聚类算法模块
"""

import numpy as np
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PairScore:
    """配对分数数据类"""
    i: int
    j: int
    cos_a: float
    cos_b: float
    ce: float
    level: str  # "high" | "mid" | "low"


class GraphBuilder:
    """图构建器"""
    
    def __init__(self, n: int, center_ce_min_for_pair: float):
        self.n = n
        self.center_ce_min_for_pair = center_ce_min_for_pair
    
    def build_graph_from_pairs(self, pairs: List[PairScore]) -> Dict[int, Set[int]]:
        """
        构建图：
        - 仅用高置信边连接为核心连通
        - 中置信边：只允许连接到已有核心节点，避免跨核心链式连接
        - 双点簇保护：若只有两点相连，CE需≥center_ce_min_for_pair
        """
        # 初始：高置信边构建核心图
        core_adj: Dict[int, Set[int]] = {i: set() for i in range(self.n)}
        for p in pairs:
            if p.level == "high":
                core_adj[p.i].add(p.j)
                core_adj[p.j].add(p.i)

        # 记录核心节点（有高置信连接或自成节点均算）
        core_nodes: Set[int] = set(i for i in range(self.n) if len(core_adj[i]) > 0)

        # 引入中置信弱边：只允许连接到核心节点
        for p in pairs:
            if p.level == "mid":
                # 至少一端是核心节点才允许加入
                if (p.i in core_nodes) ^ (p.j in core_nodes):
                    core_adj[p.i].add(p.j)
                    core_adj[p.j].add(p.i)

        # 构建边CE快速查询表
        edge_ce: Dict[Tuple[int,int], float] = {}
        for p in pairs:
            if p.level in ("high", "mid"):
                a, b = (p.i, p.j) if p.i < p.j else (p.j, p.i)
                edge_ce[(a,b)] = max(edge_ce.get((a,b), 0.0), p.ce)

        # 处理双点簇保护
        self._protect_pairs(core_adj, edge_ce)
        
        return core_adj
    
    def _protect_pairs(self, adj: Dict[int, Set[int]], edge_ce: Dict[Tuple[int,int], float]) -> None:
        """保护双点簇，断开不达标的边"""
        visited = set()
        
        def dfs(u: int, comp: List[int]) -> None:
            visited.add(u)
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    dfs(v, comp)

        # 收集连通分量
        comps: List[List[int]] = []
        for i in range(self.n):
            if i not in visited and adj[i]:
                comp: List[int] = []
                dfs(i, comp)
                comps.append(comp)

        # 断开不达标的双点簇
        for comp in comps:
            if len(comp) == 2:
                u, v = comp
                a, b = (u, v) if u < v else (v, u)
                ce = edge_ce.get((a,b), 0.0)
                if ce < self.center_ce_min_for_pair:
                    # 断开这条边
                    adj[u].discard(v)
                    adj[v].discard(u)


class ConnectedComponents:
    """连通分量分析器"""
    
    @staticmethod
    def find_components(adj: Dict[int, Set[int]]) -> List[List[int]]:
        """查找连通分量"""
        visited = set()
        comps = []
        
        for i in adj.keys():
            if i not in visited and adj[i]:
                stack = [i]
                visited.add(i)
                comp = [i]
                
                while stack:
                    u = stack.pop()
                    for v in adj[u]:
                        if v not in visited:
                            visited.add(v)
                            stack.append(v)
                            comp.append(v)
                
                comps.append(comp)
        
        return comps


class CenterSelector:
    """中心点选择器"""
    
    def __init__(self, ce_score_mat: np.ndarray, cover_threshold: float, mean_threshold: float):
        self.ce_mat = ce_score_mat
        self.cover_threshold = cover_threshold
        self.mean_threshold = mean_threshold
    
    def pick_center_by_ce(self, comp: List[int]) -> Optional[int]:
        """
        选择中心点：
        - 对每个候选点，统计其与组内其他点 CE ≥ 阈值的覆盖率
        - 选择覆盖率最高且均值较高者；若均不达标，返回 None
        """
        if len(comp) == 1:
            return comp[0]
        
        best_idx = None
        best_tuple = (-1.0, -1.0)  # (覆盖率, 平均分)
        
        for u in comp:
            scores = [self.ce_mat[min(u,v), max(u,v)] for v in comp if v != u]
            if not scores:
                continue
            
            cov = sum(1 for s in scores if s >= self.mean_threshold) / max(1, len(scores))
            mean_s = float(np.mean(scores))
            tup = (cov, mean_s)
            
            if cov >= self.cover_threshold and mean_s >= self.mean_threshold and tup > best_tuple:
                best_tuple = tup
                best_idx = u
        
        return best_idx
    
    def representative_index(self, comp: List[int]) -> int:
        """若中心点不达标，退而求其次：取与组内平均CE最高的点"""
        if len(comp) == 1:
            return comp[0]
        
        best_u = comp[0]
        best_mean = -1.0
        
        for u in comp:
            scores = [self.ce_mat[min(u,v), max(u,v)] for v in comp if v != u]
            if scores:
                m = float(np.mean(scores))
                if m > best_mean:
                    best_mean = m
                    best_u = u
        
        return best_u
