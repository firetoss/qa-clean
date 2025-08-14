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
    g: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for i, j, s in zip(pairs_df['i'], pairs_df['j'], pairs_df['ce_final']):
        if s >= threshold:
            g[int(i)].append((int(j), float(s)))
            g[int(j)].append((int(i), float(s)))
    return g


def connected_components(g: Dict[int, List[Tuple[int, float]]]) -> List[List[int]]:
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
    # choose node with max mean edge weight to others in nodes
    best, best_score = nodes[0], -1.0
    for u in nodes:
        ws = [w for v, w in g.get(u, []) if v in nodes and v != u]
        m = float(np.mean(ws)) if ws else 0.0
        if m > best_score:
            best_score = m
            best = u
    return best


def center_metrics(center: int, nodes: List[int], g: Dict[int, List[Tuple[int, float]]]) -> Dict[str, float]:
    ws = [w for v, w in g.get(center, []) if v in nodes and v != center]
    if not ws:
        return {'coverage': 0.0, 'mean': 0.0, 'median': 0.0, 'p10': 0.0}
    arr = np.asarray(ws, dtype=float)
    # coverage: 边存在比例（近似覆盖度）
    coverage = float(len(ws) / max(1, len(nodes) - 1))
    return {
        'coverage': coverage,
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'p10': float(np.percentile(arr, 10)),
    }


def validate_cluster(nodes: List[int], g: Dict[int, List[Tuple[int, float]]], cons: Dict[str, float]) -> Tuple[bool, int, Dict[str, float]]:
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
    ca = float(np.dot(emb_a[i], emb_a[j]))
    cb = float(np.dot(emb_b[i], emb_b[j]))
    cc = float(np.dot(emb_c[i], emb_c[j]))
    votes = (1 if ca >= cos_a else 0) + (1 if cb >= cos_b else 0) + (1 if cc >= cos_c else 0)
    std = float(np.std([ca, cb, cc]))
    return ((votes >= 2) if vote_2_of_3 else (votes == 3)) and (std <= std_max)


def run(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    out_dir = ensure_output_dir(cfg)
    stats = StatsRecorder(cfg.get('observe.stats_path', f"{out_dir}/stage_stats.json"))

    pairs = pd.read_parquet(f"{out_dir}/pair_scores.parquet")

    # 构核心图（高置信）
    high_th = float(cfg.get('rerank.thresholds.high', 0.83))
    g = build_graph(pairs, high_th)

    # 聚类（fallback为连通分量）
    method_used = cfg.get('cluster.method', 'leiden')
    comps: List[List[int]] = connected_components(g)
    method_used = 'connected_components'

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

    # 二次聚合
    second_cfg = cfg.get('cluster.second_merge', {})
    if second_cfg.get('enable', True) and len(valid_clusters) >= 2:
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
        while merged:
            merged = False
            K = len(valid_clusters)
            if K <= 1:
                break
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


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='src/configs/config.yaml')
    args = ap.parse_args()
    run(args.config)
