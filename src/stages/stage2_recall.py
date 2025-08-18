from __future__ import annotations

import argparse
import itertools
import time
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..recall.faiss_provider import FaissProvider
from ..utils.cn_text import normalize_zh
from ..utils.config import ensure_output_dir, load_config
from ..utils.io_utils import save_npy, write_parquet
from ..utils.metrics import StatsRecorder
from ..utils.text_sim import ngram_jaccard


def _ensure_device(device: str) -> str:
    if device == 'cuda':
        try:
            import torch  # type: ignore

            if not torch.cuda.is_available():
                print('[stage2] CUDA 不可用，自动切换到 CPU')
                return 'cpu'
        except Exception:
            return 'cpu'
    return device


FALLBACK_EMB = {
    'a': 'BAAI/bge-base-zh-v1.5',
    'b': 'moka-ai/m3e-base',
    'c': 'Alibaba-NLP/gte-base-zh',
}


def batch_encode(model_name: str, texts: List[str], device: str, batch_size: int) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"需要 sentence-transformers 库以编码嵌入，请安装：pip install sentence-transformers；错误：{e}"
        )
    device = _ensure_device(device)
    try:
        model = SentenceTransformer(model_name, device=device if device == 'cuda' else 'cpu')
    except Exception as e:
        # 尝试回退base模型
        print(f"[stage2] 加载 {model_name} 失败：{e}，尝试回退基础模型。")
        model = SentenceTransformer(FALLBACK_EMB.get('a', 'BAAI/bge-base-zh-v1.5'), device=device if device == 'cuda' else 'cpu')
    emb_list: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"encode {model_name}"):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
        emb_list.append(emb.astype(np.float32))
    X = np.concatenate(emb_list, axis=0) if emb_list else np.zeros((0, 768), dtype=np.float32)
    return X


def build_or_load_index(cfg, emb_a: np.ndarray) -> FaissProvider:
    faiss_cfg = cfg.get('recall.faiss', {})
    provider = FaissProvider(
        index_type=faiss_cfg.get('index_type', 'flat_ip'),
        nlist=faiss_cfg.get('nlist', 4096),
        nprobe=faiss_cfg.get('nprobe', 16),
        hnsw_m=faiss_cfg.get('hnsw_m', 32),
        ef_search=faiss_cfg.get('ef_search', 200),
        normalize=cfg.get('embeddings.normalize', True),
        device=_ensure_device(cfg.get('embeddings.device', 'cuda')),
    )
    persist = faiss_cfg.get('persist_path', './outputs/faiss.index')
    try:
        if provider.load(persist, xb_dim=emb_a.shape[1]):
            return provider
        provider.build(emb_a)
        try:
            provider.save(persist)
        except Exception:
            pass
        return provider
    except Exception as e:
        raise RuntimeError(
            f"[stage2] 构建或加载FAISS索引失败：{e}\n"
            "请确认已通过 conda-forge 安装FAISS：\n"
            "  GPU: conda install -c conda-forge faiss-gpu\n  CPU: conda install -c conda-forge faiss-cpu\n"
            "若仍失败，请将 recall.faiss.index_type 改为 flat_ip 并重试"
        )


def dedup_pairs(pairs: List[Tuple[int, int]]) -> np.ndarray:
    uniq = set()
    out: List[Tuple[int, int]] = []
    for i, j in pairs:
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) not in uniq:
            uniq.add((a, b))
            out.append((a, b))
    arr = np.array(out, dtype=np.int32)
    return arr


def run(cfg_path: str, input_file: str = None) -> None:
    cfg = load_config(cfg_path)
    out_dir = ensure_output_dir(cfg)
    stats = StatsRecorder(cfg.get('observe.stats_path', f"{out_dir}/stage_stats.json"))

    # 优先使用stage1的清洗输出
    stage1_path = f"{out_dir}/stage1_clean.parquet"
    try:
        df = pd.read_parquet(stage1_path)
    except Exception:
        # 使用命令行参数的输入文件，如果没有则使用配置文件中的路径
        input_path = input_file if input_file else cfg.get('data.input_path')
        from src.utils.io_utils import read_data_file
        df = read_data_file(input_path)
        q_col = cfg.get('data.q_col', 'question')
        df[q_col] = df[q_col].astype(str).map(normalize_zh)

    q_col = cfg.get('data.q_col', 'question')
    a_col = cfg.get('data.a_col', 'answer')
    
    # 确保有id列，如果没有则创建
    if 'id' not in df.columns:
        df['id'] = range(len(df))

    questions = df[q_col].tolist()

    # embeddings (timed)
    em_cfg = cfg.get('embeddings', {})
    bs = em_cfg.get('batch_size', 64)
    device = em_cfg.get('device', 'cuda')
    models = em_cfg.get('models', {})

    t0 = time.perf_counter()
    emb_a = batch_encode(models.get('a', FALLBACK_EMB['a']), questions, device, bs)
    emb_b = batch_encode(models.get('b', FALLBACK_EMB['b']), questions, device, bs)
    emb_c = batch_encode(models.get('c', FALLBACK_EMB['c']), questions, device, bs)
    t1 = time.perf_counter()

    if cfg.get('embeddings.normalize', True):
        def l2n(x):
            n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
            return x / n
        emb_a = l2n(emb_a)
        emb_b = l2n(emb_b)
        emb_c = l2n(emb_c)

    save_npy(f"{out_dir}/emb_a.npy", emb_a)
    save_npy(f"{out_dir}/emb_b.npy", emb_b)
    save_npy(f"{out_dir}/emb_c.npy", emb_c)

    # FAISS search (timed)
    provider = build_or_load_index(cfg, emb_a)
    topk = int(cfg.get('recall.topk', 200))
    
    # 分批检索减少内存峰值
    t2 = time.perf_counter()
    batch_size = 1024  # 每批检索的查询数量
    D_list, I_list = [], []
    
    for i in range(0, len(questions), batch_size):
        end_i = min(i + batch_size, len(questions))
        batch_emb = emb_a[i:end_i]
        D_batch, I_batch = provider.search(batch_emb, topk + 1)  # include self
        D_list.append(D_batch)
        I_list.append(I_batch)
    
    D = np.concatenate(D_list, axis=0)
    I = np.concatenate(I_list, axis=0)
    t3 = time.perf_counter()

    # collect candidate pairs (faiss)
    faiss_pairs: List[Tuple[int, int]] = []
    self_removed = 0
    N = len(questions)
    for i in range(N):
        for rk in range(min(topk + 1, I.shape[1])):
            j = int(I[i, rk])
            if j == i:
                self_removed += 1
                continue
            a, b = (i, j) if i < j else (j, i)
            faiss_pairs.append((a, b))

    # n-gram补召 within neighborhood
    ng_cfg = cfg.get('recall.char_ngram', {})
    n = int(ng_cfg.get('n', 3))
    th = float(ng_cfg.get('threshold', 0.45))
    th_short = float(ng_cfg.get('short_threshold', 0.50))
    short_len = int(ng_cfg.get('short_len', 10))
    topn_neighbors = int(ng_cfg.get('topn_neighbors', 50))

    ngram_pair_set: Set[Tuple[int, int]] = set()
    ngram_hits = 0
    for i in range(N):
        neigh = [int(x) for x in I[i, 1 : min(I.shape[1], topn_neighbors + 1)]]
        for j in neigh:
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            t = th_short if min(len(questions[i]), len(questions[j])) < short_len else th
            sim = ngram_jaccard(questions[i], questions[j], n=n)
            if sim >= t:
                ngram_pair_set.add((a, b))
                ngram_hits += 1

    # 合并并去重
    all_pairs = faiss_pairs + list(ngram_pair_set)
    cand_pairs = dedup_pairs(all_pairs)

    # 三嵌入一致性过滤
    cons = cfg.get('consistency', {})
    cos_a_th = float(cons.get('cos_a', 0.875))
    cos_b_th = float(cons.get('cos_b', 0.870))
    cos_c_th = float(cons.get('cos_c', 0.870))
    std_max = float(cons.get('std_max', 0.04))
    vote_2_of_3 = bool(cons.get('vote_2_of_3', True))

    keep_mask = []
    for i, j in cand_pairs.tolist():
        ca = float(np.dot(emb_a[i], emb_a[j]))
        cb = float(np.dot(emb_b[i], emb_b[j]))
        cc = float(np.dot(emb_c[i], emb_c[j]))
        votes = (1 if ca >= cos_a_th else 0) + (1 if cb >= cos_b_th else 0) + (1 if cc >= cos_c_th else 0)
        std = float(np.std([ca, cb, cc]))
        ok = (votes >= 2 if vote_2_of_3 else votes == 3) and (std <= std_max)
        keep_mask.append(ok)

    cand_pairs_kept = cand_pairs[keep_mask]

    # 构造 meta（按最终保留对）
    metas: Dict[str, List] = {k: [] for k in ['i', 'j', 'source', 'score_hint', 'ngram_hit']}
    for i, j in cand_pairs_kept.tolist():
        pair = (min(i, j), max(i, j))
        is_ng = pair in ngram_pair_set
        metas['i'].append(int(pair[0]))
        metas['j'].append(int(pair[1]))
        metas['ngram_hit'].append(bool(is_ng))
        metas['source'].append('ngram' if is_ng else 'faiss')
        if is_ng:
            metas['score_hint'].append(float(ngram_jaccard(questions[i], questions[j], n=n)))
        else:
            metas['score_hint'].append(float(np.dot(emb_a[i], emb_a[j])))

    meta = pd.DataFrame(metas)

    # 保存数据文件供stage3使用
    write_parquet(df, f"{out_dir}/stage2_data.parquet")
    write_parquet(meta, f"{out_dir}/candidate_pairs_meta.parquet")

    # outputs
    stats.update('stage2', {
        'num_questions': int(N),
        'faiss_topk': int(topk),
        'pairs_faiss_raw': int(len(faiss_pairs)),
        'pairs_ngram_raw': int(len(ngram_pair_set)),
        'pairs_after_dedup': int(cand_pairs.shape[0]),
        'pairs_kept_consistency': int(cand_pairs_kept.shape[0]),
        'self_removed': int(self_removed),
        'ngram_hits': int(ngram_hits),
        'encode_seconds': float(t1 - t0),
        'search_seconds': float(t3 - t2),
        'note': '若大模型下载失败，已尝试回退到base模型；可在config修改模型名后重跑'
    })


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='src/configs/config.yaml')
    ap.add_argument('--input', help='输入数据文件路径（覆盖配置文件中的设置）')
    args = ap.parse_args()
    run(args.config, args.input)
