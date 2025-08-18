from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils.config import ensure_output_dir, load_config
from ..utils.io_utils import write_parquet
from ..utils.metrics import StatsRecorder


def _ensure_device(device: str) -> str:
    if device == 'cuda':
        try:
            import torch  # type: ignore

            if not torch.cuda.is_available():
                print('[stage3] CUDA 不可用，自动切换到 CPU')
                return 'cpu'
        except Exception:
            return 'cpu'
    return device


FALLBACK_CE = {
    'ce_main': 'BAAI/bge-reranker-base',
    'ce_aux_1': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    'ce_aux_2': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
}


def batch_ce_scores(model_name: str, pairs: List[Tuple[str, str]], device: str, batch_size: int) -> np.ndarray:
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"需要 sentence-transformers 库以加载CrossEncoder，请安装：pip install sentence-transformers；错误：{e}"
        )
    device = _ensure_device(device)
    try:
        model = CrossEncoder(model_name, device=device if device == 'cuda' else 'cpu')
    except Exception as e:
        print(f"[stage3] 加载 {model_name} 失败：{e}，尝试回退基础CE模型。")
        model = CrossEncoder(FALLBACK_CE.get('ce_main', 'cross-encoder/ms-marco-MiniLM-L-6-v2'), device=device if device == 'cuda' else 'cpu')
    scores: List[float] = []
    for i in tqdm(range(0, len(pairs), batch_size), desc=f"rerank {model_name}"):
        batch = pairs[i : i + batch_size]
        sc = model.predict(batch, batch_size=batch_size, show_progress_bar=False)
        scores.extend([float(x) for x in sc])
    return np.asarray(scores, dtype=np.float32)


def run(cfg_path: str, input_file: str = None) -> None:
    cfg = load_config(cfg_path)
    out_dir = ensure_output_dir(cfg)
    stats = StatsRecorder(cfg.get('observe.stats_path', f"{out_dir}/stage_stats.json"))

    # load data
    stage2_df = pd.read_parquet(f"{out_dir}/stage2_data.parquet")
    meta = pd.read_parquet(f"{out_dir}/candidate_pairs_meta.parquet")
    pairs = [(int(i), int(j)) for i, j in zip(meta['i'].tolist(), meta['j'].tolist())]

    q_col = cfg.get('data.q_col', 'question')
    ques = stage2_df[q_col].tolist()

    rerank_cfg = cfg.get('rerank', {})
    device = rerank_cfg.get('device', 'cuda')
    bs = rerank_cfg.get('batch_size', 64)

    text_pairs: List[Tuple[str, str]] = [(ques[i], ques[j]) for i, j in pairs]

    w_main, w_aux = rerank_cfg.get('fusion', {}).get('weights', [0.7, 0.3])
    ce_aux_max = np.maximum(ce_aux1, ce_aux2)
    ce_final = w_main * ce_main + w_aux * ce_aux_max

    th = rerank_cfg.get('thresholds', {})
    high = float(th.get('high', 0.83))
    mid_low = float(th.get('mid_low', 0.77))

    label = np.where(ce_final >= high, 'high', np.where(ce_final >= mid_low, 'mid', 'low'))

    out = meta.copy()
    out['ce_main'] = ce_main
    out['ce_aux1'] = ce_aux1
    out['ce_aux2'] = ce_aux2
    out['ce_final'] = ce_final
    out['label'] = label

    # drop low
    out = out[out['label'] != 'low'].reset_index(drop=True)
    # stats & optional figs
    stats.update('stage3', {
        'num_pairs_input': int(len(pairs)),
        'num_pairs_output': int(out.shape[0]),
        'ce_final_p10': float(np.percentile(ce_final, 10)) if len(ce_final) else 0.0,
        'ce_final_p50': float(np.percentile(ce_final, 50)) if len(ce_final) else 0.0,
        'ce_final_p90': float(np.percentile(ce_final, 90)) if len(ce_final) else 0.0,
        'high_ratio': float((label == 'high').mean()) if len(label) else 0.0,
        'mid_ratio': float((label == 'mid').mean()) if len(label) else 0.0,
        'note': '若CE加载失败，已尝试回退基础模型；可在config修改模型名后重跑'
    })
    if cfg.get('observe.save_histograms', True):
        stats.histogram_png(ce_final.tolist(), f"{out_dir}/figs/ce_final_hist.png", title='CE final distribution')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='src/configs/config.yaml')
    ap.add_argument('--input', help='输入数据文件路径（覆盖配置文件中的设置）')
    args = ap.parse_args()
    run(args.config, args.input)
