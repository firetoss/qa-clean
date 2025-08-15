from __future__ import annotations

import argparse
import os
from typing import Dict

import pandas as pd

from ..utils.cn_text import filter_reason, normalize_zh
from ..utils.config import ensure_output_dir, load_config
from ..utils.io_utils import read_data_file, write_parquet
from ..utils.metrics import StatsRecorder


def _load_or_sample(input_path: str, q_col: str, a_col: str) -> pd.DataFrame:
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(input_path)
        # 使用智能文件读取函数，支持多种格式
        df = read_data_file(input_path)
        req = {q_col, a_col}
        if not req.issubset(df.columns):
            raise KeyError(f"缺少列: 需要 {req}，实际 {set(df.columns)}")
        
        # 如果没有id列，自动创建行索引作为id
        if 'id' not in df.columns:
            df['id'] = range(len(df))
            print(f"[stage1] 未找到id列，自动创建行索引作为id")
        
        print(f"[stage1] 成功读取数据文件: {input_path}，格式: {os.path.splitext(input_path)[1]}，行数: {len(df)}")
        return df
    except Exception as e:
        print(f"[stage1] 读取 {input_path} 失败：{e}\n将生成示例数据继续流程。请替换为你的数据并包含列: {q_col},{a_col}。")
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            q_col: [
                '如何开发票', '可以报销吗', '附近哪家店比较好', '本周几点营业', '售后流程是什么'
            ],
            a_col: [
                '请登录官网开具增值税电子发票', '支持按规定报销，请保留原始小票', '请提供更具体的位置或使用官网网点查询', '营业时间为工作日9:00-18:00', '售后请联系官方客服并提供订单号'
            ],
        })


def run(cfg_path: str, input_file: str = None) -> None:
    cfg = load_config(cfg_path)
    out_dir = ensure_output_dir(cfg)
    stats = StatsRecorder(cfg.get('observe.stats_path', f"{out_dir}/stage_stats.json"))

    # 使用命令行参数的输入文件，如果没有则使用配置文件中的路径
    input_path = input_file if input_file else cfg.get('data.input_path')
    q_col = cfg.get('data.q_col', 'question')
    a_col = cfg.get('data.a_col', 'answer')

    df = _load_or_sample(input_path, q_col, a_col)
    df[q_col] = df[q_col].astype(str).map(normalize_zh)

    keep_mask = []
    reasons: Dict[str, int] = {}
    for q in df[q_col].tolist():
        keep, r = filter_reason(q)
        keep_mask.append(keep)
        reasons[r] = reasons.get(r, 0) + 1

    df_clean = df[keep_mask].reset_index(drop=True)
    write_parquet(df_clean, f"{out_dir}/stage1_clean.parquet")

    stats.update('stage1', {
        'input_count': int(df.shape[0]),
        'kept_count': int(df_clean.shape[0]),
        'keep_rate': float(df_clean.shape[0] / max(1, df.shape[0])),
        'reason_top': dict(sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:20]),
        'note': '若生成了示例数据，请替换 data/raw/input.parquet 并确保列齐全后重跑阶段1'
    })


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='src/configs/config.yaml')
    ap.add_argument('--input', help='输入数据文件路径（覆盖配置文件中的设置）')
    args = ap.parse_args()
    run(args.config, args.input)
