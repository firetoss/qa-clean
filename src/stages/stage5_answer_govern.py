from __future__ import annotations

import argparse
import json
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils.cn_text import normalize_zh
from ..utils.config import ensure_output_dir, load_config
from ..utils.io_utils import write_parquet
from ..utils.metrics import StatsRecorder

MONEY_RE = re.compile(r"(?:人民币|RMB|￥)?\s*([0-9]+(?:\.[0-9]+)?)\s*(元|块|万元|千元|角|分)?")
PCT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)%")
DATE_YMD_RE = re.compile(r"(\d{4})年(\d{1,2})月(?:(\d{1,2})日)?")
DATE_WORDS = [
    '本周', '下周', '月底', '年初', '当月', '次月'
]
NEG_WORDS = ['不', '不可', '不支持', '禁止', '未', '无', '不能', '仅限']
COND_WORDS = ['需', '须', '前提', '仅在', '只有', '才']
ORG_LOC_SUFFIX = ['公司', '银行', '支行', '分行', '网点', '门店', '营业厅']


def extract_signals(s: str) -> Dict[str, List[str]]:
    s = normalize_zh(s)
    money = [m.group(0) for m in MONEY_RE.finditer(s)]
    pct = [m.group(0) for m in PCT_RE.finditer(s)]
    dates = [m.group(0) for m in DATE_YMD_RE.finditer(s)] + [w for w in DATE_WORDS if w in s]
    neg = [w for w in NEG_WORDS if w in s]
    cond = [w for w in COND_WORDS if w in s]
    orgloc = [w for w in ORG_LOC_SUFFIX if w in s]
    nums = re.findall(r"[0-9]+(?:\.[0-9]+)?", s)
    return {
        'money': money,
        'percent': pct,
        'date': dates,
        'neg': neg,
        'cond': cond,
        'orgloc': orgloc,
        'nums': nums,
    }


def _ymd_to_tuple(d: str) -> Tuple[int, int, int]:
    m = DATE_YMD_RE.search(d)
    if not m:
        return (0, 0, 0)
    y = int(m.group(1))
    mo = int(m.group(2))
    da = int(m.group(3)) if m.group(3) else 1
    return (y, mo, da)


def _date_within_tolerance(a: str, b: str, tol_days: int = 1) -> bool:
    # 简化：仅比较YYYY年MM月(DD日)，并允许±1天
    ya, ma, da = _ymd_to_tuple(a)
    yb, mb, db = _ymd_to_tuple(b)
    if (ya, ma, da) == (0, 0, 0) or (yb, mb, db) == (0, 0, 0):
        # 若非标准日期，退化为近似关键字：本周/下周/月内等，认为可容忍
        return True
    try:
        import datetime as dt

        da_ = dt.date(ya, ma, da)
        db_ = dt.date(yb, mb, db)
        return abs((da_ - db_).days) <= tol_days
    except Exception:
        return True


def hard_conflict(a: Dict[str, List[str]], b: Dict[str, List[str]], tol_pct: float, tol_days: int) -> bool:
    # money conflict if numeric sets differ > tol
    def nums(vals: List[str]) -> List[float]:
        out = []
        for v in vals:
            ms = re.findall(r"[0-9]+(?:\.[0-9]+)?", v)
            out.extend([float(x) for x in ms])
        return out
    am = nums(a.get('money', [])) or [float(x) for x in a.get('nums', [])]
    bm = nums(b.get('money', [])) or [float(x) for x in b.get('nums', [])]
    if am and bm:
        x, y = am[0], bm[0]
        if x == 0 and y == 0:
            pass
        else:
            if abs(x - y) / max(1e-6, max(abs(x), abs(y))) > tol_pct:
                return True
    # negation vs affirmation
    if (a.get('neg') and not b.get('neg')) or (b.get('neg') and not a.get('neg')):
        return True
    # date: check overlap/close
    ad, bd = a.get('date', []), b.get('date', [])
    if ad and bd:
        ok = False
        for x in ad:
            for y in bd:
                if _date_within_tolerance(x, y, tol_days=tol_days):
                    ok = True
                    break
            if ok:
                break
        if not ok:
            return True
    return False


def pick_representative(answers: List[str]) -> str:
    counts: Dict[str, int] = {}
    for s in answers:
        n = normalize_zh(s)
        counts[n] = counts.get(n, 0) + 1
    return max(counts.items(), key=lambda x: x[1])[0] if counts else (answers[0] if answers else '')


def merge_answers(answers: List[str]) -> str:
    keyset = []
    for s in answers:
        s = normalize_zh(s)
        keyset.append(s)
    uniq = []
    for s in keyset:
        if s not in uniq:
            uniq.append(s)
    return '；'.join(uniq)


def run(cfg_path: str, input_file: str = None) -> None:
    cfg = load_config(cfg_path)
    out_dir = ensure_output_dir(cfg)
    stats = StatsRecorder(cfg.get('observe.stats_path', f"{out_dir}/stage_stats.json"))

    stage2 = pd.read_parquet(f"{out_dir}/stage2_data.parquet")
    pairs = pd.read_parquet(f"{out_dir}/pair_scores.parquet")
    clusters = pd.read_parquet(f"{out_dir}/clusters.parquet")

    q_col = cfg.get('data.q_col', 'question')
    a_col = cfg.get('data.a_col', 'answer')

    tol_pct = float(cfg.get('govern.number_tolerance_pct', 0.05))
    tol_days = int(cfg.get('govern.date_tolerance_days', 1))

    rows = []
    conflict_count = 0
    for _, c in clusters.iterrows():
        members = [int(x) for x in c['members']]
        answers = [stage2.iloc[m][a_col] for m in members]
        signals = [extract_signals(a) for a in answers]
        conflict = False
        for i in range(len(signals)):
            for j in range(i + 1, len(signals)):
                if hard_conflict(signals[i], signals[j], tol_pct, tol_days):
                    conflict = True
                    break
            if conflict:
                break
        if conflict:
            conflict_count += 1
        rep = pick_representative(answers)
        merged = merge_answers(answers) if cfg.get('govern.merge_answers', True) and not conflict else rep
        rows.append({
            'cluster_id': int(c['cluster_id']),
            'center_question': stage2.iloc[int(c['center'])][q_col],
            'members': members,
            'representative_answer': rep,
            'merged_answer': merged,
            'conflict': bool(conflict),
        })

    out = pd.DataFrame(rows)
    write_parquet(out, f"{out_dir}/clean_answers.parquet")

    stats.update('stage5', {
        'num_clusters': int(clusters.shape[0]),
        'conflict_clusters': int(conflict_count),
        'merge_rate': float((out['merged_answer'] != out['representative_answer']).mean()) if len(out) else 0.0,
    })


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='src/configs/config.yaml')
    ap.add_argument('--input', help='输入数据文件路径（覆盖配置文件中的设置）')
    args = ap.parse_args()
    run(args.config, args.input)
