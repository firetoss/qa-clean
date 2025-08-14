import os
from typing import Any, Dict

import numpy as np
import pandas as pd


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: str) -> None:
    ensure_parent_dir(path)
    df.to_parquet(path, index=False)


def save_npy(path: str, arr: np.ndarray) -> None:
    ensure_parent_dir(path)
    np.save(path, arr)


def save_json_merge(path: str, update_obj: Dict[str, Any]) -> None:
    import json
    ensure_parent_dir(path)
    base: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                base = json.load(f)
        except Exception:
            base = {}
    # deep merge at top-level by stage name keys
    for k, v in update_obj.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k].update(v)
        else:
            base[k] = v
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(base, f, ensure_ascii=False, indent=2)
