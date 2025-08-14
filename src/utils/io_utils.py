import os
from typing import Any, Dict

import numpy as np
import pandas as pd


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def read_data_file(path: str) -> pd.DataFrame:
    """
    智能读取数据文件，支持多种格式：parquet, xlsx, csv
    
    Args:
        path: 输入文件路径
        
    Returns:
        pandas.DataFrame: 读取的数据
        
    Raises:
        ValueError: 不支持的文件格式
        FileNotFoundError: 文件不存在
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    
    # 获取文件扩展名
    _, ext = os.path.splitext(path.lower())
    
    if ext == '.parquet':
        return pd.read_parquet(path)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(path)
    elif ext == '.csv':
        # 尝试不同编码和分隔符
        for encoding in ['utf-8', 'gbk', 'gb2312']:
            for sep in [',', '\t', ';']:
                try:
                    df = pd.read_csv(path, encoding=encoding, sep=sep)
                    # 检查是否成功读取（至少有2列且行数>0）
                    if df.shape[1] >= 2 and df.shape[0] > 0:
                        print(f"[io_utils] 成功读取CSV文件，编码: {encoding}, 分隔符: '{sep}'")
                        return df
                except Exception:
                    continue
        # 如果都失败了，使用默认参数再试一次
        return pd.read_csv(path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}。支持的格式: .parquet, .xlsx, .xls, .csv")


def read_parquet(path: str) -> pd.DataFrame:
    """向后兼容的parquet读取函数"""
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
