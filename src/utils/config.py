import os
import json
import random
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import yaml


@dataclass
class Config:
    data: Dict[str, Any]

    def get(self, path: str, default: Any = None) -> Any:
        cur: Any = self.data
        for key in path.split('.'):
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur


def load_config(path: str) -> Config:
    """
    加载YAML配置文件并进行基本验证
    
    Args:
        path: 配置文件路径
        
    Returns:
        Config对象
        
    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML格式错误
        ValueError: 配置验证失败
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
        
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # 基本配置验证
    _validate_config(data)
    return Config(data=data)


def _validate_config(data: Dict[str, Any]) -> None:
    """验证配置文件的基本结构和参数合理性"""
    required_sections = ['pipeline', 'data', 'embeddings', 'recall', 'consistency', 'rerank', 'cluster', 'govern', 'observe']
    for section in required_sections:
        if section not in data:
            raise ValueError(f"缺少配置节: {section}")
    
    # 验证关键参数范围
    emb_batch = data.get('embeddings', {}).get('batch_size', 64)
    if not (1 <= emb_batch <= 512):
        raise ValueError(f"embeddings.batch_size应在1-512范围内，当前: {emb_batch}")
    
    topk = data.get('recall', {}).get('topk', 200)
    if not (10 <= topk <= 1000):
        raise ValueError(f"recall.topk应在10-1000范围内，当前: {topk}")
    
    cos_a = data.get('consistency', {}).get('cos_a', 0.875)
    if not (0.5 <= cos_a <= 1.0):
        raise ValueError(f"consistency.cos_a应在0.5-1.0范围内，当前: {cos_a}")


def ensure_seed(cfg: Config) -> None:
    seed = cfg.get('pipeline.random_seed', 42)
    random.seed(seed)
    np.random.seed(seed)


def ensure_output_dir(cfg: Config) -> str:
    out_dir = cfg.get('pipeline.output_dir', './outputs')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def dump_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
