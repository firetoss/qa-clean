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
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return Config(data=data)


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
