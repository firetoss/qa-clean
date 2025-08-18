import os
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import yaml


@dataclass
class Config:
    data: Dict[str, Any]
    source_file: Optional[str] = None

    def get(self, path: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的路径"""
        cur: Any = self.data
        for key in path.split('.'):
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur
    
    def set(self, path: str, value: Any) -> None:
        """设置配置值，支持点号分隔的路径"""
        keys = path.split('.')
        cur = self.data
        for key in keys[:-1]:
            if key not in cur:
                cur[key] = {}
            cur = cur[key]
        cur[keys[-1]] = value
    
    def merge_from_env(self, env_mapping: Dict[str, str]) -> None:
        """从环境变量合并配置"""
        for env_var, config_path in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                # 尝试转换类型
                try:
                    if value.lower() in ('true', 'false'):
                        value = value.lower() == 'true'
                    elif value.isdigit():
                        value = int(value)
                    elif '.' in value and value.replace('.', '').isdigit():
                        value = float(value)
                except:
                    pass  # 保持字符串类型
                
                self.set(config_path, value)
    
    def apply_optimizations(self, optimizations: Dict[str, Any]) -> None:
        """应用环境优化配置"""
        optimization_mapping = {
            'device': 'embeddings.device',
            'embedding_batch_size': 'embeddings.batch_size', 
            'rerank_batch_size': 'rerank.batch_size',
            'n_jobs': 'cluster.n_jobs',
            'faiss_index_type': 'recall.faiss.index_type',
        }
        
        for opt_key, config_path in optimization_mapping.items():
            if opt_key in optimizations:
                current_value = self.get(config_path)
                new_value = optimizations[opt_key]
                
                # 只在没有明确配置或使用默认值时应用优化
                if self._should_apply_optimization(config_path, current_value, new_value):
                    self.set(config_path, new_value)
    
    def _should_apply_optimization(self, config_path: str, current: Any, optimized: Any) -> bool:
        """判断是否应该应用优化配置"""
        # 如果当前值是None或使用已知默认值，则应用优化
        default_values = {
            'embeddings.device': 'cuda',
            'embeddings.batch_size': 64,
            'rerank.batch_size': 64,
            'cluster.n_jobs': -1,
            'recall.faiss.index_type': 'flat_ip',
        }
        
        if current is None:
            return True
        
        if config_path in default_values and current == default_values[config_path]:
            return True
        
        # 设备自动降级
        if config_path == 'embeddings.device' and current == 'cuda' and optimized == 'cpu':
            return True
            
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return dict(self.data)


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
    
    # 创建配置对象
    config = Config(data=data, source_file=path)
    
    # 环境变量映射
    env_mapping = {
        'QA_DEVICE': 'embeddings.device',
        'QA_BATCH_SIZE': 'embeddings.batch_size',
        'QA_RERANK_BATCH_SIZE': 'rerank.batch_size',
        'QA_OUTPUT_DIR': 'pipeline.output_dir',
        'QA_LOG_LEVEL': 'observe.log_level',
    }
    
    # 合并环境变量配置
    config.merge_from_env(env_mapping)
    
    return config


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
