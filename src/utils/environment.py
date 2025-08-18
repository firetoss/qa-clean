"""
环境管理模块 - 处理依赖检查、设备检测和环境配置

特性:
- GPU/CPU设备自动检测与配置
- 依赖库可用性检查
- 环境变量管理
- 优雅的错误处理和回退机制
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# 抑制第三方库警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class SystemInfo:
    """系统信息"""
    python_version: str
    platform: str
    cuda_available: bool
    gpu_count: int
    gpu_memory: List[float]  # GB
    cpu_count: int


@dataclass
class DependencyStatus:
    """依赖状态"""
    name: str
    available: bool
    version: Optional[str] = None
    error: Optional[str] = None


class EnvironmentManager:
    """环境管理器"""
    
    def __init__(self):
        self._sys_info: Optional[SystemInfo] = None
        self._deps_status: Dict[str, DependencyStatus] = {}
        self._setup_environment_variables()
    
    def _setup_environment_variables(self) -> None:
        """设置环境变量"""
        # HuggingFace镜像设置
        if 'HF_ENDPOINT' not in os.environ:
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        # 缓存目录设置
        if 'HF_HOME' not in os.environ:
            cache_dir = os.path.join(Path.home(), '.cache', 'huggingface')
            os.environ['HF_HOME'] = cache_dir
        
        # 确保缓存目录存在
        os.makedirs(os.environ['HF_HOME'], exist_ok=True)
        
        # CUDA相关设置
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            # 默认使用所有可用GPU
            pass
        
        # 设置tokenizers并行度（避免警告）
        if 'TOKENIZERS_PARALLELISM' not in os.environ:
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    def get_system_info(self) -> SystemInfo:
        """获取系统信息"""
        if self._sys_info is None:
            self._sys_info = self._detect_system_info()
        return self._sys_info
    
    def _detect_system_info(self) -> SystemInfo:
        """检测系统信息"""
        import platform
        import multiprocessing as mp
        
        # 基本信息
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        platform_name = platform.system()
        cpu_count = mp.cpu_count()
        
        # GPU信息检测
        cuda_available = False
        gpu_count = 0
        gpu_memory = []
        
        try:
            import torch
            if torch.cuda.is_available():
                cuda_available = True
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / (1024**3)
                    gpu_memory.append(memory_gb)
        except ImportError:
            pass
        
        return SystemInfo(
            python_version=python_version,
            platform=platform_name,
            cuda_available=cuda_available,
            gpu_count=gpu_count,
            gpu_memory=gpu_memory,
            cpu_count=cpu_count
        )
    
    def check_dependency(self, name: str, import_name: Optional[str] = None) -> DependencyStatus:
        """检查单个依赖"""
        if name in self._deps_status:
            return self._deps_status[name]
        
        import_name = import_name or name
        
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            status = DependencyStatus(name=name, available=True, version=version)
        except ImportError as e:
            status = DependencyStatus(name=name, available=False, error=str(e))
        
        self._deps_status[name] = status
        return status
    
    def check_core_dependencies(self) -> Dict[str, DependencyStatus]:
        """检查核心依赖"""
        core_deps = [
            ('numpy', 'numpy'),
            ('pandas', 'pandas'), 
            ('torch', 'torch'),
            ('transformers', 'transformers'),
            ('sentence-transformers', 'sentence_transformers'),
            ('faiss', 'faiss'),
            ('sklearn', 'sklearn'),
            ('yaml', 'yaml'),
            ('tqdm', 'tqdm'),
        ]
        
        results = {}
        for name, import_name in core_deps:
            results[name] = self.check_dependency(name, import_name)
        
        return results
    
    def check_optional_dependencies(self) -> Dict[str, DependencyStatus]:
        """检查可选依赖"""
        optional_deps = [
            ('cudf', 'cudf'),          # GPU加速DataFrame
            ('cugraph', 'cugraph'),    # GPU图算法
            ('networkx', 'networkx'),  # 图算法库
            ('igraph', 'igraph'),      # 图算法库
            ('leidenalg', 'leidenalg'), # Leiden聚类
            ('matplotlib', 'matplotlib'), # 图表
            ('rapidfuzz', 'rapidfuzz'), # 快速字符串匹配
        ]
        
        results = {}
        for name, import_name in optional_deps:
            results[name] = self.check_dependency(name, import_name)
        
        return results
    
    def get_recommended_device(self) -> str:
        """获取推荐设备配置"""
        sys_info = self.get_system_info()
        
        if sys_info.cuda_available and sys_info.gpu_count > 0:
            # 检查GPU内存是否足够
            min_memory = min(sys_info.gpu_memory) if sys_info.gpu_memory else 0
            if min_memory >= 4.0:  # 至少4GB GPU内存
                return 'cuda'
        
        return 'cpu'
    
    def get_optimal_batch_size(self, model_type: str = 'embedding') -> int:
        """获取优化的批次大小"""
        sys_info = self.get_system_info()
        device = self.get_recommended_device()
        
        if device == 'cuda' and sys_info.gpu_memory:
            # 基于GPU内存调整批次大小
            max_memory = max(sys_info.gpu_memory)
            if model_type == 'embedding':
                if max_memory >= 20:
                    return 128
                elif max_memory >= 10:
                    return 64
                elif max_memory >= 6:
                    return 32
                else:
                    return 16
            elif model_type == 'cross_encoder':
                # 交叉编码器更耗内存
                if max_memory >= 20:
                    return 64
                elif max_memory >= 10:
                    return 32
                elif max_memory >= 6:
                    return 16
                else:
                    return 8
        
        # CPU回退
        if model_type == 'embedding':
            return 32
        else:
            return 16
    
    def validate_environment(self) -> Tuple[bool, List[str]]:
        """验证环境配置，返回(是否通过, 错误列表)"""
        errors = []
        
        # 检查Python版本
        if sys.version_info < (3, 8):
            errors.append(f"Python版本过低: {sys.version_info}, 需要3.8+")
        
        # 检查核心依赖
        core_deps = self.check_core_dependencies()
        missing_core = [name for name, status in core_deps.items() if not status.available]
        
        if missing_core:
            errors.append(f"缺少核心依赖: {', '.join(missing_core)}")
        
        # 检查FAISS GPU支持（如果有GPU）
        sys_info = self.get_system_info()
        if sys_info.cuda_available:
            faiss_status = self.check_dependency('faiss-gpu', 'faiss')
            if not faiss_status.available:
                errors.append("检测到GPU但FAISS GPU版本不可用，将使用CPU版本")
        
        return len(errors) == 0, errors
    
    def get_environment_summary(self) -> Dict[str, any]:
        """获取环境摘要"""
        sys_info = self.get_system_info()
        core_deps = self.check_core_dependencies()
        optional_deps = self.check_optional_dependencies()
        
        return {
            'system': {
                'python_version': sys_info.python_version,
                'platform': sys_info.platform,
                'cpu_count': sys_info.cpu_count,
                'cuda_available': sys_info.cuda_available,
                'gpu_count': sys_info.gpu_count,
                'gpu_memory_gb': sys_info.gpu_memory,
            },
            'environment_variables': {
                'HF_ENDPOINT': os.environ.get('HF_ENDPOINT'),
                'HF_HOME': os.environ.get('HF_HOME'),
                'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
            },
            'core_dependencies': {
                name: {'available': status.available, 'version': status.version}
                for name, status in core_deps.items()
            },
            'optional_dependencies': {
                name: {'available': status.available, 'version': status.version}
                for name, status in optional_deps.items()
            },
            'recommendations': {
                'device': self.get_recommended_device(),
                'embedding_batch_size': self.get_optimal_batch_size('embedding'),
                'rerank_batch_size': self.get_optimal_batch_size('cross_encoder'),
            }
        }


# 全局环境管理器实例
_env_manager: Optional[EnvironmentManager] = None


def get_env_manager() -> EnvironmentManager:
    """获取全局环境管理器"""
    global _env_manager
    if _env_manager is None:
        _env_manager = EnvironmentManager()
    return _env_manager


def setup_project_path() -> None:
    """设置项目路径到Python路径"""
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def ensure_device_compatibility(device: str) -> str:
    """确保设备兼容性"""
    env = get_env_manager()
    sys_info = env.get_system_info()
    
    if device == 'cuda' and not sys_info.cuda_available:
        return 'cpu'
    return device


def get_optimal_config_adjustments() -> Dict[str, any]:
    """获取基于当前环境的配置调整建议"""
    env = get_env_manager()
    sys_info = env.get_system_info()
    
    adjustments = {}
    
    # 设备配置
    adjustments['device'] = env.get_recommended_device()
    
    # 批次大小调整
    adjustments['embedding_batch_size'] = env.get_optimal_batch_size('embedding')
    adjustments['rerank_batch_size'] = env.get_optimal_batch_size('cross_encoder')
    
    # 并行度配置
    if sys_info.cpu_count >= 8:
        adjustments['n_jobs'] = min(8, sys_info.cpu_count)
    else:
        adjustments['n_jobs'] = max(1, sys_info.cpu_count - 1)
    
    # FAISS配置调整
    if sys_info.cuda_available:
        adjustments['faiss_index_type'] = 'flat_ip'  # GPU上使用精确检索
    else:
        adjustments['faiss_index_type'] = 'ivf_flat_ip'  # CPU上使用近似检索
    
    return adjustments
