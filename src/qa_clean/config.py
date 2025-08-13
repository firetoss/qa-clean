"""
配置管理模块
"""

import re
import os
from typing import Dict, Any

# =========================
# 配置参数（默认值）
# =========================

DEFAULTS = dict(
    model_embed_a="BAAI/bge-large-zh-v1.5",
    model_embed_b="moka-ai/m3e-large",
    model_cross_encoder="BAAI/bge-reranker-large",
    topk=150,
    cos_a_threshold=0.86,
    cos_b_threshold=0.85,
    ce_high_threshold=0.80,
    ce_mid_threshold=0.74,
    center_ce_cover=0.80,      # 中心点需覆盖比例
    center_ce_mean=0.83,        # 中心点与成员的平均CE阈值（高置信簇）
    center_ce_min_for_pair=0.82 # 双点簇最小CE
)

# PostgreSQL 连接配置
POSTGRES_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "database": os.getenv("POSTGRES_DB", "qa_clean"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
}

# 向量表配置
VECTOR_TABLE_CONFIG = {
    "table_name": os.getenv("VECTOR_TABLE", "qa_vectors"),
    "embedding_dim_a": 1024,  # bge-large-zh 维度
    "embedding_dim_b": 1024,  # m3e-large 维度
    "embedding_dim": 1024,    # 通用维度
}

# 向量存储类型配置
VECTOR_STORE_CONFIGS = {
    "faiss_gpu": {
        "name": "FAISS GPU",
        "description": "高性能GPU向量搜索，使用FAISS 1.7.2",
        "pros": [
            "高性能向量搜索",
            "GPU加速，速度极快",
            "无需外部数据库",
            "支持Python 3.9+",
            "内存中操作，速度快"
        ],
        "cons": [
            "数据不持久化",
            "内存占用较高",
            "重启后数据丢失",
            "需要GPU资源"
        ],
        "recommended_for": ["开发测试", "高性能要求", "快速原型", "GPU环境"]
    },
    "pgvector": {
        "name": "PostgreSQL + pgvector",
        "description": "企业级向量存储，基于PostgreSQL 17",
        "pros": [
            "企业级稳定性",
            "支持复杂查询",
            "可扩展性强",
            "支持事务和ACID",
            "数据持久化"
        ],
        "cons": [
            "需要PostgreSQL环境",
            "部署相对复杂",
            "资源消耗较高"
        ],
        "recommended_for": ["大规模生产", "企业环境", "需要复杂查询"]
    }
}

# 过滤关键词和模式
FILTER_KEYWORDS = [
    "地点","地址","位置","哪里","哪儿","在哪","网点",
    "营业时间","几点","上班时间","开门","关门",
    "日期","几号","工作日","周末","节假日","法定节假日",
    "营业至","截止","期间","上午","下午","晚上","时段","时间"
]

# 可能误杀但需要放行的模式（例如：办理周期/时长并非门店时间）
WHITELIST_PATTERNS = [
    r"(办理|处理|审批|审核|到账|开通).{0,4}(周期|时长|需要多长时间|多久|用时)",
    r"(退款|退费).{0,4}(周期|多久|几天)",
]

FILTER_PATTERNS = [
    r"在.+(哪|哪里|哪儿)",
    r"几点(开门|关门)",
    r"营业(到|至)",
    r"[0-9]{1,2}[:：][0-9]{2}",
    r"(工作日|周末|节假日|法定节假日)"
]

# 预编译正则表达式
FILTER_REGEX = re.compile("|".join([re.escape(k) for k in FILTER_KEYWORDS]) + "|" + "|".join(FILTER_PATTERNS))
WHITELIST_REGEXES = [re.compile(p) for p in WHITELIST_PATTERNS]

# 条件词列表
CONDITION_WORDS = [
    "需要", "须", "必须", "提供", "满足", "仅限", 
    "不支持", "不可", "除外", "以下任一", "同时"
]

# =========================
# 向量存储配置函数
# =========================

def get_recommended_store_type(environment: str = "development") -> str:
    """
    根据环境推荐向量存储类型
    
    Args:
        environment: 环境类型 (development, production, testing)
        
    Returns:
        推荐的存储类型
    """
    if environment == "development":
        return "faiss_gpu"  # 开发环境推荐FAISS GPU
    elif environment == "testing":
        return "faiss_gpu"  # 测试环境推荐FAISS GPU
    elif environment == "production":
        if os.getenv("POSTGRES_HOST"):
            return "pgvector"  # 生产环境有PostgreSQL时推荐pgvector
        else:
            return "faiss_gpu"  # 生产环境无PostgreSQL时推荐FAISS GPU
    else:
        return "faiss_gpu"  # 默认推荐FAISS GPU


def get_store_config(store_type: str) -> Dict[str, Any]:
    """
    获取存储类型配置
    
    Args:
        store_type: 存储类型
        
    Returns:
        存储配置字典
    """
    return VECTOR_STORE_CONFIGS.get(store_type, VECTOR_STORE_CONFIGS["faiss_gpu"])


def list_available_stores() -> list:
    """
    列出所有可用的存储类型
    
    Returns:
        存储类型列表
    """
    return list(VECTOR_STORE_CONFIGS.keys())
