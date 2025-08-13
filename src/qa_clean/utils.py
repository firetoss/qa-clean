"""
工具函数模块
"""

import re
import unicodedata
from typing import Set, List, Dict, Any

from .config import FILTER_REGEX, WHITELIST_REGEXES, CONDITION_WORDS


def normalize_text(s: str) -> str:
    """文本标准化处理"""
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = re.sub(r"[ \t\r\n]+", " ", s)
    # 简单统一一些常见符号
    s = s.replace("（", "(").replace("）", ")").replace("：", ":").replace("，", ",").replace("；", ";")
    return s


def is_whitelisted(q: str) -> bool:
    """检查是否在白名单中"""
    for rgx in WHITELIST_REGEXES:
        if rgx.search(q):
            return True
    return False


def should_filter_location_time(q: str) -> bool:
    """强过滤门店/时间类问题，但白名单可放行"""
    qn = normalize_text(q)
    if not qn:
        return False
    if is_whitelisted(qn):
        return False
    return bool(FILTER_REGEX.search(qn))


def extract_trigrams(s: str) -> Set[str]:
    """提取三元组特征"""
    s = normalize_text(s)
    s = re.sub(r"\s+", "", s)
    if len(s) < 3:
        return {s} if s else set()
    return {s[i:i+3] for i in range(len(s)-2)}


def jaccard_trigram(a: str, b: str) -> float:
    """计算三元组Jaccard相似度"""
    A, B = extract_trigrams(a), extract_trigrams(b)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    uni = len(A | B)
    return inter / uni if uni > 0 else 0.0


def extract_structured_facts(text: str) -> Dict[str, Any]:
    """
    轻量级要点抽取：数字、金额、百分比、条件词等。
    可按需扩展为更强的规则或NLP组件。
    """
    t = normalize_text(text)
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", t)
    percents = re.findall(r"\b\d+(?:\.\d+)?\s*%|\b百分之\d+(?:\.\d+)?", t)
    money = re.findall(r"(?:¥|人民币|RMB|CNY)\s*\d+(?:\.\d+)?|(\d+(?:\.\d+)?\s*元)", t)
    
    # 条件词
    cond_words = [w for w in CONDITION_WORDS if w in t]
    
    return {
        "numbers": numbers,
        "percents": percents,
        "money": [m if isinstance(m, str) else "".join([x for x in m if x]) for m in money],
        "cond_words": cond_words
    }


def facts_conflict(fa: Dict[str, Any], fb: Dict[str, Any]) -> bool:
    """
    简化冲突判断：
    - 若存在显著数值差异（交集为空且均非空）→ 冲突
    - 若金额/百分比集合无交集且均非空 → 冲突
    注：此为保守规则，减少误合并
    """
    def nonempty_intersect(a: List[str], b: List[str]) -> bool:
        return bool(set(a) & set(b))
    
    # 数值冲突
    if fa["numbers"] and fb["numbers"] and not nonempty_intersect(fa["numbers"], fb["numbers"]):
        return True
    # 金额冲突
    if fa["money"] and fb["money"] and not nonempty_intersect(fa["money"], fb["money"]):
        return True
    # 百分比冲突
    if fa["percents"] and fb["percents"] and not nonempty_intersect(fa["percents"], fb["percents"]):
        return True
    return False


def merge_answers(ans_list: List[str]) -> tuple[str, bool]:
    """
    合并答案：优先众数；若并列，选信息更全（长度更长）者。
    返回：(合并答案, 是否合并过多条)
    """
    norm = [normalize_text(a) for a in ans_list if normalize_text(a)]
    if not norm:
        return "", False
    
    from collections import Counter
    cnt = Counter(norm)
    most = cnt.most_common()
    top_freq = most[0][1]
    candidates = [a for a, c in most if c == top_freq]
    
    if len(candidates) == 1:
        return candidates[0], len(set(norm)) > 1
    
    # 多个并列，选最长
    best = sorted(candidates, key=lambda x: (len(x), x))[-1]
    return best, True
