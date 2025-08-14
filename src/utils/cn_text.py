import re
import unicodedata
from typing import List, Tuple

# 统一的中文标点集合（部分）
CHINESE_PUNCTS = {
    '，': ',', '。': '.', '：': ':', '；': ';', '！': '!', '？': '?', '（': '(', '）': ')',
    '【': '[', '】': ']', '「': '"', '」': '"', '、': ',', '《': '"', '》': '"', '—': '-', '－': '-', '～': '~', '｜': '|'
}

# 黑名单关键词（强时效/地点/闲聊）
BLACK_PATTERNS = [
    r"今天|明天|后天|本周|下周|这个月|月底|年初|当月|次月",
    r"附近|哪家店|门店|网点|营业厅|分行|支行",
    r"聊聊|你好|您好|在吗|有人吗|早上好|晚上好|再见",
]

# 白名单关键词（业务相关）
WHITELIST_PATTERNS = [
    r"流程|政策|材料|资费|资格|发票|售后|退款|开发票|报销|发货|寄回|积分|账单|套餐|合同|保修|理赔|年费|手续费",
]

CONTROL_CHARS = ''.join(map(chr, list(range(0, 32)) + [127]))
CONTROL_RE = re.compile(f"[{re.escape(CONTROL_CHARS)}]")
MULTI_SPACE_RE = re.compile(r"\s+")


def to_halfwidth(text: str) -> str:
    res = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            res.append(' ')
        elif 0xFF01 <= code <= 0xFF5E:
            res.append(chr(code - 0xFEE0))
        else:
            res.append(ch)
    return ''.join(res)


def normalize_zh(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize('NFKC', text)
    text = to_halfwidth(text)
    text = CONTROL_RE.sub(' ', text)
    text = ''.join(CHINESE_PUNCTS.get(c, c) for c in text)
    text = MULTI_SPACE_RE.sub(' ', text).strip()
    return text


def match_any(patterns: List[str], text: str) -> bool:
    for p in patterns:
        if re.search(p, text):
            return True
    return False


def filter_reason(q: str) -> Tuple[bool, str]:
    qn = normalize_zh(q)
    # 白名单优先保留
    if match_any(WHITELIST_PATTERNS, qn):
        return True, 'whitelist'
    # 命中黑名单则丢弃
    if match_any(BLACK_PATTERNS, qn):
        return False, 'blacklist'
    # 其他都保留
    return True, 'keep'
