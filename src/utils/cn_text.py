"""
中文文本字符级归一化与过滤模块
无分词设计，基于字符级处理和正则模式匹配
"""
import re
import unicodedata
from typing import List, Tuple

# 统一的中文标点集合（全角->半角映射）
CHINESE_PUNCTS = {
    '，': ',', '。': '.', '：': ':', '；': ';', '！': '!', '？': '?', 
    '（': '(', '）': ')', '【': '[', '】': ']', '「': '"', '」': '"', 
    '、': ',', '《': '"', '》': '"', '—': '-', '－': '-', '～': '~', '｜': '|'
}

# 黑名单关键词（强时效性/地点性/闲聊类问题，建议过滤）
BLACK_PATTERNS = [
    r"今天|明天|后天|昨天|大前天|本周|下周|上周|这个月|下个月|月底|年初|当月|次月|季度末",
    r"附近|哪家店|哪个店|门店|网点|营业厅|分行|支行|这里|那里|本地|当地|就近",
    r"聊聊|你好|您好|在吗|有人吗|早上好|中午好|下午好|晚上好|再见|拜拜|谢谢|不客气",
    r"测试|test|hello|hi|呵呵|哈哈|嗯|嗯嗯|好的|知道了|明白了",
]

# 白名单关键词（业务相关问题，优先保留）
WHITELIST_PATTERNS = [
    r"流程|政策|规定|制度|材料|资费|资格|条件|要求|标准|规范",
    r"发票|售后|退款|开发票|报销|发货|寄回|邮寄|快递|物流|配送",
    r"积分|账单|套餐|合同|协议|保修|理赔|年费|手续费|服务费|违约金",
    r"申请|办理|手续|证明|证件|资质|审核|批准|登记|注册|变更",
]

CONTROL_CHARS = ''.join(map(chr, list(range(0, 32)) + [127]))
CONTROL_RE = re.compile(f"[{re.escape(CONTROL_CHARS)}]")
MULTI_SPACE_RE = re.compile(r"\s+")


def to_halfwidth(text: str) -> str:
    """全角字符转半角字符"""
    res = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:  # 全角空格
            res.append(' ')
        elif 0xFF01 <= code <= 0xFF5E:  # 全角ASCII
            res.append(chr(code - 0xFEE0))
        else:
            res.append(ch)
    return ''.join(res)


def normalize_zh(text: str) -> str:
    """
    中文文本字符级归一化处理
    - Unicode标准化 (NFKC)
    - 全角转半角
    - 中文标点统一
    - 去除控制字符
    - 多空格合并
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Unicode标准化：兼容字符转标准形式
    text = unicodedata.normalize('NFKC', text)
    
    # 全角转半角
    text = to_halfwidth(text)
    
    # 去除控制字符（换行、制表符等）
    text = CONTROL_RE.sub(' ', text)
    
    # 中文标点转英文标点
    text = ''.join(CHINESE_PUNCTS.get(c, c) for c in text)
    
    # 多空格合并为单空格并去除首尾空白
    text = MULTI_SPACE_RE.sub(' ', text).strip()
    
    return text


def match_any(patterns: List[str], text: str) -> bool:
    """检查文本是否匹配任一正则模式"""
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False


def filter_reason(q: str) -> Tuple[bool, str]:
    """
    基于规则过滤问题文本
    优先级：白名单 > 黑名单 > 默认保留
    
    Args:
        q: 问题文本
        
    Returns:
        (是否保留, 过滤原因)
    """
    qn = normalize_zh(q)
    
    # 白名单优先保留（业务相关）
    if match_any(WHITELIST_PATTERNS, qn):
        return True, 'whitelist'
    
    # 黑名单过滤（时效性/地点性/闲聊）
    if match_any(BLACK_PATTERNS, qn):
        return False, 'blacklist'
    
    # 其他情况默认保留
    return True, 'keep'
