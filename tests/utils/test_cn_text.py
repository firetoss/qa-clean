#!/usr/bin/env python3
"""
中文文本处理模块的单元测试
测试 normalize_zh, filter_reason, to_halfwidth 等核心功能
"""

import unittest
from src.utils.cn_text import (
    normalize_zh, 
    filter_reason, 
    to_halfwidth,
    match_any,
    CHINESE_PUNCTS,
    BLACK_PATTERNS,
    WHITELIST_PATTERNS
)


class TestChineseTextProcessing(unittest.TestCase):
    """中文文本处理功能测试"""

    def test_to_halfwidth(self):
        """测试全角转半角功能"""
        # 测试全角数字和字母
        self.assertEqual(to_halfwidth("１２３ＡＢＣ"), "123ABC")
        
        # 测试全角标点
        self.assertEqual(to_halfwidth("！？（）"), "!?()")
        
        # 测试全角空格
        self.assertEqual(to_halfwidth("你　好"), "你 好")
        
        # 测试混合情况
        self.assertEqual(to_halfwidth("Ｈｅｌｌｏ　Ｗｏｒｌｄ！"), "Hello World!")
        
        # 测试中文字符不变
        self.assertEqual(to_halfwidth("中文不变"), "中文不变")
        
        # 测试空字符串
        self.assertEqual(to_halfwidth(""), "")

    def test_normalize_zh(self):
        """测试中文文本归一化功能"""
        # 测试全角转半角
        self.assertEqual(normalize_zh("１２３"), "123")
        
        # 测试中文标点转换
        self.assertEqual(normalize_zh("你好，世界！"), "你好,世界!")
        
        # 测试多空格合并
        self.assertEqual(normalize_zh("你好    世界"), "你好 世界")
        
        # 测试控制字符去除
        self.assertEqual(normalize_zh("你好\n\t世界"), "你好 世界")
        
        # 测试首尾空白去除
        self.assertEqual(normalize_zh("  你好世界  "), "你好世界")
        
        # 测试Unicode标准化（组合字符）
        self.assertEqual(normalize_zh("é"), "é")  # 组合重音符转标准字符
        
        # 测试非字符串输入
        self.assertEqual(normalize_zh(123), "123")
        self.assertEqual(normalize_zh(None), "None")

    def test_match_any(self):
        """测试模式匹配功能"""
        patterns = [r"今天|明天", r"附近|哪家店"]
        
        # 测试匹配成功
        self.assertTrue(match_any(patterns, "今天天气如何"))
        self.assertTrue(match_any(patterns, "附近有什么店"))
        
        # 测试匹配失败
        self.assertFalse(match_any(patterns, "昨天的天气"))
        
        # 测试空模式列表
        self.assertFalse(match_any([], "任何文本"))
        
        # 测试空文本
        self.assertFalse(match_any(patterns, ""))

    def test_filter_reason_whitelist(self):
        """测试白名单过滤逻辑"""
        # 白名单关键词应该被保留
        test_cases = [
            ("请问发票流程是什么？", True, "whitelist"),
            ("售后政策如何？", True, "whitelist"),
            ("积分规定是什么？", True, "whitelist"),
            ("申请材料有哪些？", True, "whitelist"),
        ]
        
        for question, expected_keep, expected_reason in test_cases:
            with self.subTest(question=question):
                keep, reason = filter_reason(question)
                self.assertEqual(keep, expected_keep)
                self.assertEqual(reason, expected_reason)

    def test_filter_reason_blacklist(self):
        """测试黑名单过滤逻辑"""
        # 黑名单关键词应该被过滤
        test_cases = [
            ("今天营业吗？", False, "blacklist"),
            ("附近哪家店比较好？", False, "blacklist"),
            ("你好，在吗？", False, "blacklist"),
            ("测试一下", False, "blacklist"),
            ("明天几点开门？", False, "blacklist"),
        ]
        
        for question, expected_keep, expected_reason in test_cases:
            with self.subTest(question=question):
                keep, reason = filter_reason(question)
                self.assertEqual(keep, expected_keep)
                self.assertEqual(reason, expected_reason)

    def test_filter_reason_default_keep(self):
        """测试默认保留逻辑"""
        # 既不在白名单也不在黑名单的问题应该被保留
        test_cases = [
            ("这是一个普通问题", True, "keep"),
            ("如何使用产品功能？", True, "keep"),
            ("价格是多少？", True, "keep"),
        ]
        
        for question, expected_keep, expected_reason in test_cases:
            with self.subTest(question=question):
                keep, reason = filter_reason(question)
                self.assertEqual(keep, expected_keep)
                self.assertEqual(reason, expected_reason)

    def test_filter_reason_priority(self):
        """测试白名单优先级高于黑名单"""
        # 同时包含白名单和黑名单关键词时，白名单优先
        question = "今天的发票政策是什么？"  # 包含"今天"(黑名单)和"发票政策"(白名单)
        keep, reason = filter_reason(question)
        self.assertTrue(keep)
        self.assertEqual(reason, "whitelist")

    def test_chinese_puncts_mapping(self):
        """测试中文标点映射"""
        # 验证标点映射表的正确性
        test_cases = [
            ('，', ','),
            ('。', '.'),
            ('：', ':'),
            ('；', ';'),
            ('！', '!'),
            ('？', '?'),
            ('（', '('),
            ('）', ')'),
        ]
        
        for chinese_punct, english_punct in test_cases:
            self.assertEqual(CHINESE_PUNCTS[chinese_punct], english_punct)

    def test_edge_cases(self):
        """测试边界情况"""
        # 空字符串
        keep, reason = filter_reason("")
        self.assertTrue(keep)
        self.assertEqual(reason, "keep")
        
        # 只有空格
        keep, reason = filter_reason("   ")
        self.assertTrue(keep)
        self.assertEqual(reason, "keep")
        
        # 只有标点
        keep, reason = filter_reason("！？。")
        self.assertTrue(keep)
        self.assertEqual(reason, "keep")
        
        # 数字字符串
        self.assertEqual(normalize_zh("12345"), "12345")

    def test_normalize_zh_comprehensive(self):
        """测试normalize_zh的综合功能"""
        # 复杂的混合文本
        input_text = "　　你好，世界！！！\n\t这是一个测试。　　"
        expected = "你好,世界!!! 这是一个测试."
        self.assertEqual(normalize_zh(input_text), expected)

    def test_patterns_not_empty(self):
        """测试模式列表不为空"""
        self.assertTrue(len(BLACK_PATTERNS) > 0)
        self.assertTrue(len(WHITELIST_PATTERNS) > 0)
        
        # 验证每个模式都是有效的正则表达式
        import re
        for pattern in BLACK_PATTERNS + WHITELIST_PATTERNS:
            try:
                re.compile(pattern)
            except re.error:
                self.fail(f"无效的正则表达式模式: {pattern}")


if __name__ == '__main__':
    unittest.main()
