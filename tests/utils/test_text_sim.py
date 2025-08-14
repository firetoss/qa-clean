#!/usr/bin/env python3
"""
文本相似度模块的单元测试
测试字符n-gram、Jaccard相似度、编辑距离等功能
"""

import unittest
from src.utils.text_sim import (
    char_ngrams,
    jaccard,
    ngram_jaccard,
    edit_distance
)


class TestTextSimilarity(unittest.TestCase):
    """文本相似度功能测试"""

    def test_char_ngrams_basic(self):
        """测试基本字符n-gram生成"""
        # 测试3-gram
        text = "hello"
        expected = {"hel", "ell", "llo"}
        result = char_ngrams(text, 3)
        self.assertEqual(result, expected)

    def test_char_ngrams_chinese(self):
        """测试中文字符n-gram"""
        text = "你好世界"
        expected = {"你好世", "好世界"}
        result = char_ngrams(text, 3)
        self.assertEqual(result, expected)

    def test_char_ngrams_different_sizes(self):
        """测试不同n值的n-gram"""
        text = "abc"
        
        # 1-gram
        result1 = char_ngrams(text, 1)
        self.assertEqual(result1, {"a", "b", "c"})
        
        # 2-gram
        result2 = char_ngrams(text, 2)
        self.assertEqual(result2, {"ab", "bc"})
        
        # 3-gram
        result3 = char_ngrams(text, 3)
        self.assertEqual(result3, {"abc"})
        
        # 4-gram (超过文本长度)
        result4 = char_ngrams(text, 4)
        self.assertEqual(result4, set())

    def test_char_ngrams_edge_cases(self):
        """测试n-gram的边界情况"""
        # 空字符串
        self.assertEqual(char_ngrams("", 3), set())
        
        # 单字符
        self.assertEqual(char_ngrams("a", 3), set())
        
        # n=0
        self.assertEqual(char_ngrams("hello", 0), set())
        
        # n为负数
        self.assertEqual(char_ngrams("hello", -1), set())

    def test_char_ngrams_spaces_and_punctuation(self):
        """测试包含空格和标点的n-gram"""
        text = "你好，世界！"
        result = char_ngrams(text, 3)
        expected = {"你好，", "好，世", "，世界", "世界！"}
        self.assertEqual(result, expected)

    def test_jaccard_basic(self):
        """测试基本Jaccard相似度"""
        set1 = {"a", "b", "c"}
        set2 = {"b", "c", "d"}
        
        # 交集: {b, c} = 2
        # 并集: {a, b, c, d} = 4
        # Jaccard = 2/4 = 0.5
        result = jaccard(set1, set2)
        self.assertAlmostEqual(result, 0.5, places=6)

    def test_jaccard_identical_sets(self):
        """测试相同集合的Jaccard相似度"""
        set1 = {"a", "b", "c"}
        set2 = {"a", "b", "c"}
        
        result = jaccard(set1, set2)
        self.assertAlmostEqual(result, 1.0, places=6)

    def test_jaccard_disjoint_sets(self):
        """测试不相交集合的Jaccard相似度"""
        set1 = {"a", "b", "c"}
        set2 = {"d", "e", "f"}
        
        result = jaccard(set1, set2)
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_jaccard_empty_sets(self):
        """测试空集合的Jaccard相似度"""
        # 两个空集合
        result1 = jaccard(set(), set())
        self.assertAlmostEqual(result1, 0.0, places=6)
        
        # 一个空集合
        result2 = jaccard({"a"}, set())
        self.assertAlmostEqual(result2, 0.0, places=6)
        
        result3 = jaccard(set(), {"a"})
        self.assertAlmostEqual(result3, 0.0, places=6)

    def test_jaccard_with_lists(self):
        """测试Jaccard相似度接受列表输入"""
        list1 = ["a", "b", "c", "a"]  # 重复元素
        list2 = ["b", "c", "d"]
        
        # 应该自动转为集合处理
        result = jaccard(list1, list2)
        expected = jaccard({"a", "b", "c"}, {"b", "c", "d"})
        self.assertAlmostEqual(result, expected, places=6)

    def test_ngram_jaccard_identical_strings(self):
        """测试相同字符串的n-gram Jaccard相似度"""
        text = "你好世界"
        result = ngram_jaccard(text, text, 3)
        self.assertAlmostEqual(result, 1.0, places=6)

    def test_ngram_jaccard_similar_strings(self):
        """测试相似字符串的n-gram Jaccard相似度"""
        text1 = "如何开发票"
        text2 = "怎么开发票"
        
        result = ngram_jaccard(text1, text2, 3)
        
        # 手动计算验证
        ngrams1 = char_ngrams(text1, 3)
        ngrams2 = char_ngrams(text2, 3)
        expected = jaccard(ngrams1, ngrams2)
        
        self.assertAlmostEqual(result, expected, places=6)
        self.assertGreater(result, 0)  # 应该有一定相似度

    def test_ngram_jaccard_different_strings(self):
        """测试不同字符串的n-gram Jaccard相似度"""
        text1 = "完全不同的文本"
        text2 = "Another text"
        
        result = ngram_jaccard(text1, text2, 3)
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_ngram_jaccard_empty_strings(self):
        """测试空字符串的n-gram Jaccard相似度"""
        result1 = ngram_jaccard("", "", 3)
        self.assertAlmostEqual(result1, 0.0, places=6)
        
        result2 = ngram_jaccard("hello", "", 3)
        self.assertAlmostEqual(result2, 0.0, places=6)

    def test_ngram_jaccard_short_strings(self):
        """测试短字符串的n-gram Jaccard相似度"""
        # 字符串长度小于n
        result = ngram_jaccard("ab", "cd", 3)
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_edit_distance_identical_strings(self):
        """测试相同字符串的编辑距离"""
        result = edit_distance("hello", "hello")
        self.assertEqual(result, 0)

    def test_edit_distance_empty_strings(self):
        """测试空字符串的编辑距离"""
        self.assertEqual(edit_distance("", ""), 0)
        self.assertEqual(edit_distance("hello", ""), 5)
        self.assertEqual(edit_distance("", "world"), 5)

    def test_edit_distance_insertion(self):
        """测试插入操作的编辑距离"""
        # "cat" -> "cart" (插入'r')
        result = edit_distance("cat", "cart")
        self.assertEqual(result, 1)
        
        # "hello" -> "helloworld" (插入"world")
        result = edit_distance("hello", "helloworld")
        self.assertEqual(result, 5)

    def test_edit_distance_deletion(self):
        """测试删除操作的编辑距离"""
        # "cart" -> "cat" (删除'r')
        result = edit_distance("cart", "cat")
        self.assertEqual(result, 1)
        
        # "helloworld" -> "hello" (删除"world")
        result = edit_distance("helloworld", "hello")
        self.assertEqual(result, 5)

    def test_edit_distance_substitution(self):
        """测试替换操作的编辑距离"""
        # "cat" -> "bat" (替换'c'为'b')
        result = edit_distance("cat", "bat")
        self.assertEqual(result, 1)
        
        # "hello" -> "world" (多个替换)
        result = edit_distance("hello", "world")
        self.assertEqual(result, 4)

    def test_edit_distance_complex(self):
        """测试复杂的编辑距离"""
        # 经典例子: "kitten" -> "sitting"
        result = edit_distance("kitten", "sitting")
        self.assertEqual(result, 3)
        
        # 中文例子
        result = edit_distance("如何开发票", "怎么开发票")
        self.assertEqual(result, 2)  # "如何" -> "怎么"

    def test_edit_distance_chinese(self):
        """测试中文字符的编辑距离"""
        result1 = edit_distance("你好", "你好")
        self.assertEqual(result1, 0)
        
        result2 = edit_distance("你好", "您好")
        self.assertEqual(result2, 1)
        
        result3 = edit_distance("你好世界", "你好")
        self.assertEqual(result3, 2)

    def test_edit_distance_symmetric(self):
        """测试编辑距离的对称性"""
        pairs = [
            ("hello", "world"),
            ("cat", "dog"),
            ("你好", "世界"),
            ("", "test"),
        ]
        
        for s1, s2 in pairs:
            with self.subTest(s1=s1, s2=s2):
                dist1 = edit_distance(s1, s2)
                dist2 = edit_distance(s2, s1)
                self.assertEqual(dist1, dist2)

    def test_comprehensive_similarity_workflow(self):
        """测试完整的相似度计算工作流"""
        # 模拟实际应用场景
        questions = [
            "如何开发票？",
            "怎么开发票？",
            "发票如何开具？",
            "退款流程是什么？"
        ]
        
        # 计算所有问题对的相似度
        similarities = []
        for i in range(len(questions)):
            for j in range(i + 1, len(questions)):
                q1, q2 = questions[i], questions[j]
                
                # n-gram Jaccard相似度
                jaccard_sim = ngram_jaccard(q1, q2, 3)
                
                # 编辑距离（转换为相似度）
                edit_dist = edit_distance(q1, q2)
                max_len = max(len(q1), len(q2))
                edit_sim = 1 - (edit_dist / max_len) if max_len > 0 else 1.0
                
                similarities.append({
                    'pair': (q1, q2),
                    'jaccard': jaccard_sim,
                    'edit_sim': edit_sim
                })
        
        # 验证相似问题有更高的相似度
        # "如何开发票？" 和 "怎么开发票？" 应该很相似
        similar_pair = None
        different_pair = None
        
        for sim in similarities:
            q1, q2 = sim['pair']
            if "开发票" in q1 and "开发票" in q2:
                similar_pair = sim
            elif "开发票" in q1 and "退款" in q2:
                different_pair = sim
        
        if similar_pair and different_pair:
            # 相似问题的相似度应该更高
            self.assertGreater(
                similar_pair['jaccard'], 
                different_pair['jaccard']
            )

    def test_performance_characteristics(self):
        """测试性能特征（基本正确性验证）"""
        # 测试较长文本的处理
        long_text1 = "这是一个很长的文本" * 10
        long_text2 = "这是一个很长的内容" * 10
        
        # 应该能正常计算，不抛出异常
        jaccard_result = ngram_jaccard(long_text1, long_text2, 3)
        edit_result = edit_distance(long_text1[:20], long_text2[:20])  # 限制长度避免过慢
        
        self.assertIsInstance(jaccard_result, float)
        self.assertIsInstance(edit_result, int)
        self.assertGreaterEqual(jaccard_result, 0.0)
        self.assertLessEqual(jaccard_result, 1.0)
        self.assertGreaterEqual(edit_result, 0)


class TestTextSimEdgeCases(unittest.TestCase):
    """文本相似度边界情况测试"""

    def test_special_characters(self):
        """测试特殊字符处理"""
        # 包含emoji和特殊字符
        text1 = "你好😊！@#$%"
        text2 = "您好😊！@#$%"
        
        # 应该能正常处理
        jaccard_result = ngram_jaccard(text1, text2, 3)
        edit_result = edit_distance(text1, text2)
        
        self.assertIsInstance(jaccard_result, float)
        self.assertIsInstance(edit_result, int)

    def test_unicode_normalization(self):
        """测试Unicode标准化对相似度的影响"""
        # 组合字符 vs 预组合字符
        text1 = "café"  # é 作为单个字符
        text2 = "cafe\u0301"  # e + 组合重音符
        
        # 编辑距离可能不同，但应该都能正常计算
        edit_result = edit_distance(text1, text2)
        jaccard_result = ngram_jaccard(text1, text2, 2)
        
        self.assertIsInstance(edit_result, int)
        self.assertIsInstance(jaccard_result, float)

    def test_very_long_strings(self):
        """测试很长字符串的处理"""
        # 生成长字符串
        long_text = "a" * 1000
        
        # n-gram应该能处理
        ngrams = char_ngrams(long_text, 3)
        self.assertGreater(len(ngrams), 0)
        
        # Jaccard相似度
        result = ngram_jaccard(long_text, long_text, 3)
        self.assertAlmostEqual(result, 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
