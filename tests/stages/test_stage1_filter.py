#!/usr/bin/env python3
"""
Stage1过滤模块的单元测试
测试数据加载、文本规范化、过滤逻辑等功能
"""

import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
from src.stages.stage1_filter import _load_or_sample, run


class TestStage1Filter(unittest.TestCase):
    """Stage1过滤功能测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('src.stages.stage1_filter.read_data_file')
    def test_load_or_sample_success(self, mock_read_data_file):
        """测试成功加载数据文件"""
        # 模拟pandas DataFrame
        mock_df = MagicMock()
        mock_df.columns = ['id', 'question', 'answer', 'extra']
        mock_read_data_file.return_value = mock_df
        
        # 测试文件存在且列齐全
        test_path = os.path.join(self.temp_dir, "test.parquet")
        with open(test_path, 'w') as f:
            f.write("dummy")
        
        result = _load_or_sample(test_path, 'id', 'question', 'answer')
        
        self.assertEqual(result, mock_df)
        mock_read_data_file.assert_called_once_with(test_path)

    @patch('src.stages.stage1_filter.read_data_file')
    def test_load_or_sample_missing_columns(self, mock_read_data_file):
        """测试缺少必需列的情况"""
        # 模拟缺少列的DataFrame
        mock_df = MagicMock()
        mock_df.columns = ['id', 'question']  # 缺少answer列
        mock_read_data_file.return_value = mock_df
        
        test_path = os.path.join(self.temp_dir, "test.parquet")
        with open(test_path, 'w') as f:
            f.write("dummy")
        
        # 应该返回示例数据
        result = _load_or_sample(test_path, 'id', 'question', 'answer')
        
        # 验证返回的是示例数据
        self.assertIsNotNone(result)
        # 验证有正确的列
        if hasattr(result, 'columns'):
            required_cols = {'id', 'question', 'answer'}
            self.assertTrue(required_cols.issubset(set(result.columns)))

    def test_load_or_sample_file_not_exist(self):
        """测试文件不存在的情况"""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.parquet")
        
        # 应该返回示例数据而不是抛出异常
        result = _load_or_sample(nonexistent_path, 'id', 'question', 'answer')
        
        self.assertIsNotNone(result)

    @patch('src.stages.stage1_filter.read_data_file')
    def test_load_or_sample_read_error(self, mock_read_data_file):
        """测试读取文件错误的情况"""
        # 模拟读取错误
        mock_read_data_file.side_effect = Exception("读取错误")
        
        test_path = os.path.join(self.temp_dir, "test.parquet")
        with open(test_path, 'w') as f:
            f.write("dummy")
        
        # 应该返回示例数据
        result = _load_or_sample(test_path, 'id', 'question', 'answer')
        
        self.assertIsNotNone(result)

    @patch('src.stages.stage1_filter.pd')
    def test_load_or_sample_sample_data_structure(self, mock_pd):
        """测试示例数据的结构"""
        # 模拟pandas DataFrame构造
        mock_df = MagicMock()
        mock_pd.DataFrame.return_value = mock_df
        
        result = _load_or_sample("nonexistent.parquet", 'custom_id', 'custom_q', 'custom_a')
        
        # 验证使用了正确的列名
        call_args = mock_pd.DataFrame.call_args[0][0]
        self.assertIn('custom_id', call_args)
        self.assertIn('custom_q', call_args)
        self.assertIn('custom_a', call_args)
        
        # 验证示例数据有内容
        self.assertEqual(len(call_args['custom_id']), 5)
        self.assertEqual(len(call_args['custom_q']), 5)
        self.assertEqual(len(call_args['custom_a']), 5)

    @patch('src.stages.stage1_filter.load_config')
    @patch('src.stages.stage1_filter.ensure_output_dir')
    @patch('src.stages.stage1_filter.StatsRecorder')
    @patch('src.stages.stage1_filter._load_or_sample')
    @patch('src.stages.stage1_filter.normalize_zh')
    @patch('src.stages.stage1_filter.filter_reason')
    @patch('src.stages.stage1_filter.write_parquet')
    def test_run_function_complete_workflow(self, mock_write_parquet, mock_filter_reason, 
                                          mock_normalize_zh, mock_load_or_sample, 
                                          mock_stats_recorder, mock_ensure_output_dir, 
                                          mock_load_config):
        """测试run函数的完整工作流"""
        # 设置mocks
        mock_config = MagicMock()
        mock_config.get.side_effect = lambda key, default=None: {
            'observe.stats_path': f"{self.temp_dir}/stats.json",
            'data.input_path': 'test.parquet',
            'data.id_col': 'id',
            'data.q_col': 'question',
            'data.a_col': 'answer'
        }.get(key, default)
        mock_load_config.return_value = mock_config
        mock_ensure_output_dir.return_value = self.temp_dir
        
        # 模拟数据框
        mock_df = MagicMock()
        mock_df.shape = [100, 3]
        mock_df.__getitem__.return_value.astype.return_value.map.return_value = None
        mock_df.__getitem__.return_value.tolist.return_value = [
            "问题1", "问题2", "问题3"
        ]
        mock_load_or_sample.return_value = mock_df
        
        # 模拟文本处理
        mock_normalize_zh.side_effect = lambda x: f"normalized_{x}"
        mock_filter_reason.side_effect = [
            (True, 'keep'),
            (False, 'blacklist'),
            (True, 'whitelist')
        ]
        
        # 模拟过滤后的数据框
        mock_df_clean = MagicMock()
        mock_df_clean.shape = [80, 3]
        mock_df.__getitem__.return_value.reset_index.return_value = mock_df_clean
        
        # 模拟统计记录器
        mock_stats = MagicMock()
        mock_stats_recorder.return_value = mock_stats
        
        # 运行函数
        run('test_config.yaml')
        
        # 验证调用
        mock_load_config.assert_called_once_with('test_config.yaml')
        mock_ensure_output_dir.assert_called_once_with(mock_config)
        mock_load_or_sample.assert_called_once()
        mock_write_parquet.assert_called_once()
        mock_stats.update.assert_called_once()

    @patch('src.stages.stage1_filter.load_config')
    @patch('src.stages.stage1_filter.ensure_output_dir')
    @patch('src.stages.stage1_filter.StatsRecorder')
    @patch('src.stages.stage1_filter._load_or_sample')
    @patch('src.stages.stage1_filter.filter_reason')
    @patch('src.stages.stage1_filter.write_parquet')
    def test_run_filter_statistics(self, mock_write_parquet, mock_filter_reason, 
                                 mock_load_or_sample, mock_stats_recorder, 
                                 mock_ensure_output_dir, mock_load_config):
        """测试过滤统计信息的正确性"""
        # 设置配置
        mock_config = MagicMock()
        mock_config.get.side_effect = lambda key, default=None: {
            'observe.stats_path': f"{self.temp_dir}/stats.json",
            'data.input_path': 'test.parquet',
            'data.id_col': 'id',
            'data.q_col': 'question', 
            'data.a_col': 'answer'
        }.get(key, default)
        mock_load_config.return_value = mock_config
        mock_ensure_output_dir.return_value = self.temp_dir
        
        # 模拟数据
        mock_df = MagicMock()
        mock_df.shape = [5, 3]  # 5行数据
        mock_df.__getitem__.return_value.astype.return_value.map.return_value = None
        mock_df.__getitem__.return_value.tolist.return_value = [
            "问题1", "问题2", "问题3", "问题4", "问题5"
        ]
        mock_load_or_sample.return_value = mock_df
        
        # 模拟过滤结果：3个保留，2个过滤
        mock_filter_reason.side_effect = [
            (True, 'keep'),
            (True, 'whitelist'),
            (False, 'blacklist'),
            (False, 'blacklist'),
            (True, 'keep')
        ]
        
        # 模拟过滤后数据框（3行）
        mock_df_clean = MagicMock()
        mock_df_clean.shape = [3, 3]
        mock_df.__getitem__.return_value.reset_index.return_value = mock_df_clean
        
        # 模拟统计记录器
        mock_stats = MagicMock()
        mock_stats_recorder.return_value = mock_stats
        
        # 运行
        run('test_config.yaml')
        
        # 验证统计信息
        stats_call = mock_stats.update.call_args[0][1]
        self.assertEqual(stats_call['input_count'], 5)
        self.assertEqual(stats_call['kept_count'], 3)
        self.assertAlmostEqual(stats_call['keep_rate'], 0.6, places=2)
        
        # 验证过滤原因统计
        self.assertIn('reason_top', stats_call)
        reason_stats = stats_call['reason_top']
        self.assertIn('blacklist', reason_stats)
        self.assertIn('keep', reason_stats)
        self.assertEqual(reason_stats['blacklist'], 2)
        self.assertEqual(reason_stats['keep'], 2)
        self.assertEqual(reason_stats['whitelist'], 1)

    @patch('src.stages.stage1_filter.load_config')
    @patch('src.stages.stage1_filter.ensure_output_dir')
    @patch('src.stages.stage1_filter.StatsRecorder')
    @patch('src.stages.stage1_filter._load_or_sample')
    @patch('src.stages.stage1_filter.normalize_zh')
    @patch('src.stages.stage1_filter.filter_reason')
    @patch('src.stages.stage1_filter.write_parquet')
    def test_run_text_normalization(self, mock_write_parquet, mock_filter_reason,
                                  mock_normalize_zh, mock_load_or_sample,
                                  mock_stats_recorder, mock_ensure_output_dir,
                                  mock_load_config):
        """测试文本规范化处理"""
        # 设置基本mocks
        mock_config = MagicMock()
        mock_config.get.side_effect = lambda key, default=None: {
            'data.q_col': 'question'
        }.get(key, default)
        mock_load_config.return_value = mock_config
        mock_ensure_output_dir.return_value = self.temp_dir
        
        # 模拟数据框
        mock_df = MagicMock()
        mock_df.shape = [3, 3]
        
        # 设置列访问和链式调用
        mock_series = MagicMock()
        mock_df.__getitem__.return_value = mock_series
        mock_astype_result = MagicMock()
        mock_series.astype.return_value = mock_astype_result
        mock_astype_result.map.return_value = None
        mock_astype_result.tolist.return_value = ["q1", "q2", "q3"]
        
        mock_load_or_sample.return_value = mock_df
        
        # 模拟过滤
        mock_filter_reason.return_value = (True, 'keep')
        mock_df_clean = MagicMock()
        mock_df_clean.shape = [3, 3]
        mock_df.__getitem__.return_value.reset_index.return_value = mock_df_clean
        
        mock_stats = MagicMock()
        mock_stats_recorder.return_value = mock_stats
        
        # 运行
        run('test_config.yaml')
        
        # 验证normalize_zh被调用
        mock_series.astype.assert_called_once_with(str)
        mock_astype_result.map.assert_called_once_with(mock_normalize_zh)

    def test_integration_with_real_data_structure(self):
        """测试与真实数据结构的集成"""
        # 创建真实的数据结构进行测试
        try:
            import pandas as pd
            
            # 创建测试数据
            test_data = pd.DataFrame({
                'id': [1, 2, 3, 4, 5],
                'question': [
                    '如何开发票？',  # 应该保留（白名单）
                    '今天天气如何？',  # 应该过滤（黑名单）
                    '普通问题',      # 应该保留（默认）
                    '你好，在吗？',   # 应该过滤（黑名单）
                    '退款流程是什么？'  # 应该保留（白名单）
                ],
                'answer': [
                    '答案1', '答案2', '答案3', '答案4', '答案5'
                ]
            })
            
            # 保存测试数据
            test_path = os.path.join(self.temp_dir, "real_test.parquet")
            test_data.to_parquet(test_path, index=False)
            
            # 测试加载
            loaded_data = _load_or_sample(test_path, 'id', 'question', 'answer')
            
            # 验证数据完整性
            self.assertEqual(len(loaded_data), 5)
            pd.testing.assert_frame_equal(loaded_data, test_data)
            
        except ImportError:
            self.skipTest("pandas不可用，跳过真实数据集成测试")

    def test_error_handling_robustness(self):
        """测试错误处理的鲁棒性"""
        # 测试各种异常情况都能正常处理
        test_cases = [
            ("", 'id', 'question', 'answer'),  # 空路径
            ("nonexistent.file", 'id', 'q', 'a'),  # 不存在的文件
            (None, 'id', 'question', 'answer'),  # None路径
        ]
        
        for path, id_col, q_col, a_col in test_cases:
            with self.subTest(path=path):
                try:
                    result = _load_or_sample(path, id_col, q_col, a_col)
                    # 应该返回示例数据，不抛出异常
                    self.assertIsNotNone(result)
                except Exception as e:
                    self.fail(f"_load_or_sample应该处理错误而不是抛出异常: {e}")

    @patch('src.stages.stage1_filter.os.path.splitext')
    @patch('src.stages.stage1_filter.print')
    def test_file_format_logging(self, mock_print, mock_splitext):
        """测试文件格式日志记录"""
        # 模拟不同文件格式
        test_cases = [
            ("/path/test.parquet", ".parquet"),
            ("/path/test.xlsx", ".xlsx"),
            ("/path/test.csv", ".csv"),
        ]
        
        for file_path, expected_ext in test_cases:
            with self.subTest(ext=expected_ext):
                mock_splitext.return_value = ("base", expected_ext)
                
                # 模拟成功读取
                with patch('src.stages.stage1_filter.read_data_file') as mock_read:
                    mock_df = MagicMock()
                    mock_df.columns = ['id', 'question', 'answer']
                    mock_read.return_value = mock_df
                    
                    # 创建测试文件
                    test_file = os.path.join(self.temp_dir, f"test{expected_ext}")
                    with open(test_file, 'w') as f:
                        f.write("dummy")
                    
                    result = _load_or_sample(test_file, 'id', 'question', 'answer')
                    
                    # 验证日志包含格式信息
                    log_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
                    format_logged = any(expected_ext in log_msg for log_msg in log_calls)
                    self.assertTrue(format_logged, f"应该记录文件格式 {expected_ext}")


if __name__ == '__main__':
    unittest.main()
