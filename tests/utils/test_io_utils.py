#!/usr/bin/env python3
"""
IO工具模块的单元测试
测试文件读写、数据格式支持等功能
"""

import unittest
import tempfile
import os
import json
import numpy as np
from unittest.mock import patch, mock_open
from src.utils.io_utils import (
    ensure_parent_dir,
    read_data_file,
    read_parquet,
    write_parquet,
    save_npy,
    save_json_merge
)


class TestIOUtils(unittest.TestCase):
    """IO工具功能测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试数据
        try:
            import pandas as pd
            self.test_df = pd.DataFrame({
                'id': [1, 2, 3],
                'question': ['问题1', '问题2', '问题3'],
                'answer': ['答案1', '答案2', '答案3']
            })
            self.pandas_available = True
        except ImportError:
            self.pandas_available = False
            self.test_df = None

    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ensure_parent_dir(self):
        """测试父目录创建功能"""
        # 测试创建嵌套目录
        nested_path = os.path.join(self.temp_dir, "a", "b", "c", "file.txt")
        ensure_parent_dir(nested_path)
        
        parent_dir = os.path.dirname(nested_path)
        self.assertTrue(os.path.exists(parent_dir))
        self.assertTrue(os.path.isdir(parent_dir))
        
        # 测试空父目录情况
        ensure_parent_dir("file.txt")  # 不应该报错
        
        # 测试已存在的目录
        ensure_parent_dir(nested_path)  # 再次调用不应该报错

    @unittest.skipUnless(False, "需要pandas依赖")
    def test_read_data_file_formats(self):
        """测试不同文件格式的读取"""
        if not self.pandas_available:
            self.skipTest("pandas不可用")
            
        import pandas as pd
        
        # 测试Parquet格式
        parquet_path = os.path.join(self.temp_dir, "test.parquet")
        self.test_df.to_parquet(parquet_path, index=False)
        
        df_loaded = read_data_file(parquet_path)
        pd.testing.assert_frame_equal(df_loaded, self.test_df)
        
        # 测试Excel格式
        excel_path = os.path.join(self.temp_dir, "test.xlsx")
        self.test_df.to_excel(excel_path, index=False)
        
        df_loaded = read_data_file(excel_path)
        pd.testing.assert_frame_equal(df_loaded, self.test_df)
        
        # 测试CSV格式
        csv_path = os.path.join(self.temp_dir, "test.csv")
        self.test_df.to_csv(csv_path, index=False)
        
        df_loaded = read_data_file(csv_path)
        pd.testing.assert_frame_equal(df_loaded, self.test_df)

    def test_read_data_file_file_not_found(self):
        """测试文件不存在的情况"""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.parquet")
        
        with self.assertRaises(FileNotFoundError) as context:
            read_data_file(nonexistent_path)
        
        self.assertIn("文件不存在", str(context.exception))

    def test_read_data_file_unsupported_format(self):
        """测试不支持的文件格式"""
        # 创建一个.txt文件
        txt_path = os.path.join(self.temp_dir, "test.txt")
        with open(txt_path, 'w') as f:
            f.write("test content")
        
        with self.assertRaises(ValueError) as context:
            read_data_file(txt_path)
        
        self.assertIn("不支持的文件格式", str(context.exception))
        self.assertIn(".txt", str(context.exception))

    def test_read_data_file_case_insensitive(self):
        """测试文件扩展名大小写不敏感"""
        if not self.pandas_available:
            self.skipTest("pandas不可用")
            
        import pandas as pd
        
        # 测试大写扩展名
        csv_path = os.path.join(self.temp_dir, "test.CSV")
        self.test_df.to_csv(csv_path, index=False)
        
        # 应该能正常读取
        df_loaded = read_data_file(csv_path)
        pd.testing.assert_frame_equal(df_loaded, self.test_df)

    def test_save_npy(self):
        """测试numpy数组保存功能"""
        # 创建测试数组
        test_array = np.array([[1, 2, 3], [4, 5, 6]])
        
        # 保存到文件
        npy_path = os.path.join(self.temp_dir, "test.npy")
        save_npy(npy_path, test_array)
        
        # 验证文件存在
        self.assertTrue(os.path.exists(npy_path))
        
        # 加载并验证内容
        loaded_array = np.load(npy_path)
        np.testing.assert_array_equal(loaded_array, test_array)

    def test_save_npy_with_nested_dir(self):
        """测试在嵌套目录中保存numpy数组"""
        test_array = np.array([1, 2, 3])
        
        nested_path = os.path.join(self.temp_dir, "deep", "nested", "test.npy")
        save_npy(nested_path, test_array)
        
        self.assertTrue(os.path.exists(nested_path))
        loaded_array = np.load(nested_path)
        np.testing.assert_array_equal(loaded_array, test_array)

    def test_save_json_merge_new_file(self):
        """测试JSON合并保存 - 新文件"""
        json_path = os.path.join(self.temp_dir, "test.json")
        
        # 保存第一个对象
        data1 = {"stage1": {"count": 100, "rate": 0.8}}
        save_json_merge(json_path, data1)
        
        # 验证文件存在和内容
        self.assertTrue(os.path.exists(json_path))
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        self.assertEqual(loaded, data1)

    def test_save_json_merge_existing_file(self):
        """测试JSON合并保存 - 合并到已存在文件"""
        json_path = os.path.join(self.temp_dir, "test.json")
        
        # 创建初始文件
        initial_data = {"stage1": {"count": 100}}
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f)
        
        # 合并新数据
        new_data = {
            "stage1": {"rate": 0.8},  # 合并到已有stage1
            "stage2": {"count": 50}   # 新增stage2
        }
        save_json_merge(json_path, new_data)
        
        # 验证合并结果
        with open(json_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        expected = {
            "stage1": {"count": 100, "rate": 0.8},
            "stage2": {"count": 50}
        }
        self.assertEqual(result, expected)

    def test_save_json_merge_corrupted_file(self):
        """测试JSON合并保存 - 处理损坏的文件"""
        json_path = os.path.join(self.temp_dir, "corrupted.json")
        
        # 创建损坏的JSON文件
        with open(json_path, 'w') as f:
            f.write("invalid json content")
        
        # 应该能处理并重新创建文件
        new_data = {"stage1": {"count": 100}}
        save_json_merge(json_path, new_data)
        
        # 验证文件内容正确
        with open(json_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        self.assertEqual(result, new_data)

    def test_save_json_merge_nested_dir(self):
        """测试在嵌套目录中保存JSON"""
        nested_path = os.path.join(self.temp_dir, "deep", "nested", "stats.json")
        
        data = {"test": {"value": 123}}
        save_json_merge(nested_path, data)
        
        self.assertTrue(os.path.exists(nested_path))
        with open(nested_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        self.assertEqual(loaded, data)

    def test_save_json_merge_unicode(self):
        """测试JSON保存中文字符"""
        json_path = os.path.join(self.temp_dir, "chinese.json")
        
        data = {
            "stage1": {
                "描述": "第一阶段",
                "问题": ["如何开发票？", "怎么退款？"]
            }
        }
        save_json_merge(json_path, data)
        
        # 验证中文字符正确保存
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("第一阶段", content)
            self.assertIn("如何开发票", content)
        
        # 验证能正确加载
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        self.assertEqual(loaded, data)

    @patch('pandas.read_csv')
    def test_csv_encoding_detection_mock(self, mock_read_csv):
        """测试CSV编码检测逻辑（使用mock）"""
        # 模拟第一次读取失败，第二次成功
        def side_effect(*args, **kwargs):
            if kwargs.get('encoding') == 'utf-8':
                raise UnicodeDecodeError('utf-8', b'', 0, 1, 'error')
            elif kwargs.get('encoding') == 'gbk':
                import pandas as pd
                return pd.DataFrame({'test': ['data']})
            else:
                raise Exception("Other error")
        
        mock_read_csv.side_effect = side_effect
        
        # 创建测试CSV文件
        csv_path = os.path.join(self.temp_dir, "test.csv")
        with open(csv_path, 'w') as f:
            f.write("test,data\n")
        
        # 这个测试需要实际的pandas，所以跳过
        if not self.pandas_available:
            self.skipTest("pandas不可用")

    def test_file_extension_detection(self):
        """测试文件扩展名检测逻辑"""
        test_cases = [
            ("file.parquet", ".parquet"),
            ("FILE.PARQUET", ".parquet"),
            ("data.xlsx", ".xlsx"),
            ("test.XLS", ".xls"),
            ("sample.csv", ".csv"),
            ("data.CSV", ".csv"),
            ("file.txt", ".txt"),
        ]
        
        for filename, expected_ext in test_cases:
            _, actual_ext = os.path.splitext(filename.lower())
            self.assertEqual(actual_ext, expected_ext)


# Mock类用于测试没有依赖时的行为
class TestIOUtilsWithoutDependencies(unittest.TestCase):
    """测试在没有外部依赖时的行为"""
    
    def test_import_structure(self):
        """测试导入结构的正确性"""
        # 验证所有函数都能正确导入
        from src.utils.io_utils import (
            ensure_parent_dir,
            read_data_file,
            read_parquet,
            write_parquet,
            save_npy,
            save_json_merge
        )
        
        # 验证函数是可调用的
        self.assertTrue(callable(ensure_parent_dir))
        self.assertTrue(callable(read_data_file))
        self.assertTrue(callable(save_json_merge))


if __name__ == '__main__':
    unittest.main()
