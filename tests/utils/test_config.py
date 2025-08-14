#!/usr/bin/env python3
"""
配置管理模块的单元测试
测试配置加载、验证、路径解析等功能
"""

import unittest
import tempfile
import os
import yaml
import random
import numpy as np
from unittest.mock import patch
from src.utils.config import (
    Config,
    load_config,
    _validate_config,
    ensure_seed,
    ensure_output_dir,
    dump_json
)


class TestConfig(unittest.TestCase):
    """Config类功能测试"""

    def test_config_get_simple_path(self):
        """测试简单路径的get方法"""
        data = {
            'key1': 'value1',
            'key2': 123,
            'key3': True
        }
        config = Config(data)
        
        self.assertEqual(config.get('key1'), 'value1')
        self.assertEqual(config.get('key2'), 123)
        self.assertEqual(config.get('key3'), True)

    def test_config_get_nested_path(self):
        """测试嵌套路径的get方法"""
        data = {
            'level1': {
                'level2': {
                    'level3': 'deep_value'
                },
                'simple': 'value'
            }
        }
        config = Config(data)
        
        self.assertEqual(config.get('level1.level2.level3'), 'deep_value')
        self.assertEqual(config.get('level1.simple'), 'value')

    def test_config_get_default_value(self):
        """测试默认值返回"""
        data = {'existing': 'value'}
        config = Config(data)
        
        # 不存在的键返回默认值
        self.assertEqual(config.get('nonexistent', 'default'), 'default')
        self.assertIsNone(config.get('nonexistent'))
        
        # 嵌套路径不存在
        self.assertEqual(config.get('existing.nonexistent', 'default'), 'default')

    def test_config_get_invalid_path(self):
        """测试无效路径的处理"""
        data = {
            'string_value': 'not_a_dict',
            'number': 42
        }
        config = Config(data)
        
        # 尝试在非字典值上继续访问
        self.assertEqual(config.get('string_value.invalid', 'default'), 'default')
        self.assertEqual(config.get('number.invalid', 'default'), 'default')

    def test_config_get_empty_key(self):
        """测试空键的处理"""
        data = {'key': 'value'}
        config = Config(data)
        
        # 空字符串键
        self.assertEqual(config.get('', 'default'), 'default')


class TestLoadConfig(unittest.TestCase):
    """配置加载功能测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_config_success(self):
        """测试成功加载配置文件"""
        # 创建有效的配置文件
        config_data = {
            'pipeline': {'language': 'zh'},
            'data': {'input_path': 'test.parquet'},
            'embeddings': {'batch_size': 64},
            'recall': {'topk': 200},
            'consistency': {'cos_a': 0.875},
            'rerank': {'thresholds': {'high': 0.83}},
            'cluster': {'method': 'leiden'},
            'govern': {'merge_answers': True},
            'observe': {'enable': True}
        }
        
        config_path = os.path.join(self.temp_dir, 'config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, allow_unicode=True)
        
        # 加载配置
        config = load_config(config_path)
        
        self.assertIsInstance(config, Config)
        self.assertEqual(config.get('pipeline.language'), 'zh')
        self.assertEqual(config.get('embeddings.batch_size'), 64)

    def test_load_config_file_not_found(self):
        """测试配置文件不存在"""
        nonexistent_path = os.path.join(self.temp_dir, 'nonexistent.yaml')
        
        with self.assertRaises(FileNotFoundError) as context:
            load_config(nonexistent_path)
        
        self.assertIn("配置文件不存在", str(context.exception))

    def test_load_config_invalid_yaml(self):
        """测试无效的YAML文件"""
        config_path = os.path.join(self.temp_dir, 'invalid.yaml')
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with self.assertRaises(yaml.YAMLError):
            load_config(config_path)

    def test_load_config_validation_failure(self):
        """测试配置验证失败"""
        # 创建缺少必需节的配置文件
        config_data = {
            'pipeline': {'language': 'zh'},
            # 缺少其他必需节
        }
        
        config_path = os.path.join(self.temp_dir, 'incomplete.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)
        
        with self.assertRaises(ValueError) as context:
            load_config(config_path)
        
        self.assertIn("缺少配置节", str(context.exception))


class TestValidateConfig(unittest.TestCase):
    """配置验证功能测试"""

    def test_validate_config_success(self):
        """测试配置验证成功"""
        valid_config = {
            'pipeline': {},
            'data': {},
            'embeddings': {'batch_size': 64},
            'recall': {'topk': 200},
            'consistency': {'cos_a': 0.875},
            'rerank': {},
            'cluster': {},
            'govern': {},
            'observe': {}
        }
        
        # 不应该抛出异常
        _validate_config(valid_config)

    def test_validate_config_missing_section(self):
        """测试缺少配置节"""
        incomplete_config = {
            'pipeline': {},
            'data': {},
            # 缺少其他节
        }
        
        with self.assertRaises(ValueError) as context:
            _validate_config(incomplete_config)
        
        self.assertIn("缺少配置节", str(context.exception))

    def test_validate_config_invalid_batch_size(self):
        """测试无效的batch_size"""
        test_cases = [
            ({'embeddings': {'batch_size': 0}}, "应在1-512范围内"),
            ({'embeddings': {'batch_size': 1000}}, "应在1-512范围内"),
            ({'embeddings': {'batch_size': -1}}, "应在1-512范围内"),
        ]
        
        for config_section, expected_error in test_cases:
            config = self._create_minimal_config()
            config.update(config_section)
            
            with self.subTest(batch_size=config_section['embeddings']['batch_size']):
                with self.assertRaises(ValueError) as context:
                    _validate_config(config)
                self.assertIn(expected_error, str(context.exception))

    def test_validate_config_invalid_topk(self):
        """测试无效的topk"""
        test_cases = [
            ({'recall': {'topk': 5}}, "应在10-1000范围内"),
            ({'recall': {'topk': 2000}}, "应在10-1000范围内"),
        ]
        
        for config_section, expected_error in test_cases:
            config = self._create_minimal_config()
            config.update(config_section)
            
            with self.subTest(topk=config_section['recall']['topk']):
                with self.assertRaises(ValueError) as context:
                    _validate_config(config)
                self.assertIn(expected_error, str(context.exception))

    def test_validate_config_invalid_cos_a(self):
        """测试无效的cos_a值"""
        test_cases = [
            ({'consistency': {'cos_a': 0.3}}, "应在0.5-1.0范围内"),
            ({'consistency': {'cos_a': 1.5}}, "应在0.5-1.0范围内"),
        ]
        
        for config_section, expected_error in test_cases:
            config = self._create_minimal_config()
            config.update(config_section)
            
            with self.subTest(cos_a=config_section['consistency']['cos_a']):
                with self.assertRaises(ValueError) as context:
                    _validate_config(config)
                self.assertIn(expected_error, str(context.exception))

    def test_validate_config_default_values(self):
        """测试默认值的验证"""
        # 不提供可选参数，应该使用默认值
        config = self._create_minimal_config()
        
        # 不应该抛出异常
        _validate_config(config)

    def _create_minimal_config(self):
        """创建最小有效配置"""
        return {
            'pipeline': {},
            'data': {},
            'embeddings': {},
            'recall': {},
            'consistency': {},
            'rerank': {},
            'cluster': {},
            'govern': {},
            'observe': {}
        }


class TestUtilityFunctions(unittest.TestCase):
    """工具函数测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ensure_seed(self):
        """测试随机种子设置"""
        config_data = {'pipeline': {'random_seed': 12345}}
        config = Config(config_data)
        
        # 设置种子
        ensure_seed(config)
        
        # 验证随机数生成器状态一致
        random_val1 = random.random()
        numpy_val1 = np.random.random()
        
        # 重新设置相同种子
        ensure_seed(config)
        random_val2 = random.random()
        numpy_val2 = np.random.random()
        
        self.assertEqual(random_val1, random_val2)
        self.assertEqual(numpy_val1, numpy_val2)

    def test_ensure_seed_default(self):
        """测试默认种子值"""
        config = Config({'pipeline': {}})  # 不提供random_seed
        
        # 应该使用默认种子42
        ensure_seed(config)
        
        # 验证能正常工作
        val = random.random()
        self.assertIsInstance(val, float)

    def test_ensure_output_dir(self):
        """测试输出目录创建"""
        output_path = os.path.join(self.temp_dir, "test_output")
        config = Config({'pipeline': {'output_dir': output_path}})
        
        # 目录应该不存在
        self.assertFalse(os.path.exists(output_path))
        
        # 创建目录
        result_path = ensure_output_dir(config)
        
        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.isdir(output_path))

    def test_ensure_output_dir_default(self):
        """测试默认输出目录"""
        config = Config({'pipeline': {}})  # 不提供output_dir
        
        result_path = ensure_output_dir(config)
        
        self.assertEqual(result_path, './outputs')
        self.assertTrue(os.path.exists('./outputs'))

    def test_ensure_output_dir_existing(self):
        """测试已存在的输出目录"""
        output_path = os.path.join(self.temp_dir, "existing_output")
        os.makedirs(output_path)
        
        config = Config({'pipeline': {'output_dir': output_path}})
        
        # 应该正常处理已存在的目录
        result_path = ensure_output_dir(config)
        
        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))

    def test_dump_json(self):
        """测试JSON导出功能"""
        test_data = {
            'stage1': {'count': 100, 'rate': 0.8},
            'stage2': {'count': 50, 'time': 123.45},
            '中文键': '中文值'
        }
        
        json_path = os.path.join(self.temp_dir, "test_dump.json")
        dump_json(json_path, test_data)
        
        # 验证文件存在
        self.assertTrue(os.path.exists(json_path))
        
        # 验证内容正确
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        self.assertEqual(loaded_data, test_data)

    def test_dump_json_nested_dir(self):
        """测试在嵌套目录中导出JSON"""
        test_data = {'test': 'value'}
        
        nested_path = os.path.join(self.temp_dir, "deep", "nested", "test.json")
        dump_json(nested_path, test_data)
        
        self.assertTrue(os.path.exists(nested_path))
        
        import json
        with open(nested_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        self.assertEqual(loaded_data, test_data)

    def test_dump_json_unicode(self):
        """测试JSON导出中文字符"""
        test_data = {
            'description': '这是一个测试',
            'questions': ['如何开发票？', '怎么退款？'],
            'nested': {
                '中文键': '中文值'
            }
        }
        
        json_path = os.path.join(self.temp_dir, "unicode_test.json")
        dump_json(json_path, test_data)
        
        # 验证文件内容包含中文字符（未被转义）
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('这是一个测试', content)
            self.assertIn('如何开发票', content)


class TestConfigIntegration(unittest.TestCase):
    """配置模块集成测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_config_workflow(self):
        """测试完整的配置工作流"""
        # 创建完整的配置文件
        config_data = {
            'pipeline': {
                'language': 'zh',
                'random_seed': 42,
                'output_dir': os.path.join(self.temp_dir, 'outputs')
            },
            'data': {
                'input_path': 'test.parquet',
                'id_col': 'id',
                'q_col': 'question',
                'a_col': 'answer'
            },
            'embeddings': {
                'batch_size': 64,
                'device': 'cpu'
            },
            'recall': {
                'topk': 200
            },
            'consistency': {
                'cos_a': 0.875
            },
            'rerank': {
                'thresholds': {
                    'high': 0.83,
                    'mid_low': 0.77
                }
            },
            'cluster': {
                'method': 'leiden'
            },
            'govern': {
                'merge_answers': True
            },
            'observe': {
                'enable': True,
                'stats_path': os.path.join(self.temp_dir, 'stats.json')
            }
        }
        
        config_path = os.path.join(self.temp_dir, 'full_config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, allow_unicode=True)
        
        # 加载配置
        config = load_config(config_path)
        
        # 验证各种路径访问
        self.assertEqual(config.get('pipeline.language'), 'zh')
        self.assertEqual(config.get('data.id_col'), 'id')
        self.assertEqual(config.get('rerank.thresholds.high'), 0.83)
        self.assertEqual(config.get('embeddings.batch_size'), 64)
        
        # 设置种子和输出目录
        ensure_seed(config)
        output_dir = ensure_output_dir(config)
        
        self.assertTrue(os.path.exists(output_dir))
        
        # 导出统计信息
        stats_path = config.get('observe.stats_path')
        test_stats = {'test': 'stats'}
        dump_json(stats_path, test_stats)
        
        self.assertTrue(os.path.exists(stats_path))


if __name__ == '__main__':
    unittest.main()
