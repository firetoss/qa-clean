#!/usr/bin/env python3
"""
测试运行器 - 运行所有单元测试并生成报告
"""

import unittest
import sys
import os
import time
from io import StringIO

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class ColoredTextTestResult(unittest.TextTestResult):
    """带颜色输出的测试结果类"""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.verbosity = verbosity
    
    def addSuccess(self, test):
        super().addSuccess(test)
        if self.verbosity > 1:
            self.stream.writeln(f"\033[92m✓ {self.getDescription(test)}\033[0m")
    
    def addError(self, test, err):
        super().addError(test, err)
        if self.verbosity > 1:
            self.stream.writeln(f"\033[91m✗ {self.getDescription(test)} - ERROR\033[0m")
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.verbosity > 1:
            self.stream.writeln(f"\033[91m✗ {self.getDescription(test)} - FAIL\033[0m")
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.verbosity > 1:
            self.stream.writeln(f"\033[93m- {self.getDescription(test)} - SKIPPED: {reason}\033[0m")


class TestRunner:
    """测试运行器"""
    
    def __init__(self):
        self.test_dir = os.path.dirname(__file__)
        self.start_time = None
        self.end_time = None
    
    def discover_tests(self, pattern='test_*.py'):
        """发现测试用例"""
        loader = unittest.TestLoader()
        suite = loader.discover(self.test_dir, pattern=pattern)
        return suite
    
    def run_tests(self, verbosity=2, use_color=True):
        """运行测试"""
        print("🧪 QA Clean 项目单元测试")
        print("=" * 50)
        
        # 发现测试
        suite = self.discover_tests()
        test_count = suite.countTestCases()
        print(f"📊 发现 {test_count} 个测试用例\n")
        
        # 配置测试运行器
        stream = sys.stdout
        if use_color:
            runner = unittest.TextTestRunner(
                stream=stream,
                verbosity=verbosity,
                resultclass=ColoredTextTestResult
            )
        else:
            runner = unittest.TextTestRunner(
                stream=stream,
                verbosity=verbosity
            )
        
        # 运行测试
        self.start_time = time.time()
        result = runner.run(suite)
        self.end_time = time.time()
        
        # 生成报告
        self.print_summary(result)
        
        return result.wasSuccessful()
    
    def print_summary(self, result):
        """打印测试总结"""
        duration = self.end_time - self.start_time
        
        print("\n" + "=" * 50)
        print("📈 测试总结")
        print("-" * 50)
        
        total = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped)
        success = total - failures - errors - skipped
        
        print(f"总测试数: {total}")
        print(f"\033[92m✓ 成功: {success}\033[0m")
        
        if failures > 0:
            print(f"\033[91m✗ 失败: {failures}\033[0m")
        
        if errors > 0:
            print(f"\033[91m✗ 错误: {errors}\033[0m")
        
        if skipped > 0:
            print(f"\033[93m- 跳过: {skipped}\033[0m")
        
        print(f"⏱️  耗时: {duration:.2f}秒")
        
        success_rate = (success / total * 100) if total > 0 else 0
        print(f"📊 成功率: {success_rate:.1f}%")
        
        # 详细失败信息
        if failures or errors:
            print("\n" + "=" * 50)
            print("❌ 失败详情")
            print("-" * 50)
            
            for test, traceback in result.failures + result.errors:
                print(f"\n🔍 {test}")
                print("-" * 30)
                print(traceback)
    
    def run_specific_module(self, module_name):
        """运行特定模块的测试"""
        print(f"🧪 运行模块测试: {module_name}")
        print("=" * 50)
        
        loader = unittest.TestLoader()
        try:
            suite = loader.loadTestsFromName(module_name)
        except (ImportError, AttributeError) as e:
            print(f"❌ 无法加载测试模块 {module_name}: {e}")
            return False
        
        runner = unittest.TextTestRunner(verbosity=2, resultclass=ColoredTextTestResult)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    
    def run_tests_by_category(self, category):
        """按类别运行测试"""
        categories = {
            'utils': 'tests.utils',
            'stages': 'tests.stages',
            'integration': 'tests.integration'
        }
        
        if category not in categories:
            print(f"❌ 未知的测试类别: {category}")
            print(f"可用类别: {', '.join(categories.keys())}")
            return False
        
        print(f"🧪 运行 {category} 测试")
        print("=" * 50)
        
        pattern = f"test_*.py"
        test_dir = os.path.join(self.test_dir, category)
        
        if not os.path.exists(test_dir):
            print(f"❌ 测试目录不存在: {test_dir}")
            return False
        
        loader = unittest.TestLoader()
        suite = loader.discover(test_dir, pattern=pattern)
        
        runner = unittest.TextTestRunner(verbosity=2, resultclass=ColoredTextTestResult)
        result = runner.run(suite)
        
        return result.wasSuccessful()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='QA Clean 项目测试运行器')
    parser.add_argument(
        '--module', '-m',
        help='运行特定模块的测试 (例如: tests.utils.test_cn_text)'
    )
    parser.add_argument(
        '--category', '-c',
        choices=['utils', 'stages', 'integration'],
        help='按类别运行测试'
    )
    parser.add_argument(
        '--verbosity', '-v',
        type=int,
        default=2,
        choices=[0, 1, 2],
        help='详细程度 (0=静默, 1=正常, 2=详细)'
    )
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='禁用彩色输出'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='列出所有测试用例'
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.list:
        suite = runner.discover_tests()
        print("📋 发现的测试用例:")
        for test in suite:
            if hasattr(test, '_tests'):
                for subtest in test:
                    if hasattr(subtest, '_tests'):
                        for test_case in subtest:
                            print(f"  - {test_case}")
                    else:
                        print(f"  - {subtest}")
            else:
                print(f"  - {test}")
        return
    
    success = True
    
    if args.module:
        success = runner.run_specific_module(args.module)
    elif args.category:
        success = runner.run_tests_by_category(args.category)
    else:
        success = runner.run_tests(
            verbosity=args.verbosity,
            use_color=not args.no_color
        )
    
    # 退出码
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
