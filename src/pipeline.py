"""
QA清洗流水线主入口 - 重构版本

特性:
- 统一的环境检测和配置
- 结构化日志和进度追踪  
- 优雅的错误处理和回退
- 模块化的阶段管理
- 资源监控和性能优化
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# 设置项目路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.environment import get_env_manager, get_optimal_config_adjustments, setup_project_path
from src.utils.logger import LogLevel, get_logger, setup_logger
from src.utils.config import load_config


class PipelineStage:
    """流水线阶段定义"""
    
    def __init__(self, name: str, module: str, description: str):
        self.name = name
        self.module = module  
        self.description = description


class QAPipeline:
    """QA清洗流水线管理器"""
    
    STAGES = [
        PipelineStage("stage1", "src.stages.stage1_filter", "字符级预处理与过滤"),
        PipelineStage("stage2", "src.stages.stage2_recall", "三路嵌入+FAISS召回+字符n-gram补召"),
        PipelineStage("stage3", "src.stages.stage3_rerank", "多交叉编码器融合精排"),
        PipelineStage("stage4", "src.stages.stage4_cluster", "图聚类 + 中心约束 + 二次聚合"),
        PipelineStage("stage5", "src.stages.stage5_answer_govern", "答案治理与融合"),
    ]
    
    def __init__(self, config_path: str, input_file: str, output_dir: Optional[str] = None):
        self.config_path = config_path
        self.input_file = input_file
        self.output_dir = output_dir or "outputs"
        
        # 初始化环境管理器
        self.env_manager = get_env_manager()
        
        # 设置日志
        log_file = os.path.join(self.output_dir, "logs", f"pipeline_{int(time.time())}.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.logger = setup_logger(
            log_file=log_file,
            level=LogLevel.INFO,
            console_output=True,
            structured=False
        )
        
        # 加载和验证配置
        self.config = None
        self._validation_passed = False
    
    def validate_environment(self) -> bool:
        """验证运行环境"""
        self.logger.info("pipeline", "开始环境验证")
        
        # 环境验证
        is_valid, errors = self.env_manager.validate_environment()
        
        if errors:
            for error in errors:
                self.logger.warning("pipeline", f"环境警告: {error}")
        
        # 输出环境摘要
        env_summary = self.env_manager.get_environment_summary()
        self.logger.info("pipeline", f"Python版本: {env_summary['system']['python_version']}")
        self.logger.info("pipeline", f"平台: {env_summary['system']['platform']}")
        self.logger.info("pipeline", f"CPU核心数: {env_summary['system']['cpu_count']}")
        
        if env_summary['system']['cuda_available']:
            self.logger.info("pipeline", f"GPU可用: {env_summary['system']['gpu_count']} 块")
            for i, memory in enumerate(env_summary['system']['gpu_memory_gb']):
                self.logger.info("pipeline", f"GPU{i} 内存: {memory:.1f}GB")
        else:
            self.logger.info("pipeline", "GPU不可用，将使用CPU")
        
        # 依赖检查
        missing_core = [
            name for name, info in env_summary['core_dependencies'].items() 
            if not info['available']
        ]
        
        if missing_core:
            for dep in missing_core:
                self.logger.error("pipeline", f"缺少核心依赖: {dep}")
            return False
        
        # 输出可选依赖状态
        for name, info in env_summary['optional_dependencies'].items():
            if info['available']:
                self.logger.info("pipeline", f"可选依赖 {name}: 可用 (v{info['version']})")
            else:
                self.logger.debug("pipeline", f"可选依赖 {name}: 不可用")
        
        self.logger.info("pipeline", "环境验证完成")
        return True
    
    def validate_config(self) -> bool:
        """验证配置文件"""
        try:
            with self.logger.operation_timer("pipeline", "加载配置文件"):
                self.config = load_config(self.config_path)
            
            # 检查输入文件
            if not os.path.exists(self.input_file):
                self.logger.error("pipeline", f"输入文件不存在: {self.input_file}")
                return False
            
            file_size = os.path.getsize(self.input_file) / (1024 * 1024)  # MB
            self.logger.info("pipeline", f"输入文件: {self.input_file} ({file_size:.1f}MB)")
            
            # 应用环境优化配置
            optimizations = get_optimal_config_adjustments()
            self.logger.info("pipeline", f"推荐设备: {optimizations['device']}")
            self.logger.info("pipeline", f"推荐嵌入批次: {optimizations['embedding_batch_size']}")
            self.logger.info("pipeline", f"推荐精排批次: {optimizations['rerank_batch_size']}")
            
            return True
            
        except Exception as e:
            self.logger.error("pipeline", f"配置验证失败: {e}")
            return False
    
    def prepare_output_directory(self) -> None:
        """准备输出目录"""
        subdirs = ["logs", "figs"]
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
        
        self.logger.info("pipeline", f"输出目录准备完成: {self.output_dir}")
    
    def run_stage(self, stage: PipelineStage) -> bool:
        """运行单个阶段"""
        with self.logger.stage_timer(stage.name, stage.description):
            cmd = [
                sys.executable, "-m", stage.module,
                "--config", self.config_path,
                "--input", self.input_file
            ]
            
            self.logger.info(stage.name, f"执行命令: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(project_root)
                )
                
                if result.returncode != 0:
                    self.logger.error(stage.name, f"阶段失败 (退出码: {result.returncode})")
                    if result.stderr:
                        self.logger.error(stage.name, f"错误输出: {result.stderr}")
                    if result.stdout:
                        self.logger.error(stage.name, f"标准输出: {result.stdout}")
                    return False
                
                if result.stdout:
                    # 过滤掉过于详细的输出，只保留关键信息
                    lines = result.stdout.strip().split('\n')
                    important_lines = [
                        line for line in lines[-20:]  # 只保留最后20行
                        if any(keyword in line for keyword in ['完成', '错误', '警告', '成功', '['])
                    ]
                    for line in important_lines:
                        self.logger.info(stage.name, f"输出: {line}")
                
                return True
                
            except Exception as e:
                self.logger.error(stage.name, f"执行异常: {e}")
                return False
    
    def run(self, start_stage: Optional[str] = None, end_stage: Optional[str] = None) -> bool:
        """运行流水线"""
        self.logger.info("pipeline", "="*60)
        self.logger.info("pipeline", "QA清洗流水线启动")
        self.logger.info("pipeline", "="*60)
        
        # 环境验证
        if not self.validate_environment():
            self.logger.error("pipeline", "环境验证失败")
            return False
        
        # 配置验证
        if not self.validate_config():
            self.logger.error("pipeline", "配置验证失败")
            return False
        
        # 准备输出目录
        self.prepare_output_directory()
        
        # 确定要运行的阶段
        stages_to_run = self.STAGES
        if start_stage or end_stage:
            start_idx = 0
            end_idx = len(self.STAGES)
            
            if start_stage:
                start_idx = next(
                    (i for i, s in enumerate(self.STAGES) if s.name == start_stage), 0
                )
            
            if end_stage:
                end_idx = next(
                    (i + 1 for i, s in enumerate(self.STAGES) if s.name == end_stage), 
                    len(self.STAGES)
                )
            
            stages_to_run = self.STAGES[start_idx:end_idx]
        
        self.logger.info("pipeline", f"将运行 {len(stages_to_run)} 个阶段")
        
        # 运行各阶段
        pipeline_start_time = time.perf_counter()
        
        for i, stage in enumerate(stages_to_run, 1):
            self.logger.info("pipeline", f"[{i}/{len(stages_to_run)}] 开始阶段: {stage.name}")
            
            if not self.run_stage(stage):
                self.logger.error("pipeline", f"阶段 {stage.name} 失败，流水线终止")
                return False
            
            self.logger.info("pipeline", f"阶段 {stage.name} 完成")
        
        # 流水线完成
        pipeline_duration = time.perf_counter() - pipeline_start_time
        self.logger.info("pipeline", "="*60)
        self.logger.info("pipeline", f"流水线全部完成，总耗时: {pipeline_duration:.2f}s")
        self.logger.info("pipeline", "="*60)
        
        # 输出进度摘要
        progress_summary = self.logger.get_progress_summary()
        self.logger.info("pipeline", f"阶段执行摘要: {progress_summary}")
        
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="QA清洗流水线 - 重构版本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行完整流水线
  python src/pipeline.py -i data/raw/qa.xlsx
  
  # 仅运行特定阶段
  python src/pipeline.py -i data/raw/qa.xlsx --start-stage stage2 --end-stage stage3
  
  # 指定输出目录
  python src/pipeline.py -i data/raw/qa.xlsx -o /path/to/output
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        default="src/configs/config.yaml",
        help="配置文件路径 (默认: %(default)s)"
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入数据文件路径 (支持格式: .parquet, .xlsx, .xls, .csv)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="outputs",
        help="输出目录路径 (默认: %(default)s)"
    )
    
    parser.add_argument(
        "--start-stage",
        choices=["stage1", "stage2", "stage3", "stage4", "stage5"],
        help="起始阶段 (默认从stage1开始)"
    )
    
    parser.add_argument(
        "--end-stage", 
        choices=["stage1", "stage2", "stage3", "stage4", "stage5"],
        help="结束阶段 (默认到stage5结束)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别 (默认: %(default)s)"
    )
    
    args = parser.parse_args()
    
    # 设置项目路径
    setup_project_path()
    
    # 创建流水线实例
    pipeline = QAPipeline(
        config_path=args.config,
        input_file=args.input,
        output_dir=args.output
    )
    
    # 运行流水线
    try:
        success = pipeline.run(
            start_stage=args.start_stage,
            end_stage=args.end_stage
        )
        
        if success:
            print("\n✅ 流水线成功完成!")
            sys.exit(0)
        else:
            print("\n❌ 流水线执行失败!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断流水线执行")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 流水线执行异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
