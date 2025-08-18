"""
统一日志系统 - 支持结构化日志、进度追踪和性能监控

特性:
- 结构化日志输出 (JSON格式可选)
- 多级别日志 (DEBUG, INFO, WARNING, ERROR)
- 进度追踪与实时状态更新
- 性能计时器与资源监控
- 线程安全的日志写入
- 灵活的输出目标 (控制台/文件)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, TextIO, Union


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class LogRecord:
    """结构化日志记录"""
    timestamp: str
    level: str
    stage: str
    message: str
    duration: Optional[float] = None
    data: Optional[Dict[str, Any]] = None
    

class ProgressTracker:
    """进度追踪器 - 线程安全"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._current_stage = ""
        self._stage_progress = {}
        self._start_times = {}
    
    def start_stage(self, stage: str) -> None:
        """开始新阶段"""
        with self._lock:
            self._current_stage = stage
            self._start_times[stage] = time.perf_counter()
            self._stage_progress[stage] = {"status": "running", "progress": 0.0}
    
    def update_progress(self, stage: str, progress: float, message: str = "") -> None:
        """更新阶段进度 (0.0-1.0)"""
        with self._lock:
            if stage in self._stage_progress:
                self._stage_progress[stage].update({
                    "progress": max(0.0, min(1.0, progress)),
                    "message": message,
                    "updated_at": time.perf_counter()
                })
    
    def finish_stage(self, stage: str) -> float:
        """完成阶段，返回耗时"""
        with self._lock:
            duration = time.perf_counter() - self._start_times.get(stage, 0)
            self._stage_progress[stage] = {
                "status": "completed", 
                "progress": 1.0,
                "duration": duration
            }
            return duration
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前状态快照"""
        with self._lock:
            return {
                "current_stage": self._current_stage,
                "stages": dict(self._stage_progress)
            }


class QALogger:
    """QA清洗流水线专用日志器"""
    
    def __init__(
        self,
        name: str = "qa-pipeline",
        level: LogLevel = LogLevel.INFO,
        log_file: Optional[str] = None,
        console_output: bool = True,
        structured: bool = False
    ):
        self.name = name
        self.level = level
        self.structured = structured
        self.progress = ProgressTracker()
        
        # 设置输出流
        self._console_enabled = console_output
        self._file_handle: Optional[TextIO] = None
        
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            self._file_handle = open(log_file, 'w', encoding='utf-8')
        
        # 线程锁
        self._write_lock = threading.Lock()
    
    def __del__(self):
        """清理资源"""
        if self._file_handle:
            self._file_handle.close()
    
    def _should_log(self, level: LogLevel) -> bool:
        """检查是否应该记录此级别的日志"""
        level_order = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1, 
            LogLevel.WARNING: 2,
            LogLevel.ERROR: 3
        }
        return level_order[level] >= level_order[self.level]
    
    def _write_log(self, record: LogRecord) -> None:
        """线程安全的日志写入"""
        with self._write_lock:
            if self.structured:
                output = json.dumps(asdict(record), ensure_ascii=False)
            else:
                msg = f"[{record.timestamp}] [{record.level}] [{record.stage}] {record.message}"
                if record.duration is not None:
                    msg += f" (耗时: {record.duration:.2f}s)"
                output = msg
            
            # 写入控制台
            if self._console_enabled:
                print(output, flush=True)
            
            # 写入文件
            if self._file_handle:
                self._file_handle.write(output + "\n")
                self._file_handle.flush()
    
    def log(
        self, 
        level: LogLevel, 
        stage: str, 
        message: str, 
        duration: Optional[float] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """记录日志"""
        if not self._should_log(level):
            return
            
        record = LogRecord(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            level=level.value,
            stage=stage,
            message=message,
            duration=duration,
            data=data
        )
        self._write_log(record)
    
    def debug(self, stage: str, message: str, **kwargs) -> None:
        """调试级别日志"""
        self.log(LogLevel.DEBUG, stage, message, **kwargs)
    
    def info(self, stage: str, message: str, **kwargs) -> None:
        """信息级别日志"""
        self.log(LogLevel.INFO, stage, message, **kwargs)
    
    def warning(self, stage: str, message: str, **kwargs) -> None:
        """警告级别日志"""
        self.log(LogLevel.WARNING, stage, message, **kwargs)
    
    def error(self, stage: str, message: str, **kwargs) -> None:
        """错误级别日志"""
        self.log(LogLevel.ERROR, stage, message, **kwargs)
    
    @contextmanager
    def stage_timer(self, stage: str, description: str = ""):
        """阶段计时器上下文管理器"""
        full_desc = f"{description}" if description else stage
        self.progress.start_stage(stage)
        self.info(stage, f"{full_desc} 开始")
        
        start_time = time.perf_counter()
        try:
            yield self.progress
        except Exception as e:
            duration = time.perf_counter() - start_time
            self.error(stage, f"{full_desc} 失败: {e}", duration=duration)
            raise
        else:
            duration = self.progress.finish_stage(stage)
            self.info(stage, f"{full_desc} 完成", duration=duration)
    
    @contextmanager
    def operation_timer(self, stage: str, operation: str):
        """操作计时器 - 用于阶段内的子操作"""
        self.info(stage, f"{operation} 开始")
        start_time = time.perf_counter()
        try:
            yield
        except Exception as e:
            duration = time.perf_counter() - start_time
            self.error(stage, f"{operation} 失败: {e}", duration=duration)
            raise
        else:
            duration = time.perf_counter() - start_time
            self.info(stage, f"{operation} 完成", duration=duration)
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """获取进度摘要"""
        return self.progress.get_status()


# 全局日志器实例
_global_logger: Optional[QALogger] = None


def get_logger() -> QALogger:
    """获取全局日志器实例"""
    global _global_logger
    if _global_logger is None:
        raise RuntimeError("日志器未初始化，请先调用 setup_logger()")
    return _global_logger


def setup_logger(
    log_file: Optional[str] = None,
    level: LogLevel = LogLevel.INFO,
    console_output: bool = True,
    structured: bool = False
) -> QALogger:
    """设置全局日志器"""
    global _global_logger
    _global_logger = QALogger(
        log_file=log_file,
        level=level,
        console_output=console_output,
        structured=structured
    )
    return _global_logger


def cleanup_logger() -> None:
    """清理全局日志器"""
    global _global_logger
    if _global_logger:
        del _global_logger
        _global_logger = None


# 便捷函数
def log_info(stage: str, message: str, **kwargs) -> None:
    """快捷信息日志"""
    get_logger().info(stage, message, **kwargs)


def log_error(stage: str, message: str, **kwargs) -> None:
    """快捷错误日志"""
    get_logger().error(stage, message, **kwargs)


def log_warning(stage: str, message: str, **kwargs) -> None:
    """快捷警告日志"""
    get_logger().warning(stage, message, **kwargs)
