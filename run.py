#!/usr/bin/env python3
"""
QA清洗流水线启动脚本 - 简化版本

这是一个简化的启动脚本，自动处理路径和导入问题。
"""

import os
import sys
from pathlib import Path

# 自动设置项目根目录到Python路径
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 设置工作目录为项目根目录
os.chdir(project_root)

if __name__ == "__main__":
    from src.pipeline import main
    main()
