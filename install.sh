#!/bin/bash

# QA Clean 项目 conda 安装脚本

echo "🚀 开始安装 QA Clean 项目..."

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: 未找到conda命令，请先安装Anaconda或Miniconda"
    exit 1
fi

# 创建基础环境
echo "📦 创建基础环境..."
conda env create -f environment.yml

if [ $? -eq 0 ]; then
    echo "✅ 基础环境创建成功！"
else
    echo "❌ 基础环境创建失败"
    exit 1
fi

# 激活环境
echo "🔄 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qa-clean

# 安装项目
echo "📥 安装项目..."
pip install -e .

if [ $? -eq 0 ]; then
    echo "✅ 项目安装成功！"
else
    echo "❌ 项目安装失败"
    exit 1
fi

echo ""
echo "🎉 安装完成！"
echo ""
echo "使用方法:"
echo "1. 激活环境: conda activate qa-clean"
echo "2. 运行CLI: qa-clean info"
echo "3. 运行测试: python test_faiss.py"
echo ""
echo "如需开发环境，请运行: conda env create -f environment-dev.yml"
