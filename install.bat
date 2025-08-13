@echo off
REM QA Clean 项目 conda 安装脚本 (Windows)

echo 🚀 开始安装 QA Clean 项目...

REM 检查conda是否安装
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ 错误: 未找到conda命令，请先安装Anaconda或Miniconda
    pause
    exit /b 1
)

REM 创建基础环境
echo 📦 创建基础环境...
conda env create -f environment.yml

if %errorlevel% equ 0 (
    echo ✅ 基础环境创建成功！
) else (
    echo ❌ 基础环境创建失败
    pause
    exit /b 1
)

REM 激活环境
echo 🔄 激活环境...
call conda activate qa-clean

REM 安装项目
echo 📥 安装项目...
pip install -e .

if %errorlevel% equ 0 (
    echo ✅ 项目安装成功！
) else (
    echo ❌ 项目安装失败
    pause
    exit /b 1
)

echo.
echo 🎉 安装完成！
echo.
echo 使用方法:
echo 1. 激活环境: conda activate qa-clean
echo 2. 运行CLI: qa-clean info
echo 3. 运行测试: python test_faiss.py
echo.
echo 如需开发环境，请运行: conda env create -f environment-dev.yml
pause
