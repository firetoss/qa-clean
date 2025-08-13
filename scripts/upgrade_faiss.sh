#!/bin/bash

# FAISS升级脚本 - 从1.7.2升级到1.12.0

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 检查conda环境
check_conda_env() {
    if ! command -v conda &> /dev/null; then
        print_error "conda未安装或不在PATH中"
        return 1
    fi
    
    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        print_warning "当前不在任何conda环境中"
        return 1
    fi
    
    print_info "当前conda环境: $CONDA_DEFAULT_ENV"
    return 0
}

# 检查当前FAISS版本
check_current_faiss() {
    print_info "检查当前FAISS版本..."
    
    if python -c "import faiss; print('FAISS版本:', faiss.__version__)" 2>/dev/null; then
        version=$(python -c "import faiss; print(faiss.__version__)" 2>/dev/null)
        print_info "当前FAISS版本: $version"
        
        if [[ "$version" == 1.12.* ]]; then
            print_success "FAISS已经是1.12.x版本，无需升级"
            return 0
        else
            print_warning "当前FAISS版本: $version，需要升级到1.12.0"
            return 1
        fi
    else
        print_error "无法检查FAISS版本或FAISS未安装"
        return 1
    fi
}

# 升级FAISS到1.12.0
upgrade_faiss() {
    print_info "开始升级FAISS到1.12.0..."
    
    # 卸载当前FAISS版本
    print_info "卸载当前FAISS版本..."
    pip uninstall -y faiss-gpu faiss-cpu faiss 2>/dev/null || true
    
    # 安装FAISS 1.12.0 GPU版本
    print_info "安装FAISS 1.12.0 GPU版本..."
    if pip install faiss-gpu>=1.12.0 2>/dev/null; then
        print_success "FAISS 1.12.0 GPU安装成功"
        return 0
    else
        print_warning "FAISS GPU安装失败，尝试CPU版本..."
        if pip install faiss-cpu>=1.12.0 2>/dev/null; then
            print_success "FAISS 1.12.0 CPU安装成功"
            return 0
        else
            print_error "FAISS 1.12.0安装失败"
            return 1
        fi
    fi
}

# 测试升级结果
test_upgrade() {
    print_info "测试升级结果..."
    
    # 测试FAISS导入
    if python -c "import faiss; print('FAISS导入成功，版本:', faiss.__version__)" 2>/dev/null; then
        version=$(python -c "import faiss; print(faiss.__version__)" 2>/dev/null)
        if [[ "$version" == 1.12.* ]]; then
            print_success "FAISS 1.12.x导入成功"
        else
            print_error "FAISS版本不正确: $version"
            return 1
        fi
    else
        print_error "FAISS导入失败"
        return 1
    fi
    
    # 测试NumPy兼容性
    if python -c "import numpy; print('NumPy版本:', numpy.__version__)" 2>/dev/null; then
        np_version=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null)
        print_info "NumPy版本: $np_version"
        
        # 测试FAISS和NumPy的兼容性
        if python -c "import numpy; import faiss; print('NumPy和FAISS兼容性测试通过')" 2>/dev/null; then
            print_success "NumPy和FAISS兼容性测试通过"
        else
            print_error "NumPy和FAISS兼容性测试失败"
            return 1
        fi
    else
        print_error "NumPy导入失败"
        return 1
    fi
    
    # 测试项目导入
    if python -c "from qa_clean.cli import main; print('项目导入测试通过')" 2>/dev/null; then
        print_success "项目导入测试通过"
    else
        print_error "项目导入测试失败"
        return 1
    fi
    
    return 0
}

# 显示升级信息
show_upgrade_info() {
    print_info "升级信息:"
    echo "  Python版本: $(python --version 2>/dev/null || echo '未知')"
    echo "  conda环境: ${CONDA_DEFAULT_ENV:-'无'}"
    echo "  NumPy版本: $(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo '未安装')"
    echo "  FAISS版本: $(python -c "import faiss; print(faiss.__version__)" 2>/dev/null || echo '未安装')"
    
    echo ""
    print_info "升级优势:"
    echo "  ✅ 完全支持NumPy 2.x"
    echo "  ✅ 更好的性能和稳定性"
    echo "  ✅ 修复了已知bug"
    echo "  ✅ 更好的GPU支持"
}

# 主函数
main() {
    echo "🚀 FAISS升级脚本 - 升级到1.12.0"
    echo "=" * 50
    
    # 检查conda环境
    if ! check_conda_env; then
        print_error "请先激活conda环境"
        echo "使用方法: conda activate qa-clean"
        exit 1
    fi
    
    # 显示当前环境信息
    show_upgrade_info
    
    # 检查当前FAISS版本
    if check_current_faiss; then
        print_success "FAISS已经是1.12.x版本，无需升级"
        exit 0
    fi
    
    # 升级FAISS
    if ! upgrade_faiss; then
        print_error "FAISS升级失败"
        exit 1
    fi
    
    # 测试升级结果
    if test_upgrade; then
        print_success "升级完成！"
        echo ""
        echo "🎉 现在可以正常使用项目了:"
        echo "   qa-clean --help"
        echo ""
        echo "💡 FAISS 1.12.0完全支持NumPy 2.x，解决了兼容性问题"
    else
        print_error "升级失败，请检查错误信息"
        exit 1
    fi
}

# 运行主函数
main "$@"
