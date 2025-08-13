#!/bin/bash

# QA Clean 项目构建和发布脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
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

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 未安装，请先安装"
        exit 1
    fi
}

# 检查当前分支
check_branch() {
    current_branch=$(git branch --show-current)
    if [ "$current_branch" != "main" ] && [ "$current_branch" != "master" ]; then
        print_warning "当前分支是 $current_branch，建议在 main/master 分支上发布"
        read -p "是否继续？(y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# 检查工作目录是否干净
check_working_directory() {
    if [ -n "$(git status --porcelain)" ]; then
        print_error "工作目录不干净，请先提交或暂存更改"
        git status --short
        exit 1
    fi
}

# 更新版本号
update_version() {
    local version_type=$1
    local current_version=$(grep 'version = ' pyproject.toml | cut -d'"' -f2)
    
    print_info "当前版本: $current_version"
    
    case $version_type in
        "patch")
            new_version=$(echo $current_version | awk -F. '{$NF = $NF + 1;} 1' | sed 's/ /./g')
            ;;
        "minor")
            new_version=$(echo $current_version | awk -F. '{$(NF-1) = $(NF-1) + 1; $NF = 0;} 1' | sed 's/ /./g')
            ;;
        "major")
            new_version=$(echo $current_version | awk -F. '{$(NF-2) = $(NF-2) + 1; $(NF-1) = 0; $NF = 0;} 1' | sed 's/ /./g')
            ;;
        *)
            print_error "无效的版本类型: $version_type"
            print_info "支持的类型: patch, minor, major"
            exit 1
            ;;
    esac
    
    print_info "新版本: $new_version"
    
    # 更新所有配置文件中的版本号
    sed -i.bak "s/version = \"$current_version\"/version = \"$new_version\"/g" pyproject.toml
    sed -i.bak "s/version = \"$current_version\"/version = \"$new_version\"/g" setup.py
    sed -i.bak "s/version = \"$current_version\"/version = \"$new_version\"/g" meta.yaml
    
    # 清理备份文件
    rm -f pyproject.toml.bak setup.py.bak meta.yaml.bak
    
    print_success "版本号已更新为 $new_version"
}

# 构建项目
build_project() {
    print_info "开始构建项目..."
    
    # 清理之前的构建
    rm -rf build/ dist/ *.egg-info/
    
    # 构建源码分发包
    python setup.py sdist bdist_wheel
    
    print_success "项目构建完成"
}

# 测试构建的包
test_build() {
    print_info "测试构建的包..."
    
    # 创建测试环境
    conda create -n test-qa-clean python=3.9 -y
    
    # 激活测试环境
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate test-qa-clean
    
    # 安装构建的包
    pip install dist/*.whl
    
    # 测试CLI
    qa-clean --help
    
    # 测试导入
    python -c "import qa_clean; print('✅ 导入成功')"
    
    # 清理测试环境
    conda deactivate
    conda env remove -n test-qa-clean -y
    
    print_success "包测试通过"
}

# 发布到PyPI
publish_to_pypi() {
    print_info "发布到PyPI..."
    
    # 检查twine是否安装
    if ! command -v twine &> /dev/null; then
        print_info "安装twine..."
        pip install twine
    fi
    
    # 上传到PyPI
    twine upload dist/*
    
    print_success "已发布到PyPI"
}

# 发布到conda-forge
publish_to_conda_forge() {
    print_info "发布到conda-forge..."
    
    print_warning "conda-forge发布需要手动操作："
    echo "1. Fork conda-forge/qa-clean-feedstock"
    echo "2. 更新meta.yaml中的版本号"
    echo "3. 提交Pull Request"
    echo "4. 等待CI通过并合并"
    
    print_info "本地构建conda包..."
    
    # 检查conda-build是否安装
    if ! command -v conda-build &> /dev/null; then
        print_info "安装conda-build..."
        conda install conda-build -y
    fi
    
    # 构建conda包
    conda build meta.yaml --output-folder=conda-dist/
    
    print_success "conda包构建完成，位于 conda-dist/ 目录"
}

# 创建Git标签
create_git_tag() {
    local version=$1
    
    print_info "创建Git标签 v$version..."
    
    git add .
    git commit -m "Release version $version"
    git tag -a "v$version" -m "Release version $version"
    
    print_success "Git标签 v$version 已创建"
}

# 推送标签
push_tags() {
    print_info "推送标签到远程仓库..."
    
    git push origin --tags
    
    print_success "标签已推送到远程仓库"
}

# 显示帮助信息
show_help() {
    echo "QA Clean 项目构建和发布脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  build                   构建项目"
    echo "  test                    测试构建的包"
    echo "  pypi                    发布到PyPI"
    echo "  conda                   构建conda包"
    echo "  release [patch|minor|major]  发布新版本"
    echo "  tag                    创建Git标签"
    echo "  push                   推送标签"
    echo "  help                    显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 build                # 构建项目"
    echo "  $0 release patch        # 发布补丁版本"
    echo "  $0 pypi                 # 发布到PyPI"
    echo "  $0 conda                # 构建conda包"
}

# 主函数
main() {
    case "${1:-help}" in
        "build")
            check_command python
            check_command conda
            build_project
            ;;
        "test")
            check_command python
            check_command conda
            test_build
            ;;
        "pypi")
            check_command python
            check_command conda
            build_project
            test_build
            publish_to_pypi
            ;;
        "conda")
            check_command python
            check_command conda
            build_project
            publish_to_conda_forge
            ;;
        "release")
            if [ -z "$2" ]; then
                print_error "请指定版本类型: patch, minor, major"
                exit 1
            fi
            check_command git
            check_branch
            check_working_directory
            update_version "$2"
            build_project
            test_build
            ;;
        "tag")
            check_command git
            current_version=$(grep 'version = ' pyproject.toml | cut -d'"' -f2)
            create_git_tag "$current_version"
            ;;
        "push")
            check_command git
            push_tags
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# 运行主函数
main "$@"
