# QA Clean 项目发布指南

本文档详细介绍如何使用conda管理后，如何发布QA Clean项目到各个平台。

## 📋 发布平台概览

### 1. PyPI (Python Package Index)
- **用途**: Python包的标准分发平台
- **用户**: 使用pip安装的用户
- **优势**: 标准Python包管理，用户基数大

### 2. Conda-forge
- **用途**: conda包的分发平台
- **用户**: 使用conda安装的用户
- **优势**: 科学计算包生态，依赖管理更好

### 3. GitHub Releases
- **用途**: 源码发布和版本管理
- **用户**: 开发者、高级用户
- **优势**: 与代码仓库集成，支持二进制文件

## 🚀 发布流程

### 阶段1: 准备工作

#### 1.1 更新版本号
```bash
# 使用发布脚本自动更新
chmod +x scripts/build_and_release.sh
./scripts/build_and_release.sh release patch  # 补丁版本
./scripts/build_and_release.sh release minor  # 次要版本
./scripts/build_and_release.sh release major  # 主要版本
```

#### 1.2 检查代码质量
```bash
# 激活开发环境
conda activate qa-clean-dev

# 代码格式化
ruff format .

# 代码检查
ruff check .

# 类型检查
mypy src/

# 运行测试
pytest
```

#### 1.3 更新文档
- 更新 `README.md`
- 更新 `CHANGELOG.md`
- 检查所有文档链接

### 阶段2: 构建和测试

#### 2.1 构建项目
```bash
# 使用发布脚本
./scripts/build_and_release.sh build

# 或手动构建
python setup.py sdist bdist_wheel
```

#### 2.2 测试构建的包
```bash
# 使用发布脚本
./scripts/build_and_release.sh test

# 或手动测试
conda create -n test-qa-clean python=3.9 -y
conda activate test-qa-clean
pip install dist/*.whl
qa-clean --help
python -c "import qa_clean; print('Import test passed')"
conda deactivate
conda env remove -n test-qa-clean -y
```

### 阶段3: 发布到各平台

#### 3.1 发布到PyPI

**准备工作:**
1. 注册PyPI账号: https://pypi.org/account/register/
2. 创建API Token: https://pypi.org/manage/account/token/
3. 配置 `~/.pypirc` 文件

```ini
[pypi]
username = __token__
password = pypi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**发布命令:**
```bash
# 使用发布脚本
./scripts/build_and_release.sh pypi

# 或手动发布
twine upload dist/*
```

#### 3.2 发布到Conda-forge

**准备工作:**
1. Fork conda-forge/qa-clean-feedstock
2. 克隆你的fork到本地
3. 更新 `recipe/meta.yaml` 中的版本号

**发布步骤:**
```bash
# 1. 更新版本号
git checkout -b update-to-v0.1.0
# 编辑 meta.yaml 中的版本号

# 2. 提交更改
git add .
git commit -m "Update to v0.1.0"

# 3. 推送分支
git push origin update-to-v0.1.0

# 4. 创建Pull Request
# 在GitHub上创建PR，等待CI通过并合并
```

**本地构建测试:**
```bash
# 使用发布脚本
./scripts/build_and_release.sh conda

# 或手动构建
conda build meta.yaml --output-folder=conda-dist/
```

#### 3.3 发布到GitHub

**创建Release:**
```bash
# 使用发布脚本
./scripts/build_and_release.sh tag
./scripts/build_and_release.sh push

# 或手动操作
git add .
git commit -m "Release version 0.1.0"
git tag -a "v0.1.0" -m "Release version 0.1.0"
git push origin --tags
```

**在GitHub上:**
1. 进入 Releases 页面
2. 点击 "Create a new release"
3. 选择刚推送的标签
4. 填写发布说明（使用 `RELEASE_TEMPLATE.md` 模板）
5. 上传构建的二进制文件
6. 发布

## 🔧 自动化发布

### GitHub Actions

项目已配置GitHub Actions工作流，当推送标签时会自动：

1. 运行多平台测试
2. 构建Python包
3. 发布到PyPI
4. 构建conda包

**触发条件:**
```bash
git tag v0.1.0
git push origin v0.1.0
```

### 手动触发

如果需要手动触发CI/CD：

1. 进入 GitHub Actions 页面
2. 选择 "Release" 工作流
3. 点击 "Run workflow"
4. 选择分支和输入参数

## 📦 包管理策略

### 版本号规范

使用 [语义化版本控制](https://semver.org/lang/zh-CN/)：

- **MAJOR.MINOR.PATCH**
- **MAJOR**: 不兼容的API修改
- **MINOR**: 向下兼容的功能性新增
- **PATCH**: 向下兼容的问题修正

### 依赖管理

**conda优先策略:**
1. 优先使用conda安装包
2. conda没有的包使用pip
3. 在 `environment.yml` 中指定conda包
4. 在 `pip:` 部分指定pip包

**版本锁定:**
- 开发环境使用 `>=` 允许更新
- 生产环境考虑使用 `==` 锁定版本

## 🧪 测试策略

### 单元测试
```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_models.py

# 生成覆盖率报告
pytest --cov=qa_clean --cov-report=html
```

### 集成测试
```bash
# 测试CLI功能
qa-clean info
qa-clean process test_data.csv --output results.csv

# 测试不同存储后端
qa-clean process test_data.csv --vector-store faiss_gpu
qa-clean process test_data.csv --vector-store pgvector
```

### 兼容性测试
- Python 3.9, 3.10, 3.11
- Windows, macOS, Linux
- 不同conda环境

## 📋 发布检查清单

### 发布前
- [ ] 代码通过所有测试
- [ ] 代码质量检查通过
- [ ] 文档更新完成
- [ ] 版本号已更新
- [ ] CHANGELOG已更新
- [ ] 依赖版本已检查

### 发布中
- [ ] 构建成功
- [ ] 测试通过
- [ ] PyPI发布成功
- [ ] Conda-forge PR已创建
- [ ] GitHub Release已创建
- [ ] 标签已推送

### 发布后
- [ ] 验证安装
- [ ] 检查文档链接
- [ ] 通知用户
- [ ] 监控反馈

## 🚨 常见问题

### 1. PyPI发布失败
**问题**: `HTTPError: 400 Client Error: File already exists.`
**解决**: 删除 `dist/` 目录，重新构建

### 2. Conda构建失败
**问题**: 依赖冲突
**解决**: 检查 `meta.yaml` 中的依赖版本，使用conda-forge频道

### 3. 版本号不一致
**问题**: 多个配置文件版本号不同步
**解决**: 使用发布脚本自动更新所有配置文件

### 4. 测试环境问题
**问题**: 测试环境依赖缺失
**解决**: 使用 `environment-dev.yml` 创建完整的开发环境

## 📚 相关资源

- [Python打包指南](https://packaging.python.org/)
- [Conda-forge贡献指南](https://conda-forge.org/docs/maintainer/adding_pkgs.html)
- [GitHub Actions文档](https://docs.github.com/en/actions)
- [语义化版本控制](https://semver.org/lang/zh-CN/)
- [Twine文档](https://twine.readthedocs.io/)

## 🤝 获取帮助

如果在发布过程中遇到问题：

1. 查看本文档的故障排除部分
2. 搜索GitHub Issues
3. 创建新的Issue描述问题
4. 联系项目维护者

---

**记住**: 发布前一定要充分测试，确保代码质量和功能完整性！
