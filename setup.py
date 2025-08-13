#!/usr/bin/env python3
"""
QA Clean 项目安装脚本
用于PyPI发布和pip安装
"""

from setuptools import setup, find_packages
import os

# 读取README文件
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# 读取requirements.txt
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="qa-clean",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="QA 数据清洗与治理工具（聚合去重、聚类合并、代表问题输出）",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/qa-clean",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/qa-clean/issues",
        "Documentation": "https://github.com/yourusername/qa-clean",
        "Source Code": "https://github.com/yourusername/qa-clean",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "ruff>=0.5",
            "mypy>=1.8",
            "pytest>=8",
            "types-tqdm",
            "pandas-stubs",
        ],
        "gpu": [
            "faiss-gpu>=1.7.2",
        ],
        "cpu": [
            "faiss-cpu>=1.7.2",
        ],
        "postgres": [
            "psycopg2-binary>=2.9.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "qa-clean=qa_clean.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="nlp, qa, cleaning, dedup, clustering, reranker, pgvector, faiss",
)
