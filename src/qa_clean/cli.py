"""
QA Clean 命令行接口
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from .processor import QAProcessor
from .config import get_recommended_store_type, get_store_config, list_available_stores


def main():
    """主函数"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        if args.command == "process":
            process_qa_data(args)
        elif args.command == "search":
            search_similar_questions(args)
        elif args.command == "info":
            show_vector_store_info(args)
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)


def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="QA 数据清洗与治理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理QA数据（使用FAISS GPU，默认）
  qa-clean process data.csv --output results.csv
  
  # 使用PostgreSQL+pgvector存储
  qa-clean process data.csv --vector-store pgvector --output results.csv
  
  # 搜索相似问题
  qa-clean search "如何安装Python?" --vector-store faiss_gpu
  
  # 查看存储信息
  qa-clean info --vector-store faiss_gpu
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # process 命令
    process_parser = subparsers.add_parser("process", help="处理QA数据")
    process_parser.add_argument("input", help="输入文件路径 (CSV/Excel)")
    process_parser.add_argument("--output", "-o", help="输出文件路径")
    process_parser.add_argument("--vector-store", choices=["faiss_gpu", "pgvector"], 
                              default="faiss_gpu", help="向量存储类型 (默认: faiss_gpu)")
    process_parser.add_argument("--topk", type=int, default=100, help="相似度搜索top-k值 (默认: 100)")
    process_parser.add_argument("--question-col", default="question", help="问题列名 (默认: question)")
    process_parser.add_argument("--answer-col", default="answer", help="答案列名 (默认: answer)")
    
    # FAISS GPU 特定参数
    process_parser.add_argument("--gpu-id", type=int, default=0, help="GPU设备ID (默认: 0)")
    
    # PostgreSQL 特定参数
    process_parser.add_argument("--pg-host", help="PostgreSQL主机地址")
    process_parser.add_argument("--pg-port", type=int, help="PostgreSQL端口")
    process_parser.add_argument("--pg-db", help="PostgreSQL数据库名")
    process_parser.add_argument("--pg-user", help="PostgreSQL用户名")
    process_parser.add_argument("--pg-password", help="PostgreSQL密码")
    
    # search 命令
    search_parser = subparsers.add_parser("search", help="搜索相似问题")
    search_parser.add_argument("query", help="查询问题")
    search_parser.add_argument("--vector-store", choices=["faiss_gpu", "pgvector"], 
                              default="faiss_gpu", help="向量存储类型 (默认: faiss_gpu)")
    search_parser.add_argument("--topk", type=int, default=10, help="返回结果数量 (默认: 10)")
    search_parser.add_argument("--gpu-id", type=int, default=0, help="GPU设备ID (默认: 0)")
    
    # info 命令
    info_parser = subparsers.add_parser("info", help="显示向量存储信息")
    info_parser.add_argument("--vector-store", choices=["faiss_gpu", "pgvector"], 
                           help="指定存储类型，不指定则显示所有")
    
    return parser


def process_qa_data(args):
    """处理QA数据"""
    print(f"🔄 开始处理QA数据: {args.input}")
    
    # 读取数据
    data = read_input_file(args.input, args.question_col, args.answer_col)
    if not data:
        print("❌ 没有读取到有效数据")
        return
    
    print(f"📊 读取到 {len(data)} 条QA数据")
    
    # 准备向量存储参数
    store_kwargs = prepare_store_kwargs(args)
    
    # 创建处理器
    with QAProcessor(topk=args.topk, vector_store_type=args.vector_store, **store_kwargs) as processor:
        # 处理数据
        results = processor.process_qa_data(data)
        
        # 输出结果
        print(f"\n✅ 处理完成!")
        print(f"   原始数据: {results['total_count']} 条")
        print(f"   去重后: {results['unique_count']} 条")
        print(f"   聚类数量: {len(results['representative_questions'])} 个")
        
        # 保存结果
        if args.output:
            save_results(results, args.output)
            print(f"💾 结果已保存到: {args.output}")


def search_similar_questions(args):
    """搜索相似问题"""
    print(f"🔍 搜索相似问题: {args.query}")
    
    # 准备向量存储参数
    store_kwargs = prepare_store_kwargs(args)
    
    # 创建处理器
    with QAProcessor(topk=args.topk, vector_store_type=args.vector_store, **store_kwargs) as processor:
        # 搜索相似问题
        results = processor.search_similar_questions(args.query, args.topk)
        
        if not results:
            print("❌ 没有找到相似问题")
            return
        
        print(f"\n✅ 找到 {len(results)} 个相似问题:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. 相似度: {result['similarity']:.3f}")
            print(f"   问题: {result['text_content']}")
            if 'metadata' in result and result['metadata']:
                print(f"   元数据: {result['metadata']}")


def show_vector_store_info(args):
    """显示向量存储信息"""
    if args.vector_store:
        # 显示特定存储类型信息
        config = get_store_config(args.vector_store)
        print(f"📊 {config['name']} ({args.vector_store})")
        print(f"描述: {config['description']}")
        print(f"优势: {', '.join(config['pros'])}")
        print(f"劣势: {', '.join(config['cons'])}")
        print(f"适用场景: {', '.join(config['recommended_for'])}")
    else:
        # 显示所有存储类型信息
        print("🔍 可用的向量存储类型:\n")
        for store_type in list_available_stores():
            config = get_store_config(store_type)
            print(f"📊 {config['name']} ({store_type})")
            print(f"   描述: {config['description']}")
            print(f"   优势: {', '.join(config['pros'])}")
            print(f"   劣势: {', '.join(config['cons'])}")
            print(f"   适用场景: {', '.join(config['recommended_for'])}")
            print()
        
        # 显示推荐
        recommended = get_recommended_store_type()
        print(f"💡 当前环境推荐: {recommended}")


def read_input_file(file_path: str, question_col: str, answer_col: str) -> List[Dict[str, Any]]:
    """读取输入文件"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 根据文件扩展名选择读取方法
    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}")
    
    # 检查必要的列
    if question_col not in df.columns:
        raise ValueError(f"问题列 '{question_col}' 不存在")
    if answer_col not in df.columns:
        raise ValueError(f"答案列 '{answer_col}' 不存在")
    
    # 转换为字典列表
    data = []
    for _, row in df.iterrows():
        data.append({
            'question': str(row[question_col]),
            'answer': str(row[answer_col]),
            'metadata': {col: str(val) for col, val in row.items() 
                        if col not in [question_col, answer_col] and pd.notna(val)}
        })
    
    return data


def prepare_store_kwargs(args) -> Dict[str, Any]:
    """准备向量存储参数"""
    kwargs = {}
    
    if args.vector_store == "faiss_gpu":
        kwargs["gpu_id"] = args.gpu_id
    
    elif args.vector_store == "pgvector":
        # 使用环境变量或命令行参数
        kwargs["connection_params"] = {
            "host": args.pg_host or os.getenv("POSTGRES_HOST", "localhost"),
            "port": args.pg_port or int(os.getenv("POSTGRES_PORT", "5432")),
            "database": args.pg_db or os.getenv("POSTGRES_DB", "qa_clean"),
            "user": args.pg_user or os.getenv("POSTGRES_USER", "postgres"),
            "password": args.pg_password or os.getenv("POSTGRES_PASSWORD", "password"),
        }
    
    return kwargs


def save_results(results: Dict[str, Any], output_path: str):
    """保存结果到文件"""
    output_path = Path(output_path)
    
    # 准备输出数据
    output_data = []
    for i, qa_data in enumerate(results['dedup_results']):
        cluster_id = None
        for cluster_idx, cluster_indices in enumerate(results['clusters']):
            if i in cluster_indices:
                cluster_id = cluster_idx
                break
        
        output_data.append({
            'id': i,
            'question': qa_data['question'],
            'answer': qa_data['answer'],
            'cluster_id': cluster_id,
            'representative_question': results['representative_questions'].get(cluster_id, ''),
            'metadata': qa_data.get('metadata', {})
        })
    
    # 创建DataFrame并保存
    df = pd.DataFrame(output_data)
    
    if output_path.suffix.lower() == '.csv':
        df.to_csv(output_path, index=False, encoding='utf-8')
    elif output_path.suffix.lower() in ['.xlsx', '.xls']:
        df.to_excel(output_path, index=False)
    else:
        # 默认保存为CSV
        output_path = output_path.with_suffix('.csv')
        df.to_csv(output_path, index=False, encoding='utf-8')


if __name__ == "__main__":
    main()


