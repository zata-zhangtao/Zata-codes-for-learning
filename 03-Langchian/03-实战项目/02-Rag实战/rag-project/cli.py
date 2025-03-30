"""
RAG系统命令行界面
"""

import os
import sys
import argparse
import json
import logging
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from rag_system import RAGSystem
from utils import truncate_text

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag_cli")

def print_json(data: Dict[str, Any], indent: int = 2) -> None:
    """美观打印JSON数据"""
    print(json.dumps(data, ensure_ascii=False, indent=indent))

def create_rag_system(args: argparse.Namespace) -> RAGSystem:
    """创建RAG系统实例"""
    return RAGSystem(
        docs_dir=args.docs_dir,
        db_dir=args.db_dir,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        temperature=args.temperature
    )

def handle_index(args: argparse.Namespace) -> None:
    """处理索引命令"""
    rag = create_rag_system(args)
    result = rag.index_documents()
    print_json(result)

def handle_query(args: argparse.Namespace) -> None:
    """处理查询命令"""
    rag = create_rag_system(args)
    
    # 处理过滤条件
    filter_dict = None
    if args.filter:
        try:
            filter_dict = json.loads(args.filter)
        except json.JSONDecodeError:
            logger.error(f"无效的过滤条件JSON: {args.filter}")
            sys.exit(1)
    
    # 执行查询
    result = rag.process_query(
        query=args.query,
        top_k=args.top_k,
        filter_dict=filter_dict,
        return_sources=not args.no_sources
    )
    
    # 打印回答
    print("\n" + "=" * 80)
    print("回答:")
    print("-" * 80)
    print(result["answer"])
    print("=" * 80)
    
    # 打印来源
    if not args.no_sources and result.get("sources"):
        print("\n来源信息:")
        print("-" * 80)
        for i, source in enumerate(result["sources"]):
            print(f"[{i+1}] {source.get('filename', 'Unknown')}")
            if "source" in source:
                print(f"    路径: {source['source']}")
            if "page" in source and source["page"] is not None:
                print(f"    页码: {source['page']}")
            if "similarity" in source and source["similarity"] is not None:
                print(f"    相似度: {source['similarity']:.4f}")
            print(f"    预览: {source.get('text', '无预览')}")
            print()
    
    # 打印处理时间
    if "processing_time" in result:
        print(f"处理时间: {result['processing_time']:.2f}秒")

def handle_add(args: argparse.Namespace) -> None:
    """处理添加文档命令"""
    rag = create_rag_system(args)
    
    # 读取文件内容
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            logger.error(f"读取文件失败: {str(e)}")
            sys.exit(1)
    else:
        text = args.text
    
    # 处理元数据
    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            logger.error(f"无效的元数据JSON: {args.metadata}")
            sys.exit(1)
    
    # 添加文档
    result = rag.add_document(text=text, metadata=metadata)
    print_json(result)

def handle_stats(args: argparse.Namespace) -> None:
    """处理统计信息命令"""
    rag = create_rag_system(args)
    stats = rag.get_stats()
    print_json(stats)

def handle_clear(args: argparse.Namespace) -> None:
    """处理清空数据命令"""
    rag = create_rag_system(args)
    result = rag.clear_data()
    print_json(result)

def handle_customize_prompt(args: argparse.Namespace) -> None:
    """处理自定义提示模板命令"""
    rag = create_rag_system(args)
    
    # 读取模板文件
    try:
        with open(args.template_file, "r", encoding="utf-8") as f:
            template = f.read()
    except Exception as e:
        logger.error(f"读取模板文件失败: {str(e)}")
        sys.exit(1)
    
    # 自定义提示模板
    try:
        rag.customize_prompt(template)
        print({"status": "success", "message": "成功自定义提示模板"})
    except Exception as e:
        logger.error(f"自定义提示模板失败: {str(e)}")
        sys.exit(1)

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="RAG系统命令行工具")
    
    # 全局参数
    parser.add_argument("--docs-dir", default="documents", help="文档目录")
    parser.add_argument("--db-dir", default="db", help="数据库目录")
    parser.add_argument("--embedding-model", default="openai", help="嵌入模型名称")
    parser.add_argument("--llm-model", default="gpt-3.5-turbo", help="语言模型名称")
    parser.add_argument("--chunk-size", type=int, default=1000, help="分块大小")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="分块重叠")
    parser.add_argument("--temperature", type=float, default=0.1, help="模型温度")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 索引命令
    index_parser = subparsers.add_parser("index", help="索引文档")
    
    # 查询命令
    query_parser = subparsers.add_parser("query", help="查询RAG系统")
    query_parser.add_argument("query", help="查询文本")
    query_parser.add_argument("--top-k", type=int, default=5, help="检索文档数量")
    query_parser.add_argument("--filter", help="过滤条件(JSON格式)")
    query_parser.add_argument("--no-sources", action="store_true", help="不显示来源信息")
    
    # 添加文档命令
    add_parser = subparsers.add_parser("add", help="添加文档")
    add_group = add_parser.add_mutually_exclusive_group(required=True)
    add_group.add_argument("--text", help="文档文本")
    add_group.add_argument("--file", help="文档文件路径")
    add_parser.add_argument("--metadata", help="文档元数据(JSON格式)")
    
    # 统计信息命令
    stats_parser = subparsers.add_parser("stats", help="获取统计信息")
    
    # 清空数据命令
    clear_parser = subparsers.add_parser("clear", help="清空系统数据")
    
    # 自定义提示模板命令
    customize_parser = subparsers.add_parser("customize-prompt", help="自定义提示模板")
    customize_parser.add_argument("template_file", help="提示模板文件路径")
    
    return parser.parse_args()

def main() -> None:
    """主函数"""
    args = parse_args()
    
    # 根据命令调用对应的处理函数
    if args.command == "index":
        handle_index(args)
    elif args.command == "query":
        handle_query(args)
    elif args.command == "add":
        handle_add(args)
    elif args.command == "stats":
        handle_stats(args)
    elif args.command == "clear":
        handle_clear(args)
    elif args.command == "customize-prompt":
        handle_customize_prompt(args)
    else:
        print("请指定要执行的命令。使用 --help 查看帮助。")
        sys.exit(1)

if __name__ == "__main__":
    main() 