"""Neo4j 知识图谱检索演示。

用法：
    # 独立使用（关键词匹配，不需要 embedding）
    python neo4j_search_demo.py "用户注册"

    # 指定跳数
    python neo4j_search_demo.py "用户注册" --hops 3

    # 使用 GraphRAG embedding（需要配置 settings.yaml）
    python neo4j_search_demo.py "用户注册" --use-embedding

    # 进入交互模式
    python neo4j_search_demo.py --interactive
"""

import argparse
import sys
import io
from pathlib import Path

# 修复 Windows 控制台编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 确保 graphrag 包在 path 中
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "packages" / "graphrag"))

from neo4j_context_builder import Neo4jLocalContextBuilder, quick_search


def demo_keyword_search(query: str, num_hops: int = 2, top_k: int = 10):
    """使用关键词匹配进行搜索（不需要 embedding）。"""
    print(f"\n{'='*60}")
    print(f"查询: {query}")
    print(f"跳数: {num_hops}, Top-K 入口实体: {top_k}")
    print(f"{'='*60}\n")

    result = quick_search(
        query=query,
        num_hops=num_hops,
        top_k_entities=top_k,
    )

    # 打印结果
    print(result.context_chunks)

    # 打印统计信息
    if result.context_records:
        print(f"\n{'─'*60}")
        print("检索统计:")
        for key, df in result.context_records.items():
            print(f"  {key}: {len(df)} 条")


def demo_with_embedding(query: str, num_hops: int = 2, top_k: int = 10):
    """使用 embedding 向量检索进行搜索。"""
    print(f"\n{'='*60}")
    print(f"查询: {query} (向量检索模式)")
    print(f"跳数: {num_hops}, Top-K 入口实体: {top_k}")
    print(f"{'='*60}\n")

    # 初始化 GraphRAG embedding model
    try:
        from graphrag.config.load_config import load_config
        from graphrag_llm.embedding import create_embedding

        root_dir = Path(__file__).resolve().parent.parent
        config = load_config(root_dir)
        embedding_settings = config.get_embedding_model_config(
            config.local_search.embedding_model_id
        )
        embedder = create_embedding(embedding_settings)
        print(f"Embedding 模型加载成功: {embedding_settings.model}")
    except Exception as e:
        print(f"[ERROR] 无法加载 embedding model: {e}")
        print("请检查 settings.yaml 配置")
        return

    builder = Neo4jLocalContextBuilder(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="neo4j_test",
        text_embedder=embedder,
    )

    try:
        result = builder.build_context(
            query=query,
            num_hops=num_hops,
            top_k_entities=top_k,
        )

        print(result.context_chunks)

        if result.context_records:
            print(f"\n{'─'*60}")
            print("检索统计:")
            for key, df in result.context_records.items():
                print(f"  {key}: {len(df)} 条")

    finally:
        builder.close()


def demo_graphrag_integration(query: str):
    """演示如何集成到 GraphRAG 的 LocalSearch 中。"""
    print(f"\n{'='*60}")
    print("GraphRAG LocalSearch 集成演示")
    print(f"{'='*60}\n")

    try:
        import asyncio
        from graphrag.config.load_config import load_config
        from graphrag.query.factories import get_chat_model, get_text_embedder
        from graphrag.query.structured_search.local_search.search import LocalSearch

        root_dir = Path(__file__).resolve().parent.parent
        config = load_config(root_dir)

        chat_model = get_chat_model(config)
        embedder = get_text_embedder(config)

        builder = Neo4jLocalContextBuilder(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="neo4j_test",
            text_embedder=embedder,
        )

        # 创建 LocalSearch
        search = LocalSearch(
            model=chat_model,
            context_builder=builder,
            context_builder_params={
                "num_hops": 2,
                "top_k_entities": 10,
            },
        )

        # 执行搜索
        result = asyncio.run(search.search(query))

        print(f"\n查询: {query}")
        print(f"\n回答:\n{result.response}")

        builder.close()

    except ImportError as e:
        print(f"[ERROR] 需要 GraphRAG 包: {e}")
        print("确保 graphrag 在 Python path 中")
    except Exception as e:
        print(f"[ERROR] {e}")


def interactive_mode(use_embedding: bool = False, num_hops: int = 2, top_k: int = 10):
    """交互模式。"""
    print("\n" + "="*60)
    print("Neo4j 知识图谱检索 - 交互模式")
    print("输入 'quit' 或 'exit' 退出")
    print(f"模式: {'向量检索' if use_embedding else '关键词匹配'}")
    print(f"跳数: {num_hops}, Top-K: {top_k}")
    print("="*60 + "\n")

    while True:
        try:
            query = input("查询 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("退出")
            break

        if use_embedding:
            demo_with_embedding(query, num_hops=num_hops, top_k=top_k)
        else:
            demo_keyword_search(query, num_hops=num_hops, top_k=top_k)


def main():
    parser = argparse.ArgumentParser(description="Neo4j 知识图谱检索演示")
    parser.add_argument("query", nargs="?", help="查询内容")
    parser.add_argument("--hops", type=int, default=2, help="多跳扩展的跳数（默认 2）")
    parser.add_argument("--top-k", type=int, default=10, help="入口实体数量（默认 10）")
    parser.add_argument("--use-embedding", action="store_true", help="使用 embedding 向量检索（需要配置 settings.yaml）")
    parser.add_argument("--graphrag", action="store_true", help="集成到 GraphRAG LocalSearch（需要 LLM）")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")

    args = parser.parse_args()

    if args.interactive:
        interactive_mode(use_embedding=args.use_embedding, num_hops=args.hops, top_k=args.top_k)
    elif args.query:
        if args.graphrag:
            demo_graphrag_integration(args.query)
        elif args.use_embedding:
            demo_with_embedding(args.query, num_hops=args.hops, top_k=args.top_k)
        else:
            demo_keyword_search(args.query, num_hops=args.hops, top_k=args.top_k)
    else:
        parser.print_help()
        print("\n示例:")
        print('  python neo4j_search_demo.py "用户注册"')
        print('  python neo4j_search_demo.py "用户注册" --hops 3')
        print('  python neo4j_search_demo.py --interactive')


if __name__ == "__main__":
    main()
