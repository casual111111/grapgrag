#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG Neo4j Local Search — MCP Server (HTTP/SSE 模式)
====================================================

将基于 Neo4j 的 GraphRAG local search 包装为 MCP (Model Context Protocol) HTTP 服务，
任何支持 MCP 的客户端通过 HTTP SSE 连接使用。

启动方式：
    pip install mcp neo4j

    # 启动 HTTP Server（默认 localhost:8000）
    python neo4j_mcp_server.py

    # 指定端口和项目目录
    python neo4j_mcp_server.py --port 8080 --root-dir C:\保存\graphrag-github-flash\graphrag\my_kg

客户端配置示例（Claude Desktop / Cursor / 自定义 Agent）：
    {
      "mcpServers": {
        "graphrag": {
          "url": "http://localhost:8080/sse"
        }
      }
    }
   {
      "mcpServers": {
        "graphrag": {
          "url": "http://192.168.127.61:8080/mcp"
        }
      }
    }

"""

import argparse
import asyncio
import io
import os
import random
import sys
from pathlib import Path
from typing import Any

# ============================================================
# 添加本地 packages 到 sys.path（与 batch_query.py 保持一致）
# ============================================================
_project_root = Path(__file__).resolve().parent  # 脚本在项目根目录下，parent 就是项目根
# 如果脚本放在 eval/ 下，改为 .parent.parent
_packages_dir = _project_root / "packages"
if _packages_dir.is_dir():
    for pkg in sorted(_packages_dir.iterdir()):
        pkg_str = str(pkg)
        if pkg_str not in sys.path:
            sys.path.insert(0, pkg_str)

# 添加 neo4j_search 目录
_neo4j_search_dir = _project_root / "my_kg" / "neo4j_search"
if _neo4j_search_dir.is_dir():
    sys.path.insert(0, str(_neo4j_search_dir))

import pandas as pd

from graphrag.config.load_config import load_config
from graphrag_llm.completion import create_completion
from graphrag_llm.embedding import create_embedding
from graphrag.utils.api import load_search_prompt

# 导入 Neo4j context builder
from neo4j_context_builder import Neo4jLocalContextBuilder

import contextlib

# MCP SDK
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import Tool, TextContent

# HTTP 框架
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.responses import JSONResponse, Response
import uvicorn

# ============================================================
# 配置参数
# ============================================================
RESPONSE_TYPE = "multiple paragraphs"
MAX_RETRIES = 3
RETRY_DELAY = 5

# Neo4j 配置
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4j_test"


# ============================================================
# 搜索引擎构建
# ============================================================

def build_search_engine(config, text_embedder):
    """
    构建 Neo4j LocalSearch 引擎。
    """
    return Neo4jLocalContextBuilder(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        text_embedder=text_embedder,
    )


# ============================================================
# MCP Server 定义
# ============================================================

_context_builder = None
_chat_model = None
_system_prompt = None
_config = None


def get_root_dir() -> Path:
    """从命令行参数或环境变量获取 GraphRAG 项目根目录"""
    env_root = os.environ.get("GRAPHRAG_ROOT_DIR")
    if env_root:
        return Path(env_root).expanduser().resolve()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "my_kg",
        help="GraphRAG 项目根目录（settings.yaml 所在目录）",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args, _ = parser.parse_known_args()
    return args.root_dir.expanduser().resolve()


def get_server_args():
    """获取服务器启动参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")
    parser.add_argument("--root-dir", type=Path, default=None)
    args, _ = parser.parse_known_args()
    return args


def initialize_engine():
    """初始化 search engine（仅执行一次）"""
    global _context_builder, _chat_model, _system_prompt, _config

    root_dir = get_root_dir()
    print(f"[GraphRAG Neo4j MCP] 加载配置: {root_dir}", file=sys.stderr)

    _config = load_config(root_dir)

    # 加载 embedding 模型
    print("[GraphRAG Neo4j MCP] 加载 embedding 模型...", file=sys.stderr)
    embedding_settings = _config.get_embedding_model_config(
        _config.local_search.embedding_model_id
    )
    text_embedder = create_embedding(embedding_settings)

    # 加载 chat 模型
    print("[GraphRAG Neo4j MCP] 加载 chat 模型...", file=sys.stderr)
    model_settings = _config.get_completion_model_config(
        _config.local_search.completion_model_id
    )
    _chat_model = create_completion(model_settings)

    # 加载 system prompt
    _system_prompt = load_search_prompt(_config.local_search.prompt)

    # 构建 Neo4j context builder
    print("[GraphRAG Neo4j MCP] 构建 Neo4j 搜索引擎...", file=sys.stderr)
    _context_builder = build_search_engine(_config, text_embedder)
    print("[GraphRAG Neo4j MCP] ✅ 搜索引擎就绪！", file=sys.stderr)


# 创建 MCP Server 实例
server = Server("graphrag-neo4j-local-search")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """列出可用的 MCP 工具"""
    return [
        Tool(
            name="neo4j_local_search",
            description=(
                "使用基于 Neo4j 的 GraphRAG 知识图谱进行本地搜索。"
                "支持多跳图遍历，基于知识图谱中的实体、关系和原始文本块，"
                "综合检索相关上下文并生成回答。"
                "适合回答需要具体实体信息、关系推理的问题。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要搜索的问题或查询内容",
                    },
                    "num_hops": {
                        "type": "integer",
                        "description": "多跳扩展的跳数（默认 2）",
                        "default": 2,
                    },
                    "top_k_entities": {
                        "type": "integer",
                        "description": "向量检索入口实体数（默认 10）",
                        "default": 10,
                    },
                    "top_k_relationships": {
                        "type": "integer",
                        "description": "每个实体最多保留的关系数（默认 18）",
                        "default": 18,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="neo4j_local_search_with_context",
            description=(
                "使用基于 Neo4j 的 GraphRAG 知识图谱进行本地搜索，并返回检索到的上下文。"
                "除了回答外，还返回检索到的实体、关系、文本块等原始上下文信息，"
                "方便调试或需要引用来源时使用。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要搜索的问题或查询内容",
                    },
                    "num_hops": {
                        "type": "integer",
                        "description": "多跳扩展的跳数（默认 2）",
                        "default": 2,
                    },
                    "top_k_entities": {
                        "type": "integer",
                        "description": "向量检索入口实体数（默认 10）",
                        "default": 10,
                    },
                    "top_k_relationships": {
                        "type": "integer",
                        "description": "每个实体最多保留的关系数（默认 18）",
                        "default": 18,
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """执行 MCP 工具调用"""
    if name not in ("neo4j_local_search", "neo4j_local_search_with_context"):
        return [TextContent(type="text", text=f"未知工具: {name}")]

    query = arguments.get("query", "").strip()
    if not query:
        return [TextContent(type="text", text="错误：query 参数不能为空")]

    # 获取可选参数
    num_hops = arguments.get("num_hops", 2)
    top_k_entities = arguments.get("top_k_entities", 10)
    top_k_relationships = arguments.get("top_k_relationships", 18)

    answer, context_text, context_records = await _do_search(
        query, num_hops, top_k_entities, top_k_relationships
    )

    if name == "neo4j_local_search":
        return [TextContent(type="text", text=answer)]
    else:
        # 构建详细的上下文信息
        context_info = []
        if context_records:
            n_entities = len(context_records.get("entities")) if context_records.get("entities") is not None else 0
            n_relationships = len(context_records.get("relationships")) if context_records.get("relationships") is not None else 0
            n_text_units = len(context_records.get("sources")) if context_records.get("sources") is not None else 0
            context_info.append(f"【上下文统计】：实体 {n_entities} 个 | 关系 {n_relationships} 条 | 文本片段 {n_text_units} 个\n")

        result = f"## 回答\n\n{answer}\n\n## 检索上下文\n\n{context_text}"
        if context_info:
            result = "\n".join(context_info) + "\n" + result
        return [TextContent(type="text", text=result)]


async def _do_search(query: str, num_hops: int, top_k_entities: int, top_k_relationships: int) -> tuple[str, str, dict]:
    """执行 Neo4j local search，带重试逻辑"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # 构建上下文
            context_result = await asyncio.to_thread(
                _context_builder.build_context,
                query=query,
                num_hops=num_hops,
                top_k_entities=top_k_entities,
                top_k_relationships=top_k_relationships,
            )

            context_text = context_result.context_chunks
            context_records = context_result.context_records

            # 生成回答
            system_message = _system_prompt.format(
                context_data=context_text,
                response_type=RESPONSE_TYPE,
            )

            response = await _chat_model.completion_async(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query},
                ],
                stream=True,
            )

            full_response = ""
            async for chunk in response:
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    if content:
                        full_response += content

            if not full_response.strip():
                raise RuntimeError("Neo4j Local Search 返回空答案")

            return full_response, context_text, context_records

        except KeyboardInterrupt:
            raise
        except Exception as e:
            if attempt >= MAX_RETRIES:
                error_name = type(e).__name__
                return f"查询失败（重试 {MAX_RETRIES} 次）: {error_name}: {e}", "", {}

            wait = min(RETRY_DELAY * (2 ** (attempt - 1)), 60)
            jitter = random.uniform(0, wait * 0.3)
            print(
                f"[GraphRAG Neo4j MCP] 第 {attempt} 次失败，{wait + jitter:.0f}s 后重试...",
                file=sys.stderr,
            )
            await asyncio.sleep(wait + jitter)

    return "查询失败", "", {}


# ============================================================
# HTTP 应用（Starlette + SSE）
# ============================================================

def create_app() -> Starlette:
    """创建 HTTP 应用，挂载 SSE 端点"""
    sse_transport = SseServerTransport("/messages/")

    async def health(request):
        """健康检查端点"""
        return JSONResponse({
            "status": "ok",
            "service": "graphrag-neo4j-local-search",
            "engine_ready": _context_builder is not None,
        })

    # Streamable HTTP（新版 MCP 协议，MCP 2025-03-26 规范）
    # 客户端（如 Quick）直接 POST 到 /mcp。stateless=True 免去 session 管理。
    session_manager = StreamableHTTPSessionManager(
        app=server,
        event_store=None,
        json_response=False,
        stateless=True,
    )

    # 用 ASGI 类包装，让 Starlette 以原始 ASGI 方式处理（保留 scope/receive/send），
    # 并注册为精确路由 /mcp —— 避免 Mount 带来的 307 尾斜杠重定向。
    class StreamableHTTPASGIApp:
        def __init__(self, manager):
            self._manager = manager

        async def __call__(self, scope, receive, send):
            await self._manager.handle_request(scope, receive, send)

    mcp_asgi = StreamableHTTPASGIApp(session_manager)

    # 让 /sse 同时兼容两种协议：
    #   GET  /sse  → 旧版 SSE 长连接（老客户端）
    #   POST /sse  → 新版 Streamable HTTP（Quick 等新客户端）
    # 这样无论客户端把地址填成 /sse 还是 /mcp 都能连上，避免 405。
    class SseCompatASGIApp:
        def __init__(self, sse_tp, manager):
            self._sse = sse_tp
            self._manager = manager

        async def __call__(self, scope, receive, send):
            if scope.get("method") == "GET":
                # 旧版 SSE：打开长连接并运行 MCP server
                async with self._sse.connect_sse(scope, receive, send) as streams:
                    await server.run(
                        streams[0], streams[1], server.create_initialization_options()
                    )
            else:
                # 新版 Streamable HTTP（POST/DELETE）：交给 session manager 处理
                await self._manager.handle_request(scope, receive, send)

    sse_compat = SseCompatASGIApp(sse_transport, session_manager)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        # 进入 session manager 的运行上下文（启动内部 task group）
        async with session_manager.run():
            yield

    app = Starlette(
        debug=False,
        routes=[
            Route("/health", health, methods=["GET"]),
            # /sse 同时支持 GET(旧版 SSE) 和 POST/DELETE(新版 Streamable HTTP)
            Route("/sse", endpoint=sse_compat, methods=["GET", "POST", "DELETE"]),
            # /messages/ 直接挂原始 ASGI handler（它自己发送响应），
            # 不能用 Route 包装，否则返回 None 触发 NoneType 报错。
            Mount("/messages/", app=sse_transport.handle_post_message),
            # 精确匹配 /mcp，支持 Streamable HTTP 需要的 POST/GET/DELETE
            Route("/mcp", endpoint=mcp_asgi, methods=["GET", "POST", "DELETE"]),
        ],
        lifespan=lifespan,
    )
    return app


# ============================================================
# 主入口
# ============================================================

def main():
    """启动 MCP HTTP Server"""
    args = get_server_args()

    # 初始化搜索引擎
    initialize_engine()

    # 创建并启动 HTTP 应用
    app = create_app()

    print(f"[GraphRAG Neo4j MCP] 🚀 HTTP Server 启动: http://{args.host}:{args.port}", file=sys.stderr)
    print(f"[GraphRAG Neo4j MCP] 📡 SSE 端点: http://{args.host}:{args.port}/sse", file=sys.stderr)
    print(f"[GraphRAG Neo4j MCP] 💚 健康检查: http://{args.host}:{args.port}/health", file=sys.stderr)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
