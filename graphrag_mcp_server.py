
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG Local Search — MCP Server (HTTP/SSE 模式)
====================================================

将 GraphRAG 的 local search 包装为 MCP (Model Context Protocol) HTTP 服务，
任何支持 MCP 的客户端通过 HTTP SSE 连接使用。

启动方式：
    pip install mcp

    # 启动 HTTP Server（默认 localhost:8000）
    python graphrag_mcp_server.py

    # 指定端口和项目目录
    python graphrag_mcp_server.py --port 8080 --root-dir C:\保存\graphrag-github-flash\graphrag\my_kg

客户端配置示例（Claude Desktop / Cursor / 自定义 Agent）：
    {
      "mcpServers": {
        "graphrag": {
          "url": "http://localhost:8000/sse"
        }
      }
    }
   {
      "mcpServers": {
        "graphrag": {
          "url": "http://192.168.127.61:8000/mcp"
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

import pandas as pd

import graphrag.api as api
from graphrag.config.load_config import load_config
from graphrag.config.embeddings import entity_description_embedding
from graphrag.query.factory import get_local_search_engine
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.utils.api import (
    get_embedding_store,
    load_search_prompt,
)
from graphrag_storage import create_storage

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
COMMUNITY_LEVEL = 2
RESPONSE_TYPE = "multiple paragraphs"
MAX_RETRIES = 3
RETRY_DELAY = 5

# ============================================================
# 索引数据加载
# ============================================================

def storage_has_table(table_name: str, storage) -> bool:
    """检查存储中是否存在指定的表文件"""
    key = f"{table_name}.parquet"
    loop = asyncio.get_event_loop()
    if loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, storage.has(key)).result()
    return asyncio.run(storage.has(key))


async def load_table_from_storage(table_name: str, storage) -> pd.DataFrame:
    """从存储中加载 parquet 表为 DataFrame"""
    key = f"{table_name}.parquet"
    data = await storage.get(key, as_bytes=True)
    return pd.read_parquet(io.BytesIO(data))


def load_index_data(config) -> dict[str, pd.DataFrame | None]:
    """一次性加载所有索引数据"""
    storage_obj = create_storage(config.output_storage)
    data = {}

    async def _load_all():
        # 必需的表
        for table_name in ["entities", "relationships", "text_units"]:
            data[table_name] = await load_table_from_storage(table_name, storage_obj)

        # 可选的表
        for table_name in ["communities", "community_reports"]:
            key = f"{table_name}.parquet"
            if await storage_obj.has(key):
                data[table_name] = await load_table_from_storage(table_name, storage_obj)
            else:
                data[table_name] = None

        # covariates 是可选的
        if await storage_obj.has("covariates.parquet"):
            data["covariates"] = await load_table_from_storage("covariates", storage_obj)
        else:
            data["covariates"] = None

    asyncio.run(_load_all())
    return data


def build_search_engine(config, data):
    """
    构建 LocalSearch 引擎（复刻 graphrag.api.local_search_streaming 的装配逻辑）。
    """
    description_embedding_store = get_embedding_store(
        config=config.vector_store,
        embedding_name=entity_description_embedding,
    )

    has_communities = data["communities"] is not None
    entities_ = read_indexer_entities(
        data["entities"],
        data["communities"] if has_communities else pd.DataFrame({
            "community": pd.Series(dtype="str"),
            "entity_ids": pd.Series(dtype="object"),
            "level": pd.Series(dtype="int"),
        }),
        COMMUNITY_LEVEL if has_communities else None,
    )
    covariates_ = (
        read_indexer_covariates(data["covariates"])
        if data["covariates"] is not None
        else []
    )
    prompt = load_search_prompt(config.local_search.prompt)

    if data["community_reports"] is not None and data["communities"] is not None:
        reports = read_indexer_reports(
            data["community_reports"], data["communities"], COMMUNITY_LEVEL
        )
    else:
        reports = []

    return get_local_search_engine(
        config=config,
        reports=reports,
        text_units=read_indexer_text_units(data["text_units"]),
        entities=entities_,
        relationships=read_indexer_relationships(data["relationships"]),
        covariates={"claims": covariates_},
        description_embedding_store=description_embedding_store,
        response_type=RESPONSE_TYPE,
        system_prompt=prompt,
        callbacks=None,
    )


# ============================================================
# MCP Server 定义
# ============================================================

_search_engine = None
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
    global _search_engine, _config

    root_dir = get_root_dir()
    print(f"[GraphRAG MCP] 加载配置: {root_dir}", file=sys.stderr)

    _config = load_config(root_dir)
    print("[GraphRAG MCP] 加载索引数据...", file=sys.stderr)
    data = load_index_data(_config)
    print("[GraphRAG MCP] 构建搜索引擎...", file=sys.stderr)
    _search_engine = build_search_engine(_config, data)
    print("[GraphRAG MCP] ✅ 搜索引擎就绪！", file=sys.stderr)


# 创建 MCP Server 实例
server = Server("graphrag-local-search")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """列出可用的 MCP 工具"""
    return [
        Tool(
            name="local_search",
            description=(
                "使用 GraphRAG 知识图谱进行本地搜索。"
                "基于知识图谱中的实体、关系、社区报告和原始文本块，"
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
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="local_search_with_context",
            description=(
                "使用 GraphRAG 知识图谱进行本地搜索，并返回检索到的上下文。"
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
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """执行 MCP 工具调用"""
    if name not in ("local_search", "local_search_with_context"):
        return [TextContent(type="text", text=f"未知工具: {name}")]

    query = arguments.get("query", "").strip()
    if not query:
        return [TextContent(type="text", text="错误：query 参数不能为空")]

    answer, context_text = await _do_search(query)

    if name == "local_search":
        return [TextContent(type="text", text=answer)]
    else:
        result = f"## 回答\n\n{answer}\n\n## 检索上下文\n\n{context_text}"
        return [TextContent(type="text", text=result)]


async def _do_search(query: str) -> tuple[str, str]:
    """执行 local search，带重试逻辑"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = await _search_engine.search(query=query)
            answer = str(result.response)

            if not answer.strip():
                raise RuntimeError("Local Search 返回空答案")

            context_text = result.context_text
            if isinstance(context_text, list):
                context_text = "\n\n".join(str(c) for c in context_text)
            elif isinstance(context_text, dict):
                context_text = "\n\n".join(
                    f"----- {k} -----\n{v}" for k, v in context_text.items()
                )

            return answer, str(context_text)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            if attempt >= MAX_RETRIES:
                error_name = type(e).__name__
                return f"查询失败（重试 {MAX_RETRIES} 次）: {error_name}: {e}", ""

            wait = min(RETRY_DELAY * (2 ** (attempt - 1)), 60)
            jitter = random.uniform(0, wait * 0.3)
            print(
                f"[GraphRAG MCP] 第 {attempt} 次失败，{wait + jitter:.0f}s 后重试...",
                file=sys.stderr,
            )
            await asyncio.sleep(wait + jitter)

    return "查询失败", ""


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
            "service": "graphrag-local-search",
            "engine_ready": _search_engine is not None,
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

    print(f"[GraphRAG MCP] 🚀 HTTP Server 启动: http://{args.host}:{args.port}", file=sys.stderr)
    print(f"[GraphRAG MCP] 📡 SSE 端点: http://{args.host}:{args.port}/sse", file=sys.stderr)
    print(f"[GraphRAG MCP] 💚 健康检查: http://{args.host}:{args.port}/health", file=sys.stderr)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
