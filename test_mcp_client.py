#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG MCP Server 测试客户端（不依赖 Quick）
================================================

用 MCP 官方 SSE 客户端连接本地服务，验证：
  1. 能否建立连接 / 握手
  2. 能列出哪些工具
  3. 可选：真正调用一次 local_search

用法：
    # 只做握手 + 列工具
    python test_mcp_client.py

    # 指定地址
    python test_mcp_client.py --url http://192.168.127.61:8000/sse

    # 连上后真正跑一次查询
    python test_mcp_client.py --query "介绍一下xxx"

    # 用 streamable http (/mcp) 而不是 sse
    python test_mcp_client.py --url http://127.0.0.1:8000/mcp --transport http
"""

import argparse
import sys

import anyio
from mcp import ClientSession
from mcp.client.sse import sse_client


async def run(url: str, transport: str, query: str | None, timeout: float):
    print(f"[连接] transport={transport} url={url}", flush=True)

    if transport == "http":
        from mcp.client.streamable_http import streamablehttp_client
        cm = streamablehttp_client(url)
    else:
        cm = sse_client(url)

    async with cm as streams:
        read, write = streams[0], streams[1]
        async with ClientSession(read, write) as session:
            with anyio.fail_after(timeout):
                info = await session.initialize()
                print(f"[握手成功] server={info.serverInfo.name} v{info.serverInfo.version}", flush=True)

                tools = await session.list_tools()
                print(f"[工具列表] {[t.name for t in tools.tools]}", flush=True)

                if query:
                    print(f"\n[调用 local_search] query={query!r}", flush=True)
                    result = await session.call_tool("local_search", {"query": query})
                    for block in result.content:
                        text = getattr(block, "text", str(block))
                        print("----- 结果 -----", flush=True)
                        print(text, flush=True)

    print("\n[完成] 一切正常 ✅", flush=True)


def main():
    parser = argparse.ArgumentParser(description="GraphRAG MCP 测试客户端")
    parser.add_argument("--url", default="http://127.0.0.1:8000/sse", help="MCP 端点 URL")
    parser.add_argument("--transport", choices=["sse", "http"], default="sse",
                        help="传输方式：sse(默认) 或 http(streamable)")
    parser.add_argument("--query", default=None, help="可选：连上后真正跑一次查询")
    parser.add_argument("--timeout", type=float, default=120.0, help="单次操作超时秒数")
    args = parser.parse_args()

    try:
        anyio.run(run, args.url, args.transport, args.query, args.timeout)
    except Exception as e:
        print(f"[失败] {type(e).__name__}: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
