#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
 python eval\batch_query.py --input-dir eval\QA.json  --output-dir eval\eval_flash  
GraphRAG 批量查询脚本（API 直调版）— community 版本

改进点（相比 subprocess 版本）：
  1. 直接调用 graphrag API，不再每条问题启动子进程
  2. 索引数据只加载一次，大幅提速
  3. 每条问题自动重试（网络超时/断连时）
  4. 输出两个文件：
     - batch_query_results.txt  — 干净的问答结果（问题+标准答案+生成答案）
     - batch_query_detail.txt   — 详细记录（含匹配的实体/关系/原文块等上下文）

用法：
    cd C:\保存\graphrag-github - community\graphrag
    conda activate graphrag
    python eval/batch_query.py

指定输入、输出文件夹：
    python eval/batch_query.py --input-dir D:\eval-data --output-dir D:\eval-results

同时指定输入文件名和 GraphRAG 项目根目录：
    python eval/batch_query.py --input-dir D:\eval-data --input-file questions.jsonl --output-dir D:\eval-results --root-dir C:\保存\graphrag-github - community\graphrag\my_kg
"""

import argparse
import asyncio
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Any

# 添加本地 packages 到 sys.path，使用项目源码而非 conda 安装的包
_project_root = Path(__file__).resolve().parent.parent
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
from graphrag.query.structured_search.local_search import (
    mixed_context as mixed_context_module,
)
from graphrag.utils.api import (
    get_embedding_store,
    load_search_prompt,
)
from graphrag_storage import create_storage
import pandas as pd
import io


def storage_has_table(table_name: str, storage) -> bool:
    """检查存储中是否存在指定的表文件"""
    key = f"{table_name}.parquet"
    return asyncio.run(storage.has(key))


async def load_table_from_storage(table_name: str, storage) -> pd.DataFrame:
    """从存储中加载 parquet 表为 DataFrame"""
    key = f"{table_name}.parquet"
    data = await storage.get(key, as_bytes=True)
    return pd.read_parquet(io.BytesIO(data))


# ============================================================
# 可调参数（命令行未指定时使用以下默认值）
# ============================================================
SCRIPT_DIR      = Path(__file__).resolve().parent
DEFAULT_ROOT_DIR = SCRIPT_DIR.parent / "my_kg"
DEFAULT_INPUT_DIR = SCRIPT_DIR
DEFAULT_INPUT_FILE = "QA.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
METHOD          = "local"       # 查询方法
MAX_RETRIES     = 3             # 每条问题的最大重试次数
RETRY_DELAY     = 5             # 首次重试等待秒数（指数退避，最大 60s）
COMMUNITY_LEVEL = 2             # 社区层级
RESPONSE_TYPE   = "multiple paragraphs"
# ============================================================


def parse_args() -> argparse.Namespace:
    """解析输入数据、输出结果和 GraphRAG 项目目录。"""
    parser = argparse.ArgumentParser(
        description="使用 GraphRAG API 批量回答 JSONL 评测问题。",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
            "输入 JSONL 文件路径，或 JSONL 所在文件夹"
            f"（默认文件夹: {DEFAULT_INPUT_DIR}）"
        ),
    )
    parser.add_argument(
        "--input-file",
        default=DEFAULT_INPUT_FILE,
        help=f"输入 JSONL 文件名（默认: {DEFAULT_INPUT_FILE}）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"结果输出文件夹，不存在时自动创建（默认: {DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=DEFAULT_ROOT_DIR,
        help=f"GraphRAG 项目根目录，即 settings.yaml 所在目录（默认: {DEFAULT_ROOT_DIR}）",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """解析并校验命令行路径，返回（输入文件、输出目录、项目根目录）。"""
    input_source = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    root_dir = args.root_dir.expanduser().resolve()

    if input_source.is_file():
        input_path = input_source
    elif input_source.is_dir():
        input_file = Path(args.input_file)
        if input_file.name != args.input_file or input_file.is_absolute():
            raise ValueError("--input-file 只能是文件名；文件夹请通过 --input-dir 指定")
        input_path = input_source / input_file
        if not input_path.is_file():
            raise ValueError(f"输入文件不存在或不是文件: {input_path}")
    else:
        raise ValueError(f"输入路径不存在或不是文件/文件夹: {input_source}")

    if not root_dir.is_dir():
        raise ValueError(f"GraphRAG 项目根目录不存在或不是文件夹: {root_dir}")
    if not (root_dir / "settings.yaml").is_file():
        raise ValueError(f"GraphRAG 项目根目录中找不到 settings.yaml: {root_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    return input_path, output_dir, root_dir


# ------------------------------------------------------------
# Local Search 分阶段计时（仅包装当前进程中的对象，不修改 site-packages）
# ------------------------------------------------------------
TIMING_STAGE_NAMES = [
    "query_embedding",
    "entity_vector_search",
    "entity_mapping_total",
    "entity_result_mapping",
    "community_context",
    "local_context",
    "text_unit_context",
    "context_assembly",
    "context_total",
    "llm_stream_ready_last_attempt",
    "llm_ttfb_last_attempt",
    "llm_first_reasoning_last_attempt",
    "llm_first_content_last_attempt",
    "llm_stream_to_first_chunk_last_attempt",
    "llm_reasoning_to_content_last_attempt",
    "llm_visible_generation_last_attempt",
    "llm_generation_last_attempt",
    "llm_stream_ready",
    "llm_ttfb",
    "llm_first_reasoning",
    "llm_first_content",
    "llm_generation",
    "llm_ttft",
    "llm_ttft_last_attempt",
    "prompt_and_overhead",
    "retry_wait",
    "query_total",
]

TIMING_STAGE_LABELS = {
    "query_embedding": "问题向量化",
    "entity_vector_search": "实体向量检索",
    "entity_mapping_total": "实体匹配总计",
    "entity_result_mapping": "实体结果映射",
    "community_context": "社区上下文",
    "local_context": "本地上下文",
    "text_unit_context": "文本块上下文",
    "context_assembly": "上下文拼装开销",
    "context_total": "上下文构建总计",
    "llm_stream_ready_last_attempt": "LLM 流对象就绪（最后尝试）",
    "llm_ttfb_last_attempt": "LLM 首原始 Chunk/TTFB（最后尝试）",
    "llm_first_reasoning_last_attempt": "LLM 首 Reasoning Chunk（最后尝试）",
    "llm_first_content_last_attempt": "LLM 首 Content Token（最后尝试）",
    "llm_stream_to_first_chunk_last_attempt": "流就绪到首 Chunk（最后尝试）",
    "llm_reasoning_to_content_last_attempt": "Reasoning 到 Content（最后尝试）",
    "llm_visible_generation_last_attempt": "首 Content 后生成（最后尝试）",
    "llm_generation_last_attempt": "LLM 完整流（最后尝试）",
    "llm_stream_ready": "LLM 流对象就绪累计",
    "llm_ttfb": "LLM 首原始 Chunk/TTFB 累计",
    "llm_first_reasoning": "LLM 首 Reasoning Chunk 累计",
    "llm_first_content": "LLM 首 Content Token 累计",
    "llm_generation": "LLM 完整流累计",
    "llm_ttft": "兼容字段：首 Content 累计",
    "llm_ttft_last_attempt": "兼容字段：最后一次首 Content",
    "prompt_and_overhead": "Prompt/其他开销",
    "retry_wait": "重试等待",
    "query_total": "整题总计",
}

LLM_METRIC_NAMES = [
    "llm_stream_ready",
    "llm_ttfb",
    "llm_first_reasoning",
    "llm_first_content",
    "llm_generation",
    "llm_ttft",
]


class QueryTiming:
    """收集单道题的分阶段耗时；重复调用同一阶段时自动累加。"""

    def __init__(self):
        self.values = defaultdict(float)
        self.llm_last_attempt: dict[str, float | None] = {}

    def reset(self):
        self.values.clear()
        self.llm_last_attempt.clear()

    def start_llm_attempt(self):
        """开始一次脚本层 LLM 尝试，清空“最后尝试”的里程碑。"""
        self.llm_last_attempt = {name: None for name in LLM_METRIC_NAMES}

    def add(self, stage: str, elapsed: float):
        self.values[stage] += elapsed

    def record_llm_metric(self, stage: str, elapsed: float):
        """累计 LLM 指标，并保留当前尝试的值。"""
        self.values[stage] += elapsed
        self.llm_last_attempt[stage] = elapsed

    def snapshot(self) -> dict[str, float | int | None]:
        result: dict[str, float | int | None] = dict(self.values)
        for name in LLM_METRIC_NAMES:
            result.setdefault(name, None)
            result[f"{name}_last_attempt"] = self.llm_last_attempt.get(name)
        return result


def get_delta_value(delta: Any, *names: str) -> Any:
    """兼容 LiteLLM/Pydantic/dict 的流式 delta 扩展字段。"""
    if delta is None:
        return None
    for name in names:
        if isinstance(delta, dict):
            value = delta.get(name)
        else:
            value = getattr(delta, name, None)
            if value is None:
                model_extra = getattr(delta, "model_extra", None)
                if isinstance(model_extra, dict):
                    value = model_extra.get(name)
        if value not in (None, ""):
            return value
    return None


class TimedAsyncStream:
    """代理 LiteLLM 原始异步流，在 GraphRAG 过滤 chunk 前记录里程碑。"""

    def __init__(self, stream, timing: QueryTiming, started: float):
        self._stream = stream
        self._iterator = stream.__aiter__()
        self._timing = timing
        self._started = started
        self._got_raw_chunk = False
        self._got_reasoning = False
        self._got_content = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        chunk = await self._iterator.__anext__()
        elapsed = perf_counter() - self._started

        if not self._got_raw_chunk:
            self._timing.record_llm_metric("llm_ttfb", elapsed)
            self._got_raw_chunk = True

        choices = getattr(chunk, "choices", None)
        if choices:
            delta = getattr(choices[0], "delta", None)
            reasoning = get_delta_value(
                delta,
                "reasoning_content",
                "reasoning",
                "thinking",
            )
            content = get_delta_value(delta, "content")

            if reasoning is not None and not self._got_reasoning:
                self._timing.record_llm_metric("llm_first_reasoning", elapsed)
                self._got_reasoning = True

            if content is not None and not self._got_content:
                self._timing.record_llm_metric("llm_first_content", elapsed)
                # 保留旧字段，便于继续使用已有分析脚本。
                self._timing.record_llm_metric("llm_ttft", elapsed)
                self._got_content = True

        return chunk

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def wrap_sync_method(obj, method_name: str, timing: QueryTiming, stage: str):
    """包装同步实例方法，并将多次调用耗时累加到指定阶段。"""
    original = getattr(obj, method_name)

    @wraps(original)
    def wrapped(*args, **kwargs):
        started = perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            timing.add(stage, perf_counter() - started)

    setattr(obj, method_name, wrapped)


def install_search_timing(search_engine) -> QueryTiming:
    """为当前 Local Search 引擎安装非侵入式计时包装。"""
    timing = QueryTiming()
    context_builder = search_engine.context_builder

    # 完整上下文及三个互相串行的上下文子阶段。
    wrap_sync_method(context_builder, "build_context", timing, "context_total")
    wrap_sync_method(
        context_builder,
        "_build_community_context",
        timing,
        "community_context",
    )
    wrap_sync_method(context_builder, "_build_local_context", timing, "local_context")
    wrap_sync_method(
        context_builder,
        "_build_text_unit_context",
        timing,
        "text_unit_context",
    )

    # 问题 Embedding。当前 GraphRAG 版本使用 embed；兼容新版的 embedding。
    embedding_method = (
        "embed" if hasattr(context_builder.text_embedder, "embed") else "embedding"
    )
    wrap_sync_method(
        context_builder.text_embedder,
        embedding_method,
        timing,
        "query_embedding",
    )

    # similarity_search_by_text 内部会调用该方法执行真正的 LanceDB ANN 检索。
    wrap_sync_method(
        context_builder.entity_text_embeddings,
        "similarity_search_by_vector",
        timing,
        "entity_vector_search",
    )

    # map_query_to_entities 包含 Embedding、ANN 和向量结果到实体对象的映射。
    original_map_query_to_entities = mixed_context_module.map_query_to_entities

    @wraps(original_map_query_to_entities)
    def timed_map_query_to_entities(*args, **kwargs):
        started = perf_counter()
        try:
            return original_map_query_to_entities(*args, **kwargs)
        finally:
            timing.add("entity_mapping_total", perf_counter() - started)

    mixed_context_module.map_query_to_entities = timed_map_query_to_entities

    # 包装 completion_async 方法以记录 LLM 计时
    original_completion_async = search_engine.model.completion_async

    @wraps(original_completion_async)
    async def timed_completion_async(*args, **kwargs):
        timing.start_llm_attempt()
        started = perf_counter()
        response = await original_completion_async(*args, **kwargs)
        timing.record_llm_metric("llm_stream_ready", perf_counter() - started)
        if kwargs.get("stream") and hasattr(response, "__aiter__"):
            return TimedAsyncStream(response, timing, started)
        return response

    search_engine.model.completion_async = timed_completion_async

    return timing


def finalize_stage_times(
    timing: QueryTiming,
    total_elapsed: float,
    attempts: int,
) -> dict[str, float | int | None]:
    """补齐父子阶段之间的差值，形成一条可直接落盘的计时记录。"""
    stage_times = timing.snapshot()
    # 保证成功、失败和提前异常的 JSONL 记录具有相同字段。
    for stage in TIMING_STAGE_NAMES:
        if stage.startswith("llm_"):
            stage_times.setdefault(stage, None)
        else:
            stage_times.setdefault(stage, 0.0)

    embedding = float(stage_times.get("query_embedding", 0.0) or 0.0)
    vector_search = float(stage_times.get("entity_vector_search", 0.0) or 0.0)
    entity_mapping = float(stage_times.get("entity_mapping_total", 0.0) or 0.0)
    community = float(stage_times.get("community_context", 0.0) or 0.0)
    local = float(stage_times.get("local_context", 0.0) or 0.0)
    text_unit = float(stage_times.get("text_unit_context", 0.0) or 0.0)
    context_total = float(stage_times.get("context_total", 0.0) or 0.0)
    llm_generation = float(stage_times.get("llm_generation", 0.0) or 0.0)
    retry_wait = float(stage_times.get("retry_wait", 0.0) or 0.0)

    stream_ready = stage_times.get("llm_stream_ready_last_attempt")
    first_chunk = stage_times.get("llm_ttfb_last_attempt")
    first_reasoning = stage_times.get("llm_first_reasoning_last_attempt")
    first_content = stage_times.get("llm_first_content_last_attempt")
    generation_last = stage_times.get("llm_generation_last_attempt")

    stage_times["llm_stream_to_first_chunk_last_attempt"] = (
        max(float(first_chunk) - float(stream_ready), 0.0)
        if isinstance(stream_ready, (int, float))
        and isinstance(first_chunk, (int, float))
        else None
    )
    stage_times["llm_reasoning_to_content_last_attempt"] = (
        max(float(first_content) - float(first_reasoning), 0.0)
        if isinstance(first_reasoning, (int, float))
        and isinstance(first_content, (int, float))
        else None
    )
    stage_times["llm_visible_generation_last_attempt"] = (
        max(float(generation_last) - float(first_content), 0.0)
        if isinstance(first_content, (int, float))
        and isinstance(generation_last, (int, float))
        else None
    )

    stage_times["entity_result_mapping"] = max(
        entity_mapping - embedding - vector_search,
        0.0,
    )
    stage_times["context_assembly"] = max(
        context_total - entity_mapping - community - local - text_unit,
        0.0,
    )
    stage_times["prompt_and_overhead"] = max(
        total_elapsed - context_total - llm_generation - retry_wait,
        0.0,
    )
    stage_times["query_total"] = total_elapsed
    stage_times["attempts"] = attempts
    return stage_times


# ------------------------------------------------------------
# 数据加载（只执行一次）
# ------------------------------------------------------------
def load_index_data(config):
    """加载所有索引 parquet 文件，返回字典"""
    storage_obj = create_storage(config.output_storage)

    data = {}
    # 必须存在的表
    for table_name in ["text_units", "relationships", "entities"]:
        data[table_name] = asyncio.run(load_table_from_storage(table_name=table_name, storage=storage_obj))

    # communities 和 community_reports 是可选的（无社区检测时不存在）
    for table_name in ["communities", "community_reports"]:
        if storage_has_table(table_name, storage_obj):
            data[table_name] = asyncio.run(load_table_from_storage(table_name=table_name, storage=storage_obj))
        else:
            data[table_name] = None

    # covariates 是可选的
    if storage_has_table("covariates", storage_obj):
        data["covariates"] = asyncio.run(load_table_from_storage(table_name="covariates", storage=storage_obj))
    else:
        data["covariates"] = None

    return data


# ------------------------------------------------------------
# 构建 local search 引擎（只执行一次）
# ------------------------------------------------------------
def build_search_engine(config, data):
    """
    构建 LocalSearch 引擎（复刻 graphrag.api.local_search_streaming 的装配逻辑）。
    直接调用 engine.search() 才能拿到真正拼进 prompt 的 context_text。
    """
    description_embedding_store = get_embedding_store(
        config=config.vector_store,
        embedding_name=entity_description_embedding,
    )

    # 关键：无社区数据时必须传 community_level=None。
    # read_indexer_entities 会按 level 过滤实体（df[df.level <= community_level]）。
    # 若无社区，left-merge 后每个实体的 level 为 NaN，而 `NaN <= 2` 为 False，
    # 会导致所有实体被过滤掉，最终检索到的实体为空、上下文为空。
    has_communities = data["communities"] is not None
    entities_ = read_indexer_entities(
        data["entities"],
        data["communities"] if has_communities else pd.DataFrame({"community": pd.Series(dtype="str"), "entity_ids": pd.Series(dtype="object"), "level": pd.Series(dtype="int")}),
        COMMUNITY_LEVEL if has_communities else None,
    )
    covariates_ = (
        read_indexer_covariates(data["covariates"])
        if data["covariates"] is not None
        else []
    )
    prompt = load_search_prompt(config.local_search.prompt)

    # 无社区数据时传空报告列表
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


# ------------------------------------------------------------
# 带重试的单条查询
# ------------------------------------------------------------
async def query_single(
    search_engine,
    question: str,
    timing: QueryTiming,
) -> tuple[str, str, dict | None, float, dict[str, float | int | None]]:
    """
    对单条问题执行 local search，带自动重试。
    返回 (答案, LLM 上下文, 上下文数据, 整题耗时, 分阶段耗时)。
    整题耗时包含失败尝试和重试等待，阶段耗时在多次尝试间累加。
    """
    timing.reset()
    started = perf_counter()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = await search_engine.search(query=question)
            answer = str(result.response)

            # 当前 GraphRAG 的 LocalSearch.search 会捕获部分 LLM 异常并返回空答案，
            # 主动转成异常才能进入本脚本的重试流程。
            if not answer.strip():
                raise RuntimeError("Local Search 返回空答案")

            context_text = result.context_text
            if isinstance(context_text, list):
                context_text = "\n\n".join(str(c) for c in context_text)
            elif isinstance(context_text, dict):
                context_text = "\n\n".join(
                    f"----- {k} -----\n{v}" for k, v in context_text.items()
                )

            elapsed = perf_counter() - started
            stage_times = finalize_stage_times(timing, elapsed, attempt)
            return answer, str(context_text), result.context_data, elapsed, stage_times

        except KeyboardInterrupt:
            raise

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_name = type(e).__name__
            if attempt >= MAX_RETRIES:
                elapsed = perf_counter() - started
                stage_times = finalize_stage_times(timing, elapsed, attempt)
                return (
                    f"（查询失败，重试 {MAX_RETRIES} 次: {error_name}）",
                    "",
                    None,
                    elapsed,
                    stage_times,
                )

            wait = min(RETRY_DELAY * (2 ** (attempt - 1)), 60)
            jitter = random.uniform(0, wait * 0.3)
            total_wait = wait + jitter
            print(
                f"    -> 第 {attempt} 次失败 ({error_name})，"
                f"{total_wait:.0f}s 后重试...",
                flush=True,
            )
            sleep_started = perf_counter()
            await asyncio.sleep(total_wait)
            timing.add("retry_wait", perf_counter() - sleep_started)

    elapsed = perf_counter() - started
    stage_times = finalize_stage_times(timing, elapsed, MAX_RETRIES)
    return "（查询失败）", "", None, elapsed, stage_times


# ------------------------------------------------------------
# 上下文 DataFrame -> 可读文本
# ------------------------------------------------------------
def df_to_text(df: pd.DataFrame, max_rows: int = 50) -> str:
    """DataFrame -> markdown 表格（不依赖 tabulate），超长单元格截断"""
    if df is None or df.empty:
        return "（无数据）"

    truncated = len(df) > max_rows
    show = df.head(max_rows)

    def cell(val) -> str:
        s = str(val) if val is not None else ""
        if len(s) > 150:
            s = s[:147] + "..."
        return s.replace("|", "/").replace("\n", " ")

    cols = list(show.columns)
    rows = [[cell(v) for v in row] for row in show.values]

    widths = [len(c) for c in cols]
    for row in rows:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))

    def fmt(cells):
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    sep = ["-" * widths[i] for i in range(len(cols))]
    lines = [fmt(cols), fmt(sep)]
    for row in rows:
        lines.append(fmt(row))

    text = "\n".join(lines)
    if truncated:
        text += f"\n（共 {len(df)} 行，仅显示前 {max_rows} 行）"
    return text


# ------------------------------------------------------------
# 输出格式化
# ------------------------------------------------------------
def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}分{s:.1f}秒"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}时{int(m)}分{s:.1f}秒"


def write_clean_result(f, item: dict, answer: str, elapsed: float, idx: int, total: int):
    """写入干净结果文件：只有问题 + 标准答案 + 生成答案"""
    q = item["question"]
    gold = item["gold_answer"]

    f.write(f"【问题】：{q}\n")
    f.write(f"【标准答案】：{gold}\n")
    f.write(f"【被评测答案】：{answer}\n")
    f.write(f"（耗时: {elapsed:.1f}s）\n")
    f.write("\n" + "-" * 60 + "\n\n")


def write_context_result(f, item: dict, answer: str, context_text: str):
    """写入上下文文件：问题 + 检索到的参考上下文（真正发给 LLM 的 context_text）+ 被评测答案"""
    q = item["question"]
    retrieved_context = context_text if context_text else "（无检索上下文）"

    f.write(f"【问题】：{q}\n\n")
    f.write(f"【检索到的参考上下文】：{retrieved_context}\n\n")
    f.write(f"【被评测答案】：{answer}\n")
    f.write("\n" + "-" * 60 + "\n\n")


def write_timing_record(
    f,
    item: dict,
    idx: int,
    answer: str,
    stage_times: dict[str, float | int | None],
):
    """每道题写一行 JSON，便于 Pandas 或其他脚本直接汇总。"""
    record = {
        "record_type": "query",
        "id": item.get("id", f"Q{idx}"),
        "question": item["question"],
        "success": not answer.startswith("（查询失败"),
        "answer_chars": len(answer),
        "timings_seconds": {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in stage_times.items()
        },
    }
    f.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_stage_timing_line(stage_times: dict[str, float | int | None]) -> str:
    """生成适合终端显示的一行检索与原始流里程碑。"""
    def seconds(name: str) -> float:
        return float(stage_times.get(name, 0.0) or 0.0)

    def optional_seconds(name: str) -> str:
        value = stage_times.get(name)
        return f"{float(value):.3f}s" if isinstance(value, (int, float)) else "N/A"

    return (
        f"Embedding={seconds('query_embedding'):.3f}s, "
        f"向量检索={seconds('entity_vector_search'):.3f}s, "
        f"上下文={seconds('context_total'):.3f}s; "
        f"LLM[流就绪={optional_seconds('llm_stream_ready_last_attempt')}, "
        f"TTFB={optional_seconds('llm_ttfb_last_attempt')}, "
        f"首Reasoning={optional_seconds('llm_first_reasoning_last_attempt')}, "
        f"首Content={optional_seconds('llm_first_content_last_attempt')}, "
        f"Reasoning→Content="
        f"{optional_seconds('llm_reasoning_to_content_last_attempt')}, "
        f"可见内容生成={optional_seconds('llm_visible_generation_last_attempt')}, "
        f"完整流={optional_seconds('llm_generation_last_attempt')}]; "
        f"总计={seconds('query_total'):.3f}s"
    )


def calculate_timing_averages(
    records: list[dict[str, float | int | None]],
) -> dict[str, float | None]:
    """按阶段计算平均值；None（例如没有首 Token）不参与平均。"""
    averages: dict[str, float | None] = {}
    for stage in TIMING_STAGE_NAMES:
        values = [
            float(record[stage])
            for record in records
            if isinstance(record.get(stage), (int, float))
        ]
        averages[stage] = sum(values) / len(values) if values else None

    attempt_values = [
        float(record["attempts"])
        for record in records
        if isinstance(record.get("attempts"), (int, float))
    ]
    averages["attempts"] = (
        sum(attempt_values) / len(attempt_values) if attempt_values else None
    )
    return averages


def write_timing_table(
    f,
    stage_times: dict[str, float | int | None],
    heading: str = "### 分阶段耗时",
):
    """向详细报告写入 Markdown 计时表。"""
    f.write(f"{heading}\n\n")
    f.write("| 阶段 | 耗时（秒） |\n")
    f.write("|------|-----------:|\n")
    for stage in TIMING_STAGE_NAMES:
        value = stage_times.get(stage)
        display = f"{float(value):.6f}" if isinstance(value, (int, float)) else "N/A"
        f.write(f"| {TIMING_STAGE_LABELS[stage]} | {display} |\n")
    attempts = stage_times.get("attempts")
    attempts_display = (
        f"{float(attempts):.2f}" if isinstance(attempts, (int, float)) else "N/A"
    )
    f.write(f"| 尝试次数 | {attempts_display} |\n\n")


def write_detail_record(
    f,
    item: dict,
    answer: str,
    context_data: dict | None,
    elapsed: float,
    idx: int,
    stage_times: dict[str, float | int | None],
):
    """写入详细记录文件：包含问题 + 答案 + 所有上下文数据"""
    qid = item.get("id", f"Q{idx}")
    q = item["question"]
    gold = item["gold_answer"]
    qtype = item.get("type", "")
    difficulty = item.get("difficulty", "")
    hops = item.get("hops", "")
    evidence = item.get("evidence", "")

    f.write(f"## 问题 #{idx} [{qid}]\n\n")
    f.write(f"**类型**: {qtype} | **难度**: {difficulty} | **跳数**: {hops}\n\n")
    f.write(f"**问题**: {q}\n\n")
    f.write(f"**标准答案**: {gold}\n\n")
    f.write(f"**证据线索**: {evidence}\n\n")
    f.write(f"**生成的答案**: {answer}\n\n")
    f.write(f"**耗时**: {elapsed:.3f}s\n\n")
    write_timing_table(f, stage_times)

    if context_data is None:
        f.write("**上下文**: （查询失败，无上下文数据）\n\n")
        f.write("=" * 60 + "\n\n")
        return

    # 上下文概览
    f.write("### 上下文数据概览\n\n")
    f.write("| 数据类型 | 记录数 |\n")
    f.write("|---------|-------|\n")
    for name, df in context_data.items():
        if isinstance(df, pd.DataFrame):
            f.write(f"| {name} | {len(df)} |\n")
    f.write("\n")

    # 各部分详细数据
    section_names = {
        "entities":          "匹配的实体 (Entities)",
        "relationships":     "匹配的关系 (Relationships)",
        "sources":           "引用的原文块 (Sources)",
        "community_reports": "参考的社区报告 (Reports)",
        "reports":           "参考的社区报告 (Reports)",
        "claims":            "声明/协变量 (Claims)",
    }

    for name, df in context_data.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        title = section_names.get(name, name)
        f.write(f"### {title}（{len(df)} 条）\n\n")
        f.write(df_to_text(df) + "\n\n")

    f.write("=" * 60 + "\n\n")


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def main():
    args = parse_args()
    try:
        input_path, output_dir, root_dir = resolve_paths(args)
    except (OSError, ValueError) as error:
        raise SystemExit(f"路径配置错误: {error}") from error

    clean_path   = output_dir / "batch_query_results.txt"
    detail_path  = output_dir / "batch_query_detail.txt"
    context_path = output_dir / "batch_query_context.txt"
    timing_path  = output_dir / "batch_query_timing.jsonl"

    # 读取评测数据（兼容 JSON 数组和 JSONL 两种格式）
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        items = json.loads(content)
    else:
        items = [json.loads(line) for line in content.splitlines() if line.strip()]

    print(f"输入文件: {input_path}")
    print(f"输出文件夹: {output_dir}")
    print(f"GraphRAG 根目录: {root_dir}")
    print(f"共加载 {len(items)} 条问题")
    print(f"查询方法: {METHOD}")
    print(f"干净结果 -> {clean_path}")
    print(f"详细记录 -> {detail_path}")
    print(f"上下文结果 -> {context_path}")
    print(f"阶段耗时 -> {timing_path}")
    print("=" * 60)

    # 加载索引数据（只加载一次！）
    print("正在加载索引数据...")
    config = load_config(root_dir)
    data = load_index_data(config)
    print(
        f"数据加载完成: "
        f"{len(data['entities'])} 个实体, "
        f"{len(data['relationships'])} 条关系, "
        f"{len(data['text_units'])} 个文本块, "
        f"{len(data['community_reports']) if data['community_reports'] is not None else 0} 份社区报告"
    )

    # 构建查询引擎并安装分阶段计时（均只执行一次）。
    print("正在构建查询引擎...")
    search_engine = build_search_engine(config, data)
    timing = install_search_timing(search_engine)
    print("查询引擎构建完成，分阶段计时已启用")
    print("=" * 60)

    query_times = []
    successful_timing_records: list[dict[str, float | int | None]] = []
    fail_count = 0
    total_start = perf_counter()

    # 四个输出文件均逐条 flush，中断时已完成题目的结果不会丢失。
    with open(clean_path, "w", encoding="utf-8", buffering=1) as clean_f, \
         open(detail_path, "w", encoding="utf-8", buffering=1) as detail_f, \
         open(context_path, "w", encoding="utf-8", buffering=1) as context_f, \
         open(timing_path, "w", encoding="utf-8", buffering=1) as timing_f:

        # 文件头
        clean_f.write("=" * 60 + "\n")
        clean_f.write("GraphRAG 批量查询评测报告\n")
        clean_f.write(f"查询时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        clean_f.write(f"查询方法: {METHOD}\n")
        clean_f.write(f"总题数: {len(items)}\n")
        clean_f.write("=" * 60 + "\n\n")

        context_f.write("=" * 60 + "\n")
        context_f.write("GraphRAG 批量查询 — 问题/检索上下文/答案\n")
        context_f.write(f"查询时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        context_f.write(f"查询方法: {METHOD}\n")
        context_f.write(f"总题数: {len(items)}\n")
        context_f.write("=" * 60 + "\n\n")

        detail_f.write("=" * 60 + "\n")
        detail_f.write("GraphRAG 批量查询 — 详细记录\n")
        detail_f.write(f"查询时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        detail_f.write(f"查询方法: {METHOD}\n")
        detail_f.write(f"总题数: {len(items)}\n")
        detail_f.write("=" * 60 + "\n\n")

        for i, item in enumerate(items, 1):
            qid = item.get("id", f"Q{i}")
            question = item["question"]
            print(f"[{i}/{len(items)}] {qid}: {question[:50]}...", flush=True)

            # 执行查询（带重试和分阶段计时）。
            answer, context_text, context_data, elapsed, stage_times = asyncio.run(
                query_single(search_engine, question, timing)
            )
            query_times.append(elapsed)

            is_fail = answer.startswith("（查询失败")
            if is_fail:
                fail_count += 1
                print(f"  X 失败: {answer}", flush=True)
            else:
                successful_timing_records.append(stage_times)
                print(f"  OK {len(answer)} 字符, {elapsed:.3f}s", flush=True)
                print(f"  答案:\n{answer}", flush=True)
            print(f"  阶段: {format_stage_timing_line(stage_times)}", flush=True)

            # 写入四个文件。
            write_clean_result(clean_f, item, answer, elapsed, i, len(items))
            write_context_result(context_f, item, answer, context_text)
            write_detail_record(
                detail_f,
                item,
                answer,
                context_data,
                elapsed,
                i,
                stage_times,
            )
            write_timing_record(timing_f, item, i, answer, stage_times)

        # 统计摘要
        total_duration = perf_counter() - total_start
        avg_time = sum(query_times) / len(query_times) if query_times else 0
        min_time = min(query_times) if query_times else 0
        max_time = max(query_times) if query_times else 0
        timing_averages = calculate_timing_averages(successful_timing_records)

        summary = (
            f"\n{'=' * 60}\n"
            f"评测完成\n"
            f"{'=' * 60}\n"
            f"总题数: {len(items)}\n"
            f"成功: {len(items) - fail_count}, 失败: {fail_count}\n"
            f"总耗时: {format_duration(total_duration)}\n"
            f"平均每题: {format_duration(avg_time)}\n"
            f"最快: {format_duration(min_time)}, 最慢: {format_duration(max_time)}\n"
        )

        clean_f.write(summary)
        detail_f.write(summary)
        context_f.write(summary)

        if successful_timing_records:
            detail_f.write("\n")
            write_timing_table(
                detail_f,
                timing_averages,
                heading="### 成功查询的平均分阶段耗时",
            )

        timing_summary = {
            "record_type": "summary",
            "total_questions": len(items),
            "successful_questions": len(items) - fail_count,
            "failed_questions": fail_count,
            "wall_clock_seconds": round(total_duration, 6),
            "average_successful_timings_seconds": {
                key: round(value, 6) if isinstance(value, float) else value
                for key, value in timing_averages.items()
            },
        }
        timing_f.write(json.dumps(timing_summary, ensure_ascii=False) + "\n")

        print(summary)
        if successful_timing_records:
            print("成功查询的平均阶段耗时:")
            print(f"  {format_stage_timing_line(timing_averages)}")
        print(f"\n干净结果 -> {clean_path}")
        print(f"详细记录 -> {detail_path}")
        print(f"上下文结果 -> {context_path}")
        print(f"阶段耗时 -> {timing_path}")


if __name__ == "__main__":
    main()
