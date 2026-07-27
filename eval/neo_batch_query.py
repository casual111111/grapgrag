#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Neo4j 多跳检索质量评测脚本。

基于 eval/batch_query.py 改造，用 Neo4j 的多跳图检索替代 GraphRAG 默认 local search，
对比两者的检索质量和回答质量。

输出文件：
  - neo_batch_results.txt       干净结果（问题 + 标准答案 + 生成答案）
  - neo_batch_detail.txt        详细记录（含检索到的实体/关系/文档）
  - neo_batch_context.txt       检索上下文（发给 LLM 的 context）
  - neo_batch_timing.jsonl      每题计时（JSONL 格式）
  - neo_batch_retrieval.jsonl   检索质量指标（JSONL 格式，含命中率）

用法：
    cd C:\保存\graphrag-github-flash\graphrag
    python eval/neo_batch_query.py

指定跳数：
    python eval/neo_batch_query.py --hops 3

只评测检索质量（不调 LLM 生成答案）：
    python eval/neo_batch_query.py --retrieval-only

对比原始 local search 结果：
    python eval/neo_batch_query.py --compare-with eval/eval_flash/batch_query_results.txt
"""

import argparse
import asyncio
import io
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

# 修复 Windows 控制台编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加本地 packages 到 sys.path
_project_root = Path(__file__).resolve().parent.parent
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
from neo4j import GraphDatabase
from neo4j_context_builder import Neo4jLocalContextBuilder

# ============================================================
# 可调参数
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT_DIR = _project_root / "my_kg"
DEFAULT_INPUT_DIR = SCRIPT_DIR
DEFAULT_INPUT_FILE = "QA.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "eval_neo4j"

# Neo4j 配置
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4j_test"

MAX_RETRIES = 3
RETRY_DELAY = 5


# ============================================================
# 命令行参数
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 Neo4j 多跳检索评测 GraphRAG 知识图谱问答质量",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help=f"输入文件目录（默认: {DEFAULT_INPUT_DIR}）")
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE,
                        help=f"输入文件名（默认: {DEFAULT_INPUT_FILE}）")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"输出目录（默认: {DEFAULT_OUTPUT_DIR}）")
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR,
                        help=f"GraphRAG 项目根目录（默认: {DEFAULT_ROOT_DIR}）")
    parser.add_argument("--hops", type=int, default=1,
                        help="Neo4j 多跳扩展跳数（默认: 2）")
    parser.add_argument("--top-k-entities", type=int, default=10,
                        help="向量检索入口实体数，同 local 的 top_k_entities（默认: 10）")
    parser.add_argument("--top-k-relationships", type=int, default=18,
                        help="关系预算系数，同 local 的 top_k_relationships（默认: 18）")
    parser.add_argument("--max-context-tokens", type=int, default=24000,
                        help="上下文 token 上限，同 local 的 max_context_tokens（默认: 24000）")
    parser.add_argument("--top-k", type=int, default=None,
                        help="(兼容旧参数) 同 --top-k-entities")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="只评测检索质量，不调 LLM 生成答案")
    parser.add_argument("--compare-with", type=Path, default=None,
                        help="与另一份 batch_query_results.txt 对比")
    parser.add_argument("--use-embedding", action="store_true", default=True,
                        help="使用 embedding 向量检索（默认开启）")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    input_source = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    root_dir = args.root_dir.expanduser().resolve()

    if input_source.is_file():
        input_path = input_source
    elif input_source.is_dir():
        input_path = input_source / args.input_file
        if not input_path.is_file():
            raise ValueError(f"输入文件不存在: {input_path}")
    else:
        raise ValueError(f"输入路径不存在: {input_source}")

    output_dir.mkdir(parents=True, exist_ok=True)
    return input_path, output_dir, root_dir


# ============================================================
# 检索质量评估
# ============================================================
def evaluate_retrieval_quality(
    item: dict,
    context_records: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """评估检索质量：命中率、覆盖度等。"""
    q = item["question"]
    gold_keypoints = item.get("gold_keypoints", [])
    reasoning_path = item.get("reasoning_path", "")
    evidence = item.get("evidence", [])
    hops_needed = item.get("hops", 2)

    entities_df = context_records.get("entities", pd.DataFrame())
    relationships_df = context_records.get("relationships", pd.DataFrame())
    sources_df = context_records.get("sources", pd.DataFrame())

    metrics = {
        "question": q,
        "hops_needed": hops_needed,
        "num_entities": len(entities_df),
        "num_relationships": len(relationships_df),
        "num_documents": len(sources_df),
    }

    # 1. 关键词命中率：gold_keypoints 中的关键词在实体 title/description 中是否出现
    if gold_keypoints:
        entity_texts = ""
        for _, row in entities_df.iterrows():
            entity_texts += str(row.get("title", "")) + " "
            entity_texts += str(row.get("description", "")) + " "
        entity_texts = entity_texts.lower()

        matched_keypoints = 0
        matched_kp_list = []
        for kp in gold_keypoints:
            if kp.lower() in entity_texts:
                matched_keypoints += 1
                matched_kp_list.append(kp)

        metrics["keypoint_hit_rate"] = matched_keypoints / len(gold_keypoints) if gold_keypoints else 0
        metrics["matched_keypoints"] = matched_kp_list
        metrics["missed_keypoints"] = [kp for kp in gold_keypoints if kp not in matched_kp_list]

    # 2. 推理路径覆盖度：reasoning_path 中的实体/概念是否在检索结果中
    if reasoning_path:
        # 提取路径中的关键词（用 - 和 -> 分隔的部分）
        path_parts = [p.strip() for p in reasoning_path.replace("->", "-").replace("→", "-").split("-") if p.strip()]
        if path_parts:
            entity_titles = set()
            for _, row in entities_df.iterrows():
                entity_titles.add(str(row.get("title", "")).lower())

            # 关系描述也参与匹配
            rel_descriptions = ""
            for _, row in relationships_df.iterrows():
                rel_descriptions += str(row.get("description", "")) + " "
            rel_descriptions = rel_descriptions.lower()

            all_text = entity_titles.union({rel_descriptions})

            matched_parts = 0
            for part in path_parts:
                part_lower = part.lower()
                if any(part_lower in t for t in entity_titles) or part_lower in rel_descriptions:
                    matched_parts += 1

            metrics["path_coverage"] = matched_parts / len(path_parts) if path_parts else 0

    # 3. 证据文档命中率：text unit 所属的 document 是否命中 evidence
    if evidence:
        # 从 text units 的 document_id 反查 document title
        doc_ids = set()
        for _, row in sources_df.iterrows():
            doc_id = str(row.get("document_id", ""))
            if doc_id:
                doc_ids.add(doc_id)

        # 用 document_id 匹配 evidence 中的文件名（evidence 格式: "文档名.md#章节"）
        matched_evidence = 0
        for ev in evidence:
            ev_doc = ev.split("#")[0].replace("(4)", "").strip()
            # 检查是否有 text unit 来自这个文档
            # document_id 是 hash，需要和原始文件名做映射
            # 简单做法：检查 evidence 文档名是否出现在任何 text unit 的 document_id 中
            # 但 document_id 是 hash，不是文件名...
            # 需要额外查 document parquet 做 id→title 映射
            # 这里先简单标记，后续可以改进
            matched_evidence += 0  # 暂不计算，等后续用 document parquet 映射

        metrics["evidence_hit_rate"] = matched_evidence / len(evidence) if evidence else 0

    return metrics


# ============================================================
# 答案质量评估（简单的关键词匹配）
# ============================================================
def evaluate_answer_quality(
    gold_answer: str,
    generated_answer: str,
    gold_keypoints: list[str],
) -> dict[str, Any]:
    """简单评估答案质量：关键词命中率、长度比等。"""
    metrics = {
        "gold_answer_len": len(gold_answer),
        "generated_answer_len": len(generated_answer),
    }

    if not generated_answer.strip():
        metrics["keypoint_recall"] = 0.0
        metrics["length_ratio"] = 0.0
        return metrics

    # 关键词召回率
    if gold_keypoints:
        gen_lower = generated_answer.lower()
        matched = sum(1 for kp in gold_keypoints if kp.lower() in gen_lower)
        metrics["keypoint_recall"] = matched / len(gold_keypoints)
    else:
        metrics["keypoint_recall"] = None

    # 长度比（生成答案 / 标准答案）
    if len(gold_answer) > 0:
        metrics["length_ratio"] = len(generated_answer) / len(gold_answer)
    else:
        metrics["length_ratio"] = None

    return metrics


# ============================================================
# 检索 + 回答
# ============================================================
def neo4j_retrieve(
    builder: Neo4jLocalContextBuilder,
    question: str,
    num_hops: int,
    top_k_entities: int,
    top_k_relationships: int,
    max_context_tokens: int,
) -> tuple[str, dict[str, pd.DataFrame], float]:
    """调用 Neo4j 检索，返回 (context_text, context_records, elapsed)。"""
    started = perf_counter()
    result = builder.build_context(
        query=question,
        num_hops=num_hops,
        top_k_entities=top_k_entities,
        top_k_relationships=top_k_relationships,
        max_context_tokens=max_context_tokens,
    )
    elapsed = perf_counter() - started

    context_chunks = result.context_chunks
    if isinstance(context_chunks, list):
        context_chunks = "\n\n".join(context_chunks)

    return str(context_chunks), result.context_records, elapsed


async def neo4j_answer(
    builder: Neo4jLocalContextBuilder,
    context_text: str,
    question: str,
    root_dir: Path,
) -> tuple[str, float]:
    """用 LLM 根据检索上下文生成答案，返回 (answer, elapsed)。"""
    from graphrag.config.load_config import load_config
    from graphrag_llm.completion import create_completion
    from graphrag.utils.api import load_search_prompt

    config = load_config(root_dir)
    model_settings = config.get_completion_model_config(
        config.local_search.completion_model_id
    )
    chat_model = create_completion(model_settings)
    system_prompt = load_search_prompt(config.local_search.prompt)

    user_message = question
    system_message = system_prompt.format(
        context_data=context_text,
        response_type="multiple paragraphs",
    )

    started = perf_counter()
    try:
        response = await chat_model.completion_async(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
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

        elapsed = perf_counter() - started
        return full_response, elapsed

    except Exception as e:
        elapsed = perf_counter() - started
        return f"（LLM 调用失败: {type(e).__name__}: {e}）", elapsed


# ============================================================
# 输出格式化
# ============================================================
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


def df_to_text(df: pd.DataFrame, max_rows: int = 30) -> str:
    """DataFrame -> markdown 表格。"""
    if df is None or df.empty:
        return "（无数据）"
    truncated = len(df) > max_rows
    show = df.head(max_rows)

    def cell(val) -> str:
        s = str(val) if val is not None else ""
        if len(s) > 120:
            s = s[:117] + "..."
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


def write_clean_result(f, item: dict, answer: str, elapsed: float, idx: int):
    q = item["question"]
    gold = item["gold_answer"]
    f.write(f"【问题】：{q}\n")
    f.write(f"【标准答案】：{gold}\n")
    f.write(f"【被评测答案】：{answer}\n")
    f.write(f"（检索耗时 + 生成耗时: {elapsed:.1f}s）\n")
    f.write("\n" + "-" * 60 + "\n\n")


def write_context_result(f, item: dict, answer: str, context_text: str, context_records: dict = None):
    q = item["question"]
    f.write(f"【问题】：{q}\n\n")
    # 写入上下文统计
    if context_records:
        n_entities = len(context_records.get("entities")) if context_records.get("entities") is not None else 0
        n_relationships = len(context_records.get("relationships")) if context_records.get("relationships") is not None else 0
        n_text_units = len(context_records.get("sources")) if context_records.get("sources") is not None else 0
        f.write(f"【上下文统计】：实体 {n_entities} 个 | 关系 {n_relationships} 条 | 文本片段 {n_text_units} 个\n\n")
    f.write(f"【检索到的参考上下文】：\n{context_text if context_text else '（无）'}\n\n")
    f.write(f"【被评测答案】：{answer}\n")
    f.write("\n" + "-" * 60 + "\n\n")


def write_detail_record(f, item: dict, answer: str, context_records: dict, elapsed: float, retrieval_metrics: dict):
    qid = item.get("id", "")
    q = item["question"]
    gold = item["gold_answer"]
    hops = item.get("hops", "")

    f.write(f"## 问题 #{qid}\n\n")
    f.write(f"**跳数需求**: {hops}\n\n")
    f.write(f"**问题**: {q}\n\n")
    f.write(f"**标准答案**: {gold}\n\n")
    f.write(f"**关键词**: {item.get('gold_keypoints', [])}\n\n")
    f.write(f"**推理路径**: {item.get('reasoning_path', '')}\n\n")
    f.write(f"**Neo4j 答案**: {answer}\n\n")
    f.write(f"**耗时**: {elapsed:.3f}s\n\n")

    # 检索质量指标
    f.write("### 检索质量指标\n\n")
    f.write(f"- 实体数: {retrieval_metrics.get('num_entities', 0)}\n")
    f.write(f"- 关系数: {retrieval_metrics.get('num_relationships', 0)}\n")
    f.write(f"- 文档数: {retrieval_metrics.get('num_documents', 0)}\n")
    kp_rate = retrieval_metrics.get("keypoint_hit_rate")
    if kp_rate is not None:
        f.write(f"- 关键词命中率: {kp_rate:.1%}\n")
        f.write(f"  - 命中: {retrieval_metrics.get('matched_keypoints', [])}\n")
        f.write(f"  - 未命中: {retrieval_metrics.get('missed_keypoints', [])}\n")
    path_cov = retrieval_metrics.get("path_coverage")
    if path_cov is not None:
        f.write(f"- 推理路径覆盖度: {path_cov:.1%}\n")
    ev_rate = retrieval_metrics.get("evidence_hit_rate")
    if ev_rate is not None:
        f.write(f"- 证据文档命中率: {ev_rate:.1%}\n")
    f.write("\n")

    # 上下文数据
    if context_records:
        f.write("### 检索到的数据\n\n")
        for name, df in context_records.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                f.write(f"#### {name}（{len(df)} 条）\n\n")
                f.write(df_to_text(df) + "\n\n")

    f.write("=" * 60 + "\n\n")


# ============================================================
# 主流程
# ============================================================
def main():
    args = parse_args()
    try:
        input_path, output_dir, root_dir = resolve_paths(args)
    except (OSError, ValueError) as error:
        raise SystemExit(f"路径配置错误: {error}") from error

    # 输出文件
    clean_path = output_dir / "neo_batch_results.txt"
    detail_path = output_dir / "neo_batch_detail.txt"
    context_path = output_dir / "neo_batch_context.txt"
    timing_path = output_dir / "neo_batch_timing.jsonl"
    retrieval_path = output_dir / "neo_batch_retrieval.jsonl"

    # 读取评测数据
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        items = json.loads(content)
    else:
        items = [json.loads(line) for line in content.splitlines() if line.strip()]

    # 兼容旧参数 --top-k
    top_k_entities = args.top_k_entities if args.top_k is None else args.top_k
    top_k_relationships = args.top_k_relationships
    max_context_tokens = args.max_context_tokens

    print(f"输入文件: {input_path}")
    print(f"输出目录: {output_dir}")
    print(f"Neo4j: {NEO4J_URI}")
    print(f"跳数 (num_hops): {args.hops}")
    print(f"入口实体 (top_k_entities): {top_k_entities}")
    print(f"关系预算 (top_k_relationships): {top_k_relationships}")
    print(f"上下文上限 (max_context_tokens): {max_context_tokens}")
    print(f"模式: {'仅检索(不调LLM)' if args.retrieval_only else '检索 + LLM 生成'}")
    print(f"入口实体检索方式: 向量相似度(Neo4j 向量索引)")
    print(f"共 {len(items)} 条问题")
    print(f"干净结果 -> {clean_path}")
    print(f"详细记录 -> {detail_path}")
    print(f"上下文 -> {context_path}")
    print(f"计时 -> {timing_path}")
    print(f"检索质量 -> {retrieval_path}")
    print("=" * 60)

    # 初始化 Neo4j context builder（必须使用 embedding 向量检索）
    print("正在连接 Neo4j...")

    # 加载 GraphRAG 的 embedding 模型
    print("正在加载 embedding 模型...")
    from graphrag.config.load_config import load_config
    from graphrag_llm.embedding import create_embedding

    config = load_config(root_dir)
    embedding_settings = config.get_embedding_model_config(
        config.local_search.embedding_model_id
    )
    text_embedder = create_embedding(embedding_settings)
    print(f"Embedding 模型加载成功: {embedding_settings.model}")

    builder = Neo4jLocalContextBuilder(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        text_embedder=text_embedder,
    )
    print("Neo4j 连接成功")
    print("=" * 60)

    # 逐条评测
    retrieval_times = []
    generation_times = []
    total_times = []
    retrieval_metrics_list = []
    fail_count = 0
    total_start = perf_counter()

    with open(clean_path, "w", encoding="utf-8", buffering=1) as clean_f, \
         open(detail_path, "w", encoding="utf-8", buffering=1) as detail_f, \
         open(context_path, "w", encoding="utf-8", buffering=1) as context_f, \
         open(timing_path, "w", encoding="utf-8", buffering=1) as timing_f, \
         open(retrieval_path, "w", encoding="utf-8", buffering=1) as retrieval_f:

        # 文件头
        for f in [clean_f, detail_f, context_f]:
            f.write("=" * 60 + "\n")
            f.write("Neo4j 多跳检索评测报告\n")
            f.write(f"评测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"跳数: {args.hops}\n")
            f.write(f"入口实体: {args.top_k}\n")
            f.write(f"模式: {'仅检索' if args.retrieval_only else '检索 + LLM'}\n")
            f.write(f"总题数: {len(items)}\n")
            f.write("=" * 60 + "\n\n")

        for i, item in enumerate(items, 1):
            qid = item.get("id", f"Q{i}")
            question = item["question"]
            print(f"[{i}/{len(items)}] #{qid}: {question[:50]}...", flush=True)

            # Step 1: 检索
            started = perf_counter()
            context_text, context_records, retrieval_elapsed = neo4j_retrieve(
                builder, question, args.hops, top_k_entities, top_k_relationships, max_context_tokens
            )
            retrieval_times.append(retrieval_elapsed)

            # Step 2: 评估检索质量
            retrieval_metrics = evaluate_retrieval_quality(item, context_records)
            retrieval_metrics["id"] = qid
            retrieval_metrics["retrieval_time"] = round(retrieval_elapsed, 4)
            retrieval_metrics_list.append(retrieval_metrics)

            # Step 3: 生成答案（可选）
            answer = "（仅检索模式）"
            generation_elapsed = 0.0
            if not args.retrieval_only:
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        answer, generation_elapsed = asyncio.run(
                            neo4j_answer(builder, context_text, question, root_dir)
                        )
                        if answer.strip() and not answer.startswith("（"):
                            break
                    except Exception as e:
                        if attempt >= MAX_RETRIES:
                            answer = f"（查询失败: {type(e).__name__}）"
                            fail_count += 1
                            break
                        wait = min(RETRY_DELAY * (2 ** (attempt - 1)), 60)
                        print(f"    重试 {attempt}/{MAX_RETRIES}，等待 {wait:.0f}s...", flush=True)
                        time.sleep(wait)

                generation_times.append(generation_elapsed)

            total_elapsed = perf_counter() - started
            total_times.append(total_elapsed)

            # 打印结果
            kp_rate = retrieval_metrics.get("keypoint_hit_rate")
            kp_str = f"{kp_rate:.0%}" if kp_rate is not None else "N/A"
            path_cov = retrieval_metrics.get("path_coverage")
            path_str = f"{path_cov:.0%}" if path_cov is not None else "N/A"
            print(f"  检索: {retrieval_elapsed:.2f}s | "
                  f"实体: {retrieval_metrics['num_entities']} | "
                  f"关系: {retrieval_metrics['num_relationships']} | "
                  f"关键词命中: {kp_str} | 路径覆盖: {path_str}", flush=True)
            if not args.retrieval_only:
                print(f"  生成: {generation_elapsed:.2f}s | 答案: {len(answer)} 字符 | 总时间: {total_elapsed:.2f}s", flush=True)
            else:
                print(f"  总时间: {total_elapsed:.2f}s", flush=True)

            # 写入文件
            write_clean_result(clean_f, item, answer, total_elapsed, i)
            write_context_result(context_f, item, answer, context_text, context_records)
            write_detail_record(detail_f, item, answer, context_records, total_elapsed, retrieval_metrics)

            # 计时记录
            timing_record = {
                "id": qid,
                "question": question,
                "retrieval_time": round(retrieval_elapsed, 4),
                "generation_time": round(generation_elapsed, 4),
                "total_time": round(total_elapsed, 4),
                "success": not answer.startswith("（查询失败"),
            }
            timing_f.write(json.dumps(timing_record, ensure_ascii=False) + "\n")

            # 检索质量记录
            retrieval_f.write(json.dumps(retrieval_metrics, ensure_ascii=False) + "\n")

        # ======================== 统计摘要 ========================
        total_duration = perf_counter() - total_start
        avg_retrieval = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0
        avg_generation = sum(generation_times) / len(generation_times) if generation_times else 0
        avg_total = sum(total_times) / len(total_times) if total_times else 0

        # 检索质量汇总
        avg_kp_rate = None
        kp_rates = [m["keypoint_hit_rate"] for m in retrieval_metrics_list if m.get("keypoint_hit_rate") is not None]
        if kp_rates:
            avg_kp_rate = sum(kp_rates) / len(kp_rates)

        avg_path_cov = None
        path_covs = [m["path_coverage"] for m in retrieval_metrics_list if m.get("path_coverage") is not None]
        if path_covs:
            avg_path_cov = sum(path_covs) / len(path_covs)

        avg_ev_rate = None
        ev_rates = [m["evidence_hit_rate"] for m in retrieval_metrics_list if m.get("evidence_hit_rate") is not None]
        if ev_rates:
            avg_ev_rate = sum(ev_rates) / len(ev_rates)

        avg_entities = sum(m["num_entities"] for m in retrieval_metrics_list) / len(retrieval_metrics_list)
        avg_relationships = sum(m["num_relationships"] for m in retrieval_metrics_list) / len(retrieval_metrics_list)
        avg_documents = sum(m["num_documents"] for m in retrieval_metrics_list) / len(retrieval_metrics_list)

        summary = (
            f"\n{'=' * 60}\n"
            f"评测完成\n"
            f"{'=' * 60}\n"
            f"总题数: {len(items)}\n"
            f"成功: {len(items) - fail_count}, 失败: {fail_count}\n"
            f"总耗时: {format_duration(total_duration)}\n"
            f"\n--- 耗时统计 ---\n"
            f"平均检索耗时: {avg_retrieval:.2f}s\n"
        )
        if generation_times:
            summary += f"平均生成耗时: {avg_generation:.2f}s\n"
        summary += f"平均总耗时:   {avg_total:.2f}s\n"
        summary += (
            f"\n--- 检索质量统计 ---\n"
            f"平均实体数:     {avg_entities:.1f}\n"
            f"平均关系数:     {avg_relationships:.1f}\n"
            f"平均文本块数:   {avg_documents:.1f}\n"
        )
        if avg_kp_rate is not None:
            summary += f"平均关键词命中率: {avg_kp_rate:.1%}\n"
        if avg_path_cov is not None:
            summary += f"平均推理路径覆盖: {avg_path_cov:.1%}\n"

        # 按跳数分组统计
        hops_groups = defaultdict(list)
        for m in retrieval_metrics_list:
            hops_groups[m.get("hops_needed", 0)].append(m)

        if len(hops_groups) > 1:
            summary += f"\n--- 按跳数分组 ---\n"
            for hops in sorted(hops_groups.keys()):
                group = hops_groups[hops]
                group_kp = [m["keypoint_hit_rate"] for m in group if m.get("keypoint_hit_rate") is not None]
                avg_group_kp = sum(group_kp) / len(group_kp) if group_kp else 0
                group_path = [m["path_coverage"] for m in group if m.get("path_coverage") is not None]
                avg_group_path = sum(group_path) / len(group_path) if group_path else 0
                summary += (
                    f"  {hops}-hop (n={len(group)}): "
                    f"关键词命中={avg_group_kp:.1%}, "
                    f"路径覆盖={avg_group_path:.1%}\n"
                )

        for f in [clean_f, detail_f, context_f]:
            f.write(summary)

        # 写汇总 JSON
        summary_json = {
            "record_type": "summary",
            "total_questions": len(items),
            "successful": len(items) - fail_count,
            "failed": fail_count,
            "hops": args.hops,
            "top_k": args.top_k,
            "wall_clock_seconds": round(total_duration, 4),
            "avg_retrieval_time": round(avg_retrieval, 4),
            "avg_generation_time": round(avg_generation, 4) if generation_times else None,
            "avg_total_time": round(avg_total, 4),
            "avg_entities": round(avg_entities, 1),
            "avg_relationships": round(avg_relationships, 1),
            "avg_documents": round(avg_documents, 1),
            "avg_keypoint_hit_rate": round(avg_kp_rate, 4) if avg_kp_rate is not None else None,
            "avg_path_coverage": round(avg_path_cov, 4) if avg_path_cov is not None else None,
            "avg_evidence_hit_rate": round(avg_ev_rate, 4) if avg_ev_rate is not None else None,
        }
        timing_f.write(json.dumps(summary_json, ensure_ascii=False) + "\n")

        print(summary)
        print(f"\n干净结果 -> {clean_path}")
        print(f"详细记录 -> {detail_path}")
        print(f"上下文   -> {context_path}")
        print(f"计时     -> {timing_path}")
        print(f"检索质量 -> {retrieval_path}")

    builder.close()


if __name__ == "__main__":
    main()
