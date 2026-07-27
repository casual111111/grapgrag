#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python eval/evaluate_all.py --gold eval/QA.json --results eval\eval_neo4j\neo_batch_context.txt --output eval\eval_neo4j\eval_result.jsonl -c 10 
python eval/evaluate_all.py --gold eval/QA.json --results eval\eval_flash\batch_query_context.txt --output eval\eval_flash\eval_result.jsonl -c 8  

输入：标准答案文件（JSONL）+ 查询结果文件（TXT/JSONL，含上下文和生成答案）
输出：三维度评分 + 加权综合分（默认 准确率0.4 / 完整度0.3 / 忠实度0.3，可调）

用法:
    # 典型用法 A：标准答案 JSONL + 带上下文的查询结果 → 三维度全评
    python eval/evaluate_all.py \\
        --gold KB-QA-eval2.jsonl \\
        --results batch_query_context.txt \\
        --output eval_result.jsonl

    # 典型用法 B：results.txt（含标准答案）+ 单独上下文文件 → 三维度全评（不依赖 gold JSONL）
    python eval/evaluate_all.py \\
        --results batch_query_results.txt \\
        --context batch_query_context.txt \\
        --output eval_result.jsonl

    # 只有结果文件（无上下文 → 只评准确率+完整度，权重自动归一化）
    python eval/evaluate_all.py \\
        --results batch_query_results.txt \\
        --output eval_result.jsonl

    # 自定义权重（准确率:完整度:忠实度）
    python eval/evaluate_all.py \\
        --results batch_query_results.txt \\
        --context batch_query_context.txt \\
        --output eval_result.jsonl \\
        --weights 0.5,0.3,0.2

    # 断点续跑：输出文件已存在时自动跳过已评条目
    python eval/evaluate_all.py \\
        --results batch_query_results.txt \\
        --context batch_query_context.txt \\
        --output eval_result.jsonl
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Windows 控制台默认 GBK 编码，输出 ✓ 等字符会抛 UnicodeEncodeError。
# 将标准输出/错误重配为 UTF-8（Python 3.7+），无法重配时降级为忽略不可编码字符。
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is None:
        continue
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# ─────────────────────────────────────────────────────────────────────────────
# 复用三个维度评测模块的 prompt 和解析逻辑
# ─────────────────────────────────────────────────────────────────────────────

_here = Path(__file__).parent
sys.path.insert(0, str(_here))

from accuracy_judge import (
    JUDGE_SYSTEM_PROMPT as ACCURACY_PROMPT,
    build_user_prompt as build_accuracy_prompt,
    call_llm,
    parse_score_response as parse_accuracy,
)
from completeness_judge import (
    JUDGE_SYSTEM_PROMPT as COMPLETENESS_PROMPT,
    build_user_prompt as build_completeness_prompt,
    parse_score_response as parse_completeness,
)
from faithfulness_judge import (
    JUDGE_SYSTEM_PROMPT as FAITHFULNESS_PROMPT,
    build_user_prompt as build_faithfulness_prompt,
    parse_score_response as parse_faithfulness,
)

# ─────────────────────────────────────────────────────────────────────────────
# 默认配置
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "qwen3.7-plus"
DEFAULT_API_KEY = "sk-sp-f7310d61e9264799a3ec7bb574e8c939"
DEFAULT_API_BASE = "https://coding.dashscope.aliyuncs.com/v1"
DEFAULT_CONCURRENCY = 5       # 全局并发 API 调用数
DEFAULT_MAX_RETRIES = 5
DEFAULT_WEIGHTS = (0.4, 0.3, 0.3)  # 准确率 / 完整度 / 忠实度

# 综合分等级
COMBINED_LEVELS = [
    (90, "卓越"),
    (80, "优秀"),
    (70, "良好"),
    (60, "中等"),
    (40, "及格"),
    (0,  "不及格"),
]


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def parse_weights(s: str) -> Tuple[float, float, float]:
    """解析权重字符串 '0.4,0.3,0.3' 或 '0.4/0.3/0.3'，自动归一化"""
    s = s.strip()
    for sep in (",", "/", ":", " "):
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            break
    else:
        parts = [s]
    if len(parts) != 3:
        raise ValueError(f"权重必须是 3 个数字，收到: {s!r}")
    try:
        w = tuple(float(p) for p in parts)
    except ValueError as e:
        raise ValueError(f"权重必须是数字: {s!r}") from e
    total = sum(w)
    if total <= 0:
        raise ValueError(f"权重之和必须 > 0: {w}")
    return tuple(x / total for x in w)


def weighted_level(score: float) -> str:
    for threshold, level in COMBINED_LEVELS:
        if score >= threshold:
            return level
    return "不及格"


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{int(seconds // 60)}分{seconds % 60:.1f}秒"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h}时{m}分{s:.1f}秒"


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────

def load_gold(filepath: str) -> Dict[str, Dict]:
    """
    加载标准答案文件（兼容 JSON 数组和 JSONL 两种格式）。
    返回 dict，键 = question 文本，值 = 完整记录。
    """
    content = Path(filepath).read_text(encoding="utf-8").strip()
    if content.startswith("["):
        # 整体是一个 JSON 数组（可能是格式化的多行 JSON）。
        items = json.loads(content)
    else:
        # 逐行 JSONL。
        items = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    gold = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        q = (item.get("question") or "").strip()
        if not q:
            continue
        gold_answer = item.get("gold_answer", item.get("reference_answer", "")) or ""
        gold[q] = {
            "id": item.get("id", ""),
            "type": item.get("type", ""),
            "question": q,
            "gold_answer": str(gold_answer).strip(),
        }
    return gold


def load_results_txt_with_context(filepath: str) -> List[Dict]:
    """
    解析 batch_query_context.txt 格式（含【检索到的参考上下文】+【被评测答案】）。
    返回 list，保持顺序。
    """
    items = []
    content = Path(filepath).read_text(encoding="utf-8")
    blocks = re.split(r'(?m)^[ \t]*-{20,}[ \t]*(?:\r?\n|\Z)', content)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        q = re.search(r'【问题】：(.+?)(?=\n【|$)', block, re.DOTALL)
        ctx = re.search(r'【检索到的参考上下文】：(.*?)(?=\n【|$)', block, re.DOTALL)
        gen = re.search(r'【被评测答案】：(.+?)(?=\n（耗时|$)', block, re.DOTALL)
        if q and ctx and gen:
            items.append({
                "question": q.group(1).strip(),
                "retrieved_context": ctx.group(1).strip(),
                "generated_answer": gen.group(1).strip(),
            })
    return items


def load_results_txt_no_context(filepath: str) -> List[Dict]:
    """
    解析 batch_query_results.txt 格式（无上下文，有标准答案+被评测答案）。
    """
    items = []
    content = Path(filepath).read_text(encoding="utf-8")
    blocks = re.split(r'(?m)^[ \t]*-{20,}[ \t]*(?:\r?\n|\Z)', content)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        q = re.search(r'【问题】：(.+?)(?=\n【|$)', block, re.DOTALL)
        gold = re.search(r'【标准答案】：(.+?)(?=\n【|$)', block, re.DOTALL)
        gen = re.search(r'【被评测答案】：(.+?)(?=\n（耗时|$)', block, re.DOTALL)
        if q and gen:
            items.append({
                "question": q.group(1).strip(),
                "retrieved_context": "",
                "gold_answer_from_results": gold.group(1).strip() if gold else "",
                "generated_answer": gen.group(1).strip(),
            })
    return items


def load_results_jsonl(filepath: str) -> List[Dict]:
    """解析 JSONL 格式结果文件"""
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = (item.get("question") or "").strip()
            if not q:
                continue
            items.append({
                "id": item.get("id", ""),
                "type": item.get("type", ""),
                "question": q,
                "retrieved_context": (item.get("retrieved_context")
                                      or item.get("context")
                                      or item.get("retrieved_ctx")
                                      or ""),
                "generated_answer": (item.get("generated_answer")
                                     or item.get("kg_answer")
                                     or ""),
            })
    return items


def detect_and_load_results(filepath: str) -> Tuple[List[Dict], bool]:
    """
    自动识别结果文件格式，返回 (items, has_context)。
    has_context: 文件中是否含【检索到的参考上下文】字段。
    """
    if filepath.endswith(".jsonl") or filepath.endswith(".json"):
        items = load_results_jsonl(filepath)
        has_context = any(it.get("retrieved_context") for it in items)
        return items, has_context

    content = Path(filepath).read_text(encoding="utf-8")
    if "【检索到的参考上下文】" in content:
        return load_results_txt_with_context(filepath), True
    else:
        return load_results_txt_no_context(filepath), False


def merge_items(
    gold_map: Optional[Dict[str, Dict]],
    results: List[Dict],
) -> List[Dict]:
    """
    将 gold 答案按 question 文本匹配到 results。
    返回合并后的列表，保持 results 顺序。
    匹配不上的 gold_answer 为空，会触发警告。
    """
    merged = []
    unmatched = 0
    for it in results:
        q = it["question"]
        g = gold_map.get(q) if gold_map else None

        # gold_answer 优先级：gold 文件 > 结果文件自带 > 空
        gold_ans = ""
        gold_id = it.get("id", "")
        gold_type = it.get("type", "")
        if g:
            gold_ans = g.get("gold_answer", "")
            gold_id = gold_id or g.get("id", "")
            gold_type = gold_type or g.get("type", "")
        elif it.get("gold_answer_from_results"):
            gold_ans = it["gold_answer_from_results"]
        else:
            unmatched += 1

        merged.append({
            "id": gold_id,
            "type": gold_type,
            "question": q,
            "gold_answer": gold_ans,
            "retrieved_context": it.get("retrieved_context", ""),
            "generated_answer": it.get("generated_answer", ""),
        })

    if unmatched and gold_map:
        print(f"[警告] {unmatched}/{len(results)} 条结果未在 gold 文件中匹配到标准答案")
    return merged


def load_context_file(filepath: str) -> Dict[str, str]:
    """
    加载单独的上下文文件（batch_query_context.txt 或 JSONL），
    返回 dict: question -> retrieved_context。
    """
    ctx_map = {}
    if filepath.endswith(".jsonl") or filepath.endswith(".json"):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = (item.get("question") or "").strip()
                ctx = (item.get("retrieved_context")
                       or item.get("context")
                       or item.get("retrieved_ctx")
                       or "")
                if q and ctx:
                    ctx_map[q] = ctx
        return ctx_map

    content = Path(filepath).read_text(encoding="utf-8")
    blocks = re.split(r'(?m)^[ \t]*-{20,}[ \t]*(?:\r?\n|\Z)', content)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        q = re.search(r'【问题】：(.+?)(?=\n【|$)', block, re.DOTALL)
        ctx = re.search(r'【检索到的参考上下文】：(.*?)(?=\n【|$)', block, re.DOTALL)
        if q and ctx:
            ctx_text = ctx.group(1).strip()
            if ctx_text:
                ctx_map[q.group(1).strip()] = ctx_text
    return ctx_map


def merge_context_into_results(results: List[Dict], ctx_map: Dict[str, str]) -> List[Dict]:
    """
    将单独的上下文按 question 文本合并到 results 中（仅填充当前为空的字段）。
    返回更新后的 results 列表。
    """
    unmatched = 0
    for it in results:
        q = it["question"]
        if not it.get("retrieved_context") and q in ctx_map:
            it["retrieved_context"] = ctx_map[q]
        elif q not in ctx_map:
            unmatched += 1
    if unmatched:
        print(f"[警告] {unmatched}/{len(results)} 条结果未在上下文文件中匹配到检索上下文")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 单条三维度评分
# ─────────────────────────────────────────────────────────────────────────────

async def score_one_item(
    client,
    model: str,
    item: Dict,
    index: int,
    semaphore: asyncio.Semaphore,
    max_retries: int,
    eval_accuracy: bool,
    eval_completeness: bool,
    eval_faithfulness: bool,
) -> Dict:
    """对单条数据并行跑三个维度评分"""

    tasks = []
    task_names = []

    if eval_accuracy:
        tasks.append(_call_accuracy(client, model, item, semaphore, max_retries))
        task_names.append("accuracy")

    if eval_completeness:
        tasks.append(_call_completeness(client, model, item, semaphore, max_retries))
        task_names.append("completeness")

    if eval_faithfulness:
        tasks.append(_call_faithfulness(client, model, item, semaphore, max_retries))
        task_names.append("faithfulness")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 汇总各维度
    dim_results = {}
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for name, res in zip(task_names, results):
        if isinstance(res, Exception):
            print(f"  [#{index+1}] {name} 维度调用失败: {res}")
            dim_results[name] = None
        else:
            parsed, usage = res
            dim_results[name] = parsed
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)

    return {
        "index": index,
        "id": item.get("id", f"Q{index+1}"),
        "type": item.get("type", ""),
        "question": item["question"],
        "gold_answer": item.get("gold_answer", ""),
        "generated_answer": item.get("generated_answer", ""),
        "usage": total_usage,
        "dimensions": dim_results,
    }


async def _call_accuracy(client, model, item, semaphore, max_retries):
    user_prompt = build_accuracy_prompt(
        item["question"], item["gold_answer"], item["generated_answer"])
    messages = [
        {"role": "system", "content": ACCURACY_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    raw, usage = await call_llm(client=client, model=model, messages=messages,
                                semaphore=semaphore, max_retries=max_retries)
    return parse_accuracy(raw), usage


async def _call_completeness(client, model, item, semaphore, max_retries):
    user_prompt = build_completeness_prompt(
        item["question"], item["gold_answer"], item["generated_answer"])
    messages = [
        {"role": "system", "content": COMPLETENESS_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    raw, usage = await call_llm(client=client, model=model, messages=messages,
                                semaphore=semaphore, max_retries=max_retries)
    return parse_completeness(raw), usage


async def _call_faithfulness(client, model, item, semaphore, max_retries):
    user_prompt = build_faithfulness_prompt(
        item["question"], item.get("retrieved_context", ""), item["generated_answer"])
    messages = [
        {"role": "system", "content": FAITHFULNESS_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    raw, usage = await call_llm(client=client, model=model, messages=messages,
                                semaphore=semaphore, max_retries=max_retries)
    return parse_faithfulness(raw), usage


# ─────────────────────────────────────────────────────────────────────────────
# 加权计算
# ─────────────────────────────────────────────────────────────────────────────

def compute_weighted(record: Dict, weights: Tuple[float, float, float],
                     eval_dims: Dict[str, bool]) -> Dict:
    """根据可用维度计算加权分数（缺失维度按剩余权重归一化）"""
    dims = record.get("dimensions", {})

    def get_score(name):
        d = dims.get(name)
        if not d:
            return None
        return d.get("score")

    scores = {
        "accuracy": get_score("accuracy"),
        "completeness": get_score("completeness"),
        "faithfulness": get_score("faithfulness"),
    }
    levels = {
        "accuracy": dims.get("accuracy", {}).get("level", "") if dims.get("accuracy") else "",
        "completeness": dims.get("completeness", {}).get("level", "") if dims.get("completeness") else "",
        "faithfulness": dims.get("faithfulness", {}).get("level", "") if dims.get("faithfulness") else "",
    }

    # 仅对启用且有效的维度加权
    w_map = {
        "accuracy": weights[0] if eval_dims["accuracy"] else 0,
        "completeness": weights[1] if eval_dims["completeness"] else 0,
        "faithfulness": weights[2] if eval_dims["faithfulness"] else 0,
    }
    present_w = {k: w_map[k] for k in scores if scores[k] is not None and w_map[k] > 0}

    if not present_w:
        weighted = None
        weighted_level_str = "无法计算"
        used_weights = {}
    else:
        total_w = sum(present_w.values())
        used_weights = {k: v / total_w for k, v in present_w.items()}
        weighted = sum(scores[k] * used_weights[k] for k in present_w)
        weighted = int(round(weighted))
        weighted = max(0, min(100, weighted))
        weighted_level_str = weighted_level(weighted)

    return {
        "scores": scores,
        "levels": levels,
        "weights_used": {k: round(v, 4) for k, v in used_weights.items()},
        "weighted_score": weighted,
        "weighted_level": weighted_level_str,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 断点续跑
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(output_path: str) -> Tuple[set, List[Dict]]:
    scored_indices = set()
    existing = []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    scored_indices.add(r.get("index", -1))
                    existing.append(r)
                except json.JSONDecodeError:
                    continue
    return scored_indices, existing


def append_result(output_path: str, result: Dict):
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()


# ─────────────────────────────────────────────────────────────────────────────
# 报告
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    records: List[Dict],
    total_items: int,
    elapsed: float,
    weights: Tuple[float, float, float],
    eval_dims: Dict[str, bool],
) -> Dict:
    valid = [r for r in records if r.get("weighted_score") is not None]
    weighted_scores = [r["weighted_score"] for r in valid]

    def stats_for(arr):
        if not arr:
            return {"average": None, "min": None, "max": None, "median": None}
        return {
            "average": round(sum(arr) / len(arr), 2),
            "min": min(arr),
            "max": max(arr),
            "median": sorted(arr)[len(arr) // 2],
        }

    def level_dist(records, key):
        dist = {}
        for r in records:
            lv = r.get(key, "未知") if key == "weighted_level" else (r.get("levels", {}).get(key, "未知") or "未知")
            dist[lv] = dist.get(lv, 0) + 1
        return dist

    per_dim = {}
    for dim, label in [("accuracy", "准确率"), ("completeness", "完整度"), ("faithfulness", "忠实度")]:
        if not eval_dims[dim]:
            continue
        arr = [r["scores"][dim] for r in valid if r["scores"][dim] is not None]
        per_dim[dim] = {
            "label": label,
            "valid_count": len(arr),
            **stats_for(arr),
            "level_distribution": level_dist(valid, dim),
        }

    weighted_ranges = [
        ("90-100 (卓越)", 90, 100),
        ("80-89  (优秀)", 80, 89),
        ("70-79  (良好)", 70, 79),
        ("60-69  (中等)", 60, 69),
        ("40-59  (及格)", 40, 59),
        ("0-39   (不及格)", 0, 39),
    ]
    range_dist = {name: 0 for name, _, _ in weighted_ranges}
    for s in weighted_scores:
        for name, lo, hi in weighted_ranges:
            if lo <= s <= hi:
                range_dist[name] += 1
                break

    # Token 统计
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for r in records:
        for k in total_usage:
            total_usage[k] += r.get("usage", {}).get(k, 0)

    return {
        "total_items": total_items,
        "evaluated_items": len(records),
        "valid_items": len(valid),
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_formatted": format_duration(elapsed),
        "avg_time_per_item": round(elapsed / len(records), 1) if records else 0,
        "configured_weights": {
            "accuracy": round(weights[0], 4),
            "completeness": round(weights[1], 4),
            "faithfulness": round(weights[2], 4),
        },
        "enabled_dimensions": eval_dims,
        "per_dimension": per_dim,
        "weighted_score": {
            "valid_count": len(weighted_scores),
            **stats_for(weighted_scores),
        },
        "weighted_level_distribution": level_dist(valid, "weighted_level"),
        "weighted_range_distribution": range_dist,
        "token_usage": total_usage,
        "bottom_5": sorted(valid, key=lambda x: x["weighted_score"])[:5],
        "top_5": sorted(valid, key=lambda x: x["weighted_score"], reverse=True)[:5],
    }


def print_report(report: Dict):
    print("\n" + "=" * 70)
    print("QA 评测综合报告")
    print("=" * 70)

    print(f"\n总题数: {report['total_items']}, 已评分: {report['evaluated_items']}, "
          f"有效: {report['valid_items']}")
    print(f"总耗时: {report['elapsed_formatted']}")
    print(f"平均每题: {report['avg_time_per_item']}秒")

    print(f"\n--- Token 消耗 ---")
    u = report["token_usage"]
    print(f"  Prompt Tokens:     {u['prompt_tokens']:,}")
    print(f"  Completion Tokens: {u['completion_tokens']:,}")
    print(f"  Total Tokens:      {u['total_tokens']:,}")

    print(f"\n--- 配置权重 ---")
    cw = report["configured_weights"]
    enabled = report["enabled_dimensions"]
    for dim, label in [("accuracy", "准确率"), ("completeness", "完整度"), ("faithfulness", "忠实度")]:
        mark = "✓" if enabled[dim] else "✗"
        print(f"  [{mark}] {label}: {cw[dim]}")

    print(f"\n--- 各维度分数统计 ---")
    for dim, info in report["per_dimension"].items():
        if info["average"] is None:
            print(f"  {info['label']}: 无有效数据")
        else:
            print(f"  {info['label']}: 平均={info['average']}, "
                  f"中位={info['median']}, 最低={info['min']}, 最高={info['max']}")

    print(f"\n--- 加权综合分 ---")
    ws = report["weighted_score"]
    if ws["average"] is None:
        print("  无有效综合分")
    else:
        print(f"  平均: {ws['average']}")
        print(f"  中位: {ws['median']}")
        print(f"  最低: {ws['min']}")
        print(f"  最高: {ws['max']}")

    print(f"\n--- 综合等级分布 ---")
    for level, count in sorted(report["weighted_level_distribution"].items()):
        print(f"  {level}: {count}")

    print(f"\n--- 综合分区间分布 ---")
    for range_name, count in report["weighted_range_distribution"].items():
        bar = "█" * count
        print(f"  {range_name}: {count} {bar}")

    print(f"\n--- 加权分最低的 5 条 ---")
    for i, item in enumerate(report.get("bottom_5", []), 1):
        s = item["scores"]
        parts = []
        for dim, label in [("accuracy", "准确"), ("completeness", "完整"), ("faithfulness", "忠实")]:
            if s.get(dim) is not None:
                parts.append(f"{label}={s[dim]}")
        print(f"  {i}. [{item.get('id', '')}] 加权={item['weighted_score']} "
              f"({item['weighted_level']})  " + " ".join(parts))
        print(f"     问题: {item['question'][:60]}...")

    print(f"\n--- 加权分最高的 5 条 ---")
    for i, item in enumerate(report.get("top_5", []), 1):
        s = item["scores"]
        parts = []
        for dim, label in [("accuracy", "准确"), ("completeness", "完整"), ("faithfulness", "忠实")]:
            if s.get(dim) is not None:
                parts.append(f"{label}={s[dim]}")
        print(f"  {i}. [{item.get('id', '')}] 加权={item['weighted_score']} "
              f"({item['weighted_level']})  " + " ".join(parts))
        print(f"     问题: {item['question'][:60]}...")

    print("\n" + "=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

async def run_evaluation(args):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.api_base)

    # 1. 加载数据
    print(f"\n[1/4] 加载数据...")

    gold_map = None
    if args.gold:
        gold_map = load_gold(args.gold)
        print(f"  标准答案: {len(gold_map)} 条 ← {args.gold}")

    results, has_context = detect_and_load_results(args.results)
    print(f"  查询结果: {len(results)} 条 ← {args.results} "
          f"{'(含上下文)' if has_context else '(无上下文)'}")

    # 如果主结果文件无上下文但提供了 --context 文件，从那里补
    if not has_context and args.context:
        ctx_map = load_context_file(args.context)
        print(f"  检索上下文: {len(ctx_map)} 条 ← {args.context}")
        results = merge_context_into_results(results, ctx_map)
        has_context = any(it.get("retrieved_context") for it in results)
    elif args.context and has_context:
        print(f"  [提示] 主结果文件已含上下文，--context 参数被忽略")

    if not results:
        print("[错误] 查询结果为空，退出")
        return

    items = merge_items(gold_map, results)

    # 2. 确定启用的维度
    has_gold = any(it.get("gold_answer") for it in items)
    eval_accuracy = has_gold
    eval_completeness = has_gold
    eval_faithfulness = has_context
    eval_dims = {
        "accuracy": eval_accuracy,
        "completeness": eval_completeness,
        "faithfulness": eval_faithfulness,
    }

    if not any(eval_dims.values()):
        print("[错误] 既无标准答案也无上下文，无法进行任何维度评测")
        return

    print(f"\n[2/4] 启用维度:")
    for dim, label in [("accuracy", "准确率"), ("completeness", "完整度"), ("faithfulness", "忠实度")]:
        mark = "✓" if eval_dims[dim] else "✗"
        print(f"  [{mark}] {label}")

    if not eval_faithfulness:
        print("  [提示] 无上下文 → 忠实度维度不启用，权重自动归一化到其余维度")
    if not eval_accuracy or not eval_completeness:
        print("  [提示] 无标准答案 → 准确率/完整度维度不启用，权重自动归一化到其余维度")

    # 3. 断点续跑
    scored_indices, existing_records = load_checkpoint(args.output)
    if scored_indices:
        print(f"\n[3/4] 断点续跑：已评 {len(scored_indices)} 条，跳过")
    else:
        print(f"\n[3/4] 开始评测...")

    pending = [(i, it) for i, it in enumerate(items) if i not in scored_indices]
    print(f"  待评分: {len(pending)} 条")

    semaphore = asyncio.Semaphore(args.concurrency)
    all_records = list(existing_records)
    start_time = time.time()

    for idx, item in pending:
        item_id = item.get("id", f"Q{idx+1}")
        progress = f"[{len(all_records) - len(existing_records) + 1}/{len(pending)}]"
        print(f"{progress} 评分: {item_id} - {item['question'][:40]}...")

        record = await score_one_item(
            client=client,
            model=args.model,
            item=item,
            index=idx,
            semaphore=semaphore,
            max_retries=args.max_retries,
            eval_accuracy=eval_accuracy,
            eval_completeness=eval_completeness,
            eval_faithfulness=eval_faithfulness,
        )

        # 计算加权分
        weighted_info = compute_weighted(record, args.weights_tuple, eval_dims)
        record.update(weighted_info)

        append_result(args.output, record)
        all_records.append(record)

        s = record["scores"]
        parts = []
        for dim, label in [("accuracy", "准确"), ("completeness", "完整"), ("faithfulness", "忠实")]:
            if s.get(dim) is not None:
                parts.append(f"{label}={s[dim]}")
        print(f"  → 加权={record['weighted_score']} ({record['weighted_level']})  " + " ".join(parts))

    # 4. 生成报告
    elapsed = time.time() - start_time
    print(f"\n[4/4] 生成报告...")

    report = generate_report(all_records, len(items), elapsed, args.weights_tuple, eval_dims)
    print_report(report)

    report_path = args.output + ".report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n汇总报告: {report_path}")
    print(f"评分明细: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="一站式 QA 评测（准确率 + 完整度 + 忠实度，加权汇总）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 方式A：标准答案 JSONL + 带上下文的查询结果 → 三维度全评
  python eval/evaluate_all.py \\
      --gold KB-QA-eval2.jsonl \\
      --results batch_query_context.txt \\
      --output eval_result.jsonl

  # 方式B：results.txt（含标准答案+生成答案）+ 单独上下文文件 → 三维度全评
  python eval/evaluate_all.py \\
      --results batch_query_results.txt \\
      --context batch_query_context.txt \\
      --output eval_result.jsonl

  # 方式C：只有结果文件（无上下文 → 只评准确率+完整度，权重自动归一化）
  python eval/evaluate_all.py \\
      --results batch_query_results.txt \\
      --output eval_result.jsonl

  # 自定义权重（准确率:完整度:忠实度）
  python eval/evaluate_all.py \\
      --results batch_query_results.txt \\
      --context batch_query_context.txt \\
      --output eval_result.jsonl \\
      --weights 0.5,0.3,0.2
        """,
    )
    parser.add_argument("--gold", "-g", default=None,
                        help="标准答案 JSONL 文件（含 question + gold_answer 字段）")
    parser.add_argument("--results", "-r", required=True,
                        help="查询结果文件（TXT/JSONL，含被评测答案；TXT 可选含检索上下文或标准答案）")
    parser.add_argument("--context", default=None,
                        help="可选：单独的检索上下文文件（TXT/JSONL）。当主结果文件不含上下文时使用")
    parser.add_argument("--output", "-o", required=True,
                        help="输出 JSONL 文件（评分明细）")
    parser.add_argument("--weights", "-w", default="0.4,0.3,0.3",
                        help="权重，逗号分隔 3 个数字（默认: 0.4,0.3,0.3；自动归一化）")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help=f"评判模型 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE,
                        help=f"API 地址 (默认: {DEFAULT_API_BASE})")
    parser.add_argument("--api-key", default=None,
                        help="API Key（或环境变量 DASHSCOPE_API_KEY）")
    parser.add_argument("--concurrency", "-c", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"并发 API 调用数 (默认: {DEFAULT_CONCURRENCY})")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                        help=f"每维度单次调用最大重试次数 (默认: {DEFAULT_MAX_RETRIES})")

    args = parser.parse_args()

    # 解析权重
    try:
        args.weights_tuple = parse_weights(args.weights)
    except ValueError as e:
        print(f"[错误] {e}")
        sys.exit(1)

    # API Key
    if not args.api_key:
        args.api_key = os.environ.get("DASHSCOPE_API_KEY", "") or DEFAULT_API_KEY
    if not args.api_key:
        print("[错误] 请通过 --api-key 或环境变量 DASHSCOPE_API_KEY 提供 API Key")
        sys.exit(1)

    # 输入文件检查
    if args.gold and not os.path.exists(args.gold):
        print(f"[错误] 标准答案文件不存在: {args.gold}")
        sys.exit(1)
    if not os.path.exists(args.results):
        print(f"[错误] 查询结果文件不存在: {args.results}")
        sys.exit(1)
    if args.context and not os.path.exists(args.context):
        print(f"[错误] 上下文文件不存在: {args.context}")
        sys.exit(1)

    print(f"配置: model={args.model}, concurrency={args.concurrency}, "
          f"权重=准确率:{args.weights_tuple[0]:.2f}/完整度:{args.weights_tuple[1]:.2f}/忠实度:{args.weights_tuple[2]:.2f}")
    print(f"标准答案: {args.gold or '(未提供，将从 results 文件提取)'}")
    print(f"查询结果: {args.results}")
    print(f"检索上下文: {args.context or '(将从 results 文件提取或无)'}")
    print(f"输出文件: {args.output}")

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    total_start = time.time()
    asyncio.run(run_evaluation(args))
    print(f"\n脚本总运行时间: {format_duration(time.time() - total_start)}")


if __name__ == "__main__":
    main()
