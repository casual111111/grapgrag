#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
忠实度评测打分脚本

读取包含【检索到的参考上下文】的评测结果，
调用 LLM 对每条【被评测答案】的事实陈述进行幻觉检测，
判定每条陈述是否能在检索上下文中找到支撑依据。

用法:
    python faithfulness_judge.py --input batch_query_context.txt --output faithfulness_scores.jsonl
    python faithfulness_judge.py --input eval_data.jsonl --output faithfulness_scores.jsonl --api-key sk-xxx
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

# ─────────────────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "qwen3.7-plus"
DEFAULT_API_KEY = "sk-sp-f7310d61e9264799a3ec7bb574e8c939"
DEFAULT_API_BASE = "https://coding.dashscope.aliyuncs.com/v1"
DEFAULT_CONCURRENCY = 5
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BASE_DELAY = 1.0
DEFAULT_MAX_RETRY_DELAY = 60.0
DEFAULT_TEMPERATURE = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 评测 Prompt
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """你是专业QA系统评测专家，评估被评测答案每条事实陈述是否能在检索上下文找到支撑依据，检测模型幻觉。

## 评估定义

忠实度只校验事实陈述是否有上下文原文依据，不判断现实层面对错；现实正确但上下文无记录，依旧判定为无支撑幻觉；过渡连接词、纯主观建议语句无事实前提的不参与判定。

## 标准化评分步骤

1. 拆解事实陈述：拆分被评测答案为独立可验证事实断言，每条单独编号；单句混合有支撑/无支撑内容需拆分为两条陈述；
2. 逐条判定事实状态：
   ✅ supported（有支撑）：上下文存在原文，或简单逻辑推理可直接推导；
   ⚠️ unsupported（无支撑/幻觉）：上下文无任何相关信息，模型凭空生成；
   ❌ contradicted（矛盾）：陈述内容和上下文原文明确冲突；
3. 分数计算公式：
   原始分 = max(0, (有支撑条数 - 矛盾条数) ÷ 总陈述条数 × 100)
   最终得分：原始分四舍五入取整整数

## 分数区间分级标准

|分数区间|等级|判定标准|典型表现|
| ---- | ---- | ---- | ---- |
|95-100|完全忠实|全部事实陈述有上下文支撑|零幻觉，所有信息均可定位上下文原文|
|80-94|高度忠实|绝大多数有支撑，仅1-2条无支撑、无矛盾|少量无依据细节，不存在和上下文冲突内容|
|60-79|中度忠实|主体内容有据，存在多处幻觉|核心内容匹配上下文，穿插多条模型编造信息|
|40-59|低度忠实|支撑与幻觉各占一半，幻觉问题突出|大量关键事实无法从检索上下文溯源|
|20-39|严重幻觉|大部分陈述无支撑或存在矛盾|答案核心内容为模型自主生成，上下文利用率极低|
|0-19|完全不忠实|几乎全为幻觉、大量内容和上下文冲突|答案整体脱离检索上下文，多处事实冲突|

## 边界统一处理规则

1. 上下文浅层隐含信息（一步简单推理）：判定为有支撑；
2. 需要多步复杂推导才能得到的结论：判定为无支撑幻觉；
3. 通用基础常识，上下文未提及：依旧判定无支撑；
4. 对上下文原文合理归纳总结：判定为有支撑；
5. 新增上下文不存在的数字、专有名词、规则定义：判定无支撑幻觉。

## 强制输出格式（仅输出JSON，禁止额外文字、注释、换行说明）

```json
{
  "faithfulness": {
    "claims": [
      {"id": 1, "content": "<单条事实陈述文本>", "status": "supported|unsupported|contradicted", "evidence": "<有支撑则摘抄上下文片段；无支撑写'上下文未提及XX内容'；矛盾写明冲突原文>"}
    ],
    "statistics": {
      "total_claims": <总事实陈述条数整数>,
      "supported_count": <有支撑条数整数>,
      "unsupported_count": <无支撑条数整数>,
      "contradicted_count": <矛盾条数整数>
    },
    "score": <0-100整数分值>,
    "level": "<完全忠实|高度忠实|中度忠实|低度忠实|严重幻觉|完全不忠实>",
    "hallucinations": ["<按严重程度排序罗列所有幻觉、矛盾陈述>"],
    "reasoning": "<1-2句话总结整体忠实度与幻觉情况>"
  }
}
```"""


def build_user_prompt(question: str, retrieved_context: str, generated_answer: str) -> str:
    """构建用户侧 prompt"""
    # 上下文为空或空列表时给一个明确提示，避免 LLM 自行脑补
    context_text = retrieved_context.strip()
    if not context_text or context_text in ("[]", "{}", "null", "None"):
        context_text = "（检索上下文为空，无任何参考信息）"

    return f"""【问题】：{question}

【检索到的参考上下文】：{context_text}

【被评测答案】：{generated_answer}"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM 调用
# ─────────────────────────────────────────────────────────────────────────────

async def call_llm(
    client,
    model: str,
    messages: List[Dict],
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_RETRY_DELAY,
    semaphore: asyncio.Semaphore = None,
) -> Tuple[str, Dict]:
    """调用 LLM，返回 (content, usage_dict)"""
    last_error = None
    for attempt in range(max_retries):
        try:
            if semaphore:
                async with semaphore:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=6000,
                    )
            else:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=6000,
                )
            content = response.choices[0].message.content
            usage = {}
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                }
            return content, usage
        except Exception as e:
            last_error = e
            delay = min(base_delay * (2 ** attempt), max_delay)
            print(f"  [重试] 第{attempt+1}次调用失败: {e}，{delay:.1f}s 后重试...")
            await asyncio.sleep(delay)
    raise RuntimeError(f"LLM 调用失败（已重试 {max_retries} 次）: {last_error}")


# ─────────────────────────────────────────────────────────────────────────────
# JSON 解析
# ─────────────────────────────────────────────────────────────────────────────

def extract_json(text: str) -> Optional[Dict]:
    """从 LLM 返回文本中提取 JSON"""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    patterns = [
        r'```json\s*\n?(.*?)\n?\s*```',
        r'```\s*\n?(.*?)\n?\s*```',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass

    return None


def parse_score_response(raw_text: str) -> Optional[Dict]:
    """解析评分返回"""
    data = extract_json(raw_text)
    if data is None:
        return None

    if "faithfulness" in data:
        faith = data["faithfulness"]
    else:
        faith = data

    score = faith.get("score")
    if not isinstance(score, (int, float)) or score < 0 or score > 100:
        return None

    statistics = faith.get("statistics", {})
    claims = faith.get("claims", [])

    return {
        "claims": claims,
        "statistics": {
            "total_claims": int(statistics.get("total_claims", len(claims))),
            "supported_count": int(statistics.get("supported_count", 0)),
            "unsupported_count": int(statistics.get("unsupported_count", 0)),
            "contradicted_count": int(statistics.get("contradicted_count", 0)),
        },
        "score": int(score),
        "level": faith.get("level", ""),
        "hallucinations": faith.get("hallucinations", []),
        "reasoning": faith.get("reasoning", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 输入解析
# ─────────────────────────────────────────────────────────────────────────────

def parse_context_txt(filepath: str) -> List[Dict]:
    """解析 batch_query_context.txt 格式（含【检索到的参考上下文】和【被评测答案】）"""
    items = []
    content = Path(filepath).read_text(encoding="utf-8")

    # 按分隔线切分
    blocks = re.split(r'-{20,}', content)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        question_match = re.search(r'【问题】：(.+?)(?=\n【|$)', block, re.DOTALL)
        ctx_match = re.search(r'【检索到的参考上下文】：(.*?)(?=\n【|$)', block, re.DOTALL)
        gen_match = re.search(r'【被评测答案】：(.+?)(?=\n（耗时|$)', block, re.DOTALL)

        if question_match and ctx_match and gen_match:
            items.append({
                "question": question_match.group(1).strip(),
                "retrieved_context": ctx_match.group(1).strip(),
                "generated_answer": gen_match.group(1).strip(),
            })

    return items


def parse_results_txt(filepath: str) -> List[Dict]:
    """解析 batch_query_results.txt 格式（无上下文，需配合 --context 文件使用）"""
    items = []
    content = Path(filepath).read_text(encoding="utf-8")
    blocks = re.split(r'-{20,}', content)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        question_match = re.search(r'【问题】：(.+?)(?=\n【|$)', block, re.DOTALL)
        gen_match = re.search(r'【被评测答案】：(.+?)(?=\n（耗时|$)', block, re.DOTALL)

        if question_match and gen_match:
            items.append({
                "question": question_match.group(1).strip(),
                "retrieved_context": "",  # 占位，由 context 文件填充
                "generated_answer": gen_match.group(1).strip(),
            })

    return items


def parse_jsonl_results(filepath: str) -> List[Dict]:
    """解析 JSONL 格式文件"""
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    question = item.get("question", "")
                    ctx = item.get("retrieved_context",
                                   item.get("context",
                                            item.get("retrieved_ctx", "")))
                    gen = item.get("generated_answer", item.get("kg_answer", ""))
                    if question:
                        items.append({
                            "id": item.get("id", ""),
                            "type": item.get("type", ""),
                            "question": question,
                            "retrieved_context": ctx,
                            "generated_answer": gen,
                        })
                except json.JSONDecodeError:
                    continue
    return items


def merge_context(items: List[Dict], context_items: List[Dict]) -> List[Dict]:
    """按顺序/问题将上下文合并到主列表"""
    # 优先按顺序一一对应
    if len(items) == len(context_items):
        for item, ctx_item in zip(items, context_items):
            item["retrieved_context"] = ctx_item.get("retrieved_context", "")
        return items

    # 否则按问题文本匹配
    ctx_map = {it["question"]: it.get("retrieved_context", "") for it in context_items}
    for item in items:
        item["retrieved_context"] = ctx_map.get(item["question"], "")
    return items


def load_input(filepath: str, context_filepath: Optional[str] = None) -> List[Dict]:
    """自动识别文件格式并加载"""
    if filepath.endswith(".jsonl") or filepath.endswith(".json"):
        items = parse_jsonl_results(filepath)
    else:
        # 先尝试按 context 文件解析（含【检索到的参考上下文】）
        content = Path(filepath).read_text(encoding="utf-8")
        if "【检索到的参考上下文】" in content:
            items = parse_context_txt(filepath)
        else:
            items = parse_results_txt(filepath)

    # 如果主文件不含上下文但提供了单独的 context 文件，合并
    if context_filepath:
        ctx_items = parse_context_txt(context_filepath)
        items = merge_context(items, ctx_items)

    return items


# ─────────────────────────────────────────────────────────────────────────────
# 断点续跑
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(output_path: str) -> Tuple[set, List[Dict]]:
    """加载已有打分结果"""
    scored_indices = set()
    existing_results = []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        scored_indices.add(record.get("index", -1))
                        existing_results.append(record)
                    except json.JSONDecodeError:
                        continue
    return scored_indices, existing_results


def append_result(output_path: str, result: Dict):
    """追加写入一条结果"""
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()


# ─────────────────────────────────────────────────────────────────────────────
# 单条评分
# ─────────────────────────────────────────────────────────────────────────────

async def score_one(
    client,
    model: str,
    item: Dict,
    index: int,
    semaphore: asyncio.Semaphore,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Dict:
    """对单条进行忠实度评分"""
    question = item["question"]
    retrieved_context = item.get("retrieved_context", "")
    generated_answer = item.get("generated_answer", "")

    user_prompt = build_user_prompt(question, retrieved_context, generated_answer)

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    raw_response, usage = await call_llm(
        client=client,
        model=model,
        messages=messages,
        semaphore=semaphore,
        max_retries=max_retries,
    )

    parsed = parse_score_response(raw_response)

    if parsed is None:
        print(f"  [#{index+1}] 首次解析失败，重试中...")
        raw_response, usage2 = await call_llm(
            client=client,
            model=model,
            messages=messages,
            temperature=0.1,
            semaphore=semaphore,
            max_retries=2,
        )
        parsed = parse_score_response(raw_response)
        if usage2:
            usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0) + usage2.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0) + usage2.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0) + usage2.get("total_tokens", 0),
            }

    result = {
        "index": index,
        "id": item.get("id", f"Q{index+1}"),
        "type": item.get("type", ""),
        "question": question,
        "retrieved_context": retrieved_context[:200] + ("..." if len(retrieved_context) > 200 else ""),
        "generated_answer": generated_answer,
        "usage": usage,
    }

    if parsed:
        result.update({
            "score": parsed["score"],
            "level": parsed["level"],
            "claims": parsed["claims"],
            "statistics": parsed["statistics"],
            "hallucinations": parsed["hallucinations"],
            "reasoning": parsed["reasoning"],
        })
    else:
        result.update({
            "score": None,
            "level": "解析失败",
            "claims": [],
            "statistics": {},
            "hallucinations": [],
            "reasoning": "LLM 返回内容无法解析为有效 JSON",
            "raw_response": raw_response[:500],
        })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 汇总报告
# ─────────────────────────────────────────────────────────────────────────────

def format_duration(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}分{secs:.1f}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}时{minutes}分{secs:.1f}秒"


def generate_report(results: List[Dict], total_time: float) -> Dict:
    """生成汇总报告"""
    valid = [r for r in results if r.get("score") is not None]

    total_prompt_tokens = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in results)
    total_completion_tokens = sum(r.get("usage", {}).get("completion_tokens", 0) for r in results)
    total_tokens = sum(r.get("usage", {}).get("total_tokens", 0) for r in results)

    scores = [r["score"] for r in valid]
    avg_score = sum(scores) / len(scores) if scores else 0

    level_dist = {}
    for r in valid:
        level = r.get("level", "未知")
        level_dist[level] = level_dist.get(level, 0) + 1

    score_ranges = {
        "95-100 (完全忠实)": 0,
        "80-94 (高度忠实)": 0,
        "60-79 (中度忠实)": 0,
        "40-59 (低度忠实)": 0,
        "20-39 (严重幻觉)": 0,
        "0-19 (完全不忠实)": 0,
    }
    for s in scores:
        if s >= 95:
            score_ranges["95-100 (完全忠实)"] += 1
        elif s >= 80:
            score_ranges["80-94 (高度忠实)"] += 1
        elif s >= 60:
            score_ranges["60-79 (中度忠实)"] += 1
        elif s >= 40:
            score_ranges["40-59 (低度忠实)"] += 1
        elif s >= 20:
            score_ranges["20-39 (严重幻觉)"] += 1
        else:
            score_ranges["0-19 (完全不忠实)"] += 1

    # 陈述统计汇总
    total_all_claims = 0
    total_supported = 0
    total_unsupported = 0
    total_contradicted = 0
    for r in valid:
        stats = r.get("statistics", {})
        total_all_claims += stats.get("total_claims", 0)
        total_supported += stats.get("supported_count", 0)
        total_unsupported += stats.get("unsupported_count", 0)
        total_contradicted += stats.get("contradicted_count", 0)

    return {
        "total_items": len(results),
        "valid_items": len(valid),
        "failed_items": len(results) - len(valid),
        "total_time_seconds": round(total_time, 1),
        "total_time_formatted": format_duration(total_time),
        "avg_time_per_item": round(total_time / len(results), 1) if results else 0,
        "token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
        },
        "score_stats": {
            "average": round(avg_score, 2),
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "median": sorted(scores)[len(scores)//2] if scores else None,
        },
        "level_distribution": level_dist,
        "score_range_distribution": score_ranges,
        "claims_summary": {
            "total_claims": total_all_claims,
            "supported_count": total_supported,
            "unsupported_count": total_unsupported,
            "contradicted_count": total_contradicted,
            "supported_rate": round(total_supported / total_all_claims * 100, 2) if total_all_claims else 0,
            "unsupported_rate": round(total_unsupported / total_all_claims * 100, 2) if total_all_claims else 0,
            "contradicted_rate": round(total_contradicted / total_all_claims * 100, 2) if total_all_claims else 0,
        },
        "bottom_5": sorted(valid, key=lambda x: x["score"])[:5],
    }


def print_report(report: Dict):
    """打印报告"""
    print("\n" + "=" * 70)
    print("忠实度评测报告")
    print("=" * 70)

    print(f"\n总题数: {report['total_items']}, 有效评分: {report['valid_items']}, "
          f"失败: {report['failed_items']}")
    print(f"总耗时: {report['total_time_formatted']}")
    print(f"平均每题: {report['avg_time_per_item']}秒")

    print(f"\n--- Token 消耗 ---")
    usage = report["token_usage"]
    print(f"  Prompt Tokens:     {usage['prompt_tokens']:,}")
    print(f"  Completion Tokens: {usage['completion_tokens']:,}")
    print(f"  Total Tokens:      {usage['total_tokens']:,}")

    print(f"\n--- 分数统计 ---")
    stats = report["score_stats"]
    print(f"  平均分: {stats['average']}")
    print(f"  最低分: {stats['min']}")
    print(f"  最高分: {stats['max']}")
    print(f"  中位数: {stats['median']}")

    print(f"\n--- 等级分布 ---")
    for level, count in sorted(report["level_distribution"].items()):
        print(f"  {level}: {count}")

    print(f"\n--- 分数区间分布 ---")
    for range_name, count in report["score_range_distribution"].items():
        bar = "█" * count
        print(f"  {range_name}: {count} {bar}")

    print(f"\n--- 事实陈述汇总 ---")
    cs = report["claims_summary"]
    print(f"  总陈述条数: {cs['total_claims']}")
    print(f"  有支撑:     {cs['supported_count']} ({cs['supported_rate']}%)")
    print(f"  无支撑幻觉: {cs['unsupported_count']} ({cs['unsupported_rate']}%)")
    print(f"  矛盾冲突:   {cs['contradicted_count']} ({cs['contradicted_rate']}%)")

    print(f"\n--- 得分最低的 5 条 ---")
    for i, item in enumerate(report.get("bottom_5", []), 1):
        print(f"  {i}. [{item.get('id', '')}] 分数={item['score']} 等级={item.get('level', '')}")
        print(f"     问题: {item['question'][:60]}...")
        hall = item.get("hallucinations", [])
        if hall:
            print(f"     幻觉: {'; '.join(hall[:2])}{'...' if len(hall) > 2 else ''}")
        print(f"     判定: {item.get('reasoning', '')[:80]}")

    print("\n" + "=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

async def run_evaluation(args):
    """主评估流程"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
    )

    items = load_input(args.input, args.context)
    print(f"加载 {len(items)} 条待评测数据")

    if not items:
        print("没有数据，退出")
        return

    # 检查上下文是否加载
    empty_ctx = sum(1 for it in items if not it.get("retrieved_context"))
    if empty_ctx:
        print(f"[警告] {empty_ctx}/{len(items)} 条数据缺少检索上下文，将被判为全部无支撑")

    scored_indices, existing_results = load_checkpoint(args.output)
    if scored_indices:
        print(f"发现已有 {len(scored_indices)} 条打分记录（断点续跑）")

    pending = [(i, item) for i, item in enumerate(items) if i not in scored_indices]
    print(f"待打分: {len(pending)} 条")

    if not pending:
        print("全部已打分完成")
        all_results = existing_results
    else:
        semaphore = asyncio.Semaphore(args.concurrency)
        total = len(pending)
        all_results = list(existing_results)

        start_time = time.time()

        async def process(idx_item):
            idx, item = idx_item
            item_id = item.get("id", f"Q{idx+1}")
            progress = f"[{len(all_results) - len(existing_results) + 1}/{total}]"
            print(f"{progress} 评分: {item_id} - {item['question'][:40]}...")

            result = await score_one(
                client=client,
                model=args.model,
                item=item,
                index=idx,
                semaphore=semaphore,
                max_retries=args.max_retries,
            )

            append_result(args.output, result)

            score = result.get("score", "N/A")
            level = result.get("level", "N/A")
            stats = result.get("statistics", {})
            sup = stats.get("supported_count", "?")
            unsup = stats.get("unsupported_count", "?")
            contra = stats.get("contradicted_count", "?")
            print(f"  → 分数: {score}, 等级: {level} (支撑:{sup} 幻觉:{unsup} 矛盾:{contra})")

            return result

        for idx_item in pending:
            result = await process(idx_item)
            all_results.append(result)

    end_time = time.time()
    total_time = end_time - (start_time if 'start_time' in dir() else end_time)

    if not pending:
        total_time = 0

    report = generate_report(all_results, total_time)
    print_report(report)

    report_path = args.output + ".report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n详细报告已写入: {report_path}")
    print(f"评分明细已写入: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="忠实度评测打分脚本 (LLM-as-Judge) - 检测模型幻觉",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 输入含【检索到的参考上下文】的 context 文件
  python faithfulness_judge.py -i batch_query_context.txt -o faithfulness_scores.jsonl

  # 主文件无上下文，通过 --context 指定上下文文件
  python faithfulness_judge.py -i batch_query_results.txt --context batch_query_context.txt -o scores.jsonl

  # JSONL 输入（字段名：retrieved_context / context / retrieved_ctx）
  python faithfulness_judge.py -i eval_data.jsonl -o faithfulness_scores.jsonl -c 10
        """,
    )
    parser.add_argument("--input", "-i", required=True,
                        help="输入文件（支持 .txt / .jsonl；txt 可含【检索到的参考上下文】字段）")
    parser.add_argument("--context", default=None,
                        help="可选：单独的上下文文件路径（当主输入文件不含【检索到的参考上下文】时使用）")
    parser.add_argument("--output", "-o", required=True,
                        help="输出 JSONL 文件路径（评分明细）")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help=f"评判模型 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE,
                        help=f"API 地址 (默认: {DEFAULT_API_BASE})")
    parser.add_argument("--api-key", default=None,
                        help="API Key（或通过环境变量 DASHSCOPE_API_KEY 设置）")
    parser.add_argument("--concurrency", "-c", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"并发数 (默认: {DEFAULT_CONCURRENCY})")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                        help=f"最大重试次数 (默认: {DEFAULT_MAX_RETRIES})")

    args = parser.parse_args()

    if not args.api_key:
        args.api_key = os.environ.get("DASHSCOPE_API_KEY", "") or DEFAULT_API_KEY
    if not args.api_key:
        print("[错误] 请通过 --api-key 或环境变量 DASHSCOPE_API_KEY 提供 API Key")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"[错误] 输入文件不存在: {args.input}")
        sys.exit(1)

    if args.context and not os.path.exists(args.context):
        print(f"[错误] 上下文文件不存在: {args.context}")
        sys.exit(1)

    print(f"配置: model={args.model}, concurrency={args.concurrency}")
    print(f"输入: {args.input}")
    if args.context:
        print(f"上下文: {args.context}")
    print(f"输出: {args.output}")

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    start = time.time()
    asyncio.run(run_evaluation(args))
    elapsed = time.time() - start
    print(f"\n脚本总运行时间: {format_duration(elapsed)}")


if __name__ == "__main__":
    main()
