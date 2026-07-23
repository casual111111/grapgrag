#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准确率评测打分脚本

读取 batch_query_results.txt 或 JSONL 格式的评测结果，
调用 LLM 对每条【被评测答案】与【标准答案】进行准确率打分。

用法:
    python accuracy_judge.py --input batch_query_results.txt --output accuracy_scores.jsonl
    python accuracy_judge.py --input eval_data.jsonl --output accuracy_scores.jsonl --api-key sk-xxx
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

JUDGE_SYSTEM_PROMPT = """你是专业QA系统评测专家，负责评估被评测答案与标准答案的语义一致性。

## 评估定义

准确率判断被评测答案核心结论、观点、信息是否和标准答案一致，为整体语义判断，不逐字比对；语序、用词、句式不同不影响得分，核心信息一致即准确。

## 标准化评分步骤

1. 提取核心论点：分别用1-2句话概括标准答案、被评测答案的核心结论；

2. 一致性对比：判定二者核心论点匹配程度；

3. 细节偏差校验：核心一致前提下，查找事实细节误差；

4. 错误信息筛查：识别和标准答案明确冲突的内容；

5. 综合打分：对照分数区间给出整数分值与对应等级。

## 分数区间分级标准

|分数区间|等级|判定标准|典型表现|
| ---- | ---- | ---- | ---- |
|95-100|完美|语义完全一致，无任何偏差|核心结论相同，全部关键细节准确，仅表述有差异|
|85-94|优秀|核心完全正确，仅极个别细节描述模糊|主体结论无误，存在1处非关键轻微简化/模糊表述|
|70-84|良好|核心正确，存在可感知细节偏差|主干结论正确，1-2处细节失真、信息简化缺失|
|55-69|中等|正误内容共存，仅部分要点答对|回答方向贴合问题，但存在明显事实错误，仅覆盖一半核心内容|
|40-54|较差|主题相关，但核心结论出错|提及相关领域术语，但整体观点偏离标准答案|
|20-39|很差|仅有微弱关联，绝大部分信息错误|仅少量关键词重合，主体内容全部失真|
|0-19|无效|完全答非所问、全盘矛盾、无意义空话|和问题无关，核心观点与标准答案完全对立|

## 边界统一处理规则

1. 被评测答案比标准答案内容更详实、核心不变：不扣分；
2. 被评测答案简略概括、核心信息完整：不扣分，简略问题交由完整性维度评判；
3. 同义不同专业术语、转述改写：视为无偏差，不扣分；
4. 同时包含正确+错误信息：根据错误严重程度降档扣分；
5. 标准答案含多条独立结论，仅答对部分：按正确占比分配分数。

## 强制输出格式（仅输出JSON，禁止额外文字、注释、换行说明）

```json
{
  "accuracy": {
    "reference_core": "<1-2句总结标准答案核心结论>",
    "generated_core": "<1-2句总结被评测答案核心结论>",
    "consistency": "<一致/基本一致/部分一致/不一致/完全不一致>",
    "deviations": ["<列举全部偏差事实，无偏差则为空数组>"],
    "score": <0-100整数分值>,
    "level": "<完美|优秀|良好|中等|较差|很差|无效>",
    "reasoning": "<2-3句话说明打分判定逻辑>"
  }
}
```"""


def build_user_prompt(question: str, reference_answer: str, generated_answer: str) -> str:
    """构建用户侧 prompt"""
    return f"""【问题】：{question}

【标准答案】：{reference_answer}

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
                        max_tokens=2000,
                    )
            else:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=2000,
                )
            content = response.choices[0].message.content
            # 提取 usage 信息
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
    # 先去掉 think 标签内容
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 从 markdown code block 中提取
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

    # 找最外层 {}
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

    # 支持两种格式：直接是 accuracy 字段，或嵌套在顶层
    if "accuracy" in data:
        acc = data["accuracy"]
    else:
        acc = data

    # 校验必要字段
    score = acc.get("score")
    if not isinstance(score, (int, float)) or score < 0 or score > 100:
        return None

    return {
        "reference_core": acc.get("reference_core", ""),
        "generated_core": acc.get("generated_core", ""),
        "consistency": acc.get("consistency", ""),
        "deviations": acc.get("deviations", []),
        "score": int(score),
        "level": acc.get("level", ""),
        "reasoning": acc.get("reasoning", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 输入解析
# ─────────────────────────────────────────────────────────────────────────────

def parse_txt_results(filepath: str) -> List[Dict]:
    """解析 batch_query_results.txt 格式的文件"""
    items = []
    content = Path(filepath).read_text(encoding="utf-8")

    # 按分隔线切分
    blocks = re.split(r'-{20,}', content)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        question_match = re.search(r'【问题】：(.+?)(?=\n【|$)', block, re.DOTALL)
        gold_match = re.search(r'【标准答案】：(.+?)(?=\n【|$)', block, re.DOTALL)
        gen_match = re.search(r'【被评测答案】：(.+?)(?=\n（耗时|$)', block, re.DOTALL)

        if question_match and gold_match and gen_match:
            items.append({
                "question": question_match.group(1).strip(),
                "gold_answer": gold_match.group(1).strip(),
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
                    # 兼容不同字段名
                    question = item.get("question", "")
                    gold = item.get("gold_answer", item.get("reference_answer", ""))
                    gen = item.get("generated_answer", item.get("kg_answer", ""))
                    if question and gold:
                        items.append({
                            "id": item.get("id", ""),
                            "type": item.get("type", ""),
                            "question": question,
                            "gold_answer": gold,
                            "generated_answer": gen,
                        })
                except json.JSONDecodeError:
                    continue
    return items


def load_input(filepath: str) -> List[Dict]:
    """自动识别文件格式并加载"""
    if filepath.endswith(".jsonl") or filepath.endswith(".json"):
        return parse_jsonl_results(filepath)
    else:
        return parse_txt_results(filepath)


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
    """对单条进行准确率评分"""
    question = item["question"]
    gold_answer = item["gold_answer"]
    generated_answer = item.get("generated_answer", "")

    user_prompt = build_user_prompt(question, gold_answer, generated_answer)

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

    # 解析失败重试一次
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
        # 合并 usage
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
        "gold_answer": gold_answer,
        "generated_answer": generated_answer,
        "usage": usage,
    }

    if parsed:
        result.update({
            "score": parsed["score"],
            "level": parsed["level"],
            "consistency": parsed["consistency"],
            "reference_core": parsed["reference_core"],
            "generated_core": parsed["generated_core"],
            "deviations": parsed["deviations"],
            "reasoning": parsed["reasoning"],
        })
    else:
        result.update({
            "score": None,
            "level": "解析失败",
            "reasoning": "LLM 返回内容无法解析为有效 JSON",
            "raw_response": raw_response[:500],
        })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 汇总报告
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(results: List[Dict], total_time: float) -> Dict:
    """生成汇总报告"""
    valid = [r for r in results if r.get("score") is not None]

    # Token 统计
    total_prompt_tokens = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in results)
    total_completion_tokens = sum(r.get("usage", {}).get("completion_tokens", 0) for r in results)
    total_tokens = sum(r.get("usage", {}).get("total_tokens", 0) for r in results)

    # 分数统计
    scores = [r["score"] for r in valid]
    avg_score = sum(scores) / len(scores) if scores else 0

    # 等级分布
    level_dist = {}
    for r in valid:
        level = r.get("level", "未知")
        level_dist[level] = level_dist.get(level, 0) + 1

    # 分数区间分布
    score_ranges = {
        "95-100 (完美)": 0,
        "85-94 (优秀)": 0,
        "70-84 (良好)": 0,
        "55-69 (中等)": 0,
        "40-54 (较差)": 0,
        "20-39 (很差)": 0,
        "0-19 (无效)": 0,
    }
    for s in scores:
        if s >= 95:
            score_ranges["95-100 (完美)"] += 1
        elif s >= 85:
            score_ranges["85-94 (优秀)"] += 1
        elif s >= 70:
            score_ranges["70-84 (良好)"] += 1
        elif s >= 55:
            score_ranges["55-69 (中等)"] += 1
        elif s >= 40:
            score_ranges["40-54 (较差)"] += 1
        elif s >= 20:
            score_ranges["20-39 (很差)"] += 1
        else:
            score_ranges["0-19 (无效)"] += 1

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
        "bottom_5": sorted(valid, key=lambda x: x["score"])[:5],
    }


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


def print_report(report: Dict):
    """打印报告"""
    print("\n" + "=" * 70)
    print("准确率评测报告")
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

    print(f"\n--- 得分最低的 5 条 ---")
    for i, item in enumerate(report.get("bottom_5", []), 1):
        print(f"  {i}. [{item.get('id', '')}] 分数={item['score']} 等级={item.get('level', '')}")
        print(f"     问题: {item['question'][:60]}...")
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

    # 加载输入
    items = load_input(args.input)
    print(f"加载 {len(items)} 条待评测数据")

    if not items:
        print("没有数据，退出")
        return

    # 断点续跑
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
            print(f"  → 分数: {score}, 等级: {level}")

            return result

        # 逐条执行（保持顺序，便于观察进度）
        for idx_item in pending:
            result = await process(idx_item)
            all_results.append(result)

    # 生成报告
    end_time = time.time()
    total_time = end_time - (start_time if 'start_time' in dir() else end_time)

    # 如果是断点续跑完成，用文件修改时间估算
    if not pending:
        total_time = 0

    report = generate_report(all_results, total_time)
    print_report(report)

    # 写入报告文件
    report_path = args.output + ".report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n详细报告已写入: {report_path}")
    print(f"评分明细已写入: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="准确率评测打分脚本 (LLM-as-Judge)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python accuracy_judge.py --input batch_query_results.txt --output accuracy_scores.jsonl
  python accuracy_judge.py --input eval_data.jsonl --output accuracy_scores.jsonl --api-key sk-xxx
  python accuracy_judge.py --input batch_query_results.txt --output scores.jsonl --concurrency 10
        """,
    )
    parser.add_argument("--input", "-i", required=True,
                        help="输入文件（支持 .txt 和 .jsonl 格式）")
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

    # API Key
    if not args.api_key:
        args.api_key = os.environ.get("DASHSCOPE_API_KEY", "") or DEFAULT_API_KEY
    if not args.api_key:
        print("[错误] 请通过 --api-key 或环境变量 DASHSCOPE_API_KEY 提供 API Key")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"[错误] 输入文件不存在: {args.input}")
        sys.exit(1)

    print(f"配置: model={args.model}, concurrency={args.concurrency}")
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    start = time.time()
    asyncio.run(run_evaluation(args))
    elapsed = time.time() - start
    print(f"\n脚本总运行时间: {format_duration(elapsed)}")


if __name__ == "__main__":
    main()
