#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多维度加权整合脚本

读取准确率、完整度、忠实度三个维度的打分结果 JSONL，
按可配置权重计算加权综合分，输出逐条明细 + 汇总报告。

用法:
    python aggregate_scores.py \\
        --accuracy   accuracy_scores.jsonl \\
        --completeness completeness_scores.jsonl \\
        --faithfulness faithfulness_scores.jsonl \\
        --output combined_scores.jsonl

    # 自定义权重（准确率:完整度:忠实度）
    python aggregate_scores.py \\
        --accuracy a.jsonl --completeness c.jsonl --faithfulness f.jsonl \\
        --weights 0.5,0.3,0.2 --output combined.jsonl

    # 缺失某维度时自动按剩余维度重新归一化权重
    python aggregate_scores.py \\
        --accuracy a.jsonl --completeness c.jsonl \\
        --weights 0.4,0.3,0.3 --missing-mode renormalize --output combined.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# 默认配置
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = (0.4, 0.3, 0.3)  # 准确率 / 完整度 / 忠实度

# 综合分等级（基于加权后的 0-100 分数）
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
    """解析权重字符串，支持 '0.4,0.3,0.3' 或 '0.4/0.3/0.3'"""
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

    # 自动归一化
    w = tuple(x / total for x in w)
    return w


def weighted_level(score: float) -> str:
    """根据加权分数返回等级"""
    for threshold, level in COMBINED_LEVELS:
        if score >= threshold:
            return level
    return "不及格"


def format_duration(seconds: float) -> str:
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


# ─────────────────────────────────────────────────────────────────────────────
# JSONL 加载
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(filepath: str) -> Dict[int, Dict]:
    """加载 JSONL，按 index 字段建立索引"""
    result = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[警告] {filepath}:{lineno} 解析失败: {e}")
                continue
            idx = record.get("index")
            if idx is None:
                print(f"[警告] {filepath}:{lineno} 缺少 index 字段，跳过")
                continue
            result[int(idx)] = record
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 单条整合
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_one(
    index: int,
    acc_record: Optional[Dict],
    comp_record: Optional[Dict],
    faith_record: Optional[Dict],
    weights: Tuple[float, float, float],
    missing_mode: str,  # 'skip' or 'renormalize'
) -> Optional[Dict]:
    """
    整合单条记录。

    返回整合后的 dict，或在 missing_mode='skip' 时因缺分数返回 None。
    """
    # 提取公共字段（优先用准确率记录作为基准，因为它通常最全）
    base = acc_record or comp_record or faith_record
    common = {
        "index": index,
        "id": base.get("id", f"Q{index+1}"),
        "type": base.get("type", ""),
        "question": base.get("question", ""),
        "gold_answer": base.get("gold_answer", ""),
        "generated_answer": base.get("generated_answer", ""),
    }

    # 各维度分数
    scores = {
        "accuracy": acc_record.get("score") if acc_record else None,
        "completeness": comp_record.get("score") if comp_record else None,
        "faithfulness": faith_record.get("score") if faith_record else None,
    }
    levels = {
        "accuracy": acc_record.get("level", "") if acc_record else "",
        "completeness": comp_record.get("level", "") if comp_record else "",
        "faithfulness": faith_record.get("level", "") if faith_record else "",
    }
    reasonings = {
        "accuracy": acc_record.get("reasoning", "") if acc_record else "",
        "completeness": comp_record.get("reasoning", "") if comp_record else "",
        "faithfulness": faith_record.get("reasoning", "") if faith_record else "",
    }

    # 检查缺失
    missing = [k for k, v in scores.items() if v is None]
    if missing:
        if missing_mode == "skip":
            return None
        # renormalize: 只对有分数的维度加权
        present = {k: v for k, v in scores.items() if v is not None}
        w_map = dict(zip(["accuracy", "completeness", "faithfulness"], weights))
        total_w = sum(w_map[k] for k in present)
        if total_w <= 0:
            return None
        weighted = sum(scores[k] * w_map[k] / total_w for k in present)
        used_weights = {k: w_map[k] / total_w for k in present}
    else:
        weighted = sum(scores[k] * weights[i] for i, k in enumerate(
            ["accuracy", "completeness", "faithfulness"]))
        used_weights = {k: weights[i] for i, k in enumerate(
            ["accuracy", "completeness", "faithfulness"])}

    weighted = int(round(weighted))
    weighted = max(0, min(100, weighted))

    # 提取统计字段（完整度 / 忠实度的要点、陈述汇总）
    extra_stats = {}
    if comp_record:
        extra_stats["completeness_statistics"] = comp_record.get("statistics", {})
        extra_stats["missed_points"] = comp_record.get("missed_points", [])
    if faith_record:
        extra_stats["faithfulness_statistics"] = faith_record.get("statistics", {})
        extra_stats["hallucinations"] = faith_record.get("hallucinations", [])
    if acc_record:
        extra_stats["consistency"] = acc_record.get("consistency", "")
        extra_stats["deviations"] = acc_record.get("deviations", [])

    return {
        **common,
        "scores": scores,
        "levels": levels,
        "weights_used": {k: round(v, 4) for k, v in used_weights.items()},
        "missing_dimensions": missing,
        "weighted_score": weighted,
        "weighted_level": weighted_level(weighted),
        "reasonings": reasonings,
        **extra_stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 报告生成
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    combined: List[Dict],
    skipped: int,
    weights: Tuple[float, float, float],
    missing_mode: str,
) -> Dict:
    """生成汇总报告"""
    valid = [r for r in combined if r.get("weighted_score") is not None]
    scores_acc = [r["scores"]["accuracy"] for r in valid if r["scores"]["accuracy"] is not None]
    scores_comp = [r["scores"]["completeness"] for r in valid if r["scores"]["completeness"] is not None]
    scores_faith = [r["scores"]["faithfulness"] for r in valid if r["scores"]["faithfulness"] is not None]
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

    def range_dist(arr, ranges):
        d = {name: 0 for name, _, _ in ranges}
        for s in arr:
            for name, lo, hi in ranges:
                if lo <= s <= hi:
                    d[name] += 1
                    break
        return d

    weighted_ranges = [
        ("90-100 (卓越)", 90, 100),
        ("80-89  (优秀)", 80, 89),
        ("70-79  (良好)", 70, 79),
        ("60-69  (中等)", 60, 69),
        ("40-59  (及格)", 40, 59),
        ("0-39   (不及格)", 0, 39),
    ]

    return {
        "total_items": len(combined) + skipped,
        "combined_items": len(combined),
        "skipped_items": skipped,
        "missing_mode": missing_mode,
        "configured_weights": {
            "accuracy": round(weights[0], 4),
            "completeness": round(weights[1], 4),
            "faithfulness": round(weights[2], 4),
        },
        "per_dimension": {
            "accuracy": {
                "valid_count": len(scores_acc),
                **stats_for(scores_acc),
            },
            "completeness": {
                "valid_count": len(scores_comp),
                **stats_for(scores_comp),
            },
            "faithfulness": {
                "valid_count": len(scores_faith),
                **stats_for(scores_faith),
            },
        },
        "weighted_score": {
            "valid_count": len(weighted_scores),
            **stats_for(weighted_scores),
        },
        "weighted_level_distribution": level_dist(valid, "weighted_level"),
        "weighted_range_distribution": range_dist(weighted_scores, weighted_ranges),
        "per_dimension_level_distribution": {
            "accuracy": level_dist(valid, "accuracy"),
            "completeness": level_dist(valid, "completeness"),
            "faithfulness": level_dist(valid, "faithfulness"),
        },
        "bottom_5": sorted(valid, key=lambda x: x["weighted_score"])[:5],
        "top_5": sorted(valid, key=lambda x: x["weighted_score"], reverse=True)[:5],
    }


def print_report(report: Dict):
    """打印报告"""
    print("\n" + "=" * 70)
    print("多维度加权整合报告")
    print("=" * 70)

    print(f"\n总题数: {report['total_items']}, 整合成功: {report['combined_items']}, "
          f"跳过: {report['skipped_items']} (missing_mode={report['missing_mode']})")

    print(f"\n--- 配置权重 ---")
    cw = report["configured_weights"]
    print(f"  准确率:   {cw['accuracy']}")
    print(f"  完整度:   {cw['completeness']}")
    print(f"  忠实度:   {cw['faithfulness']}")

    print(f"\n--- 各维度分数统计 ---")
    for dim in ("accuracy", "completeness", "faithfulness"):
        d = report["per_dimension"][dim]
        name = {"accuracy": "准确率", "completeness": "完整度", "faithfulness": "忠实度"}[dim]
        if d["average"] is None:
            print(f"  {name}: 无有效数据")
        else:
            print(f"  {name}: 平均={d['average']}, 中位={d['median']}, "
                  f"最低={d['min']}, 最高={d['max']}, 有效={d['valid_count']}")

    print(f"\n--- 加权综合分统计 ---")
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

    print(f"\n--- 各维度等级分布 ---")
    for dim in ("accuracy", "completeness", "faithfulness"):
        name = {"accuracy": "准确率", "completeness": "完整度", "faithfulness": "忠实度"}[dim]
        print(f"  [{name}]")
        for level, count in sorted(report["per_dimension_level_distribution"][dim].items()):
            print(f"    {level}: {count}")

    print(f"\n--- 加权分最低的 5 条 ---")
    for i, item in enumerate(report.get("bottom_5", []), 1):
        s = item["scores"]
        print(f"  {i}. [{item.get('id', '')}] 加权={item['weighted_score']} "
              f"({item['weighted_level']})  "
              f"准确={s['accuracy']} 完整={s['completeness']} 忠实={s['faithfulness']}")
        print(f"     问题: {item['question'][:60]}...")

    print(f"\n--- 加权分最高的 5 条 ---")
    for i, item in enumerate(report.get("top_5", []), 1):
        s = item["scores"]
        print(f"  {i}. [{item.get('id', '')}] 加权={item['weighted_score']} "
              f"({item['weighted_level']})  "
              f"准确={s['accuracy']} 完整={s['completeness']} 忠实={s['faithfulness']}")
        print(f"     问题: {item['question'][:60]}...")

    print("\n" + "=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run_aggregate(args):
    """主流程"""
    weights = parse_weights(args.weights)
    print(f"权重: 准确率={weights[0]:.3f}, 完整度={weights[1]:.3f}, 忠实度={weights[2]:.3f}")

    # 加载三份打分结果
    inputs = {
        "accuracy": args.accuracy,
        "completeness": args.completeness,
        "faithfulness": args.faithfulness,
    }
    loaded = {}
    for dim, path in inputs.items():
        if path:
            if not os.path.exists(path):
                print(f"[错误] 文件不存在: {path}")
                sys.exit(1)
            data = load_jsonl(path)
            print(f"  [{dim}] 加载 {len(data)} 条 ← {path}")
            loaded[dim] = data
        else:
            print(f"  [{dim}] 未提供")
            loaded[dim] = {}

    if not any(loaded.values()):
        print("[错误] 至少需要提供一份打分结果")
        sys.exit(1)

    # 确定主键集合（按 accuracy > completeness > faithfulness 的优先级作为主索引）
    primary_key = None
    for k in ("accuracy", "completeness", "faithfulness"):
        if loaded[k]:
            primary_key = k
            break
    all_indices = sorted(loaded[primary_key].keys())
    print(f"\n主索引（基于 {primary_key}）: {len(all_indices)} 条")

    # 检查各维度覆盖情况
    coverage = {}
    for dim in ("accuracy", "completeness", "faithfulness"):
        hit = sum(1 for i in all_indices if i in loaded[dim])
        coverage[dim] = hit
        if hit < len(all_indices):
            print(f"  [注意] {dim} 缺失 {len(all_indices) - hit} 条")

    # 逐条整合
    combined = []
    skipped = 0
    for idx in all_indices:
        acc = loaded["accuracy"].get(idx)
        comp = loaded["completeness"].get(idx)
        faith = loaded["faithfulness"].get(idx)

        result = aggregate_one(
            index=idx,
            acc_record=acc,
            comp_record=comp,
            faith_record=faith,
            weights=weights,
            missing_mode=args.missing_mode,
        )
        if result is None:
            skipped += 1
            continue
        combined.append(result)

    print(f"\n整合: 成功 {len(combined)} 条, 跳过 {skipped} 条")

    # 写入 JSONL
    with open(args.output, "w", encoding="utf-8") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"逐条明细已写入: {args.output}")

    # 生成报告
    report = generate_report(combined, skipped, weights, args.missing_mode)
    print_report(report)

    report_path = args.output + ".report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n汇总报告已写入: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="多维度加权整合（准确率 / 完整度 / 忠实度）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认权重 0.4 / 0.3 / 0.3
  python aggregate_scores.py \\
      --accuracy accuracy_scores.jsonl \\
      --completeness completeness_scores.jsonl \\
      --faithfulness faithfulness_scores.jsonl \\
      --output combined_scores.jsonl

  # 自定义权重（准确率 0.5 / 完整度 0.3 / 忠实度 0.2）
  python aggregate_scores.py \\
      --accuracy a.jsonl --completeness c.jsonl --faithfulness f.jsonl \\
      --weights 0.5,0.3,0.2 --output combined.jsonl

  # 只跑准确率 + 完整度，按剩余维度归一化
  python aggregate_scores.py \\
      --accuracy a.jsonl --completeness c.jsonl \\
      --weights 0.4,0.3,0.3 --missing-mode renormalize --output combined.jsonl
        """,
    )
    parser.add_argument("--accuracy", "-a", default=None,
                        help="准确率打分结果 JSONL")
    parser.add_argument("--completeness", "-c", default=None,
                        help="完整度打分结果 JSONL")
    parser.add_argument("--faithfulness", "-f", default=None,
                        help="忠实度打分结果 JSONL")
    parser.add_argument("--output", "-o", required=True,
                        help="输出 JSONL 文件路径（逐条加权明细）")
    parser.add_argument("--weights", "-w", default="0.4,0.3,0.3",
                        help="权重，逗号分隔的 3 个数字（默认: 0.4,0.3,0.3；自动归一化）")
    parser.add_argument("--missing-mode", choices=["skip", "renormalize"], default="skip",
                        help="某维度缺失时的处理策略: "
                             "skip=跳过该条; renormalize=按剩余维度重新分配权重 (默认: skip)")

    args = parser.parse_args()

    # 至少提供一个输入
    if not (args.accuracy or args.completeness or args.faithfulness):
        print("[错误] 至少需要通过 --accuracy / --completeness / --faithfulness 提供一个输入文件")
        sys.exit(1)

    run_aggregate(args)


if __name__ == "__main__":
    main()
