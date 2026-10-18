#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""增量来自哪个维度？—— 域外判定 vs status 判定（诊断量，不改门槛）

**背景**：词法基线把两个维度混在一个 F1 里，看起来只有一个数。
但读金标构造方的 `lexical_baseline.py` 的 `score()` 会发现：
**status 根本不进 tp/fp/fn** —— 匹配上就是 tp，匹配不上就是 fn/fp。
所以他们的 F1 纯粹是「**域外判定**」这一维的能力，而 `status_acc` 是
「**status 判定**」这一维的能力。**两个维度被分开测量了。**

本脚本做三件事：

1. **口径对齐**：金标构造方的 F1 用的是「含 miss detection」口径
   （所有金标域外方向都在分母里），而主表口径（裁决 ②-b）把 `not_generated`
   单独拆出。两者的 F1 **不可直接比较** —— 而门槛必须同口径。
   本脚本把同一批预测用**主表口径**重算，给出门槛该用的数。
2. **维度分解**：每个变体报出
   - 域外判定维度：F1（与 status 无关）
   - status 维度：status_acc
   由此读出「词法在哪一步弱」。
3. **oracle 探针**：把 status 换成金标的（oracle_status），
   看 F1 能抬到多少 —— 直接量化「status 这一步的语义价值」。

只读金标，不修改任何东西。**这是诊断量，不改变门槛。**

用法：
    python3 eval/diagnose_increment_source.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDSET = os.path.join(os.path.dirname(HERE), "goldset")
GOLD_JSONL = os.path.join(GOLDSET, "gold_v1.jsonl")
SCAN_CSV = os.path.join(GOLDSET, "baseline_scan.csv")
BASELINE_PY = os.path.join(GOLDSET, "lexical_baseline.py")

sys.path.insert(0, HERE)
import evaluator as E  # noqa: E402


def load_baseline_module():
    spec = importlib.util.spec_from_file_location("goldset_lexical_baseline", BASELINE_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def best_points() -> dict[str, dict]:
    """从扫描表里取每个闭卷变体的最高 F1 点。"""
    best: dict[str, dict] = {}
    if not os.path.exists(SCAN_CSV):
        return best
    with open(SCAN_CSV, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("mode") != "closed":
                continue
            variant = row.get("variant", "")
            f1 = float(row.get("f1", 0) or 0)
            if variant not in best or f1 > best[variant]["f1"]:
                best[variant] = {
                    "f1": f1,
                    "oof_thr": float(row["oof_thr"]),
                    "match_thr": float(row["match_thr"]),
                    "stance_thr": float(row["stance_thr"]),
                    "status_acc": float(row.get("status_acc") or 0),
                }
    return best


def score_in_main_caliber(gold_items, preds_by_id, match_thr: float) -> dict:
    """用**主表口径**（裁决 ②-b：not_generated 不进主表）给一批预测打分。

    这是门槛该用的口径 —— 与抽取器的主表 F1 同口径。
    """
    matcher = E.make_similarity_match_fn(match_thr)
    results = []
    for item in gold_items:
        preds = preds_by_id.get(item["id"], [])
        pred_obj = {
            "id": item["id"],
            "directions": [
                {"direction": p["direction"], "in_frame": False, "status": p["status"]}
                for p in preds
            ],
        }
        results.append(E.score_prediction(item, pred_obj, match_fn=matcher))
    agg = E.aggregate(results)
    return {
        "f1": agg["f1"],
        "precision": agg["precision"],
        "recall": agg["recall"],
        "tp": agg["tp"],
        "fp": agg["fp"],
        "fn": agg["fn"],
        "status_accuracy": agg["status_accuracy"],
        "miss_detection_recall": agg["miss_detection"]["miss_detection_recall"],
    }


def score_with_oracle_status(gold_items, preds_by_id, match_thr: float) -> dict:
    """oracle_status：把**匹配上的**预测的 status 换成金标的，其余不动。

    F1 能抬到多少 = 「status 这一步的语义价值」的上界。
    """
    matcher = E.make_similarity_match_fn(match_thr)
    results = []
    for item in gold_items:
        golds = item["gold"]["out_of_frame_directions"]
        preds = preds_by_id.get(item["id"], [])
        used = set()
        fixed = []
        for p in preds:
            replacement = dict(p)
            best_gold, best_score = None, 0.0
            for g in golds:
                score = E.similarity(p["direction"], g["direction"])
                if score > best_score:
                    best_gold, best_score = g, score
            if best_gold is not None and best_score >= match_thr:
                replacement["status"] = E._dir_status(best_gold)
            fixed.append(replacement)
        pred_obj = {
            "id": item["id"],
            "directions": [
                {"direction": p["direction"], "in_frame": False, "status": p["status"]}
                for p in fixed
            ],
        }
        results.append(E.score_prediction(item, pred_obj, match_fn=matcher))
    agg = E.aggregate(results)
    return {"f1": agg["f1"], "precision": agg["precision"], "recall": agg["recall"]}


def main() -> int:
    if not (os.path.exists(GOLD_JSONL) and os.path.exists(BASELINE_PY)):
        print("找不到金标集或词法基线脚本，跳过（诊断量，不影响评测器）")
        return 0

    baseline = load_baseline_module()
    gold_items = [json.loads(line) for line in open(GOLD_JSONL, encoding="utf-8") if line.strip()]
    best = best_points()

    print("=" * 78)
    print("增量来源分解：域外判定 vs status 判定（闭卷口径）")
    print("=" * 78)
    print("读金标构造方 score() 的实现：status 根本不进 tp/fp/fn ——")
    print("  匹配上 = tp，匹配不上 = fn/fp。所以他们的 F1 纯粹是「域外判定」维度，")
    print("  status_acc 是「status 判定」维度。**两个维度被分开测量了。**")
    print()
    print(f"{'变体':<6}{'词法F1(域外维度)':<18}{'status_acc':<14}"
          f"{'主表口径F1':<14}{'oracle_status F1':<18}{'status维度还差':<14}")
    print("-" * 78)

    rows = []
    for variant in sorted(best, key=lambda v: -best[v]["f1"]):
        point = best[variant]
        oof_thr, match_thr, stance_thr = point["oof_thr"], point["match_thr"], point["stance_thr"]
        preds_by_id = {
            item["id"]: baseline.predict(item, variant, oof_thr, stance_thr)
            for item in gold_items
        }
        main_cal = score_in_main_caliber(gold_items, preds_by_id, match_thr)
        oracle = score_with_oracle_status(gold_items, preds_by_id, match_thr)
        headroom = oracle["f1"] - main_cal["f1"]
        rows.append(
            {
                "variant": variant,
                "lexical_f1_all_caliber": point["f1"],
                "status_acc": point["status_acc"],
                "main_caliber_f1": main_cal["f1"],
                "oracle_status_f1": oracle["f1"],
                "status_headroom": round(headroom, 4),
                "oof_thr": oof_thr,
                "match_thr": match_thr,
                "stance_thr": stance_thr,
            }
        )
        print(
            f"{variant:<6}{point['f1']:<18.4f}{point['status_acc']:<14.4f}"
            f"{main_cal['f1']:<14.4f}{oracle['f1']:<18.4f}{headroom:<+14.4f}"
        )

    if not rows:
        print("没有可用的扫描点")
        return 0

    lex_max = max(r["lexical_f1_all_caliber"] for r in rows)
    main_max = max(r["main_caliber_f1"] for r in rows)
    best_status = max(r["status_acc"] for r in rows)
    best_row = max(rows, key=lambda r: r["main_caliber_f1"])

    print("-" * 78)
    print("读法（对应主会话那张判读表）：")
    print(f"  · 域外判定维度（词法）：最高 F1 = {lex_max:.4f}（status 不进这个 F1）")
    print(f"  · status 维度（词法）：最高 status_acc = {best_status:.4f}")
    print(f"  · 主表口径（同抽取器可比）的词法基线最高 F1 = {main_max:.4f}"
          f"（变体 {best_row['variant']}）")
    print()
    print("  ★ 口径提醒：金标构造方的 F1 用的是「含 miss detection」口径，")
    print("    而主表口径（裁决 ②-b）把 not_generated 单独拆出 —— **两者不可直接比较**。")
    print(f"    门槛若要同口径，应取主表口径的 {main_max:.4f}，而不是 {lex_max:.4f}。")
    print()
    if best_status >= main_max - 0.10:
        print("  → 词法在 status 这步已能到较高水平，增量必须主要来自**域外判定**。")
        print("    工程含义：优先把域外判定做对（语义判断「这个动作是不是常规解」），")
        print("    status 可以更多交给词法。")
    else:
        print("  → 词法在 status 这步也不灵，增量来自**全面语义**。")
        print("    工程含义：整条链都需要语义模型，不能靠局部替换省掉。")
    print()
    print("  · oracle_status 一列 = 把 status 换成金标后 F1 能抬到多少，")
    print("    即「status 这一步的语义价值」的上界。")
    print()
    print("  ⚠ 这几列全为 0 不是 bug，是结构性的事实：")
    print("    词法基线**从不产出 not_generated**（它只在 trace 里找得到东西，")
    print("    找不到就不报）。而主表口径里，status 只通过「not_generated 与主表三类")
    print("    的那条边界」影响 F1 —— realized/selection/expression 之间的互换不改变")
    print("    tp/fp/fn。")
    print("    ⟹ **主表 F1 真正在意的是「这个方向到底有没有被生成」，")
    print("       而不是「它死在选择还是死在表达」。**")
    print("       三分类是**诊断维度**，不是计分维度 —— 这从数据上印证了裁决 ⑪-c。")
    print("=" * 78)

    # --- 导出主表口径的基线，供评测器做同口径门槛 ---
    out = {
        "note": (
            "词法基线在**主表口径**（裁决 ②-b：not_generated 不进主表）下的得分。"
            "与抽取器的主表 F1 **同口径**，门槛必须用这一份，不能用含 miss detection 的口径。"
        ),
        "caliber": "main_table（not_generated 不计入主表）",
        "source_scan": SCAN_CSV,
        "source_gold": GOLD_JSONL,
        "variants": rows,
        "max_f1": main_max,
        "max_variant": best_row["variant"],
        "all_directions_caliber_max_f1": lex_max,
        "recommended_gate_f1": round(main_max + 0.10, 4),
    }
    out_path = os.path.join(HERE, "baseline_main_caliber.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=2)
    print(f"已导出主表口径基线: {out_path}")
    print(f"  主表口径最高 F1 = {main_max:.4f}  →  同口径门槛 = {main_max + 0.10:.4f}")
    print(f"  （含 miss detection 口径的 {lex_max:.4f} 不可用于门槛）")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
