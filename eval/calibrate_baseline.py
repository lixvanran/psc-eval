"""定版：门槛基准必须由**评测器自己的打分路径**算出来。

⑯-c 的教训：同一份数据经三条独立路径得出三个数时，不要取平均、不要取最高 ——
先找出差异来源。三条路径的差异已定位为**三个不同的测量定义**（不是脏数据）：

  0.5106 = 他们的 score() + 他们的非对称匹配（预测为分母）
  0.5397 = 评测器的 score_prediction + 评测器的 Jaccard 匹配 @ 0.34
  0.5714 = 评测器的 score_prediction + 评测器的 Jaccard 匹配 @ 0.25

**门槛比较的是「抽取器 F1 vs 基线 F1」** —— 两者必须由**同一套打分路径**算出。
抽取器 F1 来自评测器，所以基线也必须来自评测器 → 规范定义 =
**评测器自己的 scorer + 评测器自己的 matcher，并对匹配阈值取最大**。

本脚本按该定义重算闭卷基线，并导出供评测器接入。
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDSET = os.path.join(os.path.dirname(HERE), "goldset")
sys.path.insert(0, HERE)
import evaluator as E

spec = importlib.util.spec_from_file_location("lb", os.path.join(GOLDSET, "lexical_baseline.py"))
lb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lb)

GOLD_JSONL = os.path.join(GOLDSET, "gold_v1.jsonl")
OUT = os.path.join(HERE, "baseline_main_caliber.json")

MATCH_THR_GRID = sorted({round(0.20 + 0.05 * i, 2) for i in range(9)} | {0.34})
# 0.34 是评测器的默认阈值（必须进网格，否则算不出「同阈值基准」）；
# 0.25 是门槛基准点（⑳-b 同点比较）。

# 候选的 (variant, oof_thr, stance_thr) 组合 —— 取他们扫描表里出现过的点
import csv
points = set()
scan = os.path.join(GOLDSET, "baseline_scan.csv")
if os.path.exists(scan):
    with open(scan, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("mode") != "closed":
                continue
            points.add((row["variant"], float(row["oof_thr"]), float(row["stance_thr"])))

gold = [json.loads(l) for l in open(GOLD_JSONL, encoding="utf-8") if l.strip()]


def my_scorer(gold_item, preds, match_fn):
    """只算主表口径的 tp/fp/fn —— 用 _score_core 绕开「每次都算 6 个词法基线变体」的
    开销（校准脚本要跑几千次，7 倍差距在这里是分钟级的）。"""
    pred_obj = {"id": gold_item["id"], "directions": [
        {"direction": p["direction"], "in_frame": False, "status": p["status"]} for p in preds]}
    r = E._score_core(gold_item, pred_obj, match_fn=match_fn)
    return r


def aggregate_fast(results):
    tp = sum(r["tp"] for r in results)
    fp = sum(r["fp"] for r in results)
    fn = sum(r["fn"] for r in results)
    p, rec, f1 = E._f1(tp, fp, fn)
    st_checked = sum(r["status_checked"] for r in results)
    st_correct = sum(sum(r["status_confusion"][s][s] for s in E.STATUSES) for r in results)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(p, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "status_accuracy": round(st_correct / st_checked, 4) if st_checked else None,
    }


best = None
rows = []
for variant, oof_thr, stance_thr in sorted(points):
    preds_by_id = {it["id"]: lb.predict(it, variant, oof_thr, stance_thr) for it in gold}
    for match_thr in MATCH_THR_GRID:
        matcher = E.make_similarity_match_fn(match_thr)
        agg = aggregate_fast([my_scorer(it, preds_by_id[it["id"]], matcher) for it in gold])
        rec = {
            "variant": variant,
            "oof_thr": oof_thr,
            "stance_thr": stance_thr,
            "match_thr": match_thr,
            "f1": agg["f1"],
            "precision": agg["precision"],
            "recall": agg["recall"],
            "status_accuracy": agg["status_accuracy"],
            "tp": agg["tp"], "fp": agg["fp"], "fn": agg["fn"],
        }
        rows.append(rec)
        if best is None or rec["f1"] > best["f1"]:
            best = rec

# ---- ⑱-d：取高的适用范围收窄 ----
#   变体（V1-V8：不同的域外判定**方法**）-> 取最高（不同的对手）
#   参数（oof/match/stance：同一方法的**调参**）-> **不取最高**（不对等）
# 因此：**主门槛**用「与抽取器同阈值」的基线（默认 match_thr），
#       **强通过线**用「基线调参后最优」的数（作为「最努力的词法对手」参照）。
DEFAULT_MATCH_THR = 0.34  # evaluator.default_match_fn 的阈值


def pick(match_thr, variant=None, oof_thr=None, stance_thr=None):
    cands = [r for r in rows
             if abs(r["match_thr"] - match_thr) < 1e-9
             and (variant is None or r["variant"] == variant)
             and (oof_thr is None or abs(r["oof_thr"] - oof_thr) < 1e-9)
             and (stance_thr is None or abs(r["stance_thr"] - stance_thr) < 1e-9)]
    if not cands:
        return None
    return max(cands, key=lambda r: r["f1"])


at_default = pick(DEFAULT_MATCH_THR)
at_020 = pick(0.20)
max_row = max(rows, key=lambda r: r["f1"])

# 抽取器基线在默认阈值上的敏感性曲线（供报告并列）
curve = []
for thr in MATCH_THR_GRID:
    r = pick(thr)
    if r:
        curve.append({"match_thr": thr, "baseline_f1": r["f1"],
                       "precision": r["precision"], "recall": r["recall"]})


print("=" * 74)
print("门槛基准定版：评测器自己的打分路径（scorer + matcher）")
print("=" * 74)
print(f"  扫描点: {len(points)} 个 (variant, oof_thr, stance_thr) × {len(MATCH_THR_GRID)} 个匹配阈值")
print(f"  最高 F1 = {best['f1']}  （{best['variant']} oof={best['oof_thr']} stance={best['stance_thr']} match={best['match_thr']}）")
print(f"  P={best['precision']} R={best['recall']} tp={best['tp']} fp={best['fp']} fn={best['fn']}")
print()
print("  三条路径的数（都是**合法的不同测量定义**，不是脏数据）：")
print("    0.5106  他们的 score() + 他们的非对称匹配")
print("    0.5397  评测器 scorer + 评测器匹配 @0.34（评测器默认阈值）")
print("    0.5714  评测器 scorer + 评测器匹配 @0.25（他们扫描网格的最低阈值）")
print(f"    {best['f1']}    ← 规范定义：评测器 scorer + 评测器匹配，**对匹配阈值取最大**")
print()
print(f"  ⟹ 主门槛（同阈值 @0.34）= {at_default['f1']} + 0.10 = {round(at_default['f1'] + 0.10, 4)}")
print(f"  ⟹ 强通过线（最努力的词法对手 @max）= {max_row['f1']} + 0.10 = {round(max_row['f1'] + 0.10, 4)}")
print("=" * 74)

out = {
    "note": (
        "词法基线在**评测器自己的打分路径**下的得分（scorer + matcher 均为 evaluator.py 的实现）。"
        "主门槛用**与抽取器同阈值**的数（完整协议一致）；"
        "强通过线用**基线调参后最优**的数（最努力的词法对手）。"
    ),
    "caliber": "main_table（not_generated 不计入主表）",
    "definition": "evaluator.score_prediction + evaluator 默认匹配器",
    "source_gold": GOLD_JSONL,
    "default_match_thr": DEFAULT_MATCH_THR,
    # 主门槛基准：与抽取器同阈值
    "at_default_threshold_f1": at_default["f1"] if at_default else None,
    "at_default_threshold_params": {k: at_default[k] for k in ("variant", "oof_thr", "stance_thr", "match_thr")} if at_default else None,
    "recommended_gate_f1": round(at_default["f1"] + 0.10, 4) if at_default else None,
    # 强通过线基准：变体取最高、参数不取高的折中（最努力的词法对手）
    "max_f1": max_row["f1"],
    "max_params": {k: max_row[k] for k in ("variant", "oof_thr", "stance_thr", "match_thr")},
    "strong_pass_line_f1": round(max_row["f1"] + 0.10, 4),
    "at_match_020_f1": at_020["f1"] if at_020 else None,
    "precision": best["precision"],
    "recall": best["recall"],
    "status_accuracy": best["status_accuracy"],
    "tp": best["tp"], "fp": best["fp"], "fn": best["fn"],
    "sensitivity_curve": curve,
    "number_classification": {
        "0.9381": "脏（fp 逐条覆盖 bug）—— 作废",
        "0.5397": "规范口径（评测器路径 @0.34）—— **主门槛基准**",
        "0.5714": "规范口径（评测器路径 @0.20）—— 强通过线参照",
        "0.5106": "干净但用 General 的 score() —— 不作门槛（跨口径），仍是重要的变体扫描数据源",
        "note": (
            "⑱-a 分类：0.9381 / 0.5397（旧 CSV 转换）/ 0.5714（旧 CSV 转换）三个作废；"
            "0.5106 干净但跨口径、不作门槛；"
            "**0.5397 @0.34 与 0.5714 @0.20 是评测器自己的路径 = 规范口径**。"
        ),
    },
    "top_rows": sorted(rows, key=lambda r: -r["f1"])[:15],
}
with open(OUT, "w", encoding="utf-8") as handle:
    json.dump(out, handle, ensure_ascii=False, indent=2)
print(f"已导出: {OUT}")
