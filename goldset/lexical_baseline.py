#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""词法捷径基线（lexical_baseline）—— 开卷对照 / 闭卷主指标。

用途（裁决 ⑦-a / ⑨-c / ⑨-b）：
  真实增量 = 抽取器 F1 − 词法基线 F1      （M2 硬门槛 ≥ 0.10）
  门槛基准 = **所有基线变体 × 所有阈值点的最高 F1**（阈值可调 → 门槛可被放水，必须取高）

★ 开卷 vs 闭卷（裁决 ⑨）：
  开卷 = initial_frame 里带「域外参考」段（等于把答案印在题面上）
         → 域外判定可以对着答案做字串比对，是**抽取器复述能力**的上限，只作对照、不进门槛
  闭卷 = initial_frame 只有四段式（【候选集】【封闭性】【评估维度】【硬约束】）
         → 域外判定只能靠「方向不在【候选集】里」这一条，是**真实域外辨识能力**，主指标

变体（全部不依赖语义模型）：
  V1 closed-bigram  闭卷：域外 iff 候选文本与【候选集】的二元组重叠 < thr；status 取最终答案 stance 词
  V2 closed-bigram-阶段4  同上，但 status 锚在取舍段的否决词（旧锚点）
  V3 closed-jaccard 闭卷：域外 iff Jaccard(候选, 候选集) < thr
  V4 closed-anymatch 闭卷：域外 iff 候选文本中没有任何 3-gram 出现在【候选集】里
  V5 open-bigram    开卷：域外 iff 候选文本与「域外参考」的二元组重叠 ≥ thr（对照组，泄漏答案）
  V6 closed-graded  闭卷：同 V1，但 status 用「与含 stance 词的最终答案句子的重叠 ≥ s_thr」判定

用法：
    python3 lexical_baseline.py                 # 报告 + 扫描（闭卷为主、开卷对照）
    python3 lexical_baseline.py --json          # 机器可读
    python3 lexical_baseline.py --scan-out baseline_scan.csv
"""
import csv
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "gold_v1.jsonl")
OPENBOOK = os.path.join(HERE, "gold_v1_openbook.jsonl")

NEG_WORDS = ("否决", "不采纳", "不执行", "不列入", "排除", "放弃", "删掉", "明确不", "不做", "不启动")
POS_WORDS = ("建议", "请", "推荐", "采用", "纳入", "优先", "执行", "启动", "立项",
             "打包", "试", "保留", "收紧", "改写", "做")
MARKERS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
MARK_RE = re.compile(f"([{MARKERS}])")
STRIP_RE = re.compile(r"[\s，。；：、（）()「」『』“”\"'《》【】…—\-0-9A-Za-z%]")


def bigrams(s):
    s = STRIP_RE.sub("", s)
    return {s[i:i + 2] for i in range(len(s) - 1)} or {s}


def trigrams(s):
    s = STRIP_RE.sub("", s)
    return {s[i:i + 3] for i in range(len(s) - 2)} or {s}


_BG_CACHE = {}


def bg_cached(s):
    """bigrams 缓存（匹配循环里同一对文本会被反复比较）。"""
    v = _BG_CACHE.get(s)
    if v is None:
        v = bigrams(s)
        if len(_BG_CACHE) < 200000:
            _BG_CACHE[s] = v
    return v


def overlap(a, b):
    A, B = bg_cached(a), bg_cached(b)
    return len(A & B) / max(1, len(A))


def jaccard(a, b):
    A, B = bigrams(a), bigrams(b)
    return len(A & B) / max(1, len(A | B))


def parse_candidates(trace):
    seg = next((t for t in trace if t.startswith("阶段3")), "")
    body = seg.split("：", 1)[1] if "：" in seg else seg
    out = []
    for chunk in body.split("；"):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = MARK_RE.search(chunk)
        if m:
            out.append((m.group(1), chunk[m.end():].strip()))
    return out


def frame_part(frame, key):
    for line in frame.split("\n"):
        if line.startswith(key):
            return line.split("】", 1)[1].strip() if "】" in line else ""
    return ""


def ref_in_frame(frame):
    """「域外参考」只允许从 initial_frame 里取。

    闭卷文件里它已被移出 initial_frame（进 frame_note）→ 闭卷下作弊变体无答案可偷，
    自然退化为 0 预测；开卷文件里它还在 initial_frame → 作对照组正常工作。
    """
    for line in frame.split("\n"):
        if line.startswith("域外参考"):
            return line.split("：", 1)[1].strip() if "：" in line else ""
    return ""


def stance_in_final(marker, final):
    for sent in re.split(r"[。；]", final):
        if marker not in sent:
            continue
        if any(w in sent for w in NEG_WORDS):
            return "selection_failure"
        if any(w in sent for w in POS_WORDS):
            return "realized"
    return None


def stance_graded(marker, text, final, thr):
    """V6：看候选文本与「含 stance 词、且点名该编号」的最终答案句子的重叠是否 ≥ thr。"""
    best, stance = 0.0, None
    for sent in re.split(r"[。；]", final):
        if marker not in sent:
            continue
        if any(w in sent for w in NEG_WORDS):
            s = "selection_failure"
        elif any(w in sent for w in POS_WORDS):
            s = "realized"
        else:
            continue
        o = overlap(text, sent)
        if o > best:
            best, stance = o, s
    if stance is None or best < thr:
        return "expression_failure"
    return stance


def predict(item, variant, oof_thr, stance_thr):
    trace = item["trace"]
    cands = parse_candidates(trace)
    dec = next((t for t in trace if t.startswith("阶段4")), "")
    final = trace[-1]
    cand_set = frame_part(item["initial_frame"], "【候选集】")
    ref = ref_in_frame(item["initial_frame"])   # 只读 initial_frame，不读金标专有字段
    preds = []
    for marker, text in cands:
        if variant == "V5":
            oof = overlap(text, ref) >= oof_thr
        elif variant == "V3":
            oof = jaccard(text, cand_set) < oof_thr
        elif variant == "V4":
            cs = STRIP_RE.sub("", cand_set)
            oof = not any(t in cs for t in trigrams(text))
        elif variant == "V7":
            ref7 = cand_set + " " + frame_part(item["initial_frame"], "【硬约束】")
            oof = overlap(text, ref7) < oof_thr
        elif variant == "V8":
            ref8 = cand_set + " " + item["task"]
            oof = overlap(text, ref8) < oof_thr
        else:                                     # V1 / V2 / V6
            oof = overlap(text, cand_set) < oof_thr
        if not oof:
            continue
        if variant == "V2":
            clause = ""
            for part in re.split(r"[；。]", dec):
                if marker in part:
                    clause = part
                    break
            st = "selection_failure" if any(w in clause for w in NEG_WORDS) else (
                "realized" if overlap(text, final) >= 0.30 else "expression_failure")
        elif variant == "V6":
            st = stance_graded(marker, text, final, stance_thr)
        else:
            st = stance_in_final(marker, final) or "expression_failure"
        preds.append({"direction": text, "status": st})
    return preds


NG = "discovery_failure"   # 裁决 ②-b：not_generated 单列，不进主表


def score(items, variant, oof_thr, match_thr, stance_thr=0.30):
    """同时算两个口径（关键：门槛必须与抽取器主表同口径，否则门槛被放松）：

    main 主表口径（**门槛基准用这个**）：分母 = 非 not_generated 的金标方向；
         匹配上 not_generated 金标的预测不进主表（它属于漏判表的误报）
    full 含漏判口径：分母 = 全部金标方向（含 5 条 not_generated），历史口径，仅供对照
    """
    tp = fp = fn = 0
    tp_f = fp_f = fn_f = 0
    st_correct = st_total = 0
    miss_tp = miss_fn = 0
    n_pred_total = 0
    n_gold_total = 0
    conf = Counter()
    for it in items:
        golds = it["gold"]["out_of_frame_directions"]
        # 裁决 ⑧-c（与评测器一致）：零方向条目（正确答案是「没有域外方向」）**不计入主表**，
        # 必须在**池层面**清空——否则它的预测会被算成主表 FP，基线就与抽取器不同口径。
        if not golds:
            continue
        preds = predict(it, variant, oof_thr, stance_thr)
        used = set()
        matched = {}
        for gi, g in enumerate(golds):
            hit = None
            for i, p in enumerate(preds):
                if i not in used and overlap(p["direction"], g["direction"]) >= match_thr:
                    hit = i
                    break
            if hit is not None:
                used.add(hit)
                matched[gi] = hit
        for gi, g in enumerate(golds):
            st_total += 1
            if gi not in matched:
                conf[(g["status"], "漏报")] += 1
                continue
            p = preds[matched[gi]]
            if p["status"] == g["status"]:
                st_correct += 1
                conf[(g["status"], "ok")] += 1
            else:
                conf[(g["status"], f"->{p['status']}")] += 1
        ng_matched = [gi for gi in matched if golds[gi]["status"] == NG]
        nonng_matched = [gi for gi in matched if golds[gi]["status"] != NG]
        # --- 主表口径 ---
        for gi, g in enumerate(golds):
            if g["status"] == NG:
                if gi in matched:
                    miss_tp += 1
                else:
                    miss_fn += 1
                continue
            if gi in matched:
                tp += 1
            else:
                fn += 1
        fp += len(preds) - len(nonng_matched) - len(ng_matched)
        n_pred_total += len(preds)
        n_gold_total += len(golds)
        # --- 含漏判口径 ---
        tp_f += len(matched)
        fn_f += len(golds) - len(matched)
        fp_f += len(preds) - len(matched)
    # --- 口径自检（防止"逐条覆盖"这类 bug 再发生；用累加值，不重算预测）---
    n_nonng = sum(1 for it in items for d in it["gold"]["out_of_frame_directions"]
                  if d["status"] != NG)   # 零方向条目在上面的 continue 里已整条跳过
    assert tp + fn == n_nonng, f"主表分母不对: tp+fn={tp+fn} 应={n_nonng}"
    assert fp + tp_f == n_pred_total, f"FP 口径不对: fp+matched={fp+tp_f} 应={n_pred_total}"
    assert tp_f + fn_f == n_gold_total, "含漏判分母不对"

    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-9, p + r)
    pf = tp_f / max(1, tp_f + fp_f)
    rf = tp_f / max(1, tp_f + fn_f)
    f1f = 2 * pf * rf / max(1e-9, pf + rf)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f1, 4), "caliber": "main(不含 not_generated，与抽取器主表同口径)",
            "f1_full": round(f1f, 4), "precision_full": round(pf, 4), "recall_full": round(rf, 4),
            "tp_full": tp_f, "fp_full": fp_f, "fn_full": fn_f,
            "miss_detection_tp": miss_tp, "miss_detection_fn": miss_fn,
            "miss_detection_recall": round(miss_tp / max(1, miss_tp + miss_fn), 4),
            "status_acc": round(st_correct / max(1, st_total), 4),
            "confusion": {f"{k[0]}|{k[1]}": v for k, v in conf.items()}}


CLOSED_THR = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30)
OPEN_THR = (0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26)
MATCH_THR = (0.25, 0.35, 0.45, 0.55)
STANCE_THR = (0.20, 0.30, 0.40)
VARIANTS = ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8")


def scan(items, label):
    rows = []
    for v in VARIANTS:
        thrs = OPEN_THR if v == "V5" else CLOSED_THR
        for oof in thrs:
            for mt in MATCH_THR:
                sts = STANCE_THR if v == "V6" else (0.30,)
                for st_thr in sts:
                    r = score(items, v, oof, mt, st_thr)
                    rows.append({"mode": label, "variant": v, "oof_thr": oof, "match_thr": mt,
                                 "stance_thr": st_thr,
                                 "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
                                 "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
                                 "f1_full": r["f1_full"], "precision_full": r["precision_full"],
                                 "recall_full": r["recall_full"],
                                 "miss_detection_recall": r["miss_detection_recall"],
                                 "status_acc": r["status_acc"]})
    return rows


def main():
    json_out = "--json" in sys.argv
    scan_out = None
    if "--scan-out" in sys.argv:
        scan_out = sys.argv[sys.argv.index("--scan-out") + 1]
    closed = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]
    openb = [json.loads(l) for l in open(OPENBOOK, encoding="utf-8") if l.strip()] \
        if os.path.exists(OPENBOOK) else []

    rows_c = scan(closed, "closed")
    rows_o = scan(openb, "open") if openb else []
    allrows = rows_c + rows_o

    if scan_out:
        with open(scan_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
            w.writeheader()
            w.writerows(allrows)
        print(f"扫描表已写出：{scan_out}（{len(allrows)} 行）", file=sys.stderr)  # 不污染 --json 的 stdout

    best_c = max(rows_c, key=lambda r: r["f1"])            # 主表口径（门槛基准）
    # 同点比较：抽取器必须在**取得该基线数的同一个匹配阈值**上被评分（⑳-b）
    same_point = {"variant": best_c["variant"], "oof_thr": best_c["oof_thr"],
                  "match_thr": best_c["match_thr"], "stance_thr": best_c["stance_thr"],
                  "f1": best_c["f1"],
                  "rule": "抽取器必须在同一 match_thr 上被评分并与之比较；"
                          "禁止抽取器在多阈值上取最优再与门槛比（那是不对等比较）"}
    # 基线在匹配阈值上的曲线（门槛的参照系；⑳-b 要求同时报出）
    curve = []
    for mt in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        vals = [float(r["f1"]) for r in rows_c
                if r["variant"] == best_c["variant"] and float(r["oof_thr"]) == float(best_c["oof_thr"])
                and float(r["match_thr"]) == mt]
        if vals:
            curve.append({"match_thr": mt, "f1": max(vals)})
    curve_spread = (max(c["f1"] for c in curve) - min(c["f1"] for c in curve)) if curve else 0.0
    best_c_full = max(rows_c, key=lambda r: r["f1_full"])  # 含漏判口径（仅对照）
    best_o = max(rows_o, key=lambda r: r["f1"]) if rows_o else None
    best_o_full = max(rows_o, key=lambda r: r["f1_full"]) if rows_o else None

    if json_out:
        print(json.dumps({
            "closed_best": best_c, "open_best": best_o,
            "n_scan_points": len(allrows),
            "closed_points": len(rows_c), "open_points": len(rows_o),
            # ↓ 评测器若优先采纳 sweep_max_f1，它现在明确是**闭卷**最高值（门槛基准）
            "sweep_max_f1": best_c["f1"],
            "sweep_max_f1_mode": "closed_book",
            "same_point": same_point,
            "match_threshold_curve": curve,
            "match_threshold_spread": round(curve_spread, 4),
            "closed_sweep_min": round(min(float(r["f1"]) for r in rows_c), 4),
            "closed_sweep_max": round(max(float(r["f1"]) for r in rows_c), 4),
            "closed_sweep_spread": round(max(float(r["f1"]) for r in rows_c)
                                         - min(float(r["f1"]) for r in rows_c), 4),
            "closed_sweep_spread_nondegenerate": round(
                max(float(r["f1"]) for r in rows_c if float(r["f1"]) > 0)
                - min(float(r["f1"]) for r in rows_c if float(r["f1"]) > 0), 4),
            "sweep_max_f1_caliber": "main（与抽取器主表同口径，不含 not_generated）",
            "sweep_max_f1_full_caliber": best_c_full["f1_full"],
            "sweep_max_f1_full_caliber_note": "含漏判口径（分母含 5 条 not_generated）——与主表口径不可比，仅供对照",
            "open_max_f1": best_o["f1"] if best_o else None,
            "open_max_f1_mode": "open_book（仅对照，不进门槛）",
            "m2_gate": {
                "rule": "真实增量 = 抽取器主表 F1 − max(所有闭卷基线变体 × 所有阈值点) ≥ 0.10",
                "closed_max_f1": best_c["f1"],
                "closed_max_f1_caliber": "main（与抽取器主表同口径）",
                "required_extractor_f1": round(best_c["f1"] + 0.10, 4),
                "note": "门槛只认闭卷（无泄漏）基线；开卷最大值仅作对照，"
                        "若被误用为门槛基准会人为抬高门槛并让『泄漏答案』变成隐形优势",
            },
        }, ensure_ascii=False, indent=2))
        return 0

    print("=" * 74)
    print("词法捷径基线：闭卷（主指标） vs 开卷（对照）")
    print("=" * 74)
    print(f"扫描点：闭卷 {len(rows_c)} / 开卷 {len(rows_o)}（变体 {len(VARIANTS)} × 阈值网格）\n")
    print("-- 闭卷（域外判定只靠【候选集】：方向不在候选集内 = 域外）--")
    for r in sorted(rows_c, key=lambda r: -r["f1"])[:8]:
        print(f"  {r['variant']} oof={r['oof_thr']:<5} match={r['match_thr']:<5} stance={r['stance_thr']:<5}"
              f" P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} status_acc={r['status_acc']:.3f}")
    if best_o:
        print("\n-- 开卷（域外判定对着「域外参考」字串比对 = 泄漏答案）--")
        for r in sorted(rows_o, key=lambda r: -r["f1"])[:4]:
            print(f"  {r['variant']} oof={r['oof_thr']:<5} match={r['match_thr']:<5}"
                  f" P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} status_acc={r['status_acc']:.3f}")
    print(f"\n→ 闭卷基线最高 F1 = {best_c['f1']:.4f}【主表口径，与抽取器同口径】"
          f"（{best_c['variant']} oof={best_c['oof_thr']} match={best_c['match_thr']}）")
    print(f"   同口径含漏判变体最高 F1 = {best_c_full['f1_full']:.4f}（分母含 5 条 not_generated，与主表不可比）")
    if best_o:
        print(f"→ 开卷基线最高 F1 = {best_o['f1']:.4f}【主表口径】（对照，不进门槛）")
        print(f"   开卷含漏判变体最高 F1 = {best_o_full['f1_full']:.4f}")
    print(f"→ M2 门槛：真实增量 = 抽取器主表 F1 − {best_c['f1']:.4f} ≥ 0.10 "
          f"⟹ 抽取器主表 F1 ≥ {best_c['f1'] + 0.10:.4f}")
    print(f"→ 同点比较（⑳-b）：抽取器必须在 match_thr={best_c['match_thr']} 上被评分"
          f"（变体 {best_c['variant']} oof={best_c['oof_thr']}）")
    print("   基线在匹配阈值上的曲线：" +
          "  ".join(f"{c['match_thr']}→{c['f1']:.4f}" for c in curve) +
          f"   跨度 {curve_spread:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
