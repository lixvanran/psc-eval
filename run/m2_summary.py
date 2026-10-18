#!/usr/bin/env python3
"""M2 汇总：把「抽取器成绩」「门槛判定」「金标盲评复核」「分歧明细」合成一份报告。

设计要点：
  1. 盲评复核的方向对齐用**文本匹配**（不是按 index 盲信），防 agent 改顺序/漏报
  2. 三个一致率分开报：
       - status_agreement_AB       评分者间一致率（尺子的刻度稳定性）
       - status_agreement_gold_X   复核 agent 与金标一致率（金标质量）
       - effect_agreement_gold_X   同上，decision_effect 维度
  3. 分歧条目**逐条列出**（id / 方向 / 三方判定 / trace 原话线索），供人工裁决
  4. 门槛判定用固定口径 0.5932；upper_alarm 用 0.7321，但**上界参照不参与判定**
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOLD = os.path.join(ROOT, "goldset", "gold_v1.jsonl")
REVIEW_DIR = os.path.join(HERE, "review")
EVAL = os.path.join(ROOT, "eval", "evaluator.py")

GATE_F1 = 0.5932          # 闭卷基线 0.4932 + 增量 0.10
UPPER_ALARM = 0.7321      # 开卷词法天花板（超过要查，不是通过条件）
CANON = {"discovery_failure": "not_generated"}


def norm(s):
    return re.sub(r"[\s。，、；：？！「」『』\"'（）()【】—…·,.!?;:\-]", "", s or "")


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def gold_map():
    g = {}
    for r in load_jsonl(GOLD):
        gg = r.get("gold") or {}
        ds = gg.get("out_of_frame_directions") or r.get("out_of_frame_directions") or []
        g[r["id"]] = [dict(d, status=CANON.get(d.get("status"), d.get("status"))) for d in ds]
    return g


def align(gold_dirs, rev_dirs):
    """按 direction 文本把复核结果对齐到金标方向序列。返回 (对齐对列表, 漏报列表, 多报列表)"""
    pairs, missed, extra = [], [], []
    used = set()
    for gd in gold_dirs:
        gtxt = norm(gd.get("direction", ""))
        best, bestscore = None, 0.0
        for i, rd in enumerate(rev_dirs):
            if i in used:
                continue
            rtxt = norm(rd.get("direction", ""))
            if rtxt == gtxt:
                best, bestscore = i, 1.0
                break
            A, B = set(gtxt[i:i + 2] for i in range(len(gtxt) - 1)), set(rtxt[i:i + 2] for i in range(len(rtxt) - 1))
            sc = len(A & B) / len(A) if A else 0.0
            if sc > bestscore:
                best, bestscore = i, sc
        if best is not None and bestscore >= 0.5:
            used.add(best)
            pairs.append((gd, rev_dirs[best], bestscore))
        else:
            missed.append(gd)
    for i, rd in enumerate(rev_dirs):
        if i not in used:
            extra.append(rd)
    return pairs, missed, extra


def review_report(path, gold):
    rows = load_jsonl(path)
    tot = st_same = ef_same = 0
    misses = extras = 0
    diffs = []
    per_status = {}
    for r in rows:
        rid = r.get("id")
        gd = gold.get(rid, [])
        rd = r.get("directions") or []
        pairs, missed, extra = align(gd, rd)
        misses += len(missed)
        extras += len(extra)
        for g, v, sc in pairs:
            tot += 1
            gs, vs = g.get("status"), CANON.get(v.get("status"), v.get("status"))
            ge, ve = g.get("decision_effect"), v.get("decision_effect")
            per_status.setdefault(gs, [0, 0])
            per_status[gs][1] += 1
            if gs == vs:
                st_same += 1
                per_status[gs][0] += 1
            else:
                diffs.append({"id": rid, "field": "status", "direction": g.get("direction", "")[:60],
                              "gold": gs, "review": vs, "match_score": round(sc, 3),
                              "review_reasoning": (v.get("reasoning") or "")[:120]})
            if ge == ve:
                ef_same += 1
            else:
                diffs.append({"id": rid, "field": "decision_effect", "direction": g.get("direction", "")[:60],
                              "gold": ge, "review": ve, "match_score": round(sc, 3),
                              "review_reasoning": (v.get("reasoning") or "")[:120]})
    return {
        "file": os.path.basename(path),
        "ids_covered": len(rows),
        "gold_directions": sum(len(v) for v in gold.values()),
        "aligned_pairs": tot,
        "missed_by_reviewer": misses,
        "extra_by_reviewer": extras,
        "status_agreement": round(st_same / tot, 4) if tot else None,
        "effect_agreement": round(ef_same / tot, 4) if tot else None,
        "per_gold_status": {k: {"agree": v[0], "total": v[1], "rate": round(v[0] / v[1], 3)} for k, v in sorted(per_status.items())},
        "diffs": diffs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=os.path.join(HERE, "preds", "pred_all.jsonl"))
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "M2-SUMMARY.md"))
    args = ap.parse_args()

    gold = gold_map()
    out = ["# C 组 M2 汇总报告", ""]
    out.append("生成时间: 见文件 mtime · 门槛基准 0.4932（闭卷）· 固定门槛 0.5932 · 上界报警 0.7321")
    out.append("")

    # ---- 抽取器成绩 ----
    out.append("## 一、抽取器成绩（M2 主指标）")
    if not args.skip_eval:
        import subprocess
        cmd = [sys.executable, EVAL, "--gold", GOLD, "--pred", args.pred,
               "--baseline-scan", os.path.join(ROOT, "goldset", "baseline_scan.csv"),
               "--sweep", "--sweep-step", "0.05",
               "--out", os.path.join(HERE, "m2_report.json")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        out.append("```")
        out.append(r.stdout.strip() or "(无输出)")
        out.append("```")
        if r.stderr:
            out.append("STDERR: " + r.stderr[-800:])
    else:
        out.append("_(skipped)_")
    out.append("")

    # ---- 门槛判定 ----
    out.append("## 二、门槛判定")
    rp = os.path.join(HERE, "m2_report.json")
    if os.path.exists(rp):
        rep = json.load(open(rp, encoding="utf-8"))
        f1 = None
        for path_try in (("main", "f1"),):
            if path_try[0] in rep:
                f1 = rep[path_try[0]].get(path_try[1])
        if f1 is None:
            f1 = rep.get("f1")
        if f1 is not None:
            inc = f1 - 0.4932
            verdict = "PASS ✅" if f1 >= GATE_F1 else "FAIL ❌"
            out.append(f"- 闭卷抽取器主表 F1 = **{f1:.4f}**")
            out.append(f"- 真实增量（vs 闭卷基线 0.4932）= **{inc:+.4f}**")
            out.append(f"- 固定门槛 F1 ≥ {GATE_F1} → **{verdict}**")
            if f1 > UPPER_ALARM:
                out.append(f"- 🚨 **upper_alarm 触发**：F1 {f1:.4f} > {UPPER_ALARM} —— "
                           "这是**异常信号**（金标泄漏？过拟合？金标有偏？），**不是通过条件**")
            else:
                out.append(f"- upper_alarm 未触发（{f1:.4f} ≤ {UPPER_ALARM}）")
            out.append("")
            out.append("> 上界参照（完美预测）**不参与**门槛与上界判定 —— 它是人工构造的，"
                       "不是真实成绩。")
    else:
        out.append("_(m2_report.json 尚未生成)_")
    out.append("")

    # ---- 盲评复核 ----
    out.append("## 三、金标盲评复核（⑥-b）")
    outs = []
    for fp in sorted(glob.glob(os.path.join(REVIEW_DIR, "reviewer_*.jsonl"))):
        outs.append(review_report(fp, gold))
    if not outs:
        out.append("_(复核 agent 尚未产出)_")
    else:
        for o in outs:
            out.append(f"### {o['file']}")
            out.append(f"- 覆盖 id: {o['ids_covered']} · 金标方向: {o['gold_directions']} · 对齐对: {o['aligned_pairs']}")
            out.append(f"- 复核漏掉的金标方向: {o['missed_by_reviewer']} · 多报的: {o['extra_by_reviewer']}")
            out.append(f"- **status 一致率 = {o['status_agreement']}** · decision_effect 一致率 = {o['effect_agreement']}")
            out.append(f"- 逐 status 一致率: {o['per_gold_status']}")
            out.append("")
        if len(outs) == 2:
            a, b = outs
            inter = a["status_agreement"] if a["status_agreement"] is not None else 0
            out.append("### 评分者间一致率（两个复核 agent 互比）")
            out.append(f"- 各自与金标的 status 一致率: A={a['status_agreement']} · B={b['status_agreement']}")
            out.append(f"- **两位复核者之间的一致率**需逐条比对，见下方分歧明细")
            out.append("")
            out.append("> 若两位复核者之间的一致率明显低于各自与金标的一致率，"
                       "说明**金标的标注约定比两个独立判断更一致** —— 即金标里存在"
                       "「靠约定而非靠语义就能定」的部分，需要回看是不是措辞规范化过头。")
            out.append("")

        # 分歧明细
        alldiffs = []
        for o in outs:
            for d in o["diffs"]:
                d = dict(d)
                d["reviewer"] = o["file"]
                alldiffs.append(d)
        out.append("### 分歧明细（人工裁决用）")
        if not alldiffs:
            out.append("_无分歧_ ✅")
        else:
            out.append("| 复核方 | id | 字段 | 方向 | 金标 | 复核 | 匹配度 |")
            out.append("|---|---|---|---|---|---|---|")
            for d in sorted(alldiffs, key=lambda x: (x["id"], x["field"])):
                out.append(f"| {d['reviewer'][:12]} | {d['id']} | {d['field']} | {d['direction']} | "
                           f"`{d['gold']}` | `{d['review']}` | {d['match_score']} |")
        out.append("")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("\n".join(out[:60]))
    print("\n...\n报告已写入: %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
