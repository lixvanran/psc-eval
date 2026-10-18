#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""前提线索泄漏审计（premise clue leakage audit）。

用途（对应 ㉔-c 的方法论杠杆）：
> 「让隐含前提在题面上留下可推的线索」是让某类能力**可测**的手段。
> 但要证明"抽取器利用了题面线索"，必须先确认：
> **该前提的线索只出现在题面、而没有出现在 trace 里。**
> 否则把线索从题面删掉（低线索版 A）也无效——trace 自己会把它讲出来。

对每道题输出：
  task_clue   = 题面各句与前提的最大二元组重叠（题面是否留了线索）
  trace_clue  = trace 各句与前提的最大二元组重叠（trace 是否把线索讲出来了）
  判定：
    ★ 线索只在题面          → 可做干净的 A/B（前提线索显隐度受控实验）
      线索也在 trace        → 改题面无效（低线索版会从 trace 泄漏）
      题面无线索            → 该前提只能来自标注者（gold_author 类），闭卷不可推

用法：
    python3 source/leak_audit.py                # 逐题表
    python3 source/leak_audit.py --json         # 机器可读
    python3 source/leak_audit.py --only G25     # 只看某题（含泄漏句原文）
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PREMISES = os.path.join(ROOT, "premises.jsonl")
GOLD = os.path.join(ROOT, "gold_v1.jsonl")

TASK_CLUE_MIN = 0.25   # 题面"留了线索"的阈值
TRACE_CLUE_MAX = 0.15  # trace"没把线索讲出来"的阈值


def bg(t):
    t = re.sub(r"[^一-鿿A-Za-z0-9]", "", t)
    return {t[i:i + 2] for i in range(len(t) - 1)} or {t}


def ov(a, b):
    A, B = bg(a), bg(b)
    return len(A & B) / max(1, len(A))


def sents(text):
    return [s for s in re.split(r"[。；\n]", text) if s.strip()]


def audit(iid, premise, task, trace):
    t_sents = sents(task)
    tr_sents = [s for seg in trace for s in sents(seg)]
    t_best = max(((ov(s, premise), s) for s in t_sents), default=(0.0, ""))
    tr_best = max(((ov(s, premise), s) for s in tr_sents), default=(0.0, ""))
    t_ov, t_s = t_best
    tr_ov, tr_s = tr_best
    if t_ov >= TASK_CLUE_MIN and tr_ov < TRACE_CLUE_MAX:
        verdict = "★ 线索只在题面 → 干净的 A/B 候选"
        clean = True
    elif t_ov >= TASK_CLUE_MIN:
        verdict = "线索也在 trace → 改题面无效（低线索版会从 trace 泄漏）"
        clean = False
    else:
        verdict = "题面无线索（只能来自标注者，闭卷不可推）"
        clean = False
    return {"id": iid, "task_clue": round(t_ov, 3), "trace_clue": round(tr_ov, 3),
            "verdict": verdict, "clean_ab_candidate": clean,
            "task_sentence": t_s[:160], "trace_sentence": tr_s[:160]}


def main():
    args = sys.argv[1:]
    prems = {json.loads(l)["id"]: json.loads(l) for l in open(PREMISES, encoding="utf-8") if l.strip()}
    items = {json.loads(l)["id"]: json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()}
    only = args[args.index("--only") + 1] if "--only" in args else None

    rows = []
    for iid, p in prems.items():
        it = items[iid]
        r = audit(iid, p["premise"], it["task"], it["trace"])
        r["premise"] = p["premise"]
        r["premise_kind"] = p["premise_kind"]
        r["expected_source"] = next((d.get("expected_source")
                                      for d in it["gold"]["out_of_frame_directions"]
                                      if d["status"] == "discovery_failure"), None)
        rows.append(r)

    if "--json" in args:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0

    if only:
        for r in rows:
            if r["id"] == only:
                print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0

    print("=" * 100)
    print("前提线索泄漏审计：题面留了线索吗？trace 讲出来了吗？")
    print("=" * 100)
    print(f"{'id':5} {'题面线索':>8} {'trace线索':>9}  判定")
    for r in rows:
        print(f"{r['id']:5} {r['task_clue']:8.3f} {r['trace_clue']:9.3f}  {r['verdict']}")
    clean = [r["id"] for r in rows if r["clean_ab_candidate"]]
    leaked = [r["id"] for r in rows
              if r["task_clue"] >= TASK_CLUE_MIN and not r["clean_ab_candidate"]]
    print(f"\n干净的 A/B 候选（线索只在题面）: {clean or '无'}")
    print(f"改题面无效（线索也在 trace）    : {leaked or '无'}")
    print("\n判读注意：trace_clue 高说明**trace 自己把前提讲出来了**——")
    print("这种题上删掉题面线索不会有任何变化（低线索版会从 trace 泄漏），")
    print("A/B 会得到无法解释的零差异（是'线索不重要'还是'实验坏了'？分不清）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
