#!/usr/bin/env python3
"""生成金标盲评复核材料：只给「方向描述」，不给 status / decision_effect / evidence / notes / frame_note。

两个独立复核 agent 各读同一份材料、各自独立判断 status 与 decision_effect，
然后与金标比对 → 同时得到：
  ① ⑥-b 人工复核（强 effect 100% + status 抽样 30%）
  ② 评分者一致性（两个复核 agent 之间的分歧率）
  ③ 金标质量的独立估计（复核 agent 与金标的一致率）
"""
from __future__ import annotations

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOLD = os.path.join(ROOT, "goldset", "gold_v1.jsonl")
OUT = os.path.join(HERE, "review", "materials")


def dirs(row):
    g = row.get("gold") or {}
    return g.get("out_of_frame_directions") or row.get("out_of_frame_directions") or []


def segs(row):
    t = row.get("trace") or []
    if not isinstance(t, list):
        return [str(t)]
    return [(x.get("content", "") if isinstance(x, dict) else str(x)) for x in t]


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    rows = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]
    n_dir = 0
    for r in rows:
        rid = r["id"]
        S = segs(r)
        final = S[-1] if S else ""
        mids = S[:-1] if len(S) > 1 else []

        lines = []
        lines.append("【TASK】")
        lines.append(r.get("task", ""))
        lines.append("")
        lines.append("【INITIAL_FRAME】")
        lines.append(r.get("initial_frame", ""))
        lines.append("")
        lines.append("【TRACE 中间段】")
        for i, m in enumerate(mids, 1):
            lines.append("--- 中间段 %d ---" % i)
            lines.append(m)
        lines.append("")
        lines.append("【FINAL_ANSWER】")
        lines.append(final)
        lines.append("")
        lines.append("【待判断的方向列表】")
        lines.append("（下面每个方向都是「疑似域外方向」。请对每一个独立判断两件事：")
        lines.append("  1) 它最终有没有进最终答案、有没有被否决 —— 给出 status")
        lines.append("  2) 若它进了最终答案，会不会改变决策 —— 给出 decision_effect）")
        lines.append("")
        ds = dirs(r)
        n_dir += len(ds)
        for i, d in enumerate(ds, 1):
            lines.append("  %d. %s" % (i, d.get("direction", "")))
        lines.append("")
        lines.append("【本条目的方向数】%d" % len(ds))

        with open(os.path.join(OUT, "%s.txt" % rid), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # 总览，方便 agent 快速定位
    idx = ["# 盲评复核材料总览", ""]
    for r in rows:
        ds = dirs(r)
        idx.append("- `%s.txt` — %d 个方向" % (r["id"], len(ds)))
    idx.append("")
    idx.append("总计 %d 条题目 / %d 个方向" % (len(rows), n_dir))
    with open(os.path.join(OUT, "INDEX.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(idx))

    print("已生成 %d 份材料 → %s" % (len(rows), os.path.abspath(OUT)))
    print("方向总数: %d" % n_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
