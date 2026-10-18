#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent verifier for reviewer_A.jsonl (does NOT reuse the builder's assumptions)."""
import json, re, os, unicodedata

MAT = "/workspace/c-group/run/review/materials"
OUT = "/workspace/c-group/run/review/reviewer_A.jsonl"

VALID_STATUS = {"realized", "selection_failure", "expression_failure", "not_generated"}
VALID_DE = {"conclusion_flip", "option_added", "ranking_only", "additive_only", "none"}

rows = []
with open(OUT, encoding="utf-8") as f:
    for ln, line in enumerate(f, 1):
        line = line.rstrip("\n")
        assert line.strip(), "empty line %d" % ln
        rows.append(json.loads(line))

errs = []
assert len(rows) == 30, "row count %d" % len(rows)
ids = [r["id"] for r in rows]
assert ids == ["G%02d" % i for i in range(1, 31)], ids

total = 0
for r in rows:
    gid = r["id"]
    src = open(os.path.join(MAT, gid + ".txt"), encoding="utf-8").read()
    # directions in the material, verbatim
    sec = re.search(r"【待判断的方向列表】(.*?)【本条目的方向数】", src, re.S).group(1)
    mat_dirs = [m.group(2).strip() for m in
                (re.match(r"^(\d+)\.\s*(.+)$", l.strip()) for l in sec.split("\n")) if m]
    out_dirs = [d["direction"] for d in r["directions"]]
    if mat_dirs != out_dirs:
        errs.append("%s: direction text mismatch\n  mat=%s\n  out=%s" % (gid, mat_dirs, out_dirs))
    for d in r["directions"]:
        total += 1
        if d["status"] not in VALID_STATUS:
            errs.append("%s: bad status %s" % (gid, d["status"]))
        if d["decision_effect"] not in VALID_DE:
            errs.append("%s: bad decision_effect %s" % (gid, d["decision_effect"]))
        q = d["evidence_quote"]
        if d["status"] == "not_generated":
            if q != "":
                errs.append("%s: not_generated with non-empty quote" % gid)
        else:
            if not q:
                errs.append("%s/%s: missing evidence_quote" % (gid, d["direction"][:20]))
            elif q not in src:
                errs.append("%s: quote not verbatim: %r" % (gid, q[:50]))
            else:
                # quote must be inside TRACE middle段 or FINAL_ANSWER
                mid = src.split("【TRACE 中间段】")[1].split("【FINAL_ANSWER】")[0] if "【TRACE 中间段】" in src else ""
                fin = src.split("【FINAL_ANSWER】")[1].split("【待判断的方向列表】")[0] if "【FINAL_ANSWER】" in src else ""
                if q not in mid and q not in fin:
                    errs.append("%s: quote outside TRACE/FINAL sections: %r" % (gid, q[:50]))
        if not d.get("reasoning", "").strip():
            errs.append("%s: empty reasoning" % gid)

print("rows=%d directions=%d" % (len(rows), total))
if errs:
    print("ERRORS (%d):" % len(errs))
    for e in errs:
        print(" -", e)
else:
    print("ALL STRUCTURAL CHECKS PASSED")

from collections import Counter
st = Counter(); de = Counter()
for r in rows:
    for d in r["directions"]:
        st[d["status"]] += 1; de[d["decision_effect"]] += 1
print("status:", dict(st))
print("decision_effect:", dict(de))
print("per-id direction counts:", {r["id"]: len(r["directions"]) for r in rows if r["directions"]})
