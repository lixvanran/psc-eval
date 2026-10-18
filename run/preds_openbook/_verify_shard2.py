import json, re, sys, unicodedata

PRED = "/workspace/c-group/run/preds_openbook/shard_2.jsonl"
PROMPTS = "/workspace/c-group/run/prompts_openbook"

STATUSES = {"realized", "selection_failure", "expression_failure", "not_generated", "discovery_failure"}
EFFECTS = {"conclusion_flip", "option_added", "ranking_only", "additive_only", "none"}


def corpus(path):
    """TRACE + FINAL_ANSWER sections only; exclude SYSTEM, INITIAL_FRAME, and the checklist."""
    txt = open(path, encoding="utf-8").read()
    user = txt.split("=== USER ===", 1)[1]
    # keep only from TRACE to end of FINAL_ANSWER block (drop the checklist tail)
    start = user.index("【TRACE】")
    end = user.index("【检查清单】")
    body = user[start:end]
    # also drop the INITIAL_FRAME / TASK header parts are before TRACE -> already excluded
    return body


def frame_note(path):
    txt = open(path, encoding="utf-8").read()
    m = re.search(r"域外参考[^\n]*", txt)
    return m.group(0) if m else ""


errors, warns = [], []
rows = []
for i, line in enumerate(open(PRED, encoding="utf-8")):
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except Exception as e:
        errors.append(f"line {i+1}: JSON parse fail: {e}")
        continue
    rows.append(obj)

print("rows:", len(rows), "ids:", [r.get("id") for r in rows])

expected = [f"G{n}" for n in range(21, 31)]
if [r.get("id") for r in rows] != expected:
    errors.append(f"id set/order mismatch: {expected}")

for r in rows:
    rid = r.get("id")
    p = f"{PROMPTS}/{rid}.txt"
    corp = corpus(p)
    note = frame_note(p)
    dirs = r.get("directions")
    if not isinstance(dirs, list):
        errors.append(f"{rid}: directions not a list")
        continue
    if len(dirs) > 8:
        errors.append(f"{rid}: too many directions {len(dirs)}")
    if r.get("frame_ambiguous") not in (True, False):
        errors.append(f"{rid}: frame_ambiguous missing")
    if r.get("confidence") not in ("high", "medium", "low"):
        errors.append(f"{rid}: bad confidence {r.get('confidence')}")
    for j, d in enumerate(dirs):
        tag = f"{rid}[{j}]"
        for f in ("direction", "in_frame", "in_frame_reason", "status", "evidence_quote",
                  "constraint_violation", "decision_effect", "reasoning"):
            if f not in d:
                errors.append(f"{tag}: missing field {f}")
        if d.get("in_frame") is not False:
            errors.append(f"{tag}: in_frame must be false (spec: only out-of-frame reported)")
        st = d.get("status")
        if st not in STATUSES:
            errors.append(f"{tag}: bad status {st}")
        de = d.get("decision_effect")
        if de not in EFFECTS:
            errors.append(f"{tag}: bad decision_effect {de}")
        if not isinstance(d.get("constraint_violation"), bool):
            errors.append(f"{tag}: constraint_violation not bool")
        q = d.get("evidence_quote")
        if not isinstance(q, str):
            errors.append(f"{tag}: quote not str")
            continue
        if st == "not_generated":
            if q != "":
                errors.append(f"{tag}: not_generated must have empty quote")
        else:
            if not q.strip():
                errors.append(f"{tag}: status {st} requires non-empty quote")
            elif q not in corp:
                errors.append(f"{tag}: QUOTE NOT VERBATIM in corpus")
            else:
                # must not be only the checklist (corpus excludes it already)
                if note and q in note:
                    errors.append(f"{tag}: quote leaks frame_note")
        if len(d.get("direction", "")) > 60:
            warns.append(f"{tag}: direction longer than 60 chars ({len(d['direction'])})")
        if len(d.get("reasoning", "")) < 10:
            errors.append(f"{tag}: reasoning < 10 chars")
        if d.get("in_frame_reason", "").strip() == "":
            errors.append(f"{tag}: empty in_frame_reason")
        # unicode sanity
        for k, v in d.items():
            if isinstance(v, str) and "\ufffd" in v:
                errors.append(f"{tag}: U+FFFD corruption in {k}")

raw = open(PRED, encoding="utf-8").read()
if "\ufffd" in raw:
    errors.append("file contains U+FFFD replacement characters")

print("\n=== STATUS / EFFECT TALLY ===")
from collections import Counter
sc, ec, cv = Counter(), Counter(), []
for r in rows:
    for d in r["directions"]:
        sc[d["status"]] += 1
        ec[d["decision_effect"]] += 1
        if d.get("constraint_violation"):
            cv.append((r["id"], d["direction"]))
print("status:", dict(sc))
print("effect :", dict(ec))
print("cv=true count:", len(cv))
for x in cv:
    print("   cv:", x[0], x[1])
print("per-id counts:", {r["id"]: len(r["directions"]) for r in rows})

print("\n=== ERRORS ===")
for e in errors:
    print("E:", e)
print("\n=== WARNINGS ===")
for w in warns:
    print("W:", w)
if not errors:
    print("ALL CHECKS PASSED")
