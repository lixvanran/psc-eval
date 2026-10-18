#!/usr/bin/env python3
"""收集抽取器分片输出 → 独立校验 → 合并 → 调评测器打分。

这一层刻意不复用 evaluator.py 的判定逻辑：
校验（evidence 逐字、status 合法、not_generated 空引用）是独立写的，
避免「用被测工具去验被测工具」。评测器只负责打分。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOLD = os.path.join(ROOT, "goldset", "gold_v1.jsonl")
PREDS = os.path.join(HERE, "preds")
EVAL = os.path.join(ROOT, "eval", "evaluator.py")

VALID_STATUS = {"realized", "selection_failure", "expression_failure", "not_generated"}
VALID_EFFECT = {"conclusion_flip", "option_added", "ranking_only", "additive_only", "none"}


def trace_corpus(row: dict) -> str:
    """抽取器可引用的语料 = trace 全部段（含末段最终答案）+ final_answer 字段。"""
    t = row.get("trace") or []
    parts = []
    if isinstance(t, list):
        for seg in t:
            parts.append(seg.get("content", "") if isinstance(seg, dict) else str(seg))
    else:
        parts.append(str(t))
    fa = row.get("final_answer")
    if isinstance(fa, list):
        parts.append("\n".join(map(str, fa)))
    elif isinstance(fa, str) and fa.strip():
        parts.append(fa)
    return "\n".join(parts)


def norm(s: str) -> str:
    return re.sub(r"[\s。，、；：？！「」『』\"'（）()【】—…·,.!?;:\-]", "", s or "")


def load_gold():
    rows = {}
    for line in open(GOLD, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def check_manifest(man_name="prompts_manifest.json"):
    """★ 版本锁校验：prompt 渲染时用的金标 md5 必须与当前金标一致。

    这条是本项目栽过最贵的一处（金标改了 trace 而 prompt 没重渲染 →
    抽取器引用的 evidence 在金标里查不到 → 9 条被判「编造」）。
    """
    import hashlib
    man_p = os.path.join(HERE, man_name)
    if not os.path.exists(man_p):
        print("⚠ 没有 prompts_manifest.json —— 无法校验 prompt 与金标的版本一致性")
        return None
    man = json.load(open(man_p, encoding="utf-8"))
    cur = hashlib.md5(open(GOLD, "rb").read()).hexdigest()
    if man.get("gold_md5") != cur:
        print("🚨🚨 金标版本不一致 —— prompt 已过期，抽取器输出不可用")
        print("   prompt 渲染时金标 md5 = %s" % man.get("gold_md5"))
        print("   当前金标 md5          = %s" % cur)
        print("   → 必须重新 render_prompts.py 并重跑抽取器")
        return False
    print("✅ 金标版本一致 (md5=%s)" % cur[:12])
    if man.get("round_tag"):
        print("   轮次标记: %s | 渲染于 %s" % (man.get("round_tag"), man.get("rendered_at", "?")))
    return True


def main() -> int:
    global PREDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PREDS, "pred_all.jsonl"))
    ap.add_argument("--no-score", action="store_true", help="只校验不调评测器")
    ap.add_argument("--manifest", default="prompts_manifest.json", help="要校验的 manifest 文件名")
    ap.add_argument("--pred-dir", default=None, help="抽取器输出目录（默认 preds/）")
    args = ap.parse_args()
    if args.pred_dir:
        PREDS = os.path.join(HERE, args.pred_dir)

    gold = load_gold()
    files = sorted(glob.glob(os.path.join(PREDS, "shard_*.jsonl")))
    if not files:
        print("没找到分片输出: %s/shard_*.jsonl" % PREDS)
        return 1

    preds, problems, stats = [], [], {"lines": 0, "ids": 0, "dirs": 0}
    seen_ids = set()

    for fp in files:
        for ln, line in enumerate(open(fp, encoding="utf-8"), 1):
            line = line.strip()
            if not line:
                continue
            stats["lines"] += 1
            try:
                p = json.loads(line)
            except Exception as e:
                problems.append("%s:%d JSON 解析失败: %s" % (os.path.basename(fp), ln, str(e)[:60]))
                continue
            rid = p.get("id")
            if rid not in gold:
                problems.append("%s:%d 未知 id %r" % (os.path.basename(fp), ln, rid))
                continue
            if rid in seen_ids:
                problems.append("id 重复: %s" % rid)
                continue
            seen_ids.add(rid)
            stats["ids"] += 1

            corpus = trace_corpus(gold[rid])
            ncorp = norm(corpus)
            dirs = p.get("directions")
            if not isinstance(dirs, list):
                problems.append("%s: directions 不是数组" % rid)
                continue
            if len(dirs) > 8:
                problems.append("%s: directions %d 条 > 上限 8" % (rid, len(dirs)))
            for d in dirs:
                stats["dirs"] += 1
                st = d.get("status")
                if st not in VALID_STATUS:
                    problems.append("%s: 非法 status %r" % (rid, st))
                    continue
                de = d.get("decision_effect")
                if de is not None and de not in VALID_EFFECT:
                    problems.append("%s: 非法 decision_effect %r" % (rid, de))
                q = d.get("evidence_quote") or ""
                if st == "not_generated":
                    if q:
                        problems.append("%s: not_generated 带引用（违规）" % rid)
                else:
                    if not q:
                        problems.append("%s: %s 但 evidence_quote 为空" % (rid, st))
                    elif q not in corpus:
                        if norm(q) in ncorp:
                            problems.append("%s: evidence 非逐字（仅标点差异）" % rid)
                        else:
                            problems.append("%s: evidence 不可核（编造?）" % rid)
                if d.get("in_frame") is True:
                    problems.append("%s: 输出了 in_frame=true 的方向（应只列域外）" % rid)
            preds.append(p)

    missing = sorted(set(gold) - seen_ids)
    print("=== 抽取器输出独立校验 ===")
    print("分片文件: %d 个 | 行 %d | 有效 id %d | 方向 %d" % (len(files), stats["lines"], stats["ids"], stats["dirs"]))
    print("覆盖: %d/%d%s" % (stats["ids"], len(gold), ("  缺: " + ",".join(missing)) if missing else "  ✅"))
    if problems:
        print("问题 %d 条:" % len(problems))
        for pr in problems[:40]:
            print("  -", pr)
        if len(problems) > 40:
            print("  ... 其余 %d 条省略" % (len(problems) - 40))
    else:
        print("问题: 0 ✅")

    # ★ 硬闸门：evidence 不可核（编造/版本脱节）> 0 → 中止，不允许跑分。
    #   本项目栽过：prompt 用旧金标渲染 → 抽取器引用在新金标里查不到 → 9 条 ungrounded。
    #   那一次如果直接跑分，就会拿一个无效的 FAIL 出来。**可核率是跑分的前置条件。**
    n_ungrounded = sum(1 for pr in problems if "不可核" in pr)
    if n_ungrounded:
        print("\n🚨 中止：%d 条 evidence 不可核（编造 或 prompt 与金标版本脱节）" % n_ungrounded)
        print("   可核率是跑分的前置条件 —— 先查清是「抽取器编造引用」还是「prompt 版本过期」")
        print("   两种情况的处理完全不同，禁止在此状态下出分。")
        return 2
    print("✅ evidence 可核率 100%% —— 允许跑分")

    os.makedirs(PREDS, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("\n合并输出 → %s" % os.path.abspath(args.out))

    if args.no_score:
        return 0

    print("\n=== 调评测器打分 ===")
    cmd = [sys.executable, EVAL, "--gold", GOLD, "--pred", args.out,
           "--out", os.path.join(HERE, "m2_report.json")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-6000:] if r.stdout else "(无 stdout)")
    if r.stderr:
        print("STDERR:", r.stderr[-1500:])
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
