#!/usr/bin/env python3
"""把 gold_v1.jsonl 渲染成抽取器 prompt 文件，供外部执行器逐条处理。

设计要点：
  1. prompt 模板从 eval/prompts/ 读纯文本，不解析 markdown（防漂移）
  2. 渲染结果与 prompt 模板分目录，prompt 只读、产物可重生成
  3. 每个 prompt 单独一个文件，便于多个执行器并行分片
  4. 自带 --check 模式：比对 eval/prompts/*.txt 与 extractor_spec.md 是否一致

用法：
  python3 run/render_prompts.py            # 渲染
  python3 run/render_prompts.py --check    # 校验 prompt 与 spec 一致
  python3 run/render_prompts.py --shard 0 3   # 第 0 片，共 3 片
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(ROOT, "..", "eval", "prompts")
OUT_DIR = os.path.join(ROOT, "prompts")
GOLD = os.path.join(ROOT, "..", "goldset", "gold_v1.jsonl")


def read(p: str) -> str:
    with open(p, encoding="utf-8") as f:
        return f.read()


def load_prompt_files():
    sysp = os.path.join(PROMPT_DIR, "system.txt")
    usrt = os.path.join(PROMPT_DIR, "user_template.txt")
    for p in (sysp, usrt):
        if not os.path.exists(p):
            sys.exit("缺少 prompt 文件: %s\n"
                     "请让 Coder 按要求导出 eval/prompts/{system.txt,user_template.txt}" % p)
    return read(sysp), read(usrt)


def trace_text(row: dict) -> str:
    """把 trace 渲染成带阶段标签的文本。

    两种形态都要支持：
      - General 金标：trace 是纯字符串数组，末段即最终答案
      - 结构化 trace：每项带 stage/content
    """
    t = row.get("trace") or []
    if not isinstance(t, list):
        return str(t)
    out = []
    for i, seg in enumerate(t):
        if isinstance(seg, dict):
            stage = seg.get("stage") or ("stage%d" % (i + 1))
            content = seg.get("content", "")
            extra = []
            if seg.get("candidates"):
                extra.append("candidates: " + " / ".join(map(str, seg["candidates"])))
            if seg.get("dropped"):
                dropped = "; ".join(
                    "%s (%s)" % (d.get("direction", ""), d.get("reason", ""))
                    for d in seg["dropped"] if isinstance(d, dict))
                extra.append("dropped: " + dropped)
            body = content + ("\n[" + "\n".join(extra) + "]" if extra else "")
        else:
            stage = "stage%d" % (i + 1)
            body = str(seg)
        out.append("--- stage: %s ---\n%s" % (stage, body))
    return "\n\n".join(out)


def final_answer(row: dict) -> str:
    """最终答案：优先独立字段；否则取 trace 末段（General 金标的形态）。"""
    fa = row.get("final_answer")
    if isinstance(fa, list):
        return "\n".join(map(str, fa))
    if isinstance(fa, str) and fa.strip():
        return fa
    t = row.get("trace") or []
    if isinstance(t, list) and t:
        last = t[-1]
        if isinstance(last, dict):
            return last.get("content", "")
        return str(last)
    return ""


def render_user(tmpl: str, row: dict) -> str:
    out = tmpl
    for ph, val in (
        ("{{TASK}}", row.get("task", "")),
        ("{{INITIAL_FRAME}}", row.get("initial_frame", "")),
        ("{{TRACE}}", trace_text(row)),
        ("{{FINAL_ANSWER}}", final_answer(row)),
    ):
        out = out.replace(ph, val)
    return out


def load_gold():
    if not os.path.exists(GOLD):
        sys.exit("找不到金标: %s" % GOLD)
    rows = []
    for line in open(GOLD, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception as e:
            print("  ! 跳过不可解析行: %s" % str(e)[:60], file=sys.stderr)
    return rows


def cmd_check(sysp: str, usrt: str) -> int:
    """比对 eval/prompts/*.txt 与 extractor_spec.md 的对应段落是否一致。"""
    spec_path = os.path.join(ROOT, "..", "eval", "extractor_spec.md")
    spec = read(spec_path)
    blocks = re.findall(r"```(?:text)?\n(.*?)```", spec, flags=re.DOTALL)
    ok = True

    def find_block(needle: str):
        for b in blocks:
            if needle in b:
                return b.strip()
        return None

    sys_block = find_block("System")
    user_block = find_block("{{TASK}}")
    if sys_block is None:
        print("  ! spec 里找不到 System Prompt 代码块", file=sys.stderr)
        ok = False
    elif sys_block != sysp.strip():
        print("  ! system.txt 与 spec §2 不一致", file=sys.stderr)
        ok = False
    else:
        print("  [OK] system.txt == spec §2")

    if user_block is None:
        print("  ! spec 里找不到 User Prompt 代码块", file=sys.stderr)
        ok = False
    elif user_block != usrt.strip():
        print("  ! user_template.txt 与 spec §4 不一致", file=sys.stderr)
        ok = False
    else:
        print("  [OK] user_template.txt == spec §4")

    if not ok:
        print("\n结论: 漂移。修 prompt 文件或 spec，不要两边都改。", file=sys.stderr)
        return 1
    print("\n结论: 一致。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验 prompt 与 spec 一致性")
    ap.add_argument("--gold", default=None, help="金标路径（默认闭卷主文件）")
    ap.add_argument("--outdir", default=None, help="输出目录（默认 prompts/）")
    ap.add_argument("--manifest", default=None, help="manifest 路径（默认 prompts_manifest.json）")
    ap.add_argument("--round-tag", default=None, help="轮次标记，写进 manifest（用于版本溯源）")
    ap.add_argument("--shard", type=int, default=None, help="分片序号，从 0 开始")
    ap.add_argument("--nshards", type=int, default=1, help="分片总数")
    args = ap.parse_args()

    sysp, usrt = load_prompt_files()
    if args.check:
        return cmd_check(sysp, usrt)

    global GOLD, OUT_DIR
    if args.gold:
        GOLD = args.gold
    if args.outdir:
        OUT_DIR = args.outdir
    rows = load_gold()
    os.makedirs(OUT_DIR, exist_ok=True)
    written = 0
    for i, row in enumerate(rows):
        if args.nshards > 1 and (i % args.nshards) != args.shard:
            continue
        rid = row.get("id", "row%03d" % i)
        content = ("=== SYSTEM ===\n" + sysp.strip() +
                   "\n\n=== USER ===\n" + render_user(usrt, row).strip() + "\n")
        with open(os.path.join(OUT_DIR, "%s.txt" % rid), "w", encoding="utf-8") as f:
            f.write(content)
        written += 1
    # ★ 版本锁定：把金标 md5 写进 manifest，评分时校验（防止 prompt 与金标脱节）
    import hashlib
    md5 = hashlib.md5(open(GOLD, "rb").read()).hexdigest()
    sys_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval", "prompts", "system.txt")
    man = {"gold_md5": md5, "gold_path": os.path.abspath(GOLD), "n": written,
           "round_tag": args.round_tag or "", "rendered_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
           "outdir": os.path.abspath(OUT_DIR),
           "system_txt_sha": hashlib.sha256(open(sys_p, "rb").read()).hexdigest()}
    man_path = args.manifest or os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts_manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    print("已渲染 %d 条 → %s" % (written, os.path.abspath(OUT_DIR)))
    print("金标 md5 = %s（已写入 prompts_manifest.json）" % md5)
    if args.nshards > 1:
        print("分片 %d/%d" % (args.shard, args.nshards))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
