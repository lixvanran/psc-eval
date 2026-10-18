#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 extractor_spec.md 抽取 prompt 全文，导出到 prompts/。

**为什么要有这一步**：`extractor_spec.md` 是给人读的、会被改；
`prompts/` 里的两个 .txt 是给执行器用的稳定接口、不能漂。
本脚本从 markdown 里**逐字**抽出 §2 的 System Prompt 与 §4 的 User Prompt 模板，
写入 prompts/system.txt 与 prompts/user_template.txt，并可随时用 --check 校验是否漂移。

用法：
    python3 eval/export_prompts.py          # 导出
    python3 eval/export_prompts.py --check  # 只校验是否与 markdown 一致（CI 用）

抽取规则：
  * System Prompt  = §2 里 `text` 代码块的**内容**（不含围栏）
  * User Template  = §4 里 `text` 代码块的**内容**（不含围栏）
  * 块内若出现「以下是示例」之类的示例性文字，导出时剔除
    （示例是给人看的，进了 prompt 会污染抽取器输出）
"""

from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "extractor_spec.md")
PROMPT_DIR = os.path.join(HERE, "prompts")
SYSTEM_OUT = os.path.join(PROMPT_DIR, "system.txt")
USER_OUT = os.path.join(PROMPT_DIR, "user_template.txt")

#: 导出时要剔除的示例性行（示例是给人看的，不该进 prompt）。
EXAMPLE_LINE_RE = re.compile(r"^\s*(?:以下(?:是)?示例|示例[:：]|例如以下)")
EXAMPLE_MARKER = "<<<EXAMPLES>>>"


def _section(text: str, title: str) -> str:
    """取 markdown 里某个 `## ` 标题到下一个 `## ` 之间的正文。"""
    pattern = re.compile(rf"^##\s+{re.escape(title)}.*?$", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        raise SystemExit(f"在 {SPEC} 里找不到标题：{title}")
    rest = text[m.end():]
    nxt = re.search(r"^##\s+", rest, re.MULTILINE)
    return rest[: nxt.start()] if nxt else rest


def _fenced_blocks(section: str) -> list[str]:
    """取一个 markdown 小节里所有 ```text 围栏块的内容（不含围栏）。"""
    blocks: list[str] = []
    for m in re.finditer(r"```text\n(.*?)\n```", section, re.DOTALL):
        blocks.append(m.group(1))
    return blocks


def _strip_examples(block: str) -> str:
    """剔除示例性文字。若原文用标记隔开示例段，则整段剔除。"""
    if EXAMPLE_MARKER in block:
        head = block.split(EXAMPLE_MARKER)[0]
        return head.rstrip() + "\n"
    lines = []
    skipping = False
    for line in block.split("\n"):
        if EXAMPLE_LINE_RE.match(line):
            skipping = True
            continue
        if skipping and not line.strip():
            skipping = False
            continue
        if not skipping:
            lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def extract() -> tuple[str, str]:
    text = open(SPEC, encoding="utf-8").read()
    sys_blocks = _fenced_blocks(_section(text, "2. System Prompt（完整可复制）"))
    user_blocks = _fenced_blocks(_section(text, "4. User Prompt 模板（完整可复制）"))
    if not sys_blocks:
        raise SystemExit("§2 里没找到 ```text 围栏块")
    if not user_blocks:
        raise SystemExit("§4 里没找到 ```text 围栏块")
    # §2 只有一个 prompt 块；§4 有一个模板块 + 一个 TRACE 渲染格式块（后者不是 prompt）
    system = _strip_examples(sys_blocks[0])
    template = _strip_examples(user_blocks[0])
    for required in ("{{TASK}}", "{{INITIAL_FRAME}}", "{{TRACE}}", "{{FINAL_ANSWER}}"):
        if required not in template:
            raise SystemExit(f"User Prompt 模板缺少占位符 {required}")
    return system, template


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 / 校验 PSC-Eval 的 prompt 稳定接口")
    parser.add_argument("--check", action="store_true", help="只校验是否与 markdown 一致")
    args = parser.parse_args()

    system, template = extract()

    if args.check:
        problems = []
        for path, expected in ((SYSTEM_OUT, system), (USER_OUT, template)):
            if not os.path.exists(path):
                problems.append(f"{path} 不存在")
                continue
            actual = open(path, encoding="utf-8").read()
            if actual != expected:
                problems.append(f"{path} 与 {SPEC} 不一致（漂移）")
        if problems:
            for p in problems:
                print(f"[FAIL] {p}")
            return 1
        print("[PASS] prompts/ 与 extractor_spec.md 逐字一致")
        return 0

    os.makedirs(PROMPT_DIR, exist_ok=True)
    with open(SYSTEM_OUT, "w", encoding="utf-8") as f:
        f.write(system)
    with open(USER_OUT, "w", encoding="utf-8") as f:
        f.write(template)
    print(f"已导出：\n  {SYSTEM_OUT}  ({len(system)} 字符)\n  {USER_OUT}  ({len(template)} 字符)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
