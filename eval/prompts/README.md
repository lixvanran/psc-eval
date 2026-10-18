# PSC-Eval Prompt 稳定接口

> **这两个 `.txt` 是给执行器用的稳定接口。以它们为准。**
> `../extractor_spec.md` 里的对应段落是**给人读的**，会被改；这两个文件是**机器读的**，不能漂。

## 文件

| 文件 | 用途 | 占位符 |
|---|---|---|
| `system.txt` | System Prompt 全文，逐字 | 无占位符 |
| `user_template.txt` | User Prompt 模板全文，逐字 | `{{TASK}}` `{{INITIAL_FRAME}}` `{{TRACE}}` `{{FINAL_ANSWER}}` |

## 怎么用

```
system.txt 的全文           -> 作为 system 消息
user_template.txt           -> 四个占位符替换后作为 user 消息
执行器输出                  -> 严格 JSON，单一对象，字段见 extractor_spec.md §1.2
```

**TRACE 的渲染格式**（建议，避免模型把阶段边界读错）：

```text
--- stage: explore ---
<content>

--- stage: select ---
<content>

--- stage: synthesize ---
<content>
```

若 `final_answer` 是 trace 数组的最后一个元素（而非独立字段），按 `{{FINAL_ANSWER}}`
原样传入即可 —— 抽取器按「最终答案段」判定 status（裁决 ⑧-a）。

## 输出契约（摘要，完整见 `../extractor_spec.md` §1.2）

```json
{
  "directions": [
    {
      "direction": "一句话描述这个方向",
      "in_frame": true,
      "in_frame_reason": "落在 initial_frame 的哪个元素内 / 逃出了哪个边界",
      "status": "realized | expression_failure | selection_failure | not_generated",
      "evidence_quote": "逐字原文片段；not_generated 时必须为 \"\"",
      "decision_effect": "conclusion_flip | option_added | ranking_only | additive_only | none",
      "reasoning": "为什么它会/不会改变最终结论"
    }
  ],
  "frame_ambiguous": false,
  "frame_ambiguity_note": "",
  "confidence": "high | medium | low",
  "notes": ""
}
```

- `status` 的 canonical 名是 `not_generated`（`discovery_failure` 是别名，评测器双向归一）。
- `decision_effect` 取代了旧的 `would_change_decision`（bool），**不要输出旧字段**。
- `directions` 最多 8 条。

## 漂移防护

两个 `.txt` 由 `../export_prompts.py` **从 `../extractor_spec.md` 逐字抽取**生成，
所以它们与规格不可能不一致（除非有人手改了 `.txt`）。

CI / 提交前校验：

```bash
python3 eval/export_prompts.py --check
# [PASS] prompts/ 与 extractor_spec.md 逐字一致
```

改了 `extractor_spec.md` 的 §2 / §4 之后，**必须**重跑：

```bash
python3 eval/export_prompts.py
```

## 为什么要有这一层

M2 要跑真实抽取器，而本组的 API key 尚未获授权 —— 用子 agent 当抽取器引擎
（子 agent 用平台自身额度，不碰别组 key）。接口固定后，将来换任何执行器
（子 agent / API / 人工）都不影响尺子本身：**尺子测的是抽取器的判别力，
不是某个具体执行器的实现**。

若未来发现 prompt 需要改：**改 `extractor_spec.md` → 重跑 `export_prompts.py` → 跑 `--check`**，
不要只改 `.txt`（那会造成 markdown 与实际 prompt 漂移，而 markdown 是给人看的依据）。

## 判定锚点提醒（v1.1）

`status` 的判定锚点是**最终答案段**，不是取舍段：

```
最终答案里提到它了吗？
├─ 提到了 → 作为可操作选项/推荐/结论？ → realized
│          └─ 明确排除/否决/推迟？        → selection_failure
└─ 没提到 → 中间段被否决 → selection_failure
            中间段保留过 → expression_failure
            全文没出现   → not_generated
```

「被讨论过」**不等于**「被采纳」。
