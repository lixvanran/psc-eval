# 抽取器执行体 · 只读清单与禁令

> **本文件是执行体纪律的唯一事实源。** 派活时请直接指向本文件，不要在消息里手抄清单（抄错过）。

---

## ✅ 允许读（仅这四类）

| # | 路径 | 说明 |
|---|---|---|
| 1 | `/workspace/c-group/eval/extractor_spec.md` | **规则**（四条判定规则、逃逸测试、输出契约） |
| 2 | `/workspace/c-group/eval/prompts/system.txt` | System Prompt 全文（判据） |
| 3 | `/workspace/c-group/eval/prompts/user_template.txt` | User Prompt 模板（占位符契约） |
| 4 | `/workspace/c-group/run/prompts/<ID>.txt` | 你负责的那几道题的渲染 prompt（TASK / INITIAL_FRAME / TRACE / FINAL_ANSWER） |

**已核验**：上述 4 类文件里**不含任何金标答案**（12-gram 全量扫描 0 命中）。

---

## ❌ 禁止读（全部）

| 路径 | 为什么禁 |
|---|---|
| `/workspace/c-group/eval/SPEC-DECISIONS-v1.md` | **裁决记录 —— 历史上曾在里面引用金标方向原文作为举例，构成答案泄漏**（裁决 ㉛）。现在已清理，但**仍必须禁读**：它是给人看的，不是给执行体看的 |
| `/workspace/c-group/eval/README.md` | 可能含示例答案 |
| `/workspace/c-group/goldset/**` | **金标本体**（gold_v1.jsonl / gold_v1_openbook.jsonl / premises.jsonl / source/ / verify.py / README 全部） |
| `/workspace/c-group/run/preds*/**` | 上一轮抽取器产物 → **锚定**（你会照抄别人的判断） |
| `/workspace/c-group/run/m2_report.json`、`run/M2-REPORT.md` | **直接是答案** |
| `/workspace/c-group/run/review/**` | 盲评复核产物 |
| `/workspace/c-group/C-GROUP-CHARTER.md`、`REPORT-v2-rewrite-proposal.md` | 组级文档，可能含金标举例 |
| 任何 `run/goldset/`、任何 `*.zip` | 归档包，含上述全部 |

**自证要求**：回报时**声明读了哪几个文件**（按路径列出）。如果发现自己误读了禁读文件，**必须在回报里主动说明**，并标注受影响的题目 id —— 该题目的结果将被标记 `contaminated: true` 并单列。

---

## 三条硬纪律

1. **不许看金标** —— 看了就是开卷作弊，这次实验的意义就没了。
2. **不许看别人的输出** —— 看了就是锚定，你的判断会退化成复读。
3. **不许自提方向** —— `directions` 里只列**在 trace / 最终答案里真实出现**的域外方向。
   唯一例外：`not_generated`（trace 全文无表达），且必须同时满足
   ① 方向确实域外（机制 ∉ 候选集 ∪ 许可杠杆）② 若被采纳会改变决策或新增可操作选项。

---

## 输出纪律

- 每行一个 JSON 对象，**文件里不许有任何 JSON 之外的文字**
- `evidence_quote` 必须是 **TRACE 或 FINAL_ANSWER 段的连续逐字子串**（一个字不改）；
  `not_generated` 的 `evidence_quote` **必须是空字符串**
- `not_generated` 的 `evidence_quote` 留空后，**依据写进 `reasoning`**
- `directions` 条数 ≤ 8
- 临时文件请用**带 session id 的唯一目录**（如 `run/preds_rN/_parts_<session_id>/`）
  —— 共享临时目录被并行清理过一次，丢过别人的中间文件
- **不要碰** `OPENROUTER_API_KEY` / `MINIMAX_SUBSCRIPTION_KEY`（别的小组资源）
