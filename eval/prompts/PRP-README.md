# PRP 三探针 prompt 模板（放在 PSC-Eval 主 prompt 旁边）

> 本目录下的三个子目录是 **PRP（前提识别能力）三探针**的 prompt 模板，
> 与上一级的 `system.txt` / `user_template.txt`（**PSC-Eval 主抽取器**的 prompt）**不是同一套**，请勿混用。

| 子目录 | 探针 | 给不给前提 P | 输出 |
|---|---|---|---|
| `A_premise_recognition/` | A 前提识别 | **不给**（只有 TASK + INITIAL_FRAME） | `{"premises": [{premise, kind, basis}], "none"}` |
| `B_violation_detection/` | B 违背检出 | **给**（`{{PREMISE}}`） | `{"verdict": "follows|violates|unclear", "violation_quote", "reasoning"}` |
| `C_escape_direction/` | C 逃逸方向生成 | **给**（`{{PREMISE}}` + `{{VIOLATION_QUOTE}}`） | `{"direction", "in_frame", "in_frame_reason", "evidence_quote"}` |

**「这个探针有没有给 P」的判定方式**：看 `user_template.txt` 里**有没有 `{{PREMISE}}` 占位符**
—— **不靠目录名**（改名不影响判定）。

**为什么这条重要**：凡 P 靠抽取器自己猜出来的方向（探针 A 的形态），
**不得计入 `expected_source = "prp"` 桶、不得进主召回分母**——
那是探针 A 的能力范围，不是漏判召回。

**源文件与配套说明**（同一份内容的另一处副本，改动请同步）：
`/workspace/c-group/goldset/prp_prompts/`（含 README）、`/workspace/c-group/goldset/PREMISES-README.md`、
`/workspace/c-group/goldset/premises.jsonl`（30 条前提清单 = 探针 A 的金标）。
