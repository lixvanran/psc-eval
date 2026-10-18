# PRP 三探针 · prompt 模板

> 位置：`/workspace/c-group/goldset/prp_prompts/`
> 配套金标：`../premises.jsonl`（30 条前提清单）、`../gold_v1.jsonl`（已冻结）
> 状态：**M3 的前置件**（不阻塞 M2）

## 三个探针

| 目录 | 探针 | 测什么 | prompt 里给不给 P |
|---|---|---|---|
| `A_premise_recognition/` | **A 前提识别** | 能不能说出候选集依赖的隐含前提 | **不给**（只给 TASK + INITIAL_FRAME） |
| `B_violation_detection/` | **B 违背检出** | 能不能发现 trace 没有按 P 走 | **给**（P 在 prompt 里） |
| `C_escape_direction/` | **C 逃逸方向生成** | 能不能命名"推翻 P"这个域外方向 | **给**（P + 违背点位置） |

**为什么要拆成三个**：前提识别本身有主观性。把它隔离在探针 A，
B/C 就变成**前提已给定**的确定性任务——于是"推翻 P"成为**闭卷可报的合法靶子**，
主召回分母不再为 0，而"前提识别"的能力被单独测量。

## 占位符

| 目录 | 占位符 |
|---|---|
| `A_premise_recognition/user_template.txt` | `{{TASK}}` `{{INITIAL_FRAME}}` |
| `B_violation_detection/user_template.txt` | `{{TASK}}` `{{INITIAL_FRAME}}` `{{PREMISE}}` `{{TRACE}}` `{{FINAL_ANSWER}}` |
| `C_escape_direction/user_template.txt` | `{{TASK}}` `{{INITIAL_FRAME}}` `{{PREMISE}}` `{{TRACE}}` `{{FINAL_ANSWER}}` `{{VIOLATION_QUOTE}}` |

`{{TRACE}}` 的渲染格式沿用 `eval/prompts/README.md` 的约定（`--- stage: xxx ---` 分段），
`{{FINAL_ANSWER}}` 若 trace 数组的最后一个元素就是最终答案，则原样传入。

## 输出契约

```jsonc
// A
{"premises": [{"premise": "…", "kind": "资产归属|承诺机制|考核口径|流程前提|数据前提|规模前提|时间前提|合规前提|其他", "basis": "…"}], "none": false}

// B
{"verdict": "follows|violates|unclear", "violation_quote": "trace 里的逐字片段", "reasoning": "…"}

// C
{"direction": "一句话的推翻方向", "in_frame": false, "in_frame_reason": "指向 initial_frame 具体措辞", "evidence_quote": "trace 里的逐字片段"}
```

## 三条硬性要求（写在 System 里，不可省）

1. **B 与 C 必须允许 `unclear`** —— 不许逼模型二选一。`unclear` 是合法答案。
2. **C 的 prompt 里不得出现"正确答案"或"应该推翻哪个前提"** —— 只给 P 与违背点位置。
3. **C 的方向必须与"推翻 P"有关**，不是"在 P 之内把事做得更好"（System 里给了正反例）。

## 与 `expected_source = "prp"` 桶的关系

凡是要计入 **主** `miss_detection_recall` 的 PRP 方向，必须满足：

> **P 在 prompt 里是给定的**（探针 B/C 的形态）。
> 凡是 P 靠抽取器自己猜出来的方向（探针 A 的形态），**一律不得进 `prp` 桶** ——
> 那属于探针 A 的能力范围，不是漏判召回。

判定方式（评测器侧）：读 `prp_prompts/` 里该题的 prompt，若其中**显式给出了 P**，
才允许计入 prp 桶；否则计入 `author_only` 或 `low_reportability`。

## 本集上的产出预期（独立复核过）

**探针 C 在本集上产出 ≈ 0。** 原因：金标构造方造 trace 时遵循"违反前提就必须提出"，
这个构造规则本身消灭了探针 C 的靶子。逐题判定见 `../PREMISES-README.md` §4。

**探针 A / B 在本集上可跑**（前提清单即探针 A 的金标；P 给定后 B 是确定性的）。
