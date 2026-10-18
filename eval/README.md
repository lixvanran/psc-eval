# PSC-Eval —— 问题空间构建评测器

> C 组（第三组）主交付物 · M1 里程碑 · **schema v2**
> 关联：组章程 `/workspace/c-group/C-GROUP-CHARTER.md` · 冻结裁决 `/workspace/c-group/eval/SPEC-DECISIONS-v1.md`
> 配套：`extractor_spec.md`（抽取器提示词规格）、`evaluator.py`（打分骨架）

---

## 0. 边界（**先读这一节**）

这五条不是「已知局限」，是**这把尺子的适用边界**。任何超出边界的解读都是误用。

1. **它测「域外方向有没有被生成、保住、呈现」** —— 不测别的。
2. **它测不了「域外方向对不对」** —— 那需要依据，而依据正是域外定义所缺的。
   要核「域外方向本身对不对」，用 B 组的 `check_claims v4`，不用 PSC-Eval。
3. **它对 selection/expression 的区分能力，取决于上游 trace 是否带候选清单与淘汰记录**
   （`candidates[]` / `dropped[]`）。没有这两项，它只能靠保守默认猜，判准率不可信。
4. **措辞规范的适用边界**：金标集的 trace 措辞规范（取舍段一律写「保留」、selection 一律带显式否决词）
   **仅适用于金标集**。接真实系统 trace 时，status 判定必须回到**语义判定**
   （有没有否决痕迹 / 保留痕迹），**不得依赖任何特定措辞**。
5. **词法捷径基线**：报告里的 `true_increment` 是 PSC-Eval 相对于关键词匹配基线的真实语义增量。
   **它如果很小，说明这把尺子在这个语料上没学到多少东西。**
6. **金标本身有代表性上限**（裁决 ⑮-b）——**这一条要写进 M2 报告第一屏**：
   - `expression_failure` 的「9/9/9 高可复现」**降级**为
     「**金标措辞模板化导致的高一致，不代表真实 trace 上的判别力**」。
     那 9 条全是「保留为 + 某种未来事项」，措辞高度模板化。
   - 报告**不得**用 9/9/9 宣称这一类可靠。
   - **该类的真实判别力尚未被测量** —— 在金标去模板化 + 重跑之前，
     `expression_failure` 的召回率**不得作为尺子质量证据**引用。
   - 佐证：同一批真实 trace 上，三个独立执行体报出的 `expression_failure` 数量是
     **1 / 7 / 0** —— 差 7 倍。盲评的一致性来自措辞模板，不来自判别力。
7. **题面不能泄题**（裁决 ⑨-d，修正了 ⑦-a 的原命题）：
   > ~~「如果 status 判据被写进 trace 的措辞里，尺子就退化成词法匹配器」~~ → **原命题过强，已修正**。
   > 真正的失效点是：**域外判定一旦在 `initial_frame` 里被明示，尺子就退化成字串匹配。**

   金标复核实测：v1.1 把 status 锚点从取舍段移到最终答案段后，基线 F1 仍是 0.696 ——
   说明**词法捷径在 status 上白拿分与锚点位置无关**。基线那 0.696 的主要来源不是
   「status 可词法化」，而是**「域外判定可以对着答案做」**。
   这是**金标设计问题，不是尺子设计问题** —— 修正手段是闭卷改造（见 §9.1）。

---

## 1. 这把尺子量什么

| 问题 | 指标 | 在哪实现 |
|---|---|---|
| Agent 到底有没有生成初始框架之外的方向？ | 主表 **precision / recall / F1**（不含 miss detection） | `score_prediction` |
| 关键漏判抓到了几个？ | **`miss_detection_recall`**（单列，**下界不是覆盖率**）+ `miss_detection_recall_author_only` | `score_prediction` |
| 如果生成了，在**哪一步**丢的？ | **status 四分类混淆矩阵** + `status_accuracy` / `status_accuracy_strict` / `status_undecidable_rate` | `score_prediction` |
| selection 与 expression 能不能分开？ | **`selection_expression_separability`** | `score_prediction` |
| 那个方向如果进了最终答案，会不会实质改变结论？ | **`decision_effect`** + `weak_effect_count` | `score_prediction` |
| 这把尺子是不是只学到了关键词？ | **`lexical_baseline`**（6 变体全表）+ **`true_increment`** | `lexical_baseline` |
| 在「正确答案是没有域外方向」的题上会不会乱报？ | **`zero_direction_precision`** / **`zero_direction_overclaim_rate`** | `score_prediction` |
| 门槛到底审到了多少金标？ | **`audit_coverage`**（应为 100%） | `aggregate` |
| 证据可不可核？ | `quote_grounding_rate`（三档） | `score_prediction` |

**这把尺子不回答**「Agent 的答案对不对」。那是 B 组 `check_claims v4` 的职责。
两把尺子，职责不重叠（章程 §4 坑 2：不自己重造核验器）。

---

## 2. 设计理念

### 2.1 为什么用 LLM 判别而不是符号化

「问题空间构建能力」测不出来，不是因为它不存在，而是因为**域外方向天然没有 ground truth** ——
它不在初始框架里，所以没有依据可查。B 组的 `check_claims v4` 测「答案里的断言有没有依据」，
在有依据的域（法律）非常好用，但对「域外的话有没有被想出来」完全无能为力。

本评测器用**三条互相独立的可核信号**替代「对不对」：

1. **是否越出框架** —— 判据是 `initial_frame` 的**显式描述**（含封闭性声明），不是「听起来新不新」。
2. **在哪一步丢失** —— 沿生成→选择→表达三道关走一遍，断在哪关就是哪种失败。
3. **是否值得进答案** —— 反事实判断，五个 `decision_effect` 取值。

**关键点：LLM 不是被信任，而是被测。**
LLM 判别器在这里扮演「可复现的、快速的、但可能有偏的标注员」。
金标集提供客观锚点，用来**量化这个标注员的 precision / recall / F1**。
判别力不够就改规格、换模型、加约束 —— 而不是把它当成真理。

**但这把尺子自己也必须被测**——这就是 `lexical_baseline` 的用途（§4）。

### 2.2 三层结构

```
┌─────────────────────────────────────────────────────────┐
│  金标集（30 条，人工定标）                                 │
│  gold: out_of_frame_directions + status + decision_effect │
└──────────────────────┬──────────────────────────────────┘
                       │  提供锚点
                       ▼
┌─────────────────────────────────────────────────────────┐
│  抽取器（LLM 判别器）  ← extractor_spec.md 规定            │
│  输入: task / initial_frame / trace / final_answer        │
│  输出: 严格 JSON（directions[] + frame_ambiguous）        │
└──────────────────────┬──────────────────────────────────┘
                       │  预测 JSONL
                       ▼
┌─────────────────────────────────────────────────────────┐
│  评测器（本目录 evaluator.py）                            │
│  主表 P/R/F1 · 漏判单表 · status 混淆矩阵 · 词法基线       │
└─────────────────────────────────────────────────────────┘
```

**与 CFC 四格的关系**：CFC（章程 §2.2）通过 F0/F1 双跑给出能力类型学，**不需要 gold**；
本评测器给出三分类归因，**需要 gold**。两层互补（M3 阶段接上），接驳点见 §6。

---

## 3. 四条判定规则（理论内核）

完整判据见 `extractor_spec.md` §3。

### 规则 1：`in_frame` —— 以 `initial_frame` 的显式描述为准

- **判据**：只认 `initial_frame` 的显式描述。**不得**用「听起来新不新」「听起来少见」「这方向很重要」判断。
- **操作化**：先建 FRAME 清单，再用**逃逸测试**：命中任一逃逸 → 域外；否则域内。
  1. 新候选 2. 新维度 3. 新约束 4. 新问题 5. 重定义
- **封闭性决定「新候选」一类能不能判**（裁决 ④）：
  - `closed` 框架：引入清单外候选 = 逃逸成功 = 域外
  - `open` 框架：「新候选」一类**自动失效**，只剩其余四类可判
- **反例**（这些都判**域内**）：换个说法、域内候选的子情形、域内候选 + 跨领域比喻包装。

### 规则 2：`status` v1.1 —— 锚点是**最终答案段**（裁决 ⑧-a）

| status | v1.1 定义 | 失败归属 |
|---|---|---|
| `not_generated` | 整个 trace 全文无该方向的任何表达 | 发现失败 |
| `selection_failure` | 最终答案中该方向被明确排除 / 否决 / 推迟；**或**中间段带推迟/搁置措辞而最终答案完全未提 | 选择失败 |
| `expression_failure` | 中间段**明确保留、无搁置措辞**，但最终答案完全未提及（⑬-b 修正后的窄定义） | 表达失败 |
| `realized` | 最终答案中该方向作为可操作选项 / 推荐 / 结论出现 | — |

```
最终答案里提到它了吗？
├─ 提到了 ──→ 作为可操作选项/推荐/结论？ → realized
│            └─ 明确排除/否决/推迟？        → selection_failure
└─ 没提到 ──→ 中间段被否决 → selection_failure
            中间段保留过 → expression_failure
            全文没出现   → not_generated
```

- **status 描述的是「对最终决策的贡献」，不是「它在哪一段被提到」。**
  「被讨论过」≠「被采纳」—— 这正是旧流水线定义的缺陷。
- **推迟与否决在决策后果上等价**（都没产生可操作增量）→ 归 `selection_failure`。
- **`expression_failure` 只覆盖「无声消失」**。若「明确推迟」也归它，该类就退化成
  「所有没落地的」，失去区分能力 —— 而这正是这把尺子要诊断的核心。
- **保守默认裁定（①-c）**：既无否决痕迹、又无保留痕迹（被静默丢弃）→ **默认判 `selection_failure`**。
- **歧义必须诚实标注**：同时标 `status_ambiguous: true`；报告里
  `status_accuracy_strict` 与 `status_undecidable_rate` **必须一起读**。

> **对 ⑧-a 的一处最小修补（需主会话追认）**：⑧-a 的**定义表**说 `not_generated` =
> 「trace 全文无该方向的任何表达」，而**判定树**的最后一支说「没提到 + 没保留 → not_generated」。
> 二者在「中间段被明确否决、最终答案没提」这一格上互相矛盾 —— 照字面执行会把「被否决」
> 并进「发现失败」，恰好抹掉这把尺子最想区分的那对失败，与 ⑧-a 理由 3 自相矛盾。
> 本实现按**定义表**取「中间段被否决 → `selection_failure`」，使四类互斥且完备。
> **未为迁就旧金标留任何后门**。

### 规则 3：`evidence_quote` —— 逐字，找不到就空

- 必须是 trace 中 `content` 的**连续逐字子串**。禁止改写、翻译、合并片段、用 `...` 省略。
- 找不到逐字匹配 ⇔ 这个方向不在 trace 里 ⇔ status 应为 `not_generated`，引用留空。
- **可核语料 = 完整 trace，**含**最终答案段**（裁决 ⑧-e，取代 ⑦-b 的「排除最终答案」做法）。
  排除最终答案的旧做法已作废 —— 它让 `realized` 无法举证（其引用按规格应引最终答案原话）。
  但「该方向是否出现在最终答案里」这一项判定**不能用裸 substring**，必须用**方向级匹配**
  （把最终答案切子句，逐句与方向比），否则方向描述与最终答案措辞不同就会漏判。
- **`task` 仍不进检索池**（裁决 ⑦-b）。把 task 纳入会让抽取器能引 task 里的内容
  伪装成「我在 trace 里见过这个方向」，把 `not_generated` 误判成有生成 ——
  污染四类里最不能错的一类。
- **防伪检查（⑧-e.3）**：若某方向的 `evidence_quote` **只在最终答案段命中**，
  而抽取器判的是 `expression_failure` / `not_generated` → 记入报告的 `violations` 桶。

### 规则 4：`decision_effect` —— 硬闸门（schema v2）

| decision_effect | 计入「有实质改变」 | 说明 |
|---|---|---|
| `conclusion_flip` | ✅ | 结论/推荐**实体**改变 |
| `option_added` | ✅ | 新增了最终答案里没有的可操作选项 |
| `ranking_only` | ❌（但记录） | 只有排序变，被选实体集合没变 |
| `additive_only` | ❌（但记录） | 只加限定条件 |
| `none` | ❌ | 纯修辞 |

`ranking_only` / `additive_only` 单列为 `weak_effect_count`，**不计入主分但完整报告**。

**为什么不要求「实体候选集变化」才算强效果**：在志愿填报、方案推荐这类场景里，
**排序变化本身就是决策变化**。强形式化会系统性误杀 `ranking_only`，
把尺子偏向「提出全新候选」这一种能力。

---

## 4. 词法捷径基线（尺子自曝偷懒空间）

**背景**：金标构造方在统一 trace 措辞时发现，可以把 status 判据**直接写进 trace 的措辞里**。
这样抽取器只需做**关键词匹配**就能判对 status —— **尺子退化成词法匹配器**。
这与 A 组踩过的坑同类：**答案的判定依据被写进了被评对象本身**。

**裁定（⑦-a / ⑧-d）**：措辞规范在合成语料阶段允许执行，但**评测器必须同时实现词法捷径基线
并报告它的得分**；且**必须跑多个实现变体，门槛以其中最高的 F1 为准**。

> 门槛的作用是「证明尺子学到了语义」—— 那就必须打败**最努力的词法匹配器**，
> 不是最容易被打败的那个。同一个「基线」在不同实现下从 0.49 浮动到 0.83，
> 直接把门槛建在某一个实现上是不稳的。

### 4.1 变体矩阵（6 个）

| 维度 | 取值 |
|---|---|
| 语料 | ① 仅 `evidence_quote` ② 整条 trace ③ **仅最终答案段** |
| 词表 | **宽**（否决/不采纳/不执行/不列入/排除/暂不推进/**不建议/不推进**）· **窄**（仅 否决/不采纳/不执行/不列入/排除） |

`evaluator.py` 的 `lexical_baseline_matrix()` 跑全部 6 个变体，报告里**全表列出**（区间本身是信息），
`true_increment` 用其中**最高**的 F1：

```
真实增量 = 抽取器 F1 − 词法基线最高 F1 = X.XX
```

**M2 硬门槛**：`true_increment.f1 ≥ 0.10`。达不到说明抽取器没学到语义 ——
**该改抽取器，不该改门槛**。

### 4.2 三种实现的差异（重要）

| 实现 | 做法 | 在真实 30 条金标上的 F1 |
|---|---|---|
| `evaluator.py` 内置（本目录） | 沿用抽取器给出的方向集合与 `in_frame`，**只把 status 换成词法推断** | 0.26 ~ **0.83**（取决于变体） |
| `goldset/lexical_baseline.py`（金标构造方） | **端到端**：自己切候选、自己判域外、自己判 status | **0.696** |
| 主会话独立实现 | 只用 `evidence_quote` 语料 | **0.487** |

本目录的实现**隔离出 status 判断那一段的语义贡献**（方向集合沿用抽取器输出），
另两个是**端到端的完整词法系统**。三者测的不是同一件事：

- 本目录的：**「如果 status 只用关键词判，会掉多少分」** —— 衡量 status 判断的语义增量
- 端到端的：**「一个纯关键词系统从零开始能做到多少分」** —— 衡量整个抽取器的下限

> **status 准确率不是这把尺子的主要区分度来源**（实测：闭卷基线 status_acc 仅 0.475–0.49，
> 开卷 0.53–0.61）—— 因为 stance 词在最终答案里是明写的，status 这一步闭卷下也能被词法蒙对一大半。
> **主表 F1 与漏判召回才是区分度的主要来源。** 出报告时别让人误读成「status_acc 0.48 = 尺子很严」。

**门槛基准的取法（重要）**：

- 门槛必须是**固定基准**，不能随抽取器的候选质量浮动。
  本目录的 6 个变体是 **status_only** 口径（方向集合沿用抽取器输出），
  它的 F1 **跟着抽取器走** —— 用它当门槛是**循环论证**。
  所以：**门槛基准只取 `end_to_end`（抽取器无关）的实现**。
- 端到端实现目前有：金标构造方的 `goldset/lexical_baseline.py`（F1 0.696，调参后最高 **0.737**）
  与主会话的独立实现（0.487）。
  用 `--baseline-result <json>` 把它们接进评测器，门槛基准自动取其中最高。
- 本地 6 个变体**仍然全表报告**（裁决 ⑧-d 要求），作为
  「如果 status 只用关键词判，抽取器会掉多少分」的诊断。
  **注意它们的主表 F1 不可与端到端 F1 比较**（方向集合沿用抽取器输出），
  对它们有区分度的是 **status 准确率**（报告里 `headline_metric` 标了）。
- 扫描表用 `--baseline-scan` 接入：门槛基准取 `mode=closed` 里的最高 F1，
  `mode=open` 只作参考（**不进门槛**，裁决 ⑨-b）。

### 4.4 ★ 门槛基准的定版（⑯-c：三个数并列，不取平均不取最高）

本轮出现过**三个基线数**。⑯-c 的纪律是：**不要取平均、不要取最高 ——
先找出差异来源**。定位结果是：三个数**都是合法的不同测量定义**，差异来自
**匹配函数 / 匹配阈值 / FP 计数口径**，**不是脏数据**。

| 数 | 测量定义 | 分类与状态 |
|---|---|---|
| **0.9381** | 金标构造方 `score()` 的 fp 逐条覆盖 bug | **脏** —— 作废（虚高近一倍，全程无报错） |
| **0.5106** | 金标构造方 `score()`，但**池未清空**（零方向条目的 15 个预测进了主表 FP） | **池口径不一致** —— 基线被低估 0.06；修完后与评测器路径同数 0.5749，此数本身作废 |
| **0.5397** | 评测器 scorer + 评测器匹配 @0.34（评测器默认阈值） | **规范口径** —— 参考点 |
| **0.5714** | 评测器 scorer + 评测器匹配 @0.20–0.25（最有利阈值），**池已清空** | **规范口径** —— 门槛基准 0.5749 |

**规范定义**：门槛比较的是「抽取器 F1 vs 基线 F1」，两者必须由**同一套打分路径**算出。
抽取器 F1 来自评测器 → 基线也必须来自评测器 →
**规范定义 = 评测器自己的 scorer + 评测器自己的 matcher，并对匹配阈值取最大**
（`calibrate_baseline.py` 负责算，导出 `baseline_main_caliber.json`）。

**⑳-a 追认**：先前那三个数**不是合法的不同测量定义，是同一个缺陷（池口径）的三个投影** ——
金标构造方的基线**没在池层面清空零方向条目**，那 15 个预测进了主表 FP → 基线被**低估** 0.06。
修完后两条独立路径同数 **0.5749** → 按 ⑯-c 的纪律，**输入是干净的**。

**⑳-b 同点比较（硬要求）**：抽取器必须在**取得 0.5749 的那个匹配阈值**上被评分
（`match_thr = 0.25`，变体 V1，`oof_thr = 0.15`），
**禁止**抽取器在多阈值上取最优再与门槛比。

```
门槛基准 F1 = 0.5749      →  M2 门槛 = 0.5749 + 0.10 = 0.6749
上界报警线 = 0.7664（开卷·主表口径）—— 超过触发人工检查，不是通过条件
```

**基线在匹配阈值上的敏感性曲线**（本项目测到的最敏感参数）：
```
match_thr  0.20  0.25  0.30  0.34  0.35  0.40  0.45  0.50  0.55  0.60
基线 F1   .5749 .5749 .5749 .5509 .5509 .5150 .4790 .4790 .4431 .3537
```
**跨度 0.2212** —— 所以 **M2 报告必须给出抽取器在整条曲线（0.20–0.60）上的 F1**，
单点数字没有意义（`--sweep` 出这条曲线）。

报告里所有基准数**并列呈现并标注来源**，不隐藏。

### 4.5 ★ 门槛必须与抽取器**同口径**（实测发现）

词法基线原始的 F1 用的是**「含 miss detection」口径**（所有金标域外方向都在分母里），
而抽取器的主表 F1（裁决 ②-b）把 `not_generated` 单独拆出 —— **两者不可直接比较**。

实测（闭卷，8 变体 × 428 点，同一批预测换口径重算）：

| 口径 | 词法基线最高 F1 | 同口径门槛 |
|---|---|---|
| 含 miss detection（扫描表原始口径） | 0.4932 | 0.5932 ← **不可用于门槛** |
| **主表口径**（与抽取器一致） | **0.5714** | **0.6714** ← 门槛应取这个 |

**拿含 miss 的基线去卡不含 miss 的抽取器，等于把门槛放松了 0.078。**
`diagnose_increment_source.py` 会重算并导出 `baseline_main_caliber.json`，
评测器用 `--baseline-main-caliber` 接入，**门槛自动用同口径的那一份**。

### 4.6 ★ 增量来自哪个维度（诊断量）

读金标构造方 `score()` 的实现会发现：**status 根本不进 tp/fp/fn** ——
匹配上就是 tp，匹配不上就是 fn/fp。所以他们的 F1 纯粹是「**域外判定**」维度，
而 `status_acc` 是「**status 判定**」维度，**两个维度被分开测量了**。

闭卷实测：

| 维度 | 词法能达到 | 结论 |
|---|---|---|
| 域外判定 | F1 **0.4932** | 明显弱 |
| status 判定 | status_acc **0.7119**（V7） | 不算弱 |

→ **词法在 status 这步已经挺强，抽取器的增量必须主要来自域外判定。**
工程含义：优先把域外判定做对（语义判断「这个动作是不是常规解」），
status 可以更多交给词法。

**另一个结构性事实**：词法基线**从不产出 `not_generated`**（它只在 trace 里找得到东西）。
而主表口径里 status 只通过「`not_generated` 与主表三类的那条边界」影响 F1 ——
realized/selection/expression 之间的互换**不改变 tp/fp/fn**。
（`oracle_status` 探针实测抬升 = 0.0000，不是 bug。）

⟹ **主表 F1 真正在意的是「这个方向到底有没有被生成」，不是「它死在选择还是死在表达」。**
**三分类是诊断维度，不是计分维度** —— 这从数据上印证了裁决 ⑪-c。

### 4.3 词表可配置

```bash
python3 evaluator.py --gold g.jsonl --pred p.jsonl --lexicon my_lexicon.json
```

文件格式：`{"rejection": ["否决", ...], "retention": ["保留", ...]}`。缺省键回落到内置值。

> 注意：示例（`examples/`）里的 trace **故意没有**遵循措辞规范，所以词法基线在示例上只有 0.00 分。
> **在真实金标集上（措辞规范被执行），词法基线会高得多** —— 那正是这个指标要暴露的东西。

## 5. 怎么用

### 5.1 跑自测（不依赖 LLM、不依赖网络）

```bash
cd /workspace/c-group/eval
python3 evaluator.py --self-test
```

预期输出末行：`159/159 passed` + `ALL PASSED`。

### 5.2 接金标集打分

```bash
# ★ M2 主跑批的最终命令（已确认）
python3 eval/evaluator.py \
  --gold      goldset/gold_v1.jsonl \
  --pred      <第三轮抽取器输出.jsonl> \
  --baseline-main-caliber eval/baseline_main_caliber.json \
  --dual-open-gold   goldset/gold_v1_openbook.jsonl \
  --dual-open-pred   <开卷对照抽取器输出.jsonl> \
  --sweep --sweep-step 0.05 \
  --out report.json
```

**M2 主跑批**不传 `--prp-probe` / `--prp-prompts` / `--premises`：
M2 测的是 PSC-Eval 抽取器、不是 PRP 探针；且本集合法 prp 靶子 = 0，探针 B/C 跑了也是空。
但**「prp 桶可准入: N/M」这行照样打**（哪怕 0/0）——
让读者知道这条路径**被检查过**，而不是不存在。

跑 PRP 探针时才加这三个参数（路径省略时自动发现同级的 `goldset/`）：
```bash
python3 eval/evaluator.py --gold goldset/gold_v1.jsonl --pred <prp输出.jsonl> \
  --prp-probe C_escape_direction \
  --prp-prompts eval/prompts --premises goldset/premises.jsonl
```

| 参数 | 说明 |
|---|---|
| `--gold` | 金标集 JSONL，每行一个任务（schema 见 §7） |
| `--pred` | 抽取器输出 JSONL，`id` 与金标对齐 |
| `--out` | 报告输出（JSON，含逐条明细） |
| `--matcher module:function` | 自定义方向匹配函数（可插拔） |
| `--match-threshold FLOAT` | 覆盖 Jaccard 阈值（默认 0.34，启发式，**待校准**） |
| `--include-miss-detection` | 把 `not_generated` 折回主表（旧行为）。**默认关闭** |
| `--lexicon path.json` | 词法捷径基线的词表 |
| `--baseline-scan path.csv` | 词法基线**扫描表**（mode/variant/…/f1）。门槛基准取 `mode=closed` 里的最高 F1（裁决 ⑨-c） |
| `--baseline-main-caliber path.json` | **主表口径**的基线（由 `diagnose_increment_source.py` 生成）。**门槛应优先用它** —— 与抽取器同口径（见 §4.4） |
| `--reference-mode` | 上界参照模式（人工构造的完美预测等）。该模式下**不参与门槛判定，也不参与 upper_alarm 判定** |
| `--dual-open-gold` / `--dual-open-pred` | 开卷对照组。同一抽取器跑两遍，**闭卷进门槛，开卷只作对照**（裁决 ⑨-b） |

现成可跑的样例：

```bash
python3 evaluator.py --gold examples/gold_sample.jsonl \
                     --pred  examples/pred_sample.jsonl \
                     --out  examples/report_sample.json
```

### 5.3 API

```python
from evaluator import (
    load_gold, load_predictions,
    score_prediction, status_agreement, status_agreement_strict, aggregate,
    lexical_baseline, frame_schema, similarity, default_match_fn,
)

gold = load_gold("gold.jsonl")
preds = {p["id"]: p for p in load_predictions("pred.jsonl")}

results = [score_prediction(g, preds[g["id"]]) for g in gold]
report = aggregate(results)

r = results[0]
r["precision"], r["recall"], r["f1"]        # 主表（不含 miss detection）
r["miss_detection"]["miss_detection_recall"] # 漏判单表
r["status_confusion"]                         # 行=gold status，列=pred status
r["selection_expression_separability"]        # selection/expression 可分性
r["lexical_baseline"]["f1"]                   # 词法捷径基线
r["true_increment"]["f1"]                     # 真实增量

status_agreement(gold[0], preds[gold[0]["id"]])        # 宽松：匹配上的对全部判对
status_agreement_strict(gold[0], preds[gold[0]["id"]])  # 严格：所有金标方向都被找到且判对
```

**`status_agreement` 的语义（有意与方向召回解耦）**：
一个抽取器可能只找对了一半方向，但在找到的那一半上把「发现/选择/表达失败」判得很准。
这个信息不应该被召回率淹没，所以 `status_agreement` **只看匹配上的那些对**：
True ⇔ 至少匹配上一对，且每一对的 gold status == pred status。
一个都没匹配上时返回 `False`（没有「判对」的证据，而不是「判错」）。

### 5.4 方向匹配

方向匹配用**语义近似**而非字符串相等：

```python
similarity(a, b)  # 字符二元组 Jaccard，0~1
default_match_fn  # similarity >= 0.34 判为同向
```

中文不依赖分词：直接对归一化后的字符串取字符二元组。匹配是**贪心最优**
（500 组随机用例验证：贪心 == 穷举最优匹配，全量与分表两种口径都成立）。

匹配函数可插拔：

```python
def my_match(gold_dir, pred_dir) -> bool:
    return ...   # 收到 dict，取方向文本自行判断

score_prediction(gold, pred, match_fn=my_match)
# CLI: --matcher mymodule:my_match
```

---

## 6. CFC 四格与 F1 构造规范

CFC（章程 §2.2）通过 F0/F1 双跑给出能力类型学，**不需要 gold**。
本评测器给出三分类归因，**需要 gold**。两层互补（M3 阶段接上）。

### 6.1 F1 诱导框架的构造规范（裁决 ⑤）

没有规范，CFC 的四格就不可比。

**F0（默认框架）**：由任务给定者按 §7.2 的四段式模板写，候选集填**题面蕴含的常规解**，
封闭性填 `open`（因为常规解通常不是穷尽的）。

**F1（诱导框架）**：**在 F0 的基础上，只做一处改动** —— 末尾追加一行：

```
【新增】请额外评估以下方向：<方向 D>
```

其余三段（候选集、评估维度、硬约束）与 F0 **完全一致**，只允许在候选集末尾追加 D，且：

- **不得**解释为什么要考虑 D（否则就成了「引导语」而不是「框架」）
- **不得**改动其他任何措辞
- 封闭性保持 F0 的取值不变

**F1 判定是二值的**：最终答案里出现 D = 命中，没出现 = 未命中。
**不依赖语义匹配阈值**（避免阈值问题污染 CFC）。

**D 的来源限制**：必须来自已知域外方向（金标集或 A/B 组提供的已知遗漏方向），
**不允许由抽取器现编 D** —— 那会变成「抽取器自己出的题自己考」。

### 6.2 接驳点

1. 每条 CFC 任务产生**两个**评测输入：`F0`（默认框架）和 `F1`（显式点名域外方向）。
2. 两次运行各产出一份 PSC-Eval 报告（同一个 `evaluator.py`，框架不同）。
3. CFC 判据只看 F1 的最终答案有没有出现被点名的那个方向 —— 用 F1 报告的 `matched` 明细定位即可。
4. 四格落表：

| | F1 带出域外方向 | F1 带不出 |
|---|---|---|
| **F0 已有域外方向** | ① 自主构建型 | ③ 偶发不稳定型 |
| **F0 无域外方向** | ② 依赖引导型 | ④ 框架锁定型 |

**F0 那一格的「已有域外方向」正好由 PSC-Eval 给出**（F0 报告里 `tp > 0` 即是）。
两层不是并列，而是咬合的。

---

## 7. 金标集 schema

一行一条 JSON（JSONL）：

```json
{
  "id": "psc-001",
  "task": "为一名山东考生推荐 5 所大学",
  "initial_frame": "【候选集】985/211 院校中的候选学校\n【封闭性】closed\n【评估维度】分数位次、就业率\n【硬约束】无",
  "trace": [
    {"stage": "explore", "content": "先按分数位次排一下。",
     "candidates": ["按分数位次排序"]},
    {"stage": "select", "content": "还可以看看家庭支持这个维度，但本题没给这个维度，先划掉。",
     "candidates": ["按分数位次排序", "把家庭支持作为选校维度"],
     "dropped": [{"direction": "把家庭支持作为选校维度", "reason": "本题未给该维度", "by": "收敛阶段自检"}]},
    {"stage": "synthesize", "content": "最终推荐：……", "candidates": ["按分数位次排序"]}
  ],
  "final_answer": "推荐山大、哈工大、南大三所。",
  "out_of_frame_directions": [
    {"direction": "把家庭支持作为选校维度", "status": "selection_failure",
     "decision_effect": "option_added", "evidence_quote": "还可以看看家庭支持这个维度"},
    {"direction": "考虑强基计划等特殊招生路径", "status": "not_generated",
     "decision_effect": "conclusion_flip", "evidence_quote": "",
     "expected_source": "task", "notes": "关键漏判：trace 全文无任何提及"}
  ],
  "in_frame_directions": [
    {"direction": "按分数位次排序", "status": "realized", "decision_effect": "none",
     "evidence_quote": "先按分数位次排一下。"}
  ]
}
```

| 字段 | 必填 | 用途 |
|---|---|---|
| `id` | ✅ | 与预测对齐的唯一键（重复会报错） |
| `initial_frame` | ✅ | **必须符合四段式模板 + 封闭性声明**，否则标 `frame_schema: non_conforming` |
| `trace[].candidates[]` / `trace[].dropped[]` | 强烈建议 | 区分 selection/expression 的证据来源（裁决 ①-a） |
| `out_of_frame_directions` | ✅ | 金标正例集。参与主表 P/R/F1 与 status 混淆矩阵 |
| `in_frame_directions` | — | 金标**陷阱集**。抽取器若把这些判成域外，即为 over-claiming（算 FP 并单独计数） |
| `out_of_frame_directions[].status` | ✅ | canonical = `not_generated`；`discovery_failure` 是别名，**双向归一**，金标可继续写别名 |
| `out_of_frame_directions[].decision_effect` | ✅ | schema v2 硬闸门字段。`not_generated` 条目**必须**是强效果（关键漏判） |
| `out_of_frame_directions[].evidence_quote` | 否 | `not_generated` 条目**必须留空**，依据写进 `notes` |
| `out_of_frame_directions[].expected_source` | 否 | 裁决 ⑦-b：记录「本该被想到」的来源 ∈ {`task`, `external`, `gold_author`}。**仅统计，不参与判定** |
| `out_of_frame_directions[].notes` | 否 | 依据说明 |

**预测文件**同构：`id` + `directions`（抽取器输出）。也兼容把输出包在
`prediction` / `extraction` / `result` / `output` 下的写法。

### 7.1 已兼容的 schema 变体

| 变体 | 处理方式 |
|---|---|
| 方向集嵌在 `gold` 对象里 | 自动提升到顶层 |
| 证据字段叫 `evidence` 而非 `evidence_quote` | 两种都认 |
| status 写 `discovery_failure` | 归一到 canonical `not_generated`（双向）。**报告输出一律用 canonical 名** |
| 旧字段 `would_change_decision`（bool） | 兼容映射：true→`conclusion_flip`，false→`none`。**金标集须升级到新字段** |
| `trace` 是纯字符串数组 | 认 |
| 最终答案内嵌在 trace 末段 | 自动识别，并**从引用语料中排除** |

> 最后一条对**规则 3 是必须的**：若把最终答案算进可引用语料，
> `expression_failure` 会被系统性误判成 `realized`，三分类归因随之失效。

### 7.2 `initial_frame` 强制模板

```
【候选集】<显式枚举的常规候选；至少列出题面蕴含的全部常规选项>
【封闭性】closed | open
          closed = 该候选集穷尽了本题的全部可能解
          open   = 该候选集只是常规解，题面允许更多
【评估维度】<显式列举用于比较候选项的维度>
【硬约束】<不可违反的边界；没有就写"无">
```

评测器会检查 `initial_frame` 是否符合本模板；不符合的标 `frame_schema: "non_conforming"`，
**报告中不得与 conforming 数据并列排名**（只能做「同一任务内的相对比较」）。

---

## 8. 报告字段

| 字段 | 含义 |
|---|---|
| `main` | **主表**（不含 miss detection）：`precision` / `recall` / `f1` / `tp` / `fp` / `fn` / `macro_f1` |
| `miss_detection` | **漏判单表**：`miss_detection_recall`（下界）、`miss_detection_precision`、`hit` / `missed` / `spurious_miss_count` + 免责句 |
| `status_confusion` | 4×4，行=gold status，列=pred status。**只在匹配上的对上统计** |
| `status_accuracy` | 全部匹配对的 status 准确率 |
| `status_accuracy_strict` | **仅可判条目**（抽取器未标 `status_ambiguous`）的准确率 |
| `status_undecidable_rate` | 标了 `status_ambiguous` 的条目占比。**必须与上面两个一起读** |
| `selection_expression_separability` | gold 为 selection/expression 的方向里，判对的比例。**尺子的质量刻度** |
| `decision_effect_agreement` / `decision_effect_exact_agreement` | 硬闸门一致性（强弱口径 / 五类精确口径） |
| `weak_effect_count` | `ranking_only` + `additive_only` 的条数（不计入主分但完整报告） |
| `lexical_baseline` | 词法捷径基线：**6 个变体全表** + F1 区间 + 门槛基准变体名 |
| `true_increment` | 抽取器 F1 − **词法基线最高** F1。**M2 硬门槛 ≥ 0.10** |
| `zero_direction_precision` / `zero_direction_overclaim_rate` | 零方向条目（正确答案是「没有域外方向」）上的正确率与过度声称率。**正向质量信号**（裁决 ⑧-c） |
| `violations` | 防伪违规桶：引用只命中最终答案段却判 `expression_failure`/`not_generated`（裁决 ⑧-e.3） |
| `constraint_violation_candidates` | **独立诊断桶**（裁决 ⑭-c，**已收窄**）：只收「**not_generated 候选**」—— 本该被想到、却因触碰禁区而**根本不可能出现**的那一类。**凡 trace 里已出现过的方向，即使触碰禁区也照常留在主表** —— 「被否决/被推迟/被采纳」是 status 维度，「禁区」是「该方向可不可行」维度，两者正交；移出主表 = 把 status 的诊断材料也扔了 |
| `constraint_violation_sensitivity` | **两桶条目数**（not_generated 候选 / 已在 trace 出现）。⑲-c 那个「不排除」上界对照**已作废** —— 收窄后桶里只剩 not_generated 候选，而它本来就不进主表 → 排除与否对 F1 无差别，对照恒等于 0、无信息量 |
| `status_support` / `zero_sample_statuses` | 各类 status 在金标里的样本数，以及**零样本的类**。零样本类**本集测不到**，其召回率/准确率**不得作为尺子质量证据**引用 |
| `match_threshold_used` | 报告用的匹配阈值。**钉在门槛点**（0.25），不随 `--sweep` 漂移 —— 一个随参数漂移的指标，测的就不是它该测的东西 |
| `unknown_source_gate_pass` | `expected_source` 取值未登记的条目数是否为 0。**不为 0 则 M2 不通过**（那是真的标注坏了） |
| `baseline_number_provenance` | 门槛基准的**四类**数及其测量定义（脏 / 池口径 / 规范口径，含那个虚高值 0.9381）—— 裁决 ⑳-a |
| `unknown_source_in_main` / `unknown_source_details` | 主召回桶里**未被显式审核**的方向数与逐条明细（`expected_source` 未标注 / 取值未登记）。㉓-e ② 降级显式化 |
| `branch_coverage` / `branches_without_real_data` | 哪些分支被真实数据走到、哪些**恒不触发**（其正确性由合成用例保证，不由本集数据保证） |
| `miss_bucket_reachability` | 每个 `expected_source` 实际路由到哪个桶（可达性断言的输出，㉓-e ①） |
| `empty_direction_responses` | 探针 C 的合法「空方向」回答数（**前提没被挑战过**）。**不计误报** —— 否则会把正确答案当成 false positive |
| `miss_detection.prp_admissibility` | **`prp` 桶的逐条可准入判定**（含不可准入的原因），可追溯为什么某条没进主分母 |
| `stance_misattribution` | 分歧分诊线索：金标 `expression_failure` 被判成 `selection_failure` 且引用含否定语气词（多为一方向歧义）。**只做分诊，不做判决** |
| `introduced_in_synthesis` / `_count` | **只在最终答案出现、中间段没有**的域外方向 —— 它是在**综合阶段新产生**的（⑲-a）。**算被生成了**，不是 `not_generated`；这恰恰是这把尺子最该测的能力（Agent 能不能在收敛中自己长出新方向） |
| `constraint_violation_details` | 触碰/违反【硬约束】禁区段（X 段）的方向**逐条明细**（⑲-c） |
| `constraint_violation_sensitivity` | **上界对照**：把违规方向**不排除**时的 F1。差值越大 =「排除」这个决定对 F1 的影响越大，越要说明清楚。**X 段判定过宽会虚高 F1** |
| `miss_detection_recall_author_only` / `_low_reportability` | 两个单列桶（⑳ 裁定）。主召回只算 `main` 桶 —— 分母不能是「标注者的领域知识」或「可报性弱」 |
| `miss_detection.prp_admissibility` | **`prp` 桶的逐条可准入判定**（含不可准入的原因），可追溯为什么某条没进主分母 |
| `m2_gate.f1_at_gate_point` | 抽取器在**门槛点匹配阈值**上重算的 F1（⑳-b 同点比较用） |
| `m2_gate.pool_inconsistent_baselines` | 被池口径守卫拦下的基线（⑳-c） |
| `audit_coverage` | **审计覆盖率**：金标方向里有多少条被 m2_gate 实际审到。**必须 = 100%**，否则 M2 不予通过 |
| `miss_detection_recall_author_only` | `expected_source = gold_author` 的漏判召回（不计入主召回，裁决 ⑧-b） |
| `quote_grounding_rate` | **应有引用且能指到原文** / 应有引用。`not_generated` 的合法空引用不进分母 |
| `gold_evidence_stats` | **金标自身**证据的三档核验（`exact` / `punct` / `ungrounded`）。漏判条目（`not_generated`）的 evidence 是「为什么它不在 trace 里」的说明而非原文引用，**不计入可核率**（`gold_evidence_skipped_not_generated`） |
| `expected_source_distribution` | `expected_source` 分布（仅统计） |
| `frame_schema` | `conforming` / `non_conforming` 计数与非合规 id 列表 |
| `m2_gate` | M2 验收清单的机检项（见 §10） |
| `per_status` / `per_effect` | 每个 status / 每个 decision_effect 的 P/R/F1 |

**报告读法建议**：

1. **先看 `quote_grounding_rate`**。低于 1 说明抽取器在编引用 —— 这种情况下 P/R/F1 不可信，先修规格。
2. **再看 `true_increment`**。接近 0 说明这把尺子没学到语义 —— 该改抽取器。
3. **再看 `per_status` 里 `not_generated` 的 recall**。这是「关键漏判抓到了几个」。
4. **`status_confusion` 的非对角线告诉你该修什么**：
   - `selection_failure → expression_failure` 多 → 抽取器分不清「被否决」和「被保留但没写进去」
     → 优先让上游 trace 带上 `candidates[]` / `dropped[]`（裁决 ①-a，根本解法）
   - `not_generated → 其他` 多 → 抽取器没认真找证据，或规格没把「找不到就留空」讲透
5. **`frame_schema` 有 non_conforming** → 这些条目不能和 conforming 放一张榜比。

---

## 9. 开卷 / 闭卷（裁决 ⑨）

### 9.1 为什么这件事决定 M2 的数有没有意义

金标复核发现：**30/30 条金标的 `initial_frame` 都含一段「域外参考（被排除在解空间之外，不作为候选）：<域外方向逐条列出>」**。

而金标构造方的词法基线判域外的方式，正是**用字符二元组与这段「域外参考」求重叠比重**。

> 即：**域外方向被直接印在题面上，基线只需做字串比对就能判域外。**
> 这批金标是**开卷考试**。

真实世界里 `initial_frame` 是「题目蕴含的常规框架」，**不会告诉 Agent 哪些是域外** ——
那恰恰是 Agent 要自己发现的东西。

**闭卷下域外判定依然完全可操作**：逃逸测试的「新候选」在 `open` 框架下是
「方向不在【候选集】里 → 域外」。**只需要知道域内是什么，不需要知道域外是什么。**
这正是裁决 ④ 强制四段式模板的意义所在。

### 9.2 改造与双口径报告

| 口径 | 说明 | 用途 |
|---|---|---|
| **闭卷** | `initial_frame` 严格四段式，不含域外参考 | **主指标** |
| 开卷 | `initial_frame` 含域外参考 | **对照组**：抽取器复述能力上限。**不进门槛** |

- 域外参考移到金标新字段 **`frame_note`**（**抽取器不可见，仅评测器可见**）。
- 评测器**断言 `frame_note` 绝不进可核语料**（`frame_note_leaks` 必须为空，否则 M2 不通过）——
  它若被并进语料，抽取器就能引它当证据，把开卷泄漏伪装成「在 trace 里见过这个方向」。
- 报告里 `gold_mode` ∈ `closed_book` / `open_book` / `mixed`；
  `m2_gate.gold_closed_book` 为 false 时，`true_increment` 门槛**直接判不通过**。

### 9.3 候选集粒度（已查，结论健康 —— 避免后续重复怀疑）

| 指标 | 实测 |
|---|---|
| 【候选集】项数 | min 6 / max 18 / **中位 9** |
| 域外方向数 / 候选集项数 | min 0.08 / max 0.67 / **中位 0.25** |
| 封闭性 | 30/30 = `open` |
| 候选集项数 ≤ 2 的畸形条目 | **0** |

### 9.4 闭卷后 `zero_direction_precision` 预期下降 —— 这是真实难度，不是 bug

闭卷下，抽取器要判「没有域外方向」，必须确认 trace 里出现的**每一个**方向都在候选集内 ——
比开卷（可以对照域外参考清单）显著更难。

> **提前声明**：闭卷下该指标下降**不判为抽取器缺陷**。
> 若下降幅度超出预期，说明的是**闭卷金标的可判定性有问题**（例如候选集描述不够具体），
> 需要回头改金标，而不是放宽门槛。**两种情况要分清。**

**判读依据（金标侧已测）**：金标的「域外方向 vs 【候选集】词汇重叠」分布 =
min 0.000 / **中位 0.056** / max 0.278 —— 贴地，说明**闭卷可分性良好、人能判**。
所以若降幅超出预期，责任在金标的候选集描述，不在抽取器。

**闭卷下 `zero_direction_overclaim_rate` 反而是更值得看的数**：
它直接反映抽取器守没守住「漏判发现失败 ≫ 误判」的非对称义务。

---

## 9.4 ★ `prp` 桶的可准入判定（前提识别探针的前置条件）

`expected_source = "prp"` 的方向**进主召回桶**，但受四条硬条件约束
（缺一不可，否则降级到 `author_only` 单列，**不进主分母**）：

| # | 条件 | 不满足的后果 |
|---|---|---|
| a | **P 在 prompt 里是给定的**（探针 B/C 形态） | 探针 A（自己猜 P）属**探针 A 的能力范围**，不是漏判召回 → 不得进 prp 桶 |
| b | 该题 `premise` **非空**（渲染后 `{{PREMISE}}` 非空） | 渲染不成立 → 不可准入 |
| c | `trace_follows_premise == false`（trace 确实违反 P） | 「推翻 P」不是合法靶子 → 不可准入 |
| d | **`premise_raised_in_trace == false`**（trace **不得已经提出过**推翻 P 的方向） | 那条方向就在金标里（realized / expression_failure），再当漏判计入 = **「算成抽取器漏了它其实说过的方向」**，把主召回算虚高 |

**c 的细节**：取值可能是 `partial`（部分遵循）—— `partial` **不等于** `false`，
按保守处理视为**不合法靶子**。

**用法**：
```bash
python3 eval/evaluator.py --gold goldset/gold_v1.jsonl --pred <prp预测.jsonl> \
  --prp-probe C_escape_direction \
  --prp-prompts goldset/prp_prompts --premises goldset/premises.jsonl
```
路径省略时自动发现同级的 `goldset/prp_prompts` 与 `goldset/premises.jsonl`。

**本集实测**：30 条前提里 `trace_follows_premise` = false 14 条 / true 11 条 / partial 5 条；
`premise_raised_in_trace` = True 22 条 / False 8 条；
**合法 prp 靶子（违反 P 且 trace 未提出）= 0 条** ——
即本集上探针 C 只会重复金标里已有的方向。
**要产出可报的漏判，必须新造「前提依赖型任务」**（trace 悄悄违反前提而不提）。

---

## 9.5 ★ 跨来源比数字之前，先核对池口径与 scorer 身份（⑳-c / ⑳-d）

本项目在**口径**上栽了三次，每次的表现都是「报出来的数比该数大或小，但**不报错**」：

| 事故 | 表现 | 后果 |
|---|---|---|
| ⑯-b `fp` 逐条覆盖 | 虚高一倍（0.9381 vs 真值 0.5106） | 门槛被反向抬高 |
| ⑰-e 从旧 CSV 转换 | 继承同一个错值 | 污染两个下游 |
| ⑳-a **池口径不一致**（零方向条目未在池层面清空） | 基线被**低估** 0.06（15 个预测进了主表 FP） | 门槛被放松 |

⟹ **硬规则**：
1. 任何**跨来源**的数字比较，必须先核对**池口径**（零方向条目是否在池层面清空）
   与 **scorer 身份**（是不是同一套 `score_prediction`），**再比大小**。
2. **池口径不一致时，该基线不得作门槛基准** —— 评测器会拒绝把
   `pool_caliber` 不一致的基线接进门槛（`m2_gate.pool_inconsistent_baselines`）。
3. 门槛比较是**同点比较**（⑳-b）。
4. 任何打分函数都必须自带口径不变式断言（§9.5.1）。

## 9.5.1 ★ 口径不变式：宁可炸掉，不要算出一个更大的数（⑯-b）

> **「报出来的数比该数大」是最危险的失败模式** —— 报错会被查，虚高不会被查。
> **数字看起来漂亮时，先查不变式，再谈结论。**

金标构造方在 `score()` 里踩过这个坑：`fp / tp_f / fn_f / fp_f` 是**逐条覆盖**
（取最后一条的值）而不是累加，闭卷基线一度报成 **0.9381**（真值 0.5106）——
**虚高近一倍、全程无报错**。它第一次看到 0.938 时应该先怀疑，而不是当成好消息。

评测器的打分路径现在自带**三条口径不变式**，错分母的数据会**抛异常**而不是算出一个更大的数：

```
assert tp + fn == 主表金标方向数        # 分母 = 非 not_generated 的金标方向数
assert fp + 已匹配数 == 计分池里的预测数   # FP 口径
assert（全量匹配口径）tp + fn == 全部金标方向数
```

每条都配了**反向测试**（用错分母的数据必须触发异常）——
「断言通过」不等于「断言有效」，「计算它的函数」也要自带不变式。

---

## 10. 当前局限

1. **Jaccard 0.34 阈值是启发式，尚未校准** ⚠️ **待校准项**
   字符二元组 Jaccard ≥ 0.34 判为同向，这个数字来自经验取值，**没有在金标集上验证过**。
   **扫描前不得用于任何跨组结论**（裁决 ⑥-a），M2 必须扫 0.20–0.60 并报告**阈值敏感区间**
   （F1 变动 < 2% 的范围）。跨组比较时三方抽取器必须落在同一区间内。
   校准前 P/R/F1 的绝对值不可跨组比较，只能在同一阈值下做相对比较。

2. **主召回这一维在闭卷金标上测不到**：4 条 `not_generated` 全在单列桶
   （`author_only` 2 条 + `low_reportability` 2 条），**主召回分母 = 0**。
   报告按「**不可度量（分母为 0）**」出，并附一句
   「**靶子不可达 ≠ 选手没打中**」—— 是闭卷金标这一维测不到，不是抽取器不行。
   （将来若引入 `expected_source = "prp"` 的前提识别探针，主分母会重新变成非零；
   评测器已留钩子：`prp` 与任何未知 `expected_source` 值都**进主桶**，不静默落进单列桶。）

3. **`not_generated` 的 recall 是下界，不是覆盖率**
   金标只能**枚举**已知的漏判方向，无法穷举。`miss_detection_recall` 的含义是
   「对已知关键漏判的命中率」。同理 `miss_detection_precision` 也不是真正的精确率 ——
   被记为「误报」的方向可能只是没被枚举到。

3. **主表分表匹配，退化解拿不到分**
   主表只让「自称属主表三类」的预测去竞争金标的主三类方向；自称 `not_generated` 的预测
   去竞争漏判方向。这样「把所有方向都说成 `not_generated`」的抽取器会得到主表全 FN，
   而不会因为「不产生主表 FP」而刷出虚高的 F1。

4. **金标本身有主观性**
   「域外」与「值不值得」都是判断。金标是**多人标 + 分歧仲裁**的结果，不是客观真值。
   报告的 P/R/F1 测的是「抽取器与金标的一致度」，不完全等于「抽取器的正确率」。

5. **合成语料没有真实域**（章程 §6）
   B 组的负结论来自法律域，A 组在教育咨询域，C 组目前是合成语料。
   **迁移性未验证。** 任何「C 组证明 X 组的方法无效」的表述都是越界的。
   本评测器提供的是**同一把尺子下的可比数据**，不是跨域判死。

6. **人工复核是强制的，但当前尚未执行**（裁决 ⑥-b）
   v1 阶段（金标集 30 条）的最低复核量：`decision_effect` 为强的条目 **100% 复核**，
   `status` 判定 **30% 随机抽样复核**，由 C 组主会话执行，结果写入 `goldset/README.md`。
   **复核发现的错误必须回改金标，并在报告里记录改了多少条** —— 金标改了却不记录，等于偷偷改了考卷。

---

## 9.7 ★ 汇总量只能定位「哪儿有差异」，不能定位「谁错了」（⑳-b / ㉘-b）

M2 第一轮 FAIL 时，根因诊断出过一个错：只看 `delta = -0.1414`（排除约束违规 vs 不排除）
就断言「抽取器判定过宽」。**逐条核之后发现 22 条里有 18 条是金标与规则不一致** ——
抽取器是按规则办事的，是金标没跟上。

> **汇总量只能定位「哪儿有差异」，不能定位「谁错了」。**

这是本项目第 6 次「结论下得比证据快」，前 5 次是工程性的，**这次是判断性的** ——
所以它必须写进硬规则区，不能只留在某次事故的记录里。

**对应的做法**：任何「排除/纳入」类的口径决定，都要
① 出**逐条明细**（谁被排除了、为什么），
② 出**两桶条目数**（不是只出一个 delta），
③ 让人能**逐条复核** —— 而不是只看到一个聚合差值。

## 10.5 ★ M3 待办：前提线索显隐度 A/B 与「漏判不可测」的两类遮蔽

**先记一条判读指引**（对任何对照实验都适用）：
> 如果 **trace 自己把前提讲出来了**，删掉题面线索不会有任何变化 → A/B 得到一个
> **无法解释的零差异**：是「线索不重要」？还是「实验坏了」？**分不清。**
> 所以做 A/B 之前必须先离线查「前提的线索到底在题面、还是在 trace 里」。

金标构造方已把这把审计器做出来了：`goldset/source/leak_audit.py`
（逐题给出 `task_clue` / `trace_clue` 与判定）。
**扫完 30 道的结论：现有题里没有一道能直接做这个 A/B**（唯一候选 G11 的 trace 主动反驳了自己的前提），
必须新造一对题。

### 10.5.1 ★ 「漏判不可测」有**两类**遮蔽，补救路径完全不同

| 遮蔽类型 | 现象 | 补救路径 | 成本 |
|---|---|---|---|
| **【候选集】措辞遮蔽** | 前提在 trace 的最终答案里**是可见的**，但【候选集】的措辞把它挡住了（如 G25②：候选集写「在既有『本地居民优先』条件下的分配方式」） | **改候选集一句** | 低，30 道都能受益 |
| **trace 从未触及** | trace 里根本没提到过这个前提（如 G24①、G06、G26–G30） | **只能新造任务** | 高 |

⟹ **「漏判不可测」里有一部分只是「框架写得太含蓄」—— 那是能修的。**
这条直接影响补救路径的选择，建议写进 M3 的判读指引。

### 10.5.2 探针 C 渲染规则（**别自动填 `{{VIOLATION_QUOTE}}`**）

探针 C 的 prompt 里 `{{VIOLATION_QUOTE}}` 这一栏**不要从题面/trace 里自动填** ——
若自动填，抽取器就拿到了「落差在哪」的提示，**等于把探针 A 的识别能力也一起送了**。
要么留空让抽取器自己找，要么只在「已确认 trace 违反 P」的题上给。

**评测器侧已配合**：探针 C 的 system 允许一个**空方向**回答
（`{"direction": "", "in_frame": true, "in_frame_reason": "trace 未出现违背 P 的位置…"}`），
这是「前提没被挑战过」的**正确答案**。评测器已把空方向**排除出主表 FP 与误报漏判**，
单独计在 `empty_direction_responses` —— 否则会把正确答案当成 false positive。

### 10.5.3 前提文件的常规体检

建议把 `leak_audit.py` 接成前提文件的**常规体检项**（新增前提时自动查泄漏），
将来长出「前提依赖型任务」时，选题能直接按「线索只在题面」筛。
（这条属金标侧的构建流程改动，我这边不动 `goldset/`。）

---

## 11. M2 验收清单（门槛）

C 组 M2 交付必须同时满足（机检项由 `evaluator.py` 的 `m2_gate` 自动给出）：

- [ ] 抽取器在 30 条金标上的 P / R / F1（**主表，不含 miss detection**）
- [ ] `miss_detection_recall` 单列 + **下界声明**
- [ ] status 混淆矩阵（4×4）+ `status_accuracy_strict` + `status_undecidable_rate` **并列**
- [ ] `selection_expression_separability`
- [ ] `weak_effect_count`（ranking_only + additive_only）单列
- [ ] **引用可核率**（exact / punct / ungrounded 三档）
- [ ] **Jaccard 阈值扫描曲线（0.20–0.60）+ 阈值敏感区间**（裁决 ⑥-a）——
      `python3 evaluator.py --gold g.jsonl --pred p.jsonl --sweep`
- [ ] **`true_increment.f1 ≥ 0.10`**（真实语义增量，裁决 ⑦-a / ⑧-d / ⑨-c）——
      门槛基准取**所有 end_to_end 实现 × 所有阈值点**里的最高 F1
- [ ] **`audit_coverage` = 100%**（金标每条方向都被 m2_gate 审到，否则不予通过）
- [ ] **`gold_closed_book` = true**（闭卷金标；开卷只作对照组，不进门槛）
- [ ] **`frame_note_leaks` 为空**（域外参考绝不能进可核语料）
- [ ] **`zero_direction_precision` / `zero_direction_overclaim_rate`** 单列报告（裁决 ⑧-c）
- [ ] `gold_evidence_stats` —— 金标自身可核率（**非逐字比例 > 5% 则不予通过**；
      若有金标条目「有 evidence 但抽不出可核片段」，也判不通过 —— 不许静默放过）
- [ ] 金标 schema 升级为 v2（`decision_effect` 取代 `would_change_decision`）
- [ ] 金标 `not_generated` 条目全部为关键漏判（`decision_effect ∈ {conclusion_flip, option_added}`）
- [ ] `initial_frame` 全部符合四段式 + 封闭性声明模板
- [ ] 人工复核完成并记录（强 effect 100%、status 抽样 30%）
- [ ] **每条 m2_gate 断言都有配对的反向测试**（见 §12.1）——「断言通过」不等于「断言有效」
- [ ] **上界报警线** `upper_alarm`：闭卷 F1 > 0.7321 触发人工检查（**不是通过条件**）
- [ ] **金标代表性警示（⑮-b）在报告第一屏** —— expression_failure 的真实判别力尚未被测量
- [ ] **门槛基准四类并列（⑳-a）** —— 脏 / 池口径 / 规范口径，含那个虚高值 0.9381，不隐藏
- [ ] **同点比较（⑳-b）** —— 抽取器在门槛点（`match_thr=0.25`）上的 F1 ≥ **0.6749**
- [ ] **匹配阈值全曲线（0.20–0.60）** —— 抽取器与基线都要出（基线跨度 0.2212，单点无意义）
- [ ] **池口径守卫生效（⑳-c）** —— `pool_caliber` 不一致的基线被拦在门槛之外
- [ ] **`introduced_in_synthesis` 计数**（⑲-a）与 **`constraint_violation` 上界对照**（⑲-c）
- [ ] **口径不变式生效（⑯-b）** —— 错分母的数据必须抛异常
- [ ] **M2 跑批说明（⑫-b，可追溯）**：本批跑批使用修订前的 prompt（判定树两分支版）；
      复核确认金标 30 条中不存在「中间段否决 + 最终答案没提」这一边界条目，
      故新旧两版 prompt 行为等价，结果可直接与金标比对。**下一批一律用新版 `prompts/`。**

---

## 12. ★ 每条 m2_gate 断言的反向测试清单

> **「断言通过」不等于「断言有效」。**
> 断言可能因为**检查的对象根本不存在**而恒真 —— 于是「通过」是假的。
> 本项目为此付过两次代价，都是同一类 bug（只审「通过的部分」/ 读错字段位置）。
> 规则（已写进 `extractor_spec.md` §7.3）：**每条 M2 门槛断言都必须配一条反向测试。**

| # | m2_gate 断言 | 反向测试 | self-test 用例 | 存在 |
|---|---|---|---|---|
| 1 | `gold_evidence_gate_pass` | 把金标 evidence 改成非逐字 | `反向R1` | ✅ |
| 2 | `gold_miss_all_key` | 把一条 not_generated 的 effect 改成弱效果 | `反向R2` | ✅ |
| 3 | `gold_uses_v2_effect` | 把金标 effect 换回旧字段 `would_change_decision` | `反向R3` | ✅ |
| 4 | `all_frames_conforming` | 把 initial_frame 改成缺封闭性声明 | `反向R4` | ✅ |
| 5 | `frame_note_leak_free` | **把 frame_note 注入 trace** → 必须报警 | `反向R5` | ✅ |
| 6 | `audit_coverage_pass` | 让一条金标方向带非法 status → 审计明细必须报出来 | `反向R6` | ✅ |
| 7 | `true_increment_gate_pass` | 给一个很低的抽取器 F1 / 一个很高的基线 | `反向R7` | ✅ |
| 8 | `gold_closed_book` | 用开卷金标 | `反向R8` | ✅ |
| 9 | `end_to_end_baseline_supplied` | 不提供外部基线 | `反向R9` | ✅ |
| 10 | `upper_alarm` | 给一个高于上界报警线的 F1 → 必须置位 | `反向R10` | ✅ |
| 11 | `sweep_max_f1_mode` 口径守卫 | 喂 `sweep_max_f1_mode: "open"` 的 JSON → 门槛基准不采用 | `反向R11` | ✅ |
| 12 | 断言清单本身可查 | —— | `R12` | ✅ |

另有两条**正例**断言（证明断言不是恒假的摆设）：
- `反向R5` 的后半：真的有 frame_note 但不在语料里 → **不报警**（`checked > 0`）
- `反向R11` 的后半：`sweep_max_f1_mode: "closed_book"` → **采用**该值

`evaluator.py` 的 `M2_GATE_ASSERTIONS` / `m2_gate_assertion_status()` 导出这份清单。

---

## 13. Prompt 稳定接口（M2 执行器用）

M2 要跑真实抽取器，而本组的 API key 尚未获授权 —— 用子 agent 当抽取器引擎
（子 agent 用平台自身额度，不碰别组 key）。为此需要**不依赖 markdown 解析**的稳定接口。

| 文件 | 内容 |
|---|---|
| `prompts/system.txt` | System Prompt 全文，逐字 |
| `prompts/user_template.txt` | User Prompt 模板全文，占位符 `{{TASK}}` `{{INITIAL_FRAME}}` `{{TRACE}}` `{{FINAL_ANSWER}}` |

这两个文件由 `export_prompts.py` **从 `extractor_spec.md` 的 §2 / §4 逐字抽取**生成，
所以**不可能与规格漂移**。改规格后必须重跑：

```bash
python3 eval/export_prompts.py          # 重新导出
python3 eval/export_prompts.py --check  # 校验（CI 用）
```

> **以 `prompts/` 为准**：`extractor_spec.md` 是**给人读的**、会被改；
> `prompts/` 里的两个 `.txt` 是**机器读的**、不能漂。
> 接口固定后，换任何执行器（子 agent / API / 人工）都不影响尺子本身 ——
> **尺子测的是抽取器的判别力，不是某个具体执行器的实现**。

---

## 14. 文件清单

| 文件 | 说明 |
|---|---|
| `extractor_spec.md` | **核心产物**。抽取器提示词规格（System/User prompt 完整可复制）、四条判定规则、防滥用 11 条、trace 规范、`initial_frame` 强制模板、`frame_ambiguous` 协议、§6 四个漏洞的裁定记录 |
| `evaluator.py` | 可运行评测器骨架。纯标准库，零依赖。`schema_version: 2` |
| `README.md` | 本文件 |
| `SPEC-DECISIONS-v1.md` | C 组主会话的冻结裁决（v1 + 补充裁决 ⑤⑥⑦⑧⑨） |
| `prompts/system.txt` | **Prompt 稳定接口**（System Prompt 全文，逐字） |
| `prompts/user_template.txt` | **Prompt 稳定接口**（User Prompt 模板全文，占位符 `{{TASK}}` 等） |
| `prompts/README.md` | prompt 接口说明 —— **以这两个 .txt 为准**，markdown 是给人读的 |
| `export_prompts.py` | 从 `extractor_spec.md` **逐字**导出上面两个 .txt；`--check` 校验是否漂移 |
| `examples/gold_sample.jsonl` | 金标样例（3 条，覆盖 selection_failure / not_generated / 陷阱 / 非合规框架） |
| `examples/pred_sample.jsonl` | 对应预测样例（含 over-claiming、弱效果、非合规框架，用于演示打分能区分） |
| `examples/report_sample.json` | 样例跑出来的报告 |
| `diagnose_increment_source.py` | **增量来源分解**：域外判定 vs status 判定 |
| `calibrate_baseline.py` | **门槛基准定版**：用评测器自己的打分路径 + 匹配阈值扫描，导出规范基线 `baseline_main_caliber.json` |
| `isolate_baseline_diff.py` | 三个基线数的**来源隔离**（匹配函数 / 贪心顺序 / FP 口径逐一拆开） |
| `baseline_main_caliber.json` | 主表口径的词法基线得分（门槛同口径基准） |

**环境约束**：Python 3.11，纯标准库，**不联网**，**不读任何 API key**。
本机 2 vCPU / 3GB，不引入 torch/transformers 等重依赖。
