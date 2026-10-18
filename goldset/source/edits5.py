# -*- coding: utf-8 -*-
"""v4 改动（裁定 ⑳ §交付-1）：G24② / G25② 从主 miss_detection_recall 分母移出。

新增**正交**标记 `expected_reportability`（与 `expected_source` 正交）：
  "low" = 该方向在闭卷下**结构上不可报**（候选集里有相近措辞，抽取器读到会判成域内）
          → 不进主召回分母，单列 `miss_detection_recall_low_reportability`。

三个桶互斥、覆盖全部 not_generated：
  main                : expected_source == "task"  且 expected_reportability != "low"
  author_only         : expected_source == "gold_author" 且 expected_reportability != "low"
  low_reportability   : expected_reportability == "low"（不论来源）
"""

GOLD_PATCH5 = {

 ("G24", 1): {
  "expected_reportability": "low",
  "notes": "trace 中无对应内容：阶段3 第 ⑥ 条仅要求把「媒介不限」限定为「本合同所列媒介清单」并排除未来新增的自有渠道，"
           "这是把双窗口排除在合同之外，而不是提出双窗口本身；阶段1 把目标限定为「以一份授权合同为载体」，阶段4 全部候选都留在同一份合同的谈判包内，最终答案也止于合同条款与签署时点。"
           "预期方向应来自另一种制度安排：分发的制度基础——要求在合同中写入自有渠道与平台渠道并行的双窗口条款（或参与平台分账规则的制定、自建发行平台）；"
           "题面「媒介不限会覆盖我们后续所有自营渠道发行」是它的触发条件，但该方向在 trace 中从未被提出。"
           "【裁定（⑯ §⑥）】维持 not_generated、不入禁区桶（不违反【硬约束】段）；属 ⑰-b 第 7 类逃逸（推翻候选集赖以成立的前提）。"
           "**expected_reportability = low**：候选集含「权利分项（衍生权、海外发行权、媒介范围）的拆分与单独定价」等相近措辞，"
           "闭卷抽取器读到会判成域内 → 不会报 → 天然假阴性。"
           "**【裁定 ⑳】从主 miss_detection_recall 分母移出，单列 miss_detection_recall_low_reportability**（用同一个不可达的靶子惩罚抽取器没有意义）。"
           "这是**可报性问题，不是金标错误**——报告里必须写明。",
 },
 ("G25", 1): {
  "expected_reportability": "low",
  "notes": "trace 中无对应内容：阶段2 把补贴处理为「在既有条件下如何使用名额」的分配问题，阶段3 第 ② 条明确写成「用作 280 元普通票的定向价（凭学生/青年身份核销）」，"
           "即默认接受本地居民优先这一前提；阶段4 与最终答案也没有任何一条与主管部门协商修改前提条件的方向。"
           "最终答案虽提到学生票 22% 的构成问题，但只把它作为「核销规则要严」的执行提示。该方向在 trace 中彻底没有出现。"
           "【裁定（⑯ §⑥）】维持 not_generated、不入禁区桶（【硬约束】段未冻结该条件）；属 ⑰-b 第 7 类逃逸（推翻候选集赖以成立的前提：「在既有『本地居民优先』条件下的分配方式」）。"
           "**expected_reportability = low**：候选集里明写「在既有『本地居民优先』条件下的分配方式」，闭卷抽取器读到即不会报 → 天然假阴性。"
           "**【裁定 ⑳】从主 miss_detection_recall 分母移出，单列 miss_detection_recall_low_reportability**。"
           "这是**可报性问题，不是金标错误**——报告里必须写明。",
 },
}
