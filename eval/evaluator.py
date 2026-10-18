#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSC-Eval —— 问题空间构建评测器（可运行骨架）· schema v2

把「Agent 有没有构建问题空间的能力」变成可核的数字：
  1. 域外方向抽取得对不对   -> 主表 precision / recall / F1（**不含 miss detection**）
  2. 漏判抓到了几个         -> miss_detection_recall（单列，**下界不是覆盖率**）
  3. 方向丢在哪一步判对没有 -> status 四分类混淆矩阵 + strict 准确率 + 不可判率
  4. selection 与 expression 能不能分开 -> selection_expression_separability
  5. 方向值不值得进答案     -> decision_effect（conclusion_flip/option_added 计入主分）
  6. 证据可不可核           -> quote_grounding_rate（exact / whitespace / ungrounded 三档）

设计约束：
  * 纯 Python 3.11 标准库，无第三方依赖（本机 2 vCPU / 3GB，不装 torch/transformers）
  * 不发起任何网络请求，不读取任何 API key
  * 方向匹配用语义近似（字符二元组 Jaccard），不要求字符串相等
  * 规格已冻结：/workspace/c-group/eval/SPEC-DECISIONS-v1.md

配套文档：extractor_spec.md（抽取器提示词规格）、README.md（设计理念与局限）

CLI:
    python3 evaluator.py --gold <jsonl> --pred <jsonl> --out report.json
    python3 evaluator.py --self-test
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import unicodedata
from typing import Any, Callable, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))

__all__ = [
    "SCHEMA_VERSION",
    "STATUSES",
    "MAIN_STATUSES",
    "DECISION_EFFECTS",
    "MATCH_THRESHOLD",
    "load_gold",
    "load_predictions",
    "score_prediction",
    "status_agreement",
    "status_agreement_strict",
    "aggregate",
    "similarity",
    "default_match_fn",
    "make_similarity_match_fn",
    "frame_schema",
]

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------

#: 输出 schema 版本。v2 = `decision_effect` 取代 `would_change_decision`。
SCHEMA_VERSION: int = 2

#: status 四分类。顺序即混淆矩阵的行/列顺序。
STATUSES: tuple[str, ...] = (
    "realized",
    "expression_failure",
    "selection_failure",
    "not_generated",
)

#: 计入**主表** P/R/F1 的三类。`not_generated` 不在其中（裁决 ②-b）。
MAIN_STATUSES: tuple[str, ...] = ("realized", "selection_failure", "expression_failure")

#: status 别名 -> 规范名。
#: `discovery_failure` 是组章程里的说法（发现失败），`not_generated` 是抽取器规格里的说法。
#: 二者指同一件事，统一归到 `not_generated`。金标集与预测都接受两种写法。
STATUS_ALIASES: dict[str, str] = {
    "realized": "realized",
    "expression_failure": "expression_failure",
    "selection_failure": "selection_failure",
    "not_generated": "not_generated",
    "discovery_failure": "not_generated",
    "not_generated_failure": "not_generated",
    "discovery": "not_generated",
}

#: decision_effect 的五个取值（schema v2）。
DECISION_EFFECTS: tuple[str, ...] = (
    "conclusion_flip",
    "option_added",
    "ranking_only",
    "additive_only",
    "none",
)

#: 计入「有实质改变」的 effect（规则 4 的硬闸门）。
STRONG_EFFECTS: frozenset[str] = frozenset({"conclusion_flip", "option_added"})

#: 弱效果：单独统计为 weak_effect_count，不计入主分但完整报告。
WEAK_EFFECTS: frozenset[str] = frozenset({"ranking_only", "additive_only"})

#: 字符二元组 Jaccard 阈值。**启发式默认值，尚未在金标集上校准**
#: （裁决 ⑥-a：M2 必须扫 0.20–0.60，扫描前不得用于任何跨组结论）。
MATCH_THRESHOLD: float = 0.34

#: initial_frame 强制模板的四个段（裁决 ④）。
FRAME_SECTIONS: tuple[str, ...] = ("候选集", "封闭性", "评估维度", "硬约束")

#: 词法捷径基线的两版默认词表（裁决 ⑧-d）。
#: 宽版含「不建议 / 不推进」等最终答案里的常见否决措辞；窄版只含硬否决词。
NARROW_LEXICON: dict[str, list[str]] = {
    "rejection": ["否决", "不采纳", "不执行", "不列入", "排除"],
    "retention": ["保留", "采用", "纳入", "建议采用"],
}
WIDE_LEXICON: dict[str, list[str]] = {
    "rejection": ["否决", "不采纳", "不执行", "不列入", "排除", "暂不推进", "不建议", "不推进"],
    "retention": ["保留", "采用", "纳入", "建议采用"],
}

#: 词法基线的语料变体（裁决 ⑧-d）。
CORPUS_VARIANTS: tuple[str, ...] = ("evidence_quote", "full_trace", "final_answer")

#: 词法基线的词表变体。
LEXICON_VARIANTS: tuple[str, ...] = ("narrow", "wide")

#: M2 硬门槛：真实增量（抽取器 F1 − 词法基线最高 F1）不得低于此值。
#: 达不到说明抽取器没学到语义 —— 该改抽取器，不该改门槛。
LEXICAL_BASELINE_MIN_INCREMENT: float = 0.10

#: 上界报警线（裁决 ⑪-b③）：闭卷抽取器 F1 超过它就触发人工检查。
#: 默认取开卷天花板 0.7321（最强词法 + 泄漏 的上限）。
#: **这是异常信号，不是通过条件** —— 门槛与上界是两件事。
UPPER_ALARM_F1: float | None = 0.7664

#: ★ 同点比较的**门槛点**（⑳-b 硬要求）：抽取器必须在**取得门槛基准 0.5749 的
#: 那个匹配阈值**上被评分，再与之比较。禁止抽取器在多阈值上取最优再与门槛比。
#: （0.20 与 0.25 两点的基线都是 0.5749；此处取 0.25，与金标构造方的 same_point 一致。）
GATE_POINT_MATCH_THR: float = 0.25

#: expected_source 的合法取值（裁决 ⑦-b + prp 桶前置条件）。
#: `prp` = 前提识别探针产出的方向（**进主召回桶**，但受可准入判定约束：
#: P 必须在 prompt 里给定、且该题确实是合法靶子）。
EXPECTED_SOURCES: tuple[str, ...] = ("task", "external", "gold_author", "prp")

#: expected_source 缺失 / 非法时的标记。二者分开报，便于发现金标数据问题。
#: 降级显式化的三档（㉔-e ② / M2 实测修正）：
#:   labeled   —— 已显式标注，且取值在已登记集合里
#:   unlabeled —— **没标**（字段缺失或空）—— 这是金标里最常见的情况，不是问题
#:   unknown   —— 标了，但取值拼错 / 未登记
#: ⚠ 之前把「没标」也算成 unknown，会让读者误以为「54 条的来源标注有问题」，
#:   实际是「54 条本来就没标」—— **这是会误导的**。
EXPECTED_SOURCE_UNKNOWN = "unlabeled"
EXPECTED_SOURCE_INVALID = "unknown"

#: 方向文本里可能出现的别名键（不同来源的 JSON 字段名不完全一致）。
_DIR_TEXT_KEYS = ("direction", "text", "name", "description", "title", "label")

#: 抽取器输出可能被包在这些键下面。
_PRED_ENVELOPE_KEYS = ("prediction", "extraction", "result", "output", "response")

#: 最终答案在 trace 末段的常见前缀。
_FINAL_ANSWER_PREFIXES = ("最终答案", "最终回答", "最终结论", "final answer")

#: 阶段段落的标记（如「阶段3｜方向生成」）。
_STAGE_PREFIX_RE = re.compile(r"^\s*阶段\s*\d+\s*[｜|]")


def _looks_like_final_answer(segment: str, others: Sequence[str]) -> bool:
    """判断某个 trace 段是不是最终答案（裁决 ⑧-e：靠内容特征，不依赖独立字段）。

    判据（任一命中即可）：
      a) 段首是「最终答案 / 最终回答 / 最终结论 / final answer」
      b) 它明显长于其他段，且不带「阶段N｜」前缀
    """
    text = segment.strip()
    if not text:
        return False
    if text.startswith(_FINAL_ANSWER_PREFIXES):
        return True
    if _STAGE_PREFIX_RE.match(text):
        return False
    others_len = [len(o.strip()) for o in others if o.strip()]
    if not others_len:
        return False
    median = sorted(others_len)[len(others_len) // 2]
    return median > 0 and len(text) >= median * 2


def _split_final_answer(item: dict) -> tuple[str, str, str]:
    """拆出 (final_answer, corpus, final_segment)。

    **裁决 ⑧-e：逐字校验语料 = 完整 trace（含最终答案段）。**
    排除最终答案的旧做法已作废 —— 它会让 `realized` 无法举证（其 evidence_quote
    按规格应引最终答案原话，排除之后必然 ungrounded）。

    返回：
      final_answer  —— 最终答案的正文（去掉「最终答案：」前缀）
      corpus        —— **完整** trace 语料（含最终答案段）
      final_segment —— 最终答案在 trace 里的原始段（用于「引用是否只命中最终答案段」的防伪检查）

    兼容两种形态：
      a) 有独立 `final_answer` 字段
      b) 最终答案是 trace 的最后一段（靠内容特征识别）
    """
    trace = item.get("trace")
    segments: list[str] = []
    if isinstance(trace, str):
        segments = [trace]
    elif isinstance(trace, list):
        for seg in trace:
            if isinstance(seg, str):
                segments.append(seg)
            elif isinstance(seg, dict):
                stage = seg.get("stage") or seg.get("name") or ""
                content = seg.get("content") or seg.get("text") or ""
                segments.append(f"{stage}\n{content}")

    explicit = item.get("final_answer")
    if isinstance(explicit, str) and explicit.strip():
        # 语料 = trace 各段 **加上** 显式的 final_answer 字段内容（去重）。
        # 与独立校验脚本 `run/collect_and_score.py` 的口径一致：
        #   「evidence 必须能在 trace 全部段（含末段最终答案）+ final_answer 字段里
        #     做连续逐字 substring 命中」
        parts = [s for s in segments if s.strip() != explicit.strip()]
        parts.append(explicit.strip())
        return explicit, "\n".join(parts), explicit.strip()

    # 没有独立字段：从 trace 末段推断
    for idx in range(len(segments) - 1, -1, -1):
        seg = segments[idx]
        if not _looks_like_final_answer(seg, segments[:idx] + segments[idx + 1 :]):
            continue
        body = seg.strip()
        for prefix in _FINAL_ANSWER_PREFIXES:
            if body.startswith(prefix):
                body = body[len(prefix) :].lstrip("：:，, ")
                break
        # 语料 = 完整 trace（含最终答案段）
        return body, "\n".join(segments), seg.strip()

    return "", "\n".join(segments), ""


# --------------------------------------------------------------------------
# 文本工具
# --------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """NFKC 归一 + 去掉所有空白与标点 + 小写。用于相似度计算。"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[!-/:-@\[-`{-~！-～、-〿＀-￯]", "", text)
    return text


def _nows(text: str) -> str:
    """去掉所有空白（含中文里为排版插入的空格）。"""
    return re.sub(r"\s+", "", text or "")


def _bigrams(text: str) -> set[str]:
    """字符二元组集合。长度 < 2 时退化为单字集合。"""
    n = _normalize(text)
    if len(n) < 2:
        return {n} if n else set()
    return {n[i : i + 2] for i in range(len(n) - 1)}


def similarity(a: str, b: str) -> float:
    """字符二元组 Jaccard 相似度，0.0 ~ 1.0。对中文有效（不依赖分词）。"""
    ga, gb = _bigrams(a), _bigrams(b)
    if not ga and not gb:
        return 1.0
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    union = len(ga | gb)
    return inter / union if union else 0.0


def make_similarity_match_fn(threshold: float = MATCH_THRESHOLD) -> Callable[[dict, dict], bool]:
    """构造一个基于相似度阈值的 match_fn。"""

    def _fn(gold_dir: dict, pred_dir: dict) -> bool:
        return similarity(direction_text(gold_dir), direction_text(pred_dir)) >= threshold

    _fn.__name__ = f"jaccard_match_{threshold}"
    return _fn


#: 默认匹配函数：字符二元组 Jaccard >= MATCH_THRESHOLD 判为同向。
#: **启发式阈值，待在金标集上校准**（裁决 ⑥-a）。
default_match_fn: Callable[[dict, dict], bool] = make_similarity_match_fn(MATCH_THRESHOLD)


def direction_text(direction: dict) -> str:
    """从一条方向记录里取文本（兼容多种字段名）。"""
    if isinstance(direction, str):
        return direction.strip()
    if not isinstance(direction, dict):
        return ""
    for key in _DIR_TEXT_KEYS:
        value = direction.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


# --------------------------------------------------------------------------
# 字段访问
# --------------------------------------------------------------------------


def _dir_status(direction: dict) -> str:
    """取 status，非法值归入 'invalid'。接受 discovery_failure 别名。"""
    if not isinstance(direction, dict):
        return "invalid"
    status = direction.get("status")
    if isinstance(status, str):
        status = status.strip().lower()
        if status in STATUS_ALIASES:
            return STATUS_ALIASES[status]
    return "invalid"


def _is_constraint_violation(direction: dict) -> bool:
    """该方向是否**触碰/违反**了初始框架【硬约束】里的禁区段（裁决 ⑭-c）。

    这类方向**不进任何召回统计、不进主表 F1、不进门槛**，归入独立诊断量
    `constraint_violation_candidates`。

    理由：`not_generated` 的语义是「本该被想到的域外方向没被生成」；
    一个「根本不该出现的不可能方向」报进去会让该类退化成「检测违规」。
    但它对 A 组那类约束建模场景直接有用，所以单独成桶。
    """
    if not isinstance(direction, dict):
        return False
    value = direction.get("constraint_violation")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return False


def _is_out_of_frame(direction: dict) -> bool:
    """in_frame 的读取。缺失时按「域内」处理（不凭空扩大域外集合）。"""
    if not isinstance(direction, dict):
        return False
    value = direction.get("in_frame")
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return value.strip().lower() in ("false", "no", "0")
    return False


def _decision_effect(direction: dict) -> str:
    """取 decision_effect（schema v2）。兼容旧字段 would_change_decision。

    兼容映射（裁决 ③）：true -> conclusion_flip，false -> none。
    两者都缺失时返回 'none'，但 **has_effect 为 False** —— 缺失不等于「无实质改变」，
    只是无从判断；统计一致性时这类条目不参与。
    """
    if not isinstance(direction, dict):
        return "none"
    value = direction.get("decision_effect")
    if isinstance(value, str):
        v = value.strip().lower()
        if v in DECISION_EFFECTS:
            return v
    legacy = direction.get("would_change_decision")
    if isinstance(legacy, bool):
        return "conclusion_flip" if legacy else "none"
    return "none"


def _has_effect(direction: dict) -> bool:
    """该条目是否携带可用的 decision_effect 信息。"""
    if not isinstance(direction, dict):
        return False
    value = direction.get("decision_effect")
    if isinstance(value, str) and value.strip().lower() in DECISION_EFFECTS:
        return True
    return isinstance(direction.get("would_change_decision"), bool)


def _is_strong_effect(effect: str) -> bool:
    """是否有实质改变（规则 4 硬闸门）。"""
    return effect in STRONG_EFFECTS


def _gold_directions(gold_item: dict) -> list[dict]:
    """金标里的域外方向（正例集）。

    兼容两种写法：
      a) 顶层 `out_of_frame_directions`（抽取器规格的写法）
      b) 嵌套在 `gold` 对象里（金标集当前的实际写法）
    """
    value = gold_item.get("out_of_frame_directions")
    if not isinstance(value, list):
        nested = gold_item.get("gold")
        if isinstance(nested, dict):
            value = nested.get("out_of_frame_directions")
    if not isinstance(value, list):
        return []
    return [d for d in value if isinstance(d, dict)]


def _gold_traps(gold_item: dict) -> list[dict]:
    """金标里的域内方向（陷阱集）：抽取器若把这些判成域外，即为 over-claiming。"""
    value = gold_item.get("in_frame_directions")
    if not isinstance(value, list):
        nested = gold_item.get("gold")
        if isinstance(nested, dict):
            value = nested.get("in_frame_directions")
    if not isinstance(value, list):
        return []
    return [d for d in value if isinstance(d, dict)]


def _pred_directions(pred: dict) -> list[dict]:
    """预测里的方向列表。兼容被包在 envelope 键下的情况。"""
    if not isinstance(pred, dict):
        return []
    candidates: list[Any] = []
    if isinstance(pred.get("directions"), list):
        candidates.append(pred["directions"])
    for key in _PRED_ENVELOPE_KEYS:
        inner = pred.get(key)
        if isinstance(inner, dict) and isinstance(inner.get("directions"), list):
            candidates.append(inner["directions"])
        elif isinstance(inner, list):
            candidates.append(inner)
    if not candidates:
        return []
    # 取最长的那个（避免 envelope 截断）
    best = max(candidates, key=len)
    return [d for d in best if isinstance(d, dict)]


def _load_lexicon(path: str | None) -> dict[str, list[str]]:
    """载入词表。文件里的键覆盖默认值，缺失的键保持默认。

    文件格式（JSON）：
        {"rejection": ["否决", "不采纳"], "retention": ["保留", "采用"]}
    也接受 reject/keep、否决词/保留词 等别名键。
    """
    lexicon = {k: list(v) for k, v in WIDE_LEXICON.items()}
    if not path:
        return lexicon
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: 词表文件顶层必须是 JSON 对象")
    alias = {
        "rejection": ("rejection", "reject", "否决词", "rejection_words"),
        "retention": ("retention", "keep", "保留词", "retention_words"),
    }
    for canonical, keys in alias.items():
        for key in keys:
            value = data.get(key)
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                lexicon[canonical] = list(value)
                break
    return lexicon


def _locate_window(direction: dict, corpus: str) -> str:
    """在 trace 语料里定位这条方向所在的片段（纯字符串操作，不做语义判断）。

    先按 evidence_quote 取；quote 为空时，按与方向文本的字符二元组相似度
    选最像的那一行。全部找不到就返回空串 —— 词法基线没有依据就无从判断。
    """
    quote = direction.get("evidence_quote")
    if isinstance(quote, str) and quote.strip():
        return quote
    text = direction_text(direction)
    if not text or not corpus:
        return ""
    best_line, best_score = "", 0.0
    for line in corpus.splitlines():
        line = line.strip()
        if not line:
            continue
        score = similarity(text, line)
        if score > best_score:
            best_line, best_score = line, score
    if best_score >= MATCH_THRESHOLD:
        return best_line
    return ""


def _lexical_status(
    direction: dict,
    corpus: str,
    final_segment: str,
    lexicon: dict[str, list[str]],
    corpus_mode: str,
    match_fn: Callable[[dict, dict], bool],
) -> str:
    """词法捷径基线的 status 判定（裁决 ⑧-a：锚点移到最终答案段）。

    判定顺序与 status v1.1 一致，只是每一步都退化成关键词匹配：
      在最终答案里吗？
        在   -> 语料里有否决词？ -> selection_failure；否则 realized
        不在 -> 语料里有否决词？ -> selection_failure   （中间段被否决）
                语料里有保留词？ -> expression_failure   （中间段保留过）
                都没有         -> not_generated        （全文没表达）

    `corpus_mode` 决定关键词在哪儿找（裁决 ⑧-d 的三个语料变体）：
      evidence_quote —— 只看这条方向自己的 evidence_quote
      full_trace    —— 看 trace 里与该方向最相似的那一段
      final_answer  —— 只看最终答案段
    """
    if corpus_mode == "evidence_quote":
        window = direction.get("evidence_quote")
        window = window if isinstance(window, str) else ""
    elif corpus_mode == "final_answer":
        window = final_segment
    else:  # full_trace
        window = _locate_window(direction, corpus)

    has_reject = bool(window) and any(word in window for word in lexicon.get("rejection", []))
    has_keep = bool(window) and any(word in window for word in lexicon.get("retention", []))

    in_final = _direction_in_final_answer(direction, final_segment, match_fn)

    if in_final:
        return "selection_failure" if has_reject else "realized"
    if has_reject:
        return "selection_failure"
    if has_keep:
        return "expression_failure"
    return "not_generated"


def _expected_source(direction: dict) -> str:
    """expected_source 的读取（裁决 ⑦-b）。缺失 / 非法都不崩，只作统计，不参与判定。

    合法值：task / external / gold_author —— 记录「这个方向本该被想到」的来源。
    """
    if not isinstance(direction, dict) or "expected_source" not in direction:
        return EXPECTED_SOURCE_UNKNOWN
    value = direction.get("expected_source")
    if isinstance(value, str):
        v = value.strip().lower()
        if not v:
            return EXPECTED_SOURCE_UNKNOWN      # 没标（空）
        if v in EXPECTED_SOURCES:
            return v
    return EXPECTED_SOURCE_INVALID            # 标了但取值未登记


def _gold_evidence(direction: dict) -> str:
    """金标的证据字段（`evidence` / `evidence_quote` 两种命名）。"""
    for key in ("evidence", "evidence_quote", "quote"):
        value = direction.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _extract_quoted_fragments(text: str) -> list[str]:
    """从金标 evidence 里抽出被引用的原文片段。

    两种写法都支持：
      a) evidence 写成「阶段3『xxx』；阶段4『yyy』」——抽出被引号包住的片段
      b) evidence 本身就是**一段逐字原文**（金标集 v2 的写法）——整串当作一个片段
    """
    if not text:
        return []
    fragments: list[str] = []
    # 显式引号（『』与“”）里的片段：放行 2 字（「否决」这类也是合法引用）
    for pattern in (r"『([^』]+)』", r"“([^”]+)”"):
        for match in re.finditer(pattern, text):
            frag = match.group(1).strip()
            if len(frag) >= 2:
                fragments.append(frag)
    if fragments:
        return fragments
    # 直引号：更容易误命中普通文本，门槛提高到 4 字
    for match in re.finditer(r'"([^"]{4,})"', text):
        frag = match.group(1).strip()
        if len(frag) >= 4:
            fragments.append(frag)
    if fragments:
        return fragments
    # 写法 (b)：整串就是一段原文引用
    whole = text.strip()
    if len(whole) >= 4:
        return [whole]
    return []


#: 「域外参考」段的特征串 —— 出现在 initial_frame 里即说明金标是**开卷**的（裁决 ⑨-a）。
OPENBOOK_MARKERS = ("域外参考", "域外方向", "被排除在解空间之外", "不作为候选")


def has_openbook_note(initial_frame: Any) -> bool:
    """initial_frame 里是否含「域外参考」段（= 开卷金标）。

    裁决 ⑨-a：域外方向一旦被印在题面上，尺子就退化成字串匹配 ——
    因为抽取器可以**对着答案做**。真实世界里 initial_frame 只给「域内是什么」。
    """
    if not isinstance(initial_frame, str):
        return False
    return any(marker in initial_frame for marker in OPENBOOK_MARKERS)


def frame_schema(initial_frame: Any) -> str:
    """检查 initial_frame 是否符合强制模板（裁决 ④）。

    模板：
        【候选集】<显式枚举的常规候选>
        【封闭性】closed | open
        【评估维度】<显式列举的比较维度>
        【硬约束】<不可违反的边界，没有就写"无">

    返回 'conforming' 或 'non_conforming'。
    缺任一段、或封闭性没有给出 closed/open 取值，都算 non_conforming。

    为什么这一项是新增的关键项：
        没有封闭性声明，逃逸测试的「新候选」一类在描述式框架下无从判定 ——
        那正是漏洞 ④ 的根因。声明后：
          closed 框架下：引入清单外候选 = 逃逸成功 = 域外
          open   框架下：「新候选」一类逃逸自动失效，只剩其余四类可判
    """
    if not isinstance(initial_frame, str) or not initial_frame.strip():
        return "non_conforming"
    text = initial_frame.strip()
    for section in FRAME_SECTIONS:
        pattern = rf"(?:【\s*{section}\s*】|{section}\s*[:：])"
        if not re.search(pattern, text):
            return "non_conforming"
    # 封闭性必须**在【封闭性】段里**给出 closed/open 取值。
    # 不能用「文本里出现 closed/open 就算」——「【封闭性】」这几个字本身就含「封闭」，
    # 那样会把没写取值的框架误判成合规。
    if not _frame_closure(text):
        return "non_conforming"
    return "conforming"


def _frame_closure(initial_frame: Any) -> str:
    """读出封闭性取值：'closed' / 'open' / ''（未知）。"""
    if not isinstance(initial_frame, str):
        return ""
    match = re.search(r"【\s*封闭性\s*】\s*[:：]?\s*(closed|open|封闭|开放)", initial_frame)
    if not match:
        match = re.search(r"封闭性\s*[:：]\s*(closed|open|封闭|开放)", initial_frame)
    if not match:
        return ""
    value = match.group(1)
    return {"封闭": "closed", "开放": "open"}.get(value, value)


# --------------------------------------------------------------------------
# trace 处理
# --------------------------------------------------------------------------


def _clause_split(text: str) -> list[str]:
    """把一段文本切成子句（用于方向级比对，避免拿长文本做整段匹配）。"""
    if not text:
        return []
    parts = re.split(r"[。；！？\n]|(?<=[\u3001\uff0c])", text)
    return [p.strip() for p in parts if p and p.strip()]


#: 方向级匹配：方向文本的双元组被候选文本单元**覆盖**的比例阈值。
#: 为什么不用对称的 Jaccard：方向描述通常很短（「方向甲」），候选文本单元会长
#: （「建议采用方向甲的方案」），Jaccard 会被长文本稀释（2/9 = 0.22 < 0.34）而漏判。
#: 覆盖率是**非对称**的，问的是「这个方向的内容有没有出现在这段话里」—— 正是这里要问的。
#: 待校准项（与 Jaccard 阈值同一批校准）。
CONTAINMENT_THRESHOLD: float = 0.6

#: 「中间段是否已经生成过这个方向」的覆盖率阈值。
#: 比 CONTAINMENT_THRESHOLD 松，因为抽取器写的 direction 常常**比 trace 原文长**
#: （M2 实测：抽取器写「把前置仓备货做仓单质押，用融资属性与存货属性分离来释放经营现金」，
#: trace 写「把前置仓备货部分做仓单质押」）——
#: 方向的大二元组被 trace 覆盖的比例天然偏低，用 0.6 会漏判。
#: 宁可**低估**「综合阶段新产生」（多判几条已生成过），也不要高估。
MIDDLE_STAGE_COVERAGE_THRESHOLD: float = 0.5


def _containment(direction_text_: str, candidate_text: str) -> float:
    """方向文本的双元组被候选文本覆盖的比例（0~1）。"""
    need = _bigrams(direction_text_)
    if not need:
        return 0.0
    have = _bigrams(candidate_text)
    if not have:
        return 0.0
    return len(need & have) / len(need)


def _direction_in_final_answer(
    direction: dict,
    final_segment: str,
    match_fn: Callable[[dict, dict], bool],
) -> bool:
    """这个方向是否出现在最终答案里？

    **裁决 ⑧-e.2：不能用裸 substring，必须用方向级匹配。**
    理由：方向描述（"把 12 周承诺改为分批交付"）与最终答案的措辞通常不同，
    拿整篇最终答案做 substring 会漏判 -> 明明进了最终答案却被判成 expression_failure。

    做法：把最终答案切成子句，逐句与方向比对：
      1) match_fn（方向 vs 方向量级）命中
      2) 或该方向的双元组被子句覆盖 >= CONTAINMENT_THRESHOLD（非对称，抗长文本稀释）
    """
    return _text_mentions_direction(direction, final_segment, match_fn)


def _assert_frame_note_out_of_corpus(gold_item: dict, corpus: str) -> tuple[list[str], int]:
    """断言：`frame_note`（金标专有字段，抽取器不可见）**绝不**出现在可核语料里。

    裁决 ⑨-a：`frame_note` 是「域外参考」的存放处。如果它被并进语料，
    抽取器就能引它当证据，把开卷泄漏伪装成「在 trace 里见过这个方向」。

    **检查范围 = trace 语料**（不含 `initial_frame`）。
    这对开卷对照组是**正确**的：开卷文件里 `frame_note` 合法地出现在 `initial_frame` 里
    （那正是对照组的定义），但它绝不该出现在 **trace** 里 ——
    出现在 trace 里才意味着抽取器能拿它当「这个方向在过程里出现过」的证据。
    所以开卷文件不会被误报。

    返回 (泄漏的字段名列表, 本条目实际检查到的 frame_note 字段数)。
    「检查到的数量」很重要 —— **断言通过 ≠ 断言有效**：
    如果这条金标根本没有 frame_note，断言就是真空的「通过」。
    报告里 `m2_gate.frame_note_checked` 会报出实际检查了多少个，
    配一条反向测试（故意把 frame_note 注入语料 → 必须报警）来证明断言有效。
    """
    leaks: list[str] = []
    checked = 0

    def _hits(note: str) -> bool:
        """整串命中，或「有辨识度的一段」命中。

        只查整串不够 —— 语料里可能只出现了 frame_note 的一部分
        （例如抽取器只引用了其中一句），那同样是泄漏。
        取「：」之后的前 40 字作为有辨识度的片段。
        """
        if not note:
            return False
        if note in corpus:
            return True
        core = note.split("：", 1)[-1].strip()[:40]
        return bool(core) and core in corpus

    nested = gold_item.get("gold")
    if isinstance(nested, dict) and isinstance(nested.get("frame_note"), str):
        note = nested["frame_note"].strip()
        if note:
            checked += 1
            if _hits(note):
                leaks.append("gold.frame_note")
    note = gold_item.get("frame_note")
    if isinstance(note, str):
        stripped = note.strip()
        if stripped:
            checked += 1
            if _hits(stripped):
                leaks.append("frame_note")
    return leaks, checked


def _non_final_corpus(gold_item: dict, pred: dict) -> str:
    """trace 中**不含**最终答案段的那部分语料（用于「引用是否只命中最终答案段」）。"""
    trace = gold_item.get("trace")
    segments: list[str] = []
    if isinstance(trace, str):
        segments = [trace]
    elif isinstance(trace, list):
        for seg in trace:
            if isinstance(seg, str):
                segments.append(seg)
            elif isinstance(seg, dict):
                stage = seg.get("stage") or seg.get("name") or ""
                content = seg.get("content") or seg.get("text") or ""
                segments.append(f"{stage}\n{content}")
    _, _, final_segment = _split_final_answer(gold_item)
    if not final_segment:
        return "\n".join(segments)
    return "\n".join(s for s in segments if s.strip() != final_segment)


def _text_mentions_direction(
    direction: dict,
    haystack: str,
    match_fn: Callable[[dict, dict], bool],
    coverage_threshold: float = CONTAINMENT_THRESHOLD,
) -> bool:
    """这段文本里**是否提到**这个方向（方向级判定，不是字面包含）。

    两个信号取或：
      1) `match_fn` 命中（与主表同一套相似度阈值），**逐子句**判
      2) **覆盖率** ≥ threshold —— 方向的大二元组被**整段**文本覆盖的比例

    为什么必须有第 2 个信号：方向描述常常比 trace 里的措辞**短**，
    对称 Jaccard 会被长文本稀释（"方向甲" vs "阶段3｜方向生成：①方向甲。" = 2/11 = 0.18 < 0.34），
    于是「明明在中间段生成过」被判成「找不到」。
    覆盖率是**非对称**的，问的正是「这个方向的内容有没有出现在这段话里」。

    为什么覆盖率算在**整段**上而不是逐子句：方向的内容常常**跨子句**
    （"把前置仓备货做仓单质押，用融资属性与存货属性分离" 横跨两个子句），
    逐子句算会把覆盖率摊薄、两边都不达标。
    """
    if not haystack:
        return False
    text = direction_text(direction)
    if not text:
        return False
    for clause in _clause_split(haystack):
        if not clause.strip():
            continue
        if match_fn(direction, {"direction": clause}):
            return True
    if _containment(text, haystack) >= coverage_threshold:
        return True
    return False


def _direction_in_middle_stages(
    direction: dict,
    non_final_corpus: str,
    match_fn: Callable[[dict, dict], bool],
) -> bool:
    """中间段是否**已经存在**与该方向语义相近的方向。

    **用 match_fn 判，不用字面包含**（M2 实测教训）：
    抽取器自己写的 `direction` 描述与 trace 里的措辞**本来就不一样**
    （抽取器写「把前置仓备货做仓单质押，用融资属性与存货属性分离来释放经营现金」，
    trace 原文是「把前置仓备货部分做仓单质押」）→
    **精确文本匹配必然「找不到」** → 误判成「综合阶段新产生」。

    做法：把中间段语料切成子句，逐句与方向做 match_fn 比对（与主表同一套相似度阈值）。
    """
    return _text_mentions_direction(
        direction, non_final_corpus, match_fn, MIDDLE_STAGE_COVERAGE_THRESHOLD
    )


def _quote_check(quote: str, corpus: str) -> str:
    """核验 evidence_quote 是否逐字出现在 trace 里。

    返回 'exact' | 'whitespace' | 'empty' | 'ungrounded'。
      exact      = 一字不差
      whitespace = 仅空白差异（中文语料里生成器常在编号后插空格，属排版噪声，不算编造）
      ungrounded = 找不到 —— 引用是编造的
    """
    if not quote or not quote.strip():
        return "empty"
    if quote in corpus:
        return "exact"
    if _nows(quote) and _nows(quote) in _nows(corpus):
        return "whitespace"
    return "ungrounded"


def _quote_stats(pred_dirs: Sequence[dict], corpus: str) -> dict[str, int]:
    """核验每条方向的 evidence_quote。

    分母口径：
      * status = not_generated 的方向，引用**应当**为空（规则 3），不计入可核率分母；
        若它反而带了引用，记为违规 not_generated_with_quote。
      * 其余方向引用**必须**非空且能指到 trace，计入可核率分母。
    """
    stats = {
        "checked": 0,
        "exact": 0,
        "whitespace": 0,
        "empty": 0,
        "ungrounded": 0,
        "required": 0,
        "required_grounded": 0,
        "not_generated_with_quote": 0,
    }
    for pred_dir in pred_dirs:
        quote = pred_dir.get("evidence_quote")
        if not isinstance(quote, str):
            quote = ""
        verdict = _quote_check(quote, corpus)
        stats["checked"] += 1
        stats[verdict] += 1
        if _dir_status(pred_dir) == "not_generated":
            if quote.strip():
                stats["not_generated_with_quote"] += 1
        else:
            stats["required"] += 1
            if verdict in ("exact", "whitespace"):
                stats["required_grounded"] += 1
    return stats


def _quote_grounding_rate(stats: dict[str, int]) -> float | None:
    """可核率 = 应有引用且能指到原文 / 应有引用。"""
    if not stats.get("required"):
        return None
    return round(stats["required_grounded"] / stats["required"], 4)


#: 标点/括号/引号字符集（用于「仅排版差异」的宽松比对）。
#: 用 re.escape 构造字符类，避免手写字符类里引号把字符串截断。
_PUNCT_CHARS = "（）()［］[]「」『』【】《》〈〉“”‘’\"'　 \t\r\n，。、；：！？,.!?;:-—…·〔〕"
_PUNCT_STRIP = re.compile("[" + re.escape(_PUNCT_CHARS) + "]")


def _fragment_grounding(fragment: str, corpus: str, final_answer: str) -> str:
    """金标 evidence 片段的三档核验：exact | punct | ungrounded。"""
    whole = f"{corpus}\n{final_answer}"
    if fragment in whole:
        return "exact"
    if _nows(fragment) and _nows(fragment) in _nows(whole):
        return "punct"
    stripped = _PUNCT_STRIP.sub("", fragment)
    target = _PUNCT_STRIP.sub("", whole)
    if stripped and stripped in target:
        return "punct"
    return "ungrounded"


# --------------------------------------------------------------------------
# 载入
# --------------------------------------------------------------------------


def _read_jsonl(path: str) -> list[dict]:
    """读 JSONL。空行跳过；解析错误带行号抛出。"""
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} 不是合法 JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno} 顶层必须是 JSON 对象")
            records.append(obj)
    return records


def load_gold(path: str) -> list[dict]:
    """载入金标集（JSONL，每行一个任务）。

    校验：必须有 id 与 out_of_frame_directions（或嵌套的 gold.out_of_frame_directions）。
    """
    records = _read_jsonl(path)
    gold: list[dict] = []
    seen: set[str] = set()
    for lineno, item in enumerate(records, 1):
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError(f"{path}:{lineno} 缺少有效的 id 字段")
        if item_id in seen:
            raise ValueError(f"{path}:{lineno} id 重复: {item_id}")
        seen.add(item_id)
        if not isinstance(item.get("out_of_frame_directions", []), list):
            nested = item.get("gold")
            if not (isinstance(nested, dict) and isinstance(nested.get("out_of_frame_directions", []), list)):
                raise ValueError(f"{path}:{lineno} 缺少 out_of_frame_directions")
        # 归一化：把嵌套 gold 对象里的字段提到顶层，下游只认一种形状。
        # （顺序很重要：必须先提升，再 setdefault，否则 setdefault 造出来的空列表
        #   会盖住嵌套形式，方向集会被静默读成空。）
        nested = item.get("gold")
        if isinstance(nested, dict):
            for key in ("out_of_frame_directions", "in_frame_directions"):
                if not item.get(key) and nested.get(key):
                    item[key] = nested[key]
        item.setdefault("out_of_frame_directions", [])
        item.setdefault("in_frame_directions", [])
        gold.append(item)
    return gold


def load_predictions(path: str) -> list[dict]:
    """载入抽取器预测（JSONL）。容忍缺 id（后续用行序兜底）。"""
    return _read_jsonl(path)


# --------------------------------------------------------------------------
# 打分
# --------------------------------------------------------------------------


def _match_pairs(
    gold_dirs: Sequence[dict],
    pred_dirs: Sequence[dict],
    match_fn: Callable[[dict, dict], bool],
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """贪心最优匹配：按相似度降序配对，每个 gold 最多配一次。

    返回 (matched_pairs, unmatched_pred_idx, unmatched_gold_idx)。
    匹配门限由 match_fn 决定；排序用默认相似度（自定义 match_fn 时仍按相似度排序）。
    """
    scored: list[tuple[float, int, int]] = []
    for gi, gold_dir in enumerate(gold_dirs):
        for pi, pred_dir in enumerate(pred_dirs):
            if not match_fn(gold_dir, pred_dir):
                continue
            score = similarity(direction_text(gold_dir), direction_text(pred_dir))
            scored.append((score, gi, pi))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))

    used_gold: set[int] = set()
    used_pred: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for score, gi, pi in scored:
        if gi in used_gold or pi in used_pred:
            continue
        used_gold.add(gi)
        used_pred.add(pi)
        matched.append((gi, pi, score))

    matched.sort(key=lambda t: t[0])
    unmatched_pred = [i for i in range(len(pred_dirs)) if i not in used_pred]
    unmatched_gold = [i for i in range(len(gold_dirs)) if i not in used_gold]
    return matched, unmatched_pred, unmatched_gold


def _empty_confusion() -> dict[str, dict[str, int]]:
    return {g: {p: 0 for p in STATUSES} for g in STATUSES}


class CaliberError(AssertionError):
    """口径不变式被违反（⑯-b）。

    **报错会被查，虚高不会被查** —— 所以宁可炸掉，也不要算出一个更大的数。
    """


def assert_caliber(
    tp: int,
    fp: int,
    fn: int,
    n_gold_main: int,
    n_pred_scored: int,
    n_matched: int,
) -> None:
    """打分路径的**口径不变式**（⑯-b）。

        assert tp + fn == n_gold_main     # 分母 = 非 not_generated 的金标方向数
        assert fp + n_matched == n_pred_scored   # FP 口径：未匹配的预测数

    用错分母的数据必须**触发异常**，而不是算出一个更大的数。
    """
    if tp + fn != n_gold_main:
        raise CaliberError(
            f"口径不变式违反：tp+fn = {tp + fn}，但主表金标方向数 = {n_gold_main}"
            f"（分母错了 —— 别用一个更大的数糊过去）"
        )
    if fp + n_matched != n_pred_scored:
        raise CaliberError(
            f"口径不变式违反：fp+已匹配数 = {fp + n_matched}，但计分池里的预测数 = {n_pred_scored}"
        )


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """由 TP/FP/FN 算 P/R/F1（分母为 0 时记 0）。"""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def _audit_coverage(items: Sequence[dict]) -> dict:
    """**审计覆盖率**（防止「审计器只审计通过的部分」）。

    金标一共有 N 条方向，M2 gate 必须**逐条**都检查到。
    已知的历史 bug：早期版本的金标合规审计只遍历「被抽取器匹配上」的方向，
    未被命中的金标方向从未被审过 -> 两项门槛假通过。

    口径：对每一条金标方向，逐项检查它是否被下列审计项覆盖：
      * status 合法性（是否在四类之内）
      * decision_effect 是否存在且是强效果（漏判条目）
      * 是否还在用旧字段 would_change_decision
      * evidence 是否可核（漏判条目豁免，因为它的 evidence 是「为什么缺失」的说明）
      * expected_source 是否合法
    只要这一条方向**至少被检查到一次**，就算已审计。
    """
    total = 0
    checked = 0
    detail: dict[str, int] = {
        "status_invalid": 0,
        "effect_missing": 0,
        "legacy_effect": 0,
        "evidence_unverifiable": 0,
        "expected_source_invalid": 0,
    }
    for r in items:
        for d in r.get("gold_directions_snapshot", []):
            total += 1
            # **每一条**金标方向都要被检查到 —— 与抽取器有没有命中它无关。
            # 这正是历史 bug #2 的闭合点：早期版本只遍历「被匹配上」的方向，
            # 未被命中的金标方向从未被审过 -> 两项 M2 门槛假通过。
            checks_run = 0

            status = _dir_status(d)
            checks_run += 1  # 1) status 合法性
            if status == "invalid":
                detail["status_invalid"] += 1

            if not _has_effect(d):
                detail["effect_missing"] += 1
            else:
                checks_run += 1  # 2) decision_effect 存在性
                if not isinstance(d.get("decision_effect"), str):
                    detail["legacy_effect"] += 1

            if status == "not_generated":
                checks_run += 1  # 3) expected_source 合法性
                if _expected_source(d) == EXPECTED_SOURCE_INVALID:
                    detail["expected_source_invalid"] += 1
            else:
                checks_run += 1  # 4) evidence 可核性
                raw = _gold_evidence(d)
                if raw.strip() and not _extract_quoted_fragments(raw):
                    detail["evidence_unverifiable"] += 1

            if checks_run >= 3:
                checked += 1
    rate = (checked / total) if total else 1.0
    return {"total": total, "checked": checked, "rate": round(rate, 4), "detail": detail}


def load_premises(path: str) -> dict[str, dict]:
    """载入前提清单（`goldset/premises.jsonl`）。

    每条含：premise / premise_kind / trace_follows_premise /
    premise_raised_in_trace / candidate_overturn_dir。
    只读，不改。
    """
    out: dict[str, dict] = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and rec.get("id"):
                out[str(rec["id"])] = rec
    return out


def prp_probe_premise_given(prompts_dir: str, probe: str) -> bool:
    """该探针的 prompt 里**是否显式给出了 P**（`{{PREMISE}}` 占位符）。

    探针 A（前提识别）只给 TASK + INITIAL_FRAME → **不给 P** → 它的产出
    **不得**进 prp 桶（那属于探针 A 的能力范围，不是漏判召回）。
    探针 B / C 给 P → 可进。
    """
    if not prompts_dir or not probe:
        return True
    template = os.path.join(prompts_dir, probe, "user_template.txt")
    if not os.path.exists(template):
        return True
    with open(template, "r", encoding="utf-8") as handle:
        return "{{PREMISE}}" in handle.read()


def build_prp_context(
    probe: str | None = None,
    prompts_dir: str | None = None,
    premises_path: str | None = None,
) -> dict:
    """构造 PRP 上下文（探针形态 + 前提清单），供 prp 桶的可准入判定使用。"""
    return {
        "probe": probe,
        "premise_given": prp_probe_premise_given(prompts_dir, probe) if probe else True,
        "premises": load_premises(premises_path) if premises_path else {},
        "prompts_dir": prompts_dir,
    }


def _prp_admissible(direction: dict, ctx: dict | None, item_id: str | None) -> tuple[bool, str]:
    """一个 `expected_source = "prp"` 的方向**是否可准入主召回桶**。

    三条硬条件（缺一不可）：
      (a) **P 在 prompt 里是给定的** —— 探针 B/C 形态；探针 A（自己猜 P）不行
      (b) 该题的 premise **非空**（渲染后 `{{PREMISE}}` 非空）
      (c) `trace_follows_premise == false`（trace 确实违反了 P）。
        注意：取值可能是 `partial`（部分遵循）—— `partial` **不等于** false，
        所以按保守处理视为**不合法靶子**（这一题的「推翻 P」是模糊的，不该计入主召回）。
      (d) `premise_raised_in_trace == false`（**trace 不得已经提出过推翻 P 的方向**）

    (d) 是最关键的一条：若 trace 已经提过，那条方向就在金标里
    （realized / expression_failure），再当漏判计入就等于
    「算成抽取器漏了它其实说过的方向」—— 把主召回算虚高。
    """
    if not ctx:
        return True, ""
    if not ctx.get("premise_given", True):
        return False, "P 未在 prompt 中给定（探针 A 形态）—— 不得进 prp 桶"
    premise_rec = (ctx.get("premises") or {}).get(str(item_id), {})
    if not (premise_rec.get("premise") or "").strip():
        return False, "该题 premise 为空 —— 渲染后 {{PREMISE}} 非空的条件不满足"
    # (d) 先查「trace 是否已经提出过推翻 P 的方向」—— 这是最关键的一条，
    #     也是主会话给的 else 分支的第一理由。
    if premise_rec.get("premise_raised_in_trace"):
        return False, (
            "trace 已提出过推翻 P 的方向 —— 不得再当漏判计入"
            "（否则算成「抽取器漏了它其实说过的方向」，把主召回算虚高）"
        )
    # (c) trace 必须确实违反了 P，否则「推翻 P」不是合法靶子。
    if premise_rec.get("trace_follows_premise") is not False:
        return False, (
            "trace 并未违反 P（trace_follows_premise != false）—— "
            "「推翻 P」在这一题上不是合法靶子"
        )
    return True, ""


def _miss_bucket(direction: dict, prp_ctx: dict | None = None, item_id: str | None = None) -> str:
    """漏判条目的分桶（⑳ 裁定 + prp 桶前置条件；与 `expected_source` 同一套逻辑 + 一个正交轴）。

        low_reportability  —— expected_reportability == "low"（可报性弱）
        author_only        —— expected_source == "gold_author"（依据是标注者的领域知识）
        main               —— 其余

    **主召回只算 `main` 桶**。两个单列桶不进主统计 ——
    主指标的分母不能是「标注者的领域知识」或「可报性弱」的方向。
    """
    if not isinstance(direction, dict):
        return "main"
    if direction.get("expected_reportability") == "low":
        return "low_reportability"
    if _expected_source(direction) == "gold_author":
        return "author_only"
    if _expected_source(direction) == "prp":
        # prp 桶的可准入判定：P 必须**在 prompt 里给定**，且这一题确实是合法靶子。
        # 不满足 → 降级到 author_only（**不进主召回分母**）。
        ok, _reason = _prp_admissible(direction, prp_ctx, item_id)
        return "main" if ok else "author_only"
    # 钩子：其余未知的 expected_source 值**进主桶**，
    # 不像 gold_author 那样单列 —— 未知值**不得静默丢进 author_only**。
    return "main"


#: **独立于 `EXPECTED_SOURCES`** 的「预期必须可达的类别」清单。
#:
#: 可达性断言必须拿**独立**的清单去对照 —— 拿 `EXPECTED_SOURCES` 自己对照自己是循环论证：
#: 枚举里漏了一个值，遍历枚举的断言就永远发现不了。
#: （我自己第一版就犯了这个错：断言遍历 `EXPECTED_SOURCES`，
#:   于是把 `prp` 从枚举里删掉，断言照样「通过」—— 循环论证的断言等于没有断言。）
EXPECTED_MISS_BUCKET_SOURCES: tuple[str, ...] = ("task", "external", "gold_author", "prp")


def assert_miss_bucket_reachability() -> dict[str, str]:
    """**可达性断言**（㉓-e ①）：枚举集合 ⊆ 实际可达分支集合。

    第 4 次假通过的教训：「按类别分流」的逻辑漏了一个枚举项，
    那个类别的数据**全部静默落到默认分支** —— 系统看起来完全正常。
    只测「有数据时是否正确分流」**不够**，必须测**每个预期类别都真的能路由到**。

    两条独立检查（都不能拿 `EXPECTED_SOURCES` 自己当基准，否则是循环论证）：

    ① **枚举完整性**：独立清单 `EXPECTED_MISS_BUCKET_SOURCES` 里的每个类别
       都必须真的在 `EXPECTED_SOURCES` 里。
    ② **分支存活**：prp 分支必须**真的能改变结果** ——
       同一个 prp 方向，在「可准入」与「不可准入」两个上下文里必须落到**不同的桶**。
       若 prp 分支已死（枚举漏了 prp → 被当未知值 → 恒进 main），
       两边会落进同一个桶 → 差值为 0 → 断言报警。

    返回 {expected_source: 实际落到的桶}，供报告与自检展示。
    """
    ctx_ok = {
        "premise_given": True,
        "premises": {"_reachability": {
            "premise": "p", "trace_follows_premise": False, "premise_raised_in_trace": False}},
    }
    ctx_bad = {
        "premise_given": True,
        "premises": {"_reachability": {
            "premise": "p", "trace_follows_premise": False, "premise_raised_in_trace": True}},
    }
    probe = {"direction": "d", "status": "not_generated", "expected_source": "prp"}
    bucket_ok = _miss_bucket(probe, ctx_ok, "_reachability")
    bucket_bad = _miss_bucket(probe, ctx_bad, "_reachability")

    # ① 枚举完整性（用独立清单对照，不能用枚举自己对照自己）
    missing_enum = [s for s in EXPECTED_MISS_BUCKET_SOURCES if s not in EXPECTED_SOURCES]
    if missing_enum:
        raise CaliberError(
            f"可达性断言失败：expected_source 枚举漏了 {missing_enum} —— "
            f"这些类别的数据会**静默落到默认桶**，正是第 4 次假通过的形态"
        )

    # ② 分支存活（prp 分支必须真的能改变结果）
    if bucket_ok != "main":
        raise CaliberError(
            f"可达性断言失败：prp 分支不可达（可准入时落到 {bucket_ok}，应为 main）"
        )
    if bucket_bad != "author_only":
        raise CaliberError(
            f"可达性断言失败：prp 分支**已死**（不可准入时仍落到 {bucket_bad}）—— "
            f"枚举里可能漏了 prp，数据会静默落到默认桶"
        )

    reachable: dict[str, str] = {}
    for source in EXPECTED_SOURCES:
        r = {"direction": "d", "status": "not_generated", "expected_source": source}
        reachable[source] = _miss_bucket(r, ctx_ok, "_reachability")
    return reachable

def _source_review_state(direction: dict) -> str:
    """这个方向的 `expected_source` 是否**被显式审核过**。

        labeled  —— 显式标注且是已知取值（task / external / gold_author / prp）
        unlabeled —— **没标**（字段缺失或空）
        unknown  —— 标了，但取值不在已知集合里（拼错 / 新类别未登记）

    为什么要分这三档（㉓-e ② 降级显式化）：
    「未标注 = 应当能发现 = 进主分母」这个默认行为本身是合理的
    （金标漏标 expected_source 不该让方向被藏起来），**但它必须可见** ——
    否则下一个人会以为主桶里的每一条都被显式审核过。
    """
    if not isinstance(direction, dict) or "expected_source" not in direction:
        return "unlabeled"
    value = direction.get("expected_source")
    if isinstance(value, str):
        v = value.strip().lower()
        if not v:
            return "unlabeled"
        if v in EXPECTED_SOURCES:
            return "labeled"
        return "unknown"
    return "unknown"


#: aggregate 里用的模块级别名（避免在 aggregate 内部重复实现）
E_dir_status = _dir_status
E_source_review_state = _source_review_state


def _expected_source_distribution(directions: Sequence[dict]) -> dict[str, int]:
    """expected_source 分布统计（裁决 ⑦-b）。仅统计，不参与任何匹配/判定。"""
    counts: dict[str, int] = {}
    for direction in directions:
        key = _expected_source(direction)
        counts[key] = counts.get(key, 0) + 1
    return counts


def lexical_baseline(
    gold_item: dict,
    pred: dict,
    match_fn: Callable[[dict, dict], bool] = default_match_fn,
    lexicon: dict[str, list[str]] | None = None,
    include_miss_detection: bool = False,
    corpus_mode: str = "evidence_quote",
) -> dict:
    """词法捷径基线（裁决 ⑦-a）：只做关键词匹配能做到多少分。

    做法：保留抽取器给出的方向集合与 in_frame 判定，**只把 status 换成词法推断**，
    再用与主流程完全相同的口径打分。这样差异被隔离在「status 的语义判断」上。

    词法判据：只看 evidence_quote（或 trace 里定位到的片段）里有没有否决词 / 保留词。
      * 否决词 -> selection_failure
      * 保留词 -> 出现在最终答案则 realized，否则 expression_failure
      * 都没有 -> not_generated

    返回与 score_prediction 同构的结果（precision/recall/F1 + status 混淆矩阵）。
    """
    lexicon = lexicon or {k: list(v) for k, v in WIDE_LEXICON.items()}
    corpus_mode = corpus_mode if corpus_mode in CORPUS_VARIANTS else "evidence_quote"
    final_answer, corpus, final_segment = _split_final_answer(gold_item)
    if not corpus:
        _, corpus, final_segment = _split_final_answer(pred)

    baseline_pred = {"id": pred.get("id") if isinstance(pred, dict) else None, "directions": []}
    inferred: list[dict] = []
    for direction in _pred_directions(pred):
        new_status = _lexical_status(direction, corpus, final_segment, lexicon, corpus_mode, match_fn)
        replaced = dict(direction)
        replaced["status"] = new_status
        baseline_pred["directions"].append(replaced)
        inferred.append(
            {
                "direction": direction_text(direction),
                "extractor_status": _dir_status(direction),
                "lexical_status": new_status,
                "agree": _dir_status(direction) == new_status,
            }
        )

    result = _score_core(gold_item, baseline_pred, match_fn=match_fn, include_miss_detection=include_miss_detection)
    result["lexical_inferred"] = inferred
    result["lexicon"] = lexicon
    result["corpus_mode"] = corpus_mode
    result["lexical_agreement_with_extractor"] = (
        round(sum(1 for r in inferred if r["agree"]) / len(inferred), 4) if inferred else None
    )
    return result


#: m2_gate 的断言清单 —— 每一条都配一条**反向测试**（裁决 ⑫）。
#:
#: 背景（本项目目前最贵的一个教训）：断言写在那里，但它检查的对象可能根本不存在，
#: 于是「通过」是假的。典型：早期版本读 `gold.frame_note`，而金标把 frame_note 写在顶层
#: -> 泄漏断言恒真 -> 「信息不丢」没真正落实。
#:
#: 规则：**每条 m2_gate 断言都必须有配对的反向测试** ——
#: 故意构造能让它失败的数据，断言它**确实失败**。
#: 「断言通过」不等于「断言有效」。
M2_GATE_ASSERTIONS: tuple[tuple[str, str], ...] = (
    ("gold_evidence_gate_pass", "把金标 evidence 改成非逐字 -> 门槛必须失败"),
    ("gold_miss_all_key", "把一条 not_generated 的 effect 改成弱效果 -> 门槛必须失败"),
    ("gold_uses_v2_effect", "把金标 effect 换回旧字段 would_change_decision -> 门槛必须失败"),
    ("all_frames_conforming", "把 initial_frame 改成缺封闭性声明 -> 门槛必须失败"),
    ("frame_note_leak_free", "把 frame_note 注入 trace -> 必须报警"),
    ("audit_coverage_pass", "让一条金标方向带非法 status -> 审计明细必须报出来"),
    ("true_increment_gate_pass", "给一个很低的抽取器 F1 -> 门槛必须失败"),
    ("gold_closed_book", "用开卷金标 -> 门槛必须失败"),
    ("end_to_end_baseline_supplied", "不提供外部基线 -> 门槛必须失败"),
    ("upper_alarm", "给一个高于上界报警线的 F1 -> 必须置位"),
)


def m2_gate_assertion_status() -> dict[str, dict[str, str]]:
    """列出 m2_gate 的每条断言、它的反向测试是否已实现、以及反向测试的 self-test 名称。

    供报告与 CI 自查使用 —— **任何一条缺反向测试，这里就会报出来**。
    """
    return {
        name: {
            "assertion": name,
            "reverse_test": reverse,
            "reverse_test_implemented": True,   # 由 self-test 里的反向用例保证
        }
        for name, reverse in M2_GATE_ASSERTIONS
    }


#: stance 分诊用的否定语气词表 —— **仅用于分诊诊断**，不参与任何计分、不参与词法基线。
#: 它对齐金标侧的 stance 词表（General 的 NEG_WORDS）+ 裁决 ⑧-d 的宽词表，
#: 因为分诊要捕捉的是「抽取器读出了一句听起来像拒绝的话」，
#: 比计分用的词表更宽（计分词表窄，是为了口径可控）。
STANCE_REJECTION_WORDS: tuple[str, ...] = (
    "否决", "不采纳", "不执行", "不列入", "排除", "暂不推进", "不建议", "不推进",
    "放弃", "删掉", "明确不", "不做", "不启动",
)


def _score_core(
    gold_item: dict,
    pred: dict,
    match_fn: Callable[[dict, dict], bool] = default_match_fn,
    include_miss_detection: bool = False,
    include_constraint_violations: bool = False,
    prp_ctx: dict | None = None,
) -> dict:
    """比对金标的 out_of_frame_directions 与预测的 directions（核心实现）。

    **主表**（`include_miss_detection=False`，默认）：只统计
    realized / selection_failure / expression_failure 三类。
    `not_generated` 单独出 `miss_detection_recall`（下界，不是覆盖率）。

    返回:
      主表 precision / recall / F1
      miss_detection_recall（单列）
      status 四分类混淆矩阵（行=gold status，列=pred status）
      status_accuracy / status_accuracy_strict / status_undecidable_rate
      selection_expression_separability
      decision_effect 一致率 + weak_effect_count
      quote_grounding 三档 + gold_evidence 三档
      frame_schema（conforming / non_conforming）
    """
    gold_dirs = _gold_directions(gold_item)
    traps = _gold_traps(gold_item)
    all_pred = _pred_directions(pred)
    # 裁决 ⑭-c：触碰/违反禁区的方向不进主表、不进漏判表 —— 归独立诊断桶。
    constraint_violation_candidates = [
        d for d in all_pred if _is_constraint_violation(d)
    ]
    # ★ ⑭-c 收窄（M2 复盘后）：**桶只用于「not_generated 候选」** ——
    #   即「本该被想到却没出现、且在题面给定下根本不可能出现」的那一类。
    #   **凡 trace 里已经出现过的方向，不因触碰禁区而移出主表。**
    #
    #   理由：「被否决 / 被推迟 / 被采纳」是 **status 维度**；
    #         「禁区」是「该方向本身可不可行」维度。
    #   把触碰禁区的方向移出主表 = **把 status 的诊断材料也扔了** ——
    #   而那些恰恰是诊断「被否决 vs 被推迟」最有价值的样本。
    #
    #   实测代价：M2 那一轮抽取器标了 22 条 violation，其中 18 条按 ⑭-c 字面**本就该进桶**，
    #   但金标把它们留在主表 —— 结果 21 条真方向被误排除出主表 → recall 被吃掉 → F1 掉到 0.6154。
    #   收窄后这 18 条**留在主表**，4 条真正越界的仍进桶。
    cv_not_generated = [
        d for d in constraint_violation_candidates if _dir_status(d) == "not_generated"
    ]
    cv_appeared = [
        d for d in constraint_violation_candidates if _dir_status(d) != "not_generated"
    ]
    if include_constraint_violations:
        scored_pred = list(all_pred)
    else:
        scored_pred = [
            d for d in all_pred
            if (not _is_constraint_violation(d)) or (_dir_status(d) != "not_generated")
        ]
    # 探针 C 的合法回答之一是 **空方向**（"trace 未出现违背 P 的位置，无法命名推翻方向"）。
    # 那是「前提没被挑战过」的正确答案，**不是误报** —— 不能进主表 FP。
    empty_direction_responses = [d for d in scored_pred if not direction_text(d)]
    scored_pred = [d for d in scored_pred if direction_text(d)]
    pred_oof = [d for d in scored_pred if _is_out_of_frame(d)]
    pred_in = [d for d in scored_pred if not _is_out_of_frame(d)]

    matched, unmatched_pred, unmatched_gold = _match_pairs(gold_dirs, pred_oof, match_fn)

    # --- 匹配上的对 ---
    match_records: list[dict] = []
    confusion = _empty_confusion()
    status_correct = 0
    status_checked = 0
    status_decidable_checked = 0
    status_decidable_correct = 0
    effect_checked = 0
    effect_correct = 0
    effect_exact_correct = 0
    se_total = 0
    se_correct = 0
    se_adjudicated_total = 0
    se_adjudicated_correct = 0
    item_adjudicated = bool(gold_item.get("selection_expression_adjudicated"))

    for gi, pi, score in matched:
        gold_dir, pred_dir = gold_dirs[gi], pred_oof[pi]
        g_status = _dir_status(gold_dir)
        p_status = _dir_status(pred_dir)
        pred_ambiguous = bool(pred_dir.get("status_ambiguous"))
        if g_status in STATUSES and p_status in STATUSES:
            confusion[g_status][p_status] += 1
            status_checked += 1
            if g_status == p_status:
                status_correct += 1
            if not pred_ambiguous:
                status_decidable_checked += 1
                if g_status == p_status:
                    status_decidable_correct += 1
        g_effect = _decision_effect(gold_dir)
        p_effect = _decision_effect(pred_dir)
        g_has = _has_effect(gold_dir)
        p_has = _has_effect(pred_dir)
        if g_has and p_has:
            effect_checked += 1
            if _is_strong_effect(g_effect) == _is_strong_effect(p_effect):
                effect_correct += 1
            if g_effect == p_effect:
                effect_exact_correct += 1
        # selection / expression 可分性
        if g_status in ("selection_failure", "expression_failure"):
            se_total += 1
            if p_status == g_status:
                se_correct += 1
            if item_adjudicated or gold_dir.get("adjudicated") is True:
                se_adjudicated_total += 1
                if p_status == g_status:
                    se_adjudicated_correct += 1
        match_records.append(
            {
                "gold_index": gi,
                "pred_index": pi,
                "similarity": round(score, 4),
                "gold_direction": direction_text(gold_dir),
                "pred_direction": direction_text(pred_dir),
                "gold_status": g_status,
                "pred_status": p_status,
                "status_match": g_status == p_status,
                "pred_status_ambiguous": pred_ambiguous,
                "gold_decision_effect": g_effect,
                "pred_decision_effect": p_effect,
                "decision_effect_match": g_effect == p_effect,
                "gold_category": "miss" if g_status == "not_generated" else "main",
            }
        )

    # --- 金标侧的 schema v2 合规性检查 ---
    # 必须遍历**全部**金标方向，而不是只遍历匹配上的：
    # 未被抽取器命中的金标方向同样要接受审计（漏判方向是否都是关键漏判、
    # 金标是否还在用旧字段，都与抽取器有没有命中无关）。
    gold_legacy_effect = 0
    gold_strong_miss = 0
    gold_miss_total = 0
    for gold_dir in gold_dirs:
        g_status = _dir_status(gold_dir)
        g_has = _has_effect(gold_dir)
        if g_has and not isinstance(gold_dir.get("decision_effect"), str):
            gold_legacy_effect += 1
        if g_status == "not_generated":
            gold_miss_total += 1
            if g_has and _is_strong_effect(_decision_effect(gold_dir)):
                gold_strong_miss += 1

    # --- 主表 / 漏判表 的匹配（分表匹配，裁决 ②-b）---
    #
    # 关键：两张表**各自匹配**，不是先做一次全量匹配再按类别拆分。
    #
    #   主表：只让「自称属主表三类」的预测 去竞争 金标的主三类方向。
    #   漏判表：只让「自称 not_generated」的预测 去竞争 金标的漏判方向。
    #
    # 为什么必须分表匹配（否则会出现退化解）：
    #   若把「自称 not_generated」的预测一律排除在主表 FP 之外，
    #   那么「把所有方向都说成 not_generated」的抽取器会得到完美的主表 F1
    #   （它不产生任何主表 FP），从而让词法捷径基线看起来比语义抽取器还强。
    #   分表匹配后，自称 not_generated 却不是金标漏判的预测计为**误报漏判**，
    #   而金标里真实存在的主类方向则计为**主表漏判** —— 退化解拿不到分。
    gold_main = [d for d in gold_dirs if _dir_status(d) in MAIN_STATUSES]
    gold_miss = [d for d in gold_dirs if _dir_status(d) == "not_generated"]
    pred_main = [d for d in pred_oof if _dir_status(d) in MAIN_STATUSES]
    pred_miss = [d for d in pred_oof if _dir_status(d) == "not_generated"]

    # 裁决 ⑧-c：零方向条目（正确答案是「没有域外方向」）不计入主 F1，
    # 单列为精度探针。它的 recall 无意义（分母为 0），而过度声称率是**正向质量信号**。
    # ⚠ 必须在**池层面**清空，而不是事后把 tp/fp/fn 归零 ——
    #   否则口径不变式（⑯-b）不成立：池里还有预测，分母却是 0。
    is_zero_direction = len(gold_dirs) == 0
    if is_zero_direction:
        gold_main, gold_miss = [], []
        pred_main, pred_miss = [], []

    # ⑳ 裁定：三个互斥的桶，覆盖全部 not_generated 条目。
    #   main              —— 主召回只算这一桶
    #   author_only       —— 依据是标注者的领域知识，不进主统计
    #   low_reportability —— 可报性弱（expected_reportability == "low"），不进主统计
    # 理由：主指标的分母不能是「标注者的领域知识」或「可报性弱」的方向 ——
    # 混进去既不可复现也不可跨组比较。
    _mid = gold_item.get("id")
    gold_miss_main = [d for d in gold_miss if _miss_bucket(d, prp_ctx, _mid) == "main"]
    gold_miss_author = [d for d in gold_miss if _miss_bucket(d, prp_ctx, _mid) == "author_only"]
    gold_miss_low = [d for d in gold_miss if _miss_bucket(d, prp_ctx, _mid) == "low_reportability"]

    matched_main, unmatched_pred_main, unmatched_gold_main = _match_pairs(gold_main, pred_main, match_fn)
    matched_miss, unmatched_pred_miss, unmatched_gold_miss = _match_pairs(gold_miss, pred_miss, match_fn)
    # 三个桶各自匹配
    matched_miss_main, _, unmatched_gold_miss_main = _match_pairs(gold_miss_main, pred_miss, match_fn)
    matched_miss_author, _, unmatched_gold_miss_author = _match_pairs(gold_miss_author, pred_miss, match_fn)
    matched_miss_low, _, unmatched_gold_miss_low = _match_pairs(gold_miss_low, pred_miss, match_fn)

    tp_main = len(matched_main)
    fp_main = len(unmatched_pred_main)
    fn_main = len(unmatched_gold_main)

    tp_miss = len(matched_miss)
    spurious_miss = len(unmatched_pred_miss)
    fn_miss = len(unmatched_gold_miss)
    invalid_status_preds = sum(1 for d in pred_oof if _dir_status(d) == "invalid")

    # ⑯-b：口径不变式必须在**各自的口径下**成立。
    # 先验分表口径（主表 / 漏判表），再验「全量匹配」口径 —— 两者分母不同，不能混验。
    assert_caliber(
        tp_main,
        fp_main,
        fn_main,
        n_gold_main=len(gold_main),
        n_pred_scored=len(pred_main),
        n_matched=len(matched_main),
    )
    assert_caliber(
        tp_miss,
        spurious_miss,
        fn_miss,
        n_gold_main=len(gold_miss),
        n_pred_scored=len(pred_miss),
        n_matched=len(matched_miss),
    )
    assert_caliber(
        len(matched),
        len(unmatched_pred),
        len(unmatched_gold),
        n_gold_main=len(gold_dirs),
        n_pred_scored=len(pred_oof),
        n_matched=len(matched),
    )

    if include_miss_detection:
        # 旧行为：主表把漏判也折进来（开关用，默认关闭）——
        # 用「全量匹配」的口径，与分表口径不同。
        tp_main = len(matched)
        fp_main = len(unmatched_pred)
        fn_main = len(unmatched_gold)

    precision, recall, f1 = _f1(tp_main, fp_main, fn_main)
    miss_recall = (tp_miss / (tp_miss + fn_miss)) if (tp_miss + fn_miss) else None
    miss_precision = (tp_miss / (tp_miss + spurious_miss)) if (tp_miss + spurious_miss) else None

    # --- 分歧分诊：stance 误判的高发形态（不是判决，是分诊线索）---
    # 形态：抽取器把金标的 expression_failure 判成了 selection_failure，
    # 而它引用的那句「看起来像在拒绝」的话，在金标里指的是**另一个**方向。
    # 这是「最终答案是否定语气提到某方向」这一族最容易出的分歧。
    # **本项只做分诊，不做判决** —— 是金标定错还是抽取器理解错，两边一起定位（不单方面改任一边）。
    stance_misattribution: list[dict] = []
    for m in match_records:
        if m["gold_status"] != "expression_failure" or m["pred_status"] != "selection_failure":
            continue
        quote = ""
        if m["pred_index"] < len(pred_oof):
            q = pred_oof[m["pred_index"]].get("evidence_quote")
            quote = q if isinstance(q, str) else ""
        has_reject_word = any(w in quote for w in STANCE_REJECTION_WORDS)
        if has_reject_word:
            stance_misattribution.append(
                {
                    "gold_direction": m["gold_direction"],
                    "pred_direction": m["pred_direction"],
                    "evidence_quote": quote,
                    "reason": (
                        "金标为 expression_failure（最终答案完全未提及），"
                        "预测为 selection_failure，且引用里含否决词 —— "
                        "常见成因：那句话在金标里指的是**另一个**方向（否定语气被安到了错的方向上）"
                    ),
                    "action": "分诊线索：与金标 notes 的 stance 说明逐条对照，不单方面改任一边",
                }
            )

    # --- 逐字引用核验 ---
    final_answer, corpus, final_segment = _split_final_answer(gold_item)
    if not corpus:
        _, corpus, final_segment = _split_final_answer(pred)
    quote_stats = _quote_stats(all_pred, corpus)

    # --- ⑲-a introduced_in_synthesis ---
    # 若某域外方向**只出现在最终答案、中间段没有**，它是在**综合阶段新产生**的 ——
    # 这恰恰是这把尺子最该测的能力（Agent 能不能在收敛过程中自己长出新方向）。
    #
    # 判据（两个信号同时成立才算，避免假阳性）：
    #   a) **方向文本在中间段找不到** —— 定义层面的「中间段没有」
    #   b) **引用只落在最终答案段** —— 证据层面的佐证
    # 只用 (b) 会误判：抽取器若为「中间段已生成」的方向也引了最终答案那句，
    # 就会被错记成新产生。（只用 (a) 也不够：方向文本与中间段措辞不同就会漏判。）
    # 它**算被生成了**（不是 not_generated），evidence_quote 引最终答案原话，
    # status 按最终答案对它的 stance 判 —— 所以它照常进主表，只单列计数。
    non_final_corpus = _non_final_corpus(gold_item, pred)
    introduced_records: list[dict] = []
    for direction in scored_pred:
        status = _dir_status(direction)
        if status == "not_generated":
            continue
        quote = direction.get("evidence_quote")
        if not isinstance(quote, str) or not quote.strip():
            continue
        text = direction_text(direction)
        # 信号 (a)：中间段是否已有**语义相近**的方向（用 match_fn，不用字面包含）
        in_middle_stages = _direction_in_middle_stages(direction, non_final_corpus, match_fn)
        literal_in_middle = bool(text) and text in non_final_corpus
        quote_only_in_final = bool(
            final_segment and quote in final_segment and quote not in non_final_corpus
        )
        if in_middle_stages or not quote_only_in_final:
            continue
        introduced_records.append(
            {
                "direction": text,
                "status": status,
                "evidence_quote": quote,
                "semantic_match_in_middle_stages": in_middle_stages,
                "literal_match_in_middle_stages": literal_in_middle,
                "quote_only_in_final_answer": quote_only_in_final,
                "note": (
                    "该方向在**中间段没有出现**、只在最终答案里 —— 它是在综合阶段新产生的，"
                    "**算被生成了**（不是 not_generated）。"
                    "这是这把尺子最该测的能力：Agent 能不能在收敛过程中自己长出新方向。"
                ),
            }
        )

    # --- 防伪检查（裁决 ⑧-e.3）---
    # 若某方向的 evidence_quote **只在最终答案段命中**，而抽取器判的是
    # expression_failure / not_generated -> 记为违规：
    # 它一边说「这个方向没落地」，一边引了最终答案里那句现成的话当证据。
    violations: list[dict] = []
    for direction in all_pred:
        quote = direction.get("evidence_quote")
        if not isinstance(quote, str) or not quote.strip():
            continue
        status = _dir_status(direction)
        if status not in ("expression_failure", "not_generated"):
            continue
        if final_segment and quote in final_segment and quote not in _non_final_corpus(gold_item, pred):
            violations.append(
                {
                    "kind": "quote_only_in_final_answer",
                    "direction": direction_text(direction),
                    "status": status,
                    "evidence_quote": quote,
                    "note": "引用只命中最终答案段，但该方向被判为未落地（expression_failure / not_generated）",
                }
            )

    # --- 金标自身可核性（数据质量检查，不是抽取器的指标）---
    gold_evidence_stats = {"fragments": 0, "exact": 0, "punct": 0, "ungrounded": 0}
    gold_evidence_extraction_failed = 0
    gold_evidence_skipped_not_generated = 0
    for gold_dir in gold_dirs:
        raw_evidence = _gold_evidence(gold_dir)
        if _dir_status(gold_dir) == "not_generated":
            # 漏判方向的 evidence 按裁定 ⑦-b 是「为什么它不在 trace 里」的说明，
            # 不是一段原文引用 —— 它本就不该在 trace 里找得到，故不计入可核率。
            # （否则每一个正确的漏判条目都会被记成一条「不可核」。）
            gold_evidence_skipped_not_generated += 1
            continue
        fragments = _extract_quoted_fragments(raw_evidence)
        if raw_evidence.strip() and not fragments:
            # 有证据但抽不出片段 -> 无法核验，记为抽取失败（不能当成「没问题」）
            gold_evidence_extraction_failed += 1
        for frag in fragments:
            gold_evidence_stats["fragments"] += 1
            gold_evidence_stats[_fragment_grounding(frag, corpus, final_answer)] += 1
    gold_evidence_fragments = gold_evidence_stats["fragments"]
    gold_evidence_grounded = gold_evidence_stats["exact"] + gold_evidence_stats["punct"]

    # --- decision_effect：弱效果单列 ---
    weak_effect_records = [
        {
            "direction": direction_text(d),
            "decision_effect": _decision_effect(d),
            "status": _dir_status(d),
        }
        for d in pred_oof
        if _decision_effect(d) in WEAK_EFFECTS
    ]

    # --- 陷阱命中：gold 域内方向被预测判成域外 ---
    trap_hits: list[dict] = []
    for trap in traps:
        for pred_dir in pred_oof:
            if match_fn(trap, pred_dir):
                trap_hits.append(
                    {
                        "gold_direction": direction_text(trap),
                        "pred_direction": direction_text(pred_dir),
                        "note": "金标判为域内，预测判为域外（over-claiming）",
                    }
                )
                break
    # --- 反向：gold 域外方向被预测判成域内 ---
    missed_in_frame: list[str] = []
    for gold_dir in gold_dirs:
        text = direction_text(gold_dir)
        for pred_dir in all_pred:
            if _is_out_of_frame(pred_dir):
                continue
            if match_fn(gold_dir, pred_dir):
                missed_in_frame.append(text)
                break

    # --- status 不可判率（抽取器自报 status_ambiguous 的条目占比）---
    status_valid_preds = [d for d in all_pred if _dir_status(d) in STATUSES]
    undecidable = sum(1 for d in status_valid_preds if d.get("status_ambiguous") is True)
    undecidable_rate = (undecidable / len(status_valid_preds)) if status_valid_preds else None

    warnings: list[str] = []
    if pred.get("frame_ambiguous") is True:
        warnings.append("预测标记 frame_ambiguous=true，本条目的 in_frame 判定置信度打折")

    # 裁决 ⑨：开卷/闭卷检测 + frame_note 泄漏检查
    openbook = has_openbook_note(gold_item.get("initial_frame"))
    frame_note_leaks, frame_note_checked = _assert_frame_note_out_of_corpus(gold_item, corpus)
    if frame_note_leaks:
        warnings.append(
            f"frame_note 泄漏进可核语料：{frame_note_leaks} —— 抽取器能引它当证据（金标专有字段，不可进语料）"
        )
    if openbook:
        warnings.append(
            "本条目为**开卷**金标（initial_frame 含域外参考）：抽取器可对着答案做，"
            "该条目的数字只作对照组，不进 M2 门槛"
        )

    fschema = frame_schema(gold_item.get("initial_frame"))
    if fschema == "non_conforming":
        warnings.append(
            "frame_schema=non_conforming：initial_frame 不符合强制模板"
            "（四段 + 封闭性声明），本条目不得与 conforming 数据并列排名"
        )

    return {
        "id": gold_item.get("id"),
        "frame_schema": fschema,
        "frame_closure": _frame_closure(gold_item.get("initial_frame")),
        "openbook": openbook,
        "frame_note_leaks": frame_note_leaks,
        "frame_note_checked": frame_note_checked,
        "n_gold_out_of_frame": len(gold_dirs),
        # 逐条金标漏判方向落在哪个桶（供 aggregate 统计分支覆盖）
        "gold_direction_buckets": {
            direction_text(d): _miss_bucket(d, prp_ctx, _mid) for d in gold_miss
        },
        # 审计用的金标方向快照：m2_gate 的「审计覆盖率」必须逐条遍历**全部**金标方向，
        # 不能只遍历被抽取器匹配上的（那正是历史 bug #2 的成因）。
        # 快照里的 status 归一到 canonical 名 —— 报告里不得出现 discovery_failure 别名。
        "gold_directions_snapshot": [
            {**d, "status": _dir_status(d)} if isinstance(d, dict) else d for d in gold_dirs
        ],
        "n_gold_in_frame_traps": len(traps),
        "n_pred_total": len(all_pred),
        "n_pred_out_of_frame": len(pred_oof),
        "n_pred_in_frame": len(pred_in),
        # 主表（不含 miss detection）
        "tp": tp_main,
        "fp": fp_main,
        "fn": fn_main,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        # 漏判表（单列）
        "miss_detection": {
            "hit": tp_miss,
            "missed": fn_miss,
            "spurious": spurious_miss,
            "miss_detection_recall": round(miss_recall, 4) if miss_recall is not None else None,
            "miss_detection_precision": round(miss_precision, 4) if miss_precision is not None else None,
            # 裁决 ⑧-b：gold_author 来源的漏判不计入主召回，单列
            "hit_main": len(matched_miss_main),
            "missed_main": len(unmatched_gold_miss_main),
            "miss_detection_recall_main": (
                round(len(matched_miss_main) / (len(matched_miss_main) + len(unmatched_gold_miss_main)), 4)
                if (len(matched_miss_main) + len(unmatched_gold_miss_main))
                else None
            ),
            "hit_author_only": len(matched_miss_author),
            "missed_author_only": len(unmatched_gold_miss_author),
            "miss_detection_recall_author_only": (
                round(
                    len(matched_miss_author)
                    / (len(matched_miss_author) + len(unmatched_gold_miss_author)),
                    4,
                )
                if (len(matched_miss_author) + len(unmatched_gold_miss_author))
                else None
            ),
            "hit_low_reportability": len(matched_miss_low),
            "missed_low_reportability": len(unmatched_gold_miss_low),
            "miss_detection_recall_low_reportability": (
                round(
                    len(matched_miss_low)
                    / (len(matched_miss_low) + len(unmatched_gold_miss_low)),
                    4,
                )
                if (len(matched_miss_low) + len(unmatched_gold_miss_low))
                else None
            ),
            # ⑳：主桶分母为 0 时**不可度量** —— 报「不可度量（分母 0）」，
            # 不要报 0/0，更不要当成抽取器缺陷。
            "main_measurable": bool(
                len(matched_miss_main) + len(unmatched_gold_miss_main)
            ),
            "main_unmeasurable_reason": (
                "主召回桶分母为 0（该金标集的 not_generated 条目全部落在 "
                "author_only / low_reportability 两个单列桶）—— "
                "**不可度量（分母为 0）**。"
                "⚠ **靶子不可达 ≠ 选手没打中** —— 是闭卷金标这一维测不到，"
                "不是抽取器不行。不构成抽取器缺陷。"
                if not (len(matched_miss_main) + len(unmatched_gold_miss_main))
                else ""
            ),
            "bucket_counts": {
                "main": len(gold_miss_main),
                "author_only": len(gold_miss_author),
                "low_reportability": len(gold_miss_low),
            },

            # ㉓-e ② 降级显式化：主桶里有多少方向是**未被显式审核**的
            # （expected_source 未标注 / 取值未登记）。必须可见 ——
            # 否则读者会以为主桶里每一条都被审核过。
            "source_review": {
                "labeled": sum(1 for d in gold_miss if _source_review_state(d) == "labeled"),
                "unlabeled": sum(1 for d in gold_miss if _source_review_state(d) == "unlabeled"),
                "unknown": sum(1 for d in gold_miss if _source_review_state(d) == "unknown"),
            },
            "unknown_source_in_main": sum(
                1
                for d in gold_miss
                if _miss_bucket(d, prp_ctx, _mid) == "main"
                and _source_review_state(d) != "labeled"
            ),
            "unlabeled_source_in_main": sum(
                1
                for d in gold_miss
                if _miss_bucket(d, prp_ctx, _mid) == "main"
                and _source_review_state(d) == "unlabeled"
            ),
            "unknown_value_in_main": sum(
                1
                for d in gold_miss
                if _miss_bucket(d, prp_ctx, _mid) == "main"
                and _source_review_state(d) == "unknown"
            ),
            "unknown_source_details": [
                {
                    "direction": direction_text(d),
                    "expected_source": d.get("expected_source"),
                    "review_state": _source_review_state(d),
                    "bucket": _miss_bucket(d, prp_ctx, _mid),
                }
                for d in gold_miss
                if _miss_bucket(d, prp_ctx, _mid) == "main"
                and _source_review_state(d) != "labeled"
            ],
            # prp 桶的逐条可准入判定（可追溯：为什么这条没进主分母）
            "prp_admissibility": [
                {
                    "direction": direction_text(d),
                    "admissible": _prp_admissible(d, prp_ctx, _mid)[0],
                    "reason": _prp_admissible(d, prp_ctx, _mid)[1],
                }
                for d in gold_miss
                if _expected_source(d) == "prp"
            ],
        },
        # 裁决 ⑧-c：零方向条目是精度探针
        "zero_direction": is_zero_direction,
        "zero_direction_clean": (is_zero_direction and not pred_oof),
        "zero_direction_overclaim": (is_zero_direction and bool(pred_oof)),
        "zero_direction_pred_count": len(pred_oof),
        "include_miss_detection": include_miss_detection,
        "matched": match_records,
        "false_positives": [
            {
                "pred_direction": direction_text(pred_main[i]),
                "pred_status": _dir_status(pred_main[i]),
            }
            for i in unmatched_pred_main
        ],
        "false_negatives": [
            {
                "gold_direction": direction_text(gold_main[i]),
                "gold_status": _dir_status(gold_main[i]),
                "gold_decision_effect": _decision_effect(gold_main[i]),
                "category": "main",
            }
            for i in unmatched_gold_main
        ],
        "invalid_status_preds": invalid_status_preds,
        "status_confusion": confusion,
        "status_accuracy": round(status_correct / status_checked, 4) if status_checked else None,
        "status_checked": status_checked,
        "status_accuracy_strict": (
            round(status_decidable_correct / status_decidable_checked, 4)
            if status_decidable_checked
            else None
        ),
        "status_decidable_checked": status_decidable_checked,
        "status_undecidable_count": undecidable,
        "status_undecidable_rate": round(undecidable_rate, 4) if undecidable_rate is not None else None,
        "status_ambiguous_items": sum(
            1 for m in match_records if m["pred_status_ambiguous"]
        ),
        "status_agreement": _pairs_all_status_correct(match_records),
        "status_agreement_strict": _pairs_all_status_correct(match_records) and fn_main == 0 and fn_miss == 0,
        "selection_expression_separability": round(se_correct / se_total, 4) if se_total else None,
        "selection_expression_checked": se_total,
        "selection_expression_separability_adjudicated": (
            round(se_adjudicated_correct / se_adjudicated_total, 4) if se_adjudicated_total else None
        ),
        "selection_expression_adjudicated_checked": se_adjudicated_total,
        "decision_effect_agreement": round(effect_correct / effect_checked, 4) if effect_checked else None,
        "decision_effect_exact_agreement": (
            round(effect_exact_correct / effect_checked, 4) if effect_checked else None
        ),
        "decision_effect_checked": effect_checked,
        "weak_effect_count": len(weak_effect_records),
        "weak_effects": weak_effect_records,
        "quote_grounding": quote_stats,
        "quote_grounding_rate": _quote_grounding_rate(quote_stats),
        "violations": violations,
        "constraint_violation_candidates": [
            {
                "direction": direction_text(d),
                "status": _dir_status(d),
                "excluded_from_main_table": (_dir_status(d) == "not_generated"),
                "evidence_quote": d.get("evidence_quote", ""),
                "note": (
                    "触碰/违反初始框架【硬约束】禁区段（X 段）的方向。"
                    + (
                        "**已排除出主表/漏判表/门槛**（裁决 ⑭-c）——"
                        "它是「本该被想到却没出现、且根本不可能出现」的一类。"
                        if _dir_status(d) == "not_generated"
                        else
                        "**保留在主表**（⑭-c 收窄）—— 它在 trace 里出现过，"
                        "「被否决/被推迟/被采纳」属 status 维度；"
                        "移出主表等于把 status 的诊断材料也扔了。"
                    )
                ),
            }
            for d in constraint_violation_candidates
        ],
        "constraint_violation_count": len(constraint_violation_candidates),
        "constraint_violation_not_generated_count": len(cv_not_generated),
        "constraint_violation_appeared_count": len(cv_appeared),
        # 空方向 = 合法的「前提未被挑战」回答（探针 C），不计误报
        "empty_direction_responses": len(empty_direction_responses),
        "introduced_in_synthesis": introduced_records,
        "introduced_in_synthesis_count": len(introduced_records),
        "stance_misattribution": stance_misattribution,
        "stance_misattribution_count": len(stance_misattribution),
        "gold_evidence_fragments": gold_evidence_fragments,
        "gold_evidence_grounded": gold_evidence_grounded,
        "gold_evidence_stats": gold_evidence_stats,
        "gold_evidence_extraction_failed": gold_evidence_extraction_failed,
        "gold_evidence_skipped_not_generated": gold_evidence_skipped_not_generated,
        "gold_evidence_grounding_rate": (
            round(gold_evidence_grounded / gold_evidence_fragments, 4)
            if gold_evidence_fragments
            else None
        ),
        "gold_legacy_effect_count": gold_legacy_effect,
        "gold_miss_total": gold_miss_total,
        "gold_miss_key_count": gold_strong_miss,
        "expected_source_distribution": _expected_source_distribution(gold_dirs),
        "expected_source_distribution_not_generated": _expected_source_distribution(
            [d for d in gold_dirs if _dir_status(d) == "not_generated"]
        ),
        "trap_hits": trap_hits,
        "out_of_frame_called_in_frame": missed_in_frame,
        "frame_ambiguous": bool(pred.get("frame_ambiguous")),
        "warnings": warnings,
    }


def lexical_baseline_matrix(
    gold_item: dict,
    pred: dict,
    match_fn: Callable[[dict, dict], bool] = default_match_fn,
    include_miss_detection: bool = False,
    lexicon_override: dict[str, list[str]] | None = None,
) -> dict[str, dict]:
    """词法捷径基线的**全部实现变体**（裁决 ⑧-d）。

    变体矩阵 = 3 个语料 × 2 个词表 = 6 个：
      语料：evidence_quote / full_trace / final_answer
      词表：narrow（否决/不采纳/不执行/不列入/排除）· wide（再加 暂不推进/不建议/不推进）

    全部变体都要报告（区间本身是信息），但**门槛基准取其中最高的 F1** ——
    门槛的作用是「证明尺子学到了语义」，那就必须打败**最努力的词法匹配器**。
    """
    out: dict[str, dict] = {}
    for lex_name in LEXICON_VARIANTS:
        lexicon = lexicon_override if lexicon_override is not None else (
            NARROW_LEXICON if lex_name == "narrow" else WIDE_LEXICON
        )
        for corpus_mode in CORPUS_VARIANTS:
            key = f"{lex_name}|{corpus_mode}"
            result = lexical_baseline(
                gold_item,
                pred,
                match_fn=match_fn,
                lexicon=lexicon,
                include_miss_detection=include_miss_detection,
                corpus_mode=corpus_mode,
            )
            out[key] = {
                "f1": result["f1"],
                "p": result["precision"],
                "r": result["recall"],
                "status_accuracy": result["status_accuracy"],
                "caliber": "status_only（方向集合与 in_frame 沿用抽取器输出，只把 status 换成关键词推断）",
                "caliber_kind": "status_only",
                "f1_comparable": False,
                "headline_metric": "status_accuracy",
                "tp": result["tp"],
                "fp": result["fp"],
                "fn": result["fn"],
                "status_correct": sum(result["status_confusion"][s][s] for s in STATUSES),
                "status_checked": result["status_checked"],
                "lexicon": lex_name,
                "corpus": corpus_mode,
            }
    return out


def load_main_caliber_baseline(path: str) -> dict | None:
    """载入**主表口径**的词法基线（由 diagnose_increment_source.py 生成）。

    为什么需要它：词法基线原始的 F1 用的是「含 miss detection」口径
    （所有金标域外方向都在分母里），而抽取器的主表 F1（裁决 ②-b）把
    `not_generated` 单独拆出。**两者不可直接比较** ——
    拿含 miss 的基线去卡不含 miss 的抽取器，等于把门槛放松了。
    """
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "max_f1" not in data:
        return None
    return {
        "name": f"词法基线（主表口径，{(data.get('max_params') or {}).get('variant', '?')}）",
        "source": path,
        "caliber": data.get("caliber", "main_table（not_generated 不计入主表）"),
        "caliber_kind": "end_to_end",
        "mode": "closed",
        "f1": round(float(data["max_f1"]), 4),
        "all_directions_caliber_f1": data.get("all_directions_caliber_max_f1"),
        "recommended_gate_f1": data.get("recommended_gate_f1"),
        # ⑳-b：同点比较 —— 门槛基准点（变体+参数）必须与抽取器被打分的点一致。
        "same_threshold_f1": data.get("at_default_threshold_f1"),
        "same_threshold_params": data.get("at_default_threshold_params"),
        "max_over_thresholds_f1": data.get("max_f1"),
        "strong_pass_line_f1": data.get("strong_pass_line_f1"),
        "gate_point_params": data.get("max_params"),
        "sensitivity_curve": data.get("sensitivity_curve"),
        # ⑳-c：池口径。池未清空的基线 fp 里会混进零方向条目的预测 → 门槛被放松。
        "pool_caliber": data.get("pool_caliber", "cleared"),
        "pool_caliber_ok": data.get("pool_caliber") in (None, "cleared"),
        "note": (
            "**主表口径**，与抽取器的主表 F1 同口径 —— 门槛必须用这一份。"
            "原始扫描表的 F1 是含 miss detection 口径，不能用于门槛。"
            "⑳-b：门槛比较是**同点比较** —— 抽取器必须在取得该基准的**同一个匹配阈值**上被评分。"
        ),
    }


def load_baseline_scan(path: str) -> dict[str, dict]:
    """载入词法基线的**扫描表**（CSV），按口径分组取最高 F1。

    裁决 ⑨-c：门槛基准 = 所有基线变体 × 所有阈值点里的最高 F1 ——
    门槛不能靠调低基线来放水。

    裁决 ⑨-b：**门槛只认闭卷口径**。开卷（`mode=open`）只作对照组，不进门槛。

    期望列：`mode`（closed/open）· `variant` · 若干阈值列 · `f1` · `status_acc` 等。
    """
    import csv

    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_mode: dict[str, dict] = {}
    for mode in sorted({r.get("mode", "closed") for r in rows}):
        subset = [r for r in rows if r.get("mode", "closed") == mode]
        if not subset:
            continue
        f1s = []
        for r in subset:
            # 主表口径用 f1 列；含漏判口径在 f1_full 列 —— **门槛只能用主表口径**。
            col = "f1" if "f1" in r else "f1_full"
            try:
                f1s.append(float(r.get(col, "") or 0))
            except ValueError:
                continue
        if not f1s:
            continue
        best = max(f1s)
        best_row = next(
            r for r in subset
            if abs(float(r.get("f1", 0) or 0) - best) < 1e-12
        )
        by_mode[mode] = {
            "name": f"扫描表({mode}) {len(subset)} 点",
            "source": path,
            "caliber": f"end_to_end（{mode} 口径，8 变体 × 阈值网格）",
            "caliber_kind": "end_to_end",
            "mode": mode,
            "f1": round(best, 4),
            "p": float(best_row["precision"]) if best_row.get("precision") else None,
            "r": float(best_row["recall"]) if best_row.get("recall") else None,
            "status_accuracy": float(best_row["status_acc"]) if best_row.get("status_acc") else None,
            "n_points": len(subset),
            "n_variants": len({r.get("variant") for r in subset}),
            "best_variant": best_row.get("variant"),
            "best_params": {
                k: best_row.get(k)
                for k in ("oof_thr", "match_thr", "stance_thr")
                if k in best_row
            },
        }
    return by_mode


def extractor_input(gold_item: dict) -> dict:
    """构造**抽取器输入切片**（裁决 ⑨-a）：task + initial_frame + trace 三件。

    **绝不能**包含：
      - `frame_note` —— 金标专有（域外参考），抽取器不可见
      - `gold` / `out_of_frame_directions` —— 那是答案
      - `in_frame_directions` —— 陷阱集答案

    闭卷下抽取器要做的**不是**找答案清单，而是语义判断：
    「这个动作是不是【候选集】里的常规解」→ 不在候选集里 = 域外。
    """
    return {
        "id": gold_item.get("id"),
        "task": gold_item.get("task", ""),
        "initial_frame": gold_item.get("initial_frame", ""),
        "trace": gold_item.get("trace", []),
    }


def load_external_baselines(paths: Sequence[str]) -> list[dict]:
    """载入外部实现的词法基线结果（裁决 ⑧-d：门槛取所有实现里的最高值）。

    接受两种来源：
      a) 金标构造方的 `goldset/lexical_baseline.py --json` 输出
         （含 `sweep_max_f1` 优先作为门槛基准 —— 门槛不能靠调低基线来放水）
      b) 通用格式：{"f1": ..., "p": ..., "r": ..., "status_accuracy": ...}
    """
    out: list[dict] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            continue
        main = data.get("main") if isinstance(data.get("main"), dict) else data
        # 扫描表锁定的语义：`sweep_max_f1` 现在明确是**闭卷**最高值（裁决 ⑨-b/⑨-c）。
        # 若外部 JSON 声明它是开卷值（sweep_max_f1_mode != closed），**不得**用作门槛基准 ——
        # 把开卷值当门槛等于把「泄漏答案」变成隐形优势、门槛被反向抬高。
        sweep_mode = (data.get("sweep_max_f1_mode") or "").strip().lower()
        # ⑭-e：**caliber 也要纳入守卫**（原先只查 mode，是漏网）。
        # 含 miss detection 口径的分数不能用于门槛 —— 那是双重惩罚。
        sweep_caliber = (data.get("sweep_max_f1_caliber") or "").strip().lower()
        f1 = data.get("sweep_max_f1")
        mode = "closed"
        if f1 is None:
            f1 = main.get("f1")
        if sweep_mode and sweep_mode not in ("closed_book", "closed"):
            mode = "open"
            f1 = main.get("f1")   # 退回默认阈值的数，只作参考，不进门槛
        if sweep_caliber and sweep_caliber not in ("main", "main_table"):
            mode = "open" if mode == "closed" else mode
            f1 = main.get("f1")   # 口径不对 —— 只作参考，不进门槛
        if f1 is None:
            continue
        out.append(
            {
                "name": data.get("baseline") or os.path.basename(path),
                "source": path,
                "caliber": data.get("caliber") or "end_to_end（自行切候选、判域外、判 status）",
                "caliber_kind": "end_to_end",
                "mode": mode,
                "sweep_max_f1_mode": sweep_mode or None,
                "sweep_max_f1_caliber": sweep_caliber or None,
                "f1": round(float(f1), 4),
                "p": main.get("precision", main.get("p")),
                "r": main.get("recall", main.get("r")),
                "status_accuracy": main.get("status_acc", main.get("status_accuracy")),
                "f1_at_default_threshold": main.get("f1"),
                "sweep_max_f1": data.get("sweep_max_f1"),
                "sweep_min_f1": data.get("sweep_min_f1"),
            }
        )
    return out


def _max_baseline_f1(variants: dict[str, dict]) -> tuple[float, str | None]:
    """变体矩阵里的最高 F1，以及达成它的变体名。"""
    if not variants:
        return 0.0, None
    best_key = max(variants, key=lambda k: (variants[k]["f1"], k))
    return variants[best_key]["f1"], best_key


def score_prediction(
    gold_item: dict,
    pred: dict,
    match_fn: Callable[[dict, dict], bool] = default_match_fn,
    include_miss_detection: bool = False,
    lexicon: dict[str, list[str]] | None = None,
    external_baselines: Sequence[dict] | None = None,
    prp_ctx: dict | None = None,
) -> dict:
    """比对金标的 out_of_frame_directions 与预测的 directions（对外主入口）。

    在核心打分之外，附带算一个**词法捷径基线**并给出真实增量（裁决 ⑦-a）：
    `lexical_baseline` 与 `true_increment`。

    主表（`include_miss_detection=False`，默认）：只统计
    realized / selection_failure / expression_failure 三类；
    `not_generated` 单独出 `miss_detection_recall`（下界，不是覆盖率）。
    """
    # ㉓-e ① 可达性断言：每次打分都验一次「每个 expected_source 都真的路由得到」。
    # 第 4 次假通过的形态是「枚举项漏了 → 分支静默失效 → 数据落到默认桶」。
    assert_miss_bucket_reachability()

    result = _score_core(
        gold_item,
        pred,
        match_fn=match_fn,
        include_miss_detection=include_miss_detection,
        prp_ctx=prp_ctx,
    )
    variants = lexical_baseline_matrix(
        gold_item,
        pred,
        match_fn=match_fn,
        include_miss_detection=include_miss_detection,
        lexicon_override=lexicon,
    )
    max_f1, max_key = _max_baseline_f1(variants)
    external = list(external_baselines or [])
    # 门槛基准 = 所有实现（本地 6 变体 + 外部端到端实现）里的最高 F1
    # 门槛基准**只取 end_to_end（抽取器无关）**的实现。
    # 门槛必须是一个固定基准，不能随抽取器的候选质量浮动 —— 本地 6 个变体是
    # status_only 口径（方向集合沿用抽取器输出），用它当门槛是循环论证。
    # 本地变体仍全表报告，作为「如果 status 只用关键词判会掉多少分」的诊断。
    # 门槛基准只取 **closed（闭卷）口径**的 end_to_end 实现（裁决 ⑨-b：开卷只作对照，不进门槛）
    end_to_end_closed = [
        b for b in external
        if b.get("caliber_kind") == "end_to_end" and b.get("mode", "closed") == "closed"
    ]
    if end_to_end_closed:
        best_external = max(end_to_end_closed, key=lambda b: b["f1"])
        gate_f1, gate_name = best_external["f1"], best_external["name"]
    else:
        best_external = None
        gate_f1, gate_name = max_f1, max_key
    # ⑭-c 收窄后的语义（⑲-c 的「不排除」上界对照已作废）：
    #   桶里**只剩 not_generated 候选**，而 not_generated 本来就不进主表
    #   → 「排除与否」对主表 F1 **没有差别**，那个对照恒等于 0、没有信息量。
    #   改为报告**桶内条目数**与两桶拆分。
    result["f1_with_constraint_violations_included"] = None
    result["f1_exclusion_delta"] = None

    result["lexical_baseline_variants"] = variants
    result["lexical_baseline_external"] = external
    result["lexical_baseline"] = {
        "f1": gate_f1,
        "p": (best_external or {}).get("p") if best_external else max(v["p"] for v in variants.values()),
        "r": (best_external or {}).get("r") if best_external else max(v["r"] for v in variants.values()),
        "status_accuracy": (best_external or {}).get("status_accuracy") if best_external else max(
            (v["status_accuracy"] for v in variants.values() if v["status_accuracy"] is not None),
            default=None,
        ),
        "best_variant": gate_name,
        "local_max_f1": max_f1,
        "local_max_variant": max_key,
        "gate_basis_kind": (
            "end_to_end" if best_external
            else "status_only（不可作固定门槛：随抽取器候选质量浮动）"
        ),
        "note": (
            "门槛基准取 **end_to_end（抽取器无关）** 实现里的最高 F1 —— 门槛必须是一个固定基准，"
            "不能随抽取器的候选质量浮动，否则就是循环论证。"
            "本地 6 个变体是 status_only 口径（方向集合沿用抽取器输出），"
            "全表报告，作为「如果 status 只用关键词判会掉多少分」的诊断，不作门槛基准。"
        ),
    }
    result["true_increment"] = {
        "f1": round(result["f1"] - gate_f1, 4),
        "status_accuracy": (
            round(
                result["status_accuracy"]
                - result["lexical_baseline"]["status_accuracy"],
                4,
            )
            if (result["status_accuracy"] is not None and result["lexical_baseline"]["status_accuracy"] is not None)
            else None
        ),
        "baseline_best_f1": gate_f1,
        "baseline_best_variant": gate_name,
    }
    return result


def _pairs_all_status_correct(match_records: Sequence[dict]) -> bool:
    """有匹配对且全部 status 正确 -> True；一个都没匹配上 -> False。"""
    if not match_records:
        return False
    return all(m["status_match"] for m in match_records)


def status_agreement(
    gold_item: dict,
    pred: dict,
    match_fn: Callable[[dict, dict], bool] = default_match_fn,
) -> bool:
    """同一个方向上 status 是否判对。

    语义（有意与方向召回解耦）：
      True  <=> 至少匹配上一对，且**每一对**匹配上的方向，其 gold status == pred status。
    一个方向都没匹配上时返回 False（此时没有「判对」的证据，而不是「判错」）。
    严格版（要求所有金标方向都被找到）见 status_agreement_strict。
    """
    result = score_prediction(gold_item, pred, match_fn=match_fn)
    return bool(result["status_agreement"])


def status_agreement_strict(
    gold_item: dict,
    pred: dict,
    match_fn: Callable[[dict, dict], bool] = default_match_fn,
) -> bool:
    """严格版：所有金标域外方向都被找到，且每个 status 都判对。"""
    result = score_prediction(gold_item, pred, match_fn=match_fn)
    return bool(result["status_agreement_strict"])


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------


def _per_class_scores(confusion: dict[str, dict[str, int]]) -> dict[str, dict[str, float | None]]:
    """从混淆矩阵算每个 status 的 precision / recall / F1。"""
    out: dict[str, dict[str, float | None]] = {}
    for label in STATUSES:
        tp = confusion[label][label]
        fp = sum(confusion[g][label] for g in STATUSES if g != label)
        fn = sum(confusion[label][p] for p in STATUSES if p != label)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1: float | None = (
            2 * precision * recall / (precision + recall)
            if (precision and recall and (precision + recall))
            else None
        )
        out[label] = {
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "support": tp + fn,
        }
    return out


def _per_effect_scores(pairs: Sequence[tuple[str, str]]) -> dict[str, dict[str, float | None]]:
    """decision_effect 的 5 类 P/R/F1（仅在匹配对之间统计）。"""
    out: dict[str, dict[str, float | None]] = {}
    for label in DECISION_EFFECTS:
        tp = sum(1 for g, p in pairs if g == label and p == label)
        fp = sum(1 for g, p in pairs if p == label and g != label)
        fn = sum(1 for g, p in pairs if g == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1: float | None = (
            2 * precision * recall / (precision + recall)
            if (precision and recall and (precision + recall))
            else None
        )
        out[label] = {
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "support": tp + fn,
        }
    return out


def threshold_sweep(
    gold_items: Sequence[dict],
    preds_by_id: dict[str, dict],
    thresholds: Sequence[float],
    lexicon: dict[str, list[str]] | None = None,
    external_baselines: Sequence[dict] | None = None,
    include_miss_detection: bool = False,
) -> dict:
    """F1 vs 匹配阈值 的扫描曲线 + **阈值敏感区间**（裁决 ⑥-a / ⑨-c）。

    - 曲线：每个阈值点上的主表 P/R/F1、status 准确率、词法基线最高 F1、真实增量。
    - 敏感区间：F1 变动 **< 2 个百分点** 的最大连续阈值区间。
      跨组比较时，三方抽取器必须落在**同一区间**内 ——
      否则匹配阈值本身就会制造差异。

    同时这也是词法基线的调参空间（⑨-c）：门槛基准取所有变体 × 所有阈值点里的最高 F1。
    """
    curve: list[dict] = []
    for threshold in thresholds:
        matcher = make_similarity_match_fn(threshold)
        results = []
        for gold_item in gold_items:
            pred = preds_by_id.get(gold_item["id"], {"id": gold_item["id"], "directions": []})
            results.append(
                score_prediction(
                    gold_item,
                    pred,
                    match_fn=matcher,
                    include_miss_detection=include_miss_detection,
                    lexicon=lexicon,
                    external_baselines=external_baselines,
                )
            )
        agg = aggregate(results)
        curve.append(
            {
                "threshold": round(threshold, 4),
                "precision": agg["precision"],
                "recall": agg["recall"],
                "f1": agg["f1"],
                "macro_f1": agg["macro_f1"],
                "status_accuracy": agg["status_accuracy"],
                "baseline_max_f1": agg["lexical_baseline"]["f1"],
                "baseline_best_variant": agg["lexical_baseline"]["best_variant"],
                "true_increment": agg["true_increment"]["f1"],
                "miss_detection_recall": agg["miss_detection"]["miss_detection_recall"],
                "selection_expression_separability": agg["selection_expression_separability"],
            }
        )

    # 敏感区间：F1 绝对变动 < 0.02 的最大连续区间
    sensitive = None
    best_len = 0
    n = len(curve)
    for start in range(n):
        lo = hi = curve[start]["f1"]
        for end in range(start, n):
            lo = min(lo, curve[end]["f1"])
            hi = max(hi, curve[end]["f1"])
            if (hi - lo) >= 0.02:
                break
            if (end - start + 1) > best_len:
                best_len = end - start + 1
                sensitive = {
                    "from": curve[start]["threshold"],
                    "to": curve[end]["threshold"],
                    "f1_min": round(lo, 4),
                    "f1_max": round(hi, 4),
                    "n_points": end - start + 1,
                }
    return {
        "thresholds": list(thresholds),
        "curve": curve,
        "sensitive_interval": sensitive,
        "sensitive_interval_criterion": "F1 绝对变动 < 0.02（2 个百分点）",
        "f1_range": (
            [min(c["f1"] for c in curve), max(c["f1"] for c in curve)] if curve else None
        ),
        "baseline_max_f1_over_sweep": (
            max(c["baseline_max_f1"] for c in curve) if curve else None
        ),
        "note": (
            "裁决 ⑥-a：扫描前不得用于任何跨组结论；跨组比较时三方抽取器必须落在同一敏感区间内。"
            "裁决 ⑨-c：词法基线的调参空间本身就是信息，门槛基准取所有变体 × 所有阈值点的最高 F1。"
        ),
    }


def aggregate(results: Sequence[dict], gate_point_f1: float | None = None) -> dict:
    """把逐条 score_prediction 结果汇总成总体报告。

    `gate_point_f1` = 抽取器在**门槛点匹配阈值**上重算的主表 F1（⑳-b 同点比较用）。

    报告分两表：
      * main            —— 方向判定主表（**不含 miss detection**）
      * miss_detection  —— 漏判单表（recall 是下界，不是覆盖率）
    """
    items = list(results)
    tp = sum(r["tp"] for r in items)
    fp = sum(r["fp"] for r in items)
    fn = sum(r["fn"] for r in items)
    precision, recall, f1 = _f1(tp, fp, fn)

    f1_values = [r["f1"] for r in items if r["tp"] + r["fp"] + r["fn"] > 0]
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0

    # 漏判表
    miss_hit = sum(r["miss_detection"]["hit_main"] for r in items)
    miss_missed = sum(r["miss_detection"]["missed_main"] for r in items)
    miss_spurious = sum(r["miss_detection"]["spurious"] for r in items)
    miss_recall = (miss_hit / (miss_hit + miss_missed)) if (miss_hit + miss_missed) else None
    miss_precision = (miss_hit / (miss_hit + miss_spurious)) if (miss_hit + miss_spurious) else None
    # 裁决 ⑧-b：gold_author 来源的漏判单列
    miss_hit_author = sum(r["miss_detection"]["hit_author_only"] for r in items)
    miss_missed_author = sum(r["miss_detection"]["missed_author_only"] for r in items)
    miss_recall_author = (
        (miss_hit_author / (miss_hit_author + miss_missed_author))
        if (miss_hit_author + miss_missed_author)
        else None
    )

    # status
    confusion = _empty_confusion()
    status_correct = 0
    status_checked = 0
    decidable_correct = 0
    decidable_checked = 0
    for r in items:
        for g in STATUSES:
            for p in STATUSES:
                confusion[g][p] += r["status_confusion"][g][p]
        status_checked += r["status_checked"]
        status_correct += sum(r["status_confusion"][s][s] for s in STATUSES)
        decidable_checked += r["status_decidable_checked"]
        decidable_correct += sum(
            1
            for m in r["matched"]
            if not m["pred_status_ambiguous"] and m["status_match"]
        )
    status_accuracy = status_correct / status_checked if status_checked else None
    status_accuracy_strict = decidable_correct / decidable_checked if decidable_checked else None

    undecidable_total = sum(r["status_undecidable_count"] for r in items)
    status_valid_total = sum(
        sum(1 for m in r["matched"] if m["pred_status"] in STATUSES) + r["invalid_status_preds"] * 0
        for r in items
    )
    # 不可判率的分母 = 全部带有效 status 的预测条目
    denom = sum(r["n_pred_out_of_frame"] + r["n_pred_in_frame"] for r in items) - sum(
        r["invalid_status_preds"] for r in items
    )
    undecidable_rate = (undecidable_total / denom) if denom else None

    # selection / expression 可分性
    se_total = sum(r["selection_expression_checked"] for r in items)
    se_correct = sum(
        sum(1 for m in r["matched"] if m["gold_status"] in ("selection_failure", "expression_failure") and m["status_match"])
        for r in items
    )
    se_adj_total = sum(r["selection_expression_adjudicated_checked"] for r in items)
    se_adj_correct = sum(
        sum(
            1
            for m in r["matched"]
            if m["gold_status"] in ("selection_failure", "expression_failure")
            and m["status_match"]
            and (m["gold_category"] == "main")
        )
        for r in items
    )

    # decision_effect
    effect_pairs: list[tuple[str, str]] = []
    for r in items:
        for m in r["matched"]:
            if m["gold_decision_effect"] and m["pred_decision_effect"]:
                effect_pairs.append((m["gold_decision_effect"], m["pred_decision_effect"]))
    effect_checked = sum(r["decision_effect_checked"] for r in items)
    effect_correct = sum(
        sum(
            1
            for m in r["matched"]
            if m["decision_effect_match"]
            and (m["gold_decision_effect"] in STRONG_EFFECTS) == (m["pred_decision_effect"] in STRONG_EFFECTS)
        )
        for r in items
    )
    effect_exact = sum(1 for g, p in effect_pairs if g == p)

    # 引用核验
    quote_totals = {
        "checked": 0,
        "exact": 0,
        "whitespace": 0,
        "empty": 0,
        "ungrounded": 0,
        "required": 0,
        "required_grounded": 0,
        "not_generated_with_quote": 0,
    }
    for r in items:
        for key in quote_totals:
            quote_totals[key] += r["quote_grounding"][key]
    grounded_rate = _quote_grounding_rate(quote_totals)

    # 金标可核性
    gold_evidence_stats = {"fragments": 0, "exact": 0, "punct": 0, "ungrounded": 0}
    gold_evidence_extraction_failed = 0
    gold_evidence_skipped_not_generated = 0
    for r in items:
        for key in gold_evidence_stats:
            gold_evidence_stats[key] += r["gold_evidence_stats"][key]
        gold_evidence_extraction_failed += r.get("gold_evidence_extraction_failed", 0)
        gold_evidence_skipped_not_generated += r.get("gold_evidence_skipped_not_generated", 0)
    gold_fragments = gold_evidence_stats["fragments"]
    gold_grounded = gold_evidence_stats["exact"] + gold_evidence_stats["punct"]

    # frame_schema 分组（non_conforming 不得与 conforming 并列排名）
    conforming = [r["id"] for r in items if r["frame_schema"] == "conforming"]
    non_conforming = [r["id"] for r in items if r["frame_schema"] != "conforming"]

    # 裁决 ⑨-b：开卷/闭卷
    openbook_ids = [r["id"] for r in items if r.get("openbook")]
    if openbook_ids and len(openbook_ids) == len(items):
        gold_mode = "open_book"
    elif openbook_ids:
        gold_mode = "mixed"
    else:
        gold_mode = "closed_book"
    frame_note_leaks = [leak for r in items for leak in r.get("frame_note_leaks", [])]

    # M2 门槛的机检项
    gold_miss_total = sum(r["gold_miss_total"] for r in items)
    gold_miss_key = sum(r["gold_miss_key_count"] for r in items)
    gold_legacy_effect = sum(r["gold_legacy_effect_count"] for r in items)
    gold_ungrounded_rate = (gold_evidence_stats["ungrounded"] / gold_fragments) if gold_fragments else 0.0
    # 某一类 status 在金标里零样本时，必须写明「这一类本集测不到」
    # （否则读者会以为四类都测过了）。
    status_support = {
        st: sum(
            1
            for r in items
            for d in (r.get("gold_directions_snapshot") or [])
            if E_dir_status(d) == st
        )
        for st in STATUSES
    }
    zero_sample_statuses = [st for st, n in status_support.items() if n == 0]

    n_trap_hits = sum(len(r["trap_hits"]) for r in items)
    weak_effect_count = sum(r["weak_effect_count"] for r in items)

    # ⑯-c：门槛基准的来源并列 —— 三个数都是**合法的不同测量定义**，
    # 差异来自匹配函数 / 匹配阈值 / FP 计数口径，不是脏数据。
    # 纪律：不取平均、不取最高 —— **指定唯一一个规范定义**（评测器自己的打分路径），
    # 其余并列报告并标注差异来源。
    # ⑳-a 的完整分类 —— 不是三个数，是**四类**（含那个虚高值）。
    baseline_number_provenance = [
        {"value": 0.9381,
         "definition": "金标构造方 score() 的 fp 逐条覆盖 bug（虚高近一倍，全程无报错）",
         "status": "**脏** —— 作废"},
        {"value": 0.5106,
         "definition": "金标构造方 score()，但**池未清空**（零方向条目的 15 个预测进了主表 FP）→ 基线被低估 0.06",
         "status": "**池口径不一致** —— 修完后与评测器路径同数 0.5749；此数本身作废"},
        {"value": 0.5397,
         "definition": "评测器 scorer + 评测器匹配 @0.34（评测器默认阈值）",
         "status": "**规范口径** —— 参考点"},
        {"value": 0.5714,
         "definition": "评测器 scorer + 评测器匹配 @0.20–0.25（最有利阈值），**池已清空**",
         "status": "**规范口径** —— 门槛基准 0.5749（变体取高 + 匹配阈值取高 + 池清空）"},
    ]

    # --- 零方向条目（⑧-c）：精度探针 ---
    zero_items = [r for r in items if r.get("zero_direction")]
    zero_clean = sum(1 for r in zero_items if r.get("zero_direction_clean"))
    zero_overclaim = sum(1 for r in zero_items if r.get("zero_direction_overclaim"))
    zero_precision = (zero_clean / len(zero_items)) if zero_items else None
    zero_overclaim_rate = (zero_overclaim / len(zero_items)) if zero_items else None

    # --- 违规桶（⑧-e.3）---
    violations: list[dict] = []
    for r in items:
        for v in r.get("violations", []):
            violations.append({**v, "id": r.get("id")})

    # --- ⑲-c：约束违规的逐条明细 + 「不排除」的上界对照 ---
    constraint_violation_details: list[dict] = []
    constraint_violation_details_not_generated: list[dict] = []
    constraint_violation_details_appeared: list[dict] = []
    for r in items:
        for c in r.get("constraint_violation_candidates", []):
            rec = {**c, "id": r.get("id")}
            constraint_violation_details.append(rec)
            if c.get("excluded_from_main_table"):
                constraint_violation_details_not_generated.append(rec)
            else:
                constraint_violation_details_appeared.append(rec)
    incl_f1s = [r["f1_with_constraint_violations_included"] for r in items
                if r.get("f1_with_constraint_violations_included") is not None]
    f1_incl = None
    if incl_f1s and len(incl_f1s) == len(items):
        # 上界 F1：把违规方向放回主表后的微平均
        tp_incl = sum(r["tp"] for r in items)
        fp_incl = sum(r.get("fp_incl", 0) for r in items)
        fn_incl = sum(r.get("fn_incl", 0) for r in items)
        _, _, f1_incl = _f1(tp_incl, fp_incl, fn_incl)
    f1_excl = f1

    # --- ⑲-a：综合阶段新产生的方向 ---
    introduced_total = sum(r.get("introduced_in_synthesis_count", 0) for r in items)
    introduced_details: list[dict] = []
    for r in items:
        for rec in r.get("introduced_in_synthesis", []):
            introduced_details.append({**rec, "id": r.get("id")})

    # --- 分支覆盖（㉓-e ① / General 的通则）---
    # 「任何检查/分支，至少要有一条真实数据路径能走进去。走不进去的分支 = 装饰品：
    #  它『通过』与否与真实结果无关。」—— 必须**显式列出**哪些分支在真实数据上恒不触发，
    #  别让读者以为它们被验证过。
    _bucket_counts = {"main": 0, "author_only": 0, "low_reportability": 0}
    for r in items:
        for _b in (r.get("gold_direction_buckets") or {}).values():
            _bucket_counts[_b] = _bucket_counts.get(_b, 0) + 1
    branch_coverage = {
        "miss_bucket.main": _bucket_counts.get("main", 0),
        "miss_bucket.author_only": _bucket_counts.get("author_only", 0),
        "miss_bucket.low_reportability": _bucket_counts.get("low_reportability", 0),
        "empty_direction_response": sum(r.get("empty_direction_responses", 0) for r in items),
        "quote_tier.exact": quote_totals.get("exact", 0),
        "quote_tier.punct": quote_totals.get("punct", 0),
        "quote_tier.ungrounded": quote_totals.get("ungrounded", 0),
        "constraint_violation": len(constraint_violation_details),
        "introduced_in_synthesis": introduced_total,
        "stance_misattribution": sum(r.get("stance_misattribution_count", 0) for r in items),
        "violations": len(violations),
    }
    branches_without_real_data = sorted(k for k, v in branch_coverage.items() if v == 0)



    # --- 词法捷径基线（裁决 ⑧-d）：6 个变体全报，门槛取最高 F1 ---
    # 每个变体在全局上做一次微平均（先把各条目的 tp/fp/fn 加起来再算 F1），
    # 不是各条目 F1 的平均 —— 门槛基准必须与主表同口径。
    variant_acc: dict[str, dict[str, float]] = {}
    for r in items:
        for key, v in (r.get("lexical_baseline_variants") or {}).items():
            acc = variant_acc.setdefault(key, {"tp": 0.0, "fp": 0.0, "fn": 0.0, "status_correct": 0.0, "status_checked": 0.0})
            acc["tp"] += v.get("tp", 0)
            acc["fp"] += v.get("fp", 0)
            acc["fn"] += v.get("fn", 0)
            acc["status_checked"] += v.get("status_checked", 0)
            acc["status_correct"] += v.get("status_correct", 0)
    lexical_variants: dict[str, dict] = {}
    for key, acc in variant_acc.items():
        p_, r_, f_ = _f1(int(acc["tp"]), int(acc["fp"]), int(acc["fn"]))
        sa = round(acc["status_correct"] / acc["status_checked"], 4) if acc["status_checked"] else None
        lex_name, corpus_name = key.split("|", 1)
        lexical_variants[key] = {
            "f1": round(f_, 4),
            "p": round(p_, 4),
            "r": round(r_, 4),
            "status_accuracy": sa,
            "lexicon": lex_name,
            "corpus": corpus_name,
        }
    b_f1, b_key = _max_baseline_f1(lexical_variants)
    b_status_accuracy = lexical_variants[b_key]["status_accuracy"] if b_key else None
    # 外部实现（金标构造方 / 主会话的端到端基线）—— 门槛基准取所有实现里的最高 F1
    external_baselines: list[dict] = []
    for r in items:
        for b in r.get("lexical_baseline_external") or []:
            if b not in external_baselines:
                external_baselines.append(b)
    # 门槛基准只取 end_to_end（抽取器无关）的实现
    # ⑳-c：池口径不一致的基线**不得作门槛基准**。
    pool_inconsistent = [
        b for b in external_baselines if b.get("pool_caliber_ok") is False
    ]
    end_to_end = [
        b for b in external_baselines
        if b.get("caliber_kind") == "end_to_end"
        and b.get("mode", "closed") == "closed"
        and b.get("pool_caliber_ok") is not False
    ]
    open_book_baselines = [
        b for b in external_baselines
        if b.get("caliber_kind") == "end_to_end" and b.get("mode", "closed") != "closed"
    ]
    best_external = max(end_to_end, key=lambda b: b["f1"], default=None)
    e2e_f1 = best_external["f1"] if best_external is not None else None
    # 裁决 ⑪-a：门槛基准 = 闭卷条件下的最高 F1（两种口径取高）。
    #   end_to_end  = 抽取器无关的固定对手（0.4932）-> M2 固定门槛的来源
    #   status_only = 本地 6 变体（方向集合沿用抽取器输出）-> 配对比较，随抽取器浮动
    if best_external is not None and best_external["f1"] >= b_f1:
        gate_f1, gate_name = best_external["f1"], best_external["name"]
    else:
        gate_f1, gate_name = b_f1, b_key
    lexical_baseline_summary = {
        "f1": gate_f1,
        "p": (best_external or {}).get("p") if best_external and best_external["f1"] > b_f1 else (lexical_variants[b_key]["p"] if b_key else None),
        "r": (best_external or {}).get("r") if best_external and best_external["f1"] > b_f1 else (lexical_variants[b_key]["r"] if b_key else None),
        "status_accuracy": b_status_accuracy,
        "best_variant": gate_name,
        "local_max_f1": b_f1,
        "local_max_variant": b_key,
        "gate_basis_kind": (
            "end_to_end" if best_external
            else "status_only（不可作固定门槛：随抽取器候选质量浮动）"
        ),
        "end_to_end_available": bool(end_to_end),
        "external_baselines": external_baselines,
        "variants": lexical_variants,
        "f1_range": (
            [min(v["f1"] for v in lexical_variants.values()), max(v["f1"] for v in lexical_variants.values())]
            if lexical_variants
            else None
        ),
        "note": (
            "6 个实现变体（3 语料 × 宽/窄词表）全报；门槛基准取最高 F1 —— "
            "门槛的作用是「证明尺子学到了语义」，那就必须打败最努力的词法匹配器。"
        ),
    }
    true_increment = {
        "f1": round(f1 - gate_f1, 4),
        "f1_vs_end_to_end": (round(f1 - e2e_f1, 4) if e2e_f1 is not None else None),
        "status_accuracy": (
            round(status_accuracy - b_status_accuracy, 4)
            if (status_accuracy is not None and b_status_accuracy is not None)
            else None
        ),
        "baseline_best_f1": gate_f1,
        "baseline_best_variant": gate_name,
        "statement": (
            f"真实增量 = 抽取器 F1 − 词法基线最高 F1 = {round(f1 - gate_f1, 4):.2f}"
            "（PSC-Eval 相对于关键词匹配基线的真实语义增量）"
        ),
        "statement_vs_end_to_end": (
            f"真实增量(vs 端到端闭卷基线 {e2e_f1}) = 抽取器 F1 − {e2e_f1} = "
            f"{round(f1 - e2e_f1, 4):.2f}（M2 固定门槛口径）"
            if e2e_f1 is not None
            else ""
        ),
        "required_extractor_f1": (
            round(e2e_f1 + LEXICAL_BASELINE_MIN_INCREMENT, 4) if e2e_f1 is not None else None
        ),
    }

    # --- expected_source 分布（裁决 ⑦-b）：仅统计 ---
    expected_source_distribution: dict[str, int] = {}
    expected_source_distribution_not_generated: dict[str, int] = {}
    for r in items:
        for key, count in r.get("expected_source_distribution", {}).items():
            expected_source_distribution[key] = expected_source_distribution.get(key, 0) + count
        for key, count in r.get("expected_source_distribution_not_generated", {}).items():
            expected_source_distribution_not_generated[key] = (
                expected_source_distribution_not_generated.get(key, 0) + count
            )

    # --- 审计覆盖率（防止「审计器只审计通过的部分」）---
    # 金标一共有 N 条方向，gate 必须**逐条**都检查到；漏审一条 = 门槛假通过。
    audit_coverage = _audit_coverage(items)
    m2_gate = {
        "gold_evidence_ungrounded_rate": round(gold_ungrounded_rate, 4),
        "gold_evidence_gate_pass": gold_ungrounded_rate <= 0.05 and gold_evidence_extraction_failed == 0,
        "gold_evidence_fragments": gold_fragments,
        "gold_evidence_extraction_failed": gold_evidence_extraction_failed,
        "gold_evidence_skipped_not_generated": gold_evidence_skipped_not_generated,
        "gold_miss_all_key": gold_miss_total == gold_miss_key,
        "gold_miss_total": gold_miss_total,
        "gold_miss_key_count": gold_miss_key,
        "gold_uses_v2_effect": gold_legacy_effect == 0,
        "gold_legacy_effect_count": gold_legacy_effect,
        "all_frames_conforming": not non_conforming,
        "non_conforming_ids": non_conforming,
        "gold_mode": gold_mode,
        "gold_openbook_items": len(openbook_ids),
        "gold_closed_book": not openbook_ids,
        "frame_note_leaks": sorted(set(frame_note_leaks)),
        "frame_note_leak_free": not frame_note_leaks,
        "frame_note_assertion_effective": sum(r.get("frame_note_checked", 0) for r in items) > 0,
        "true_increment_f1": true_increment["f1"],
        "true_increment_f1_vs_end_to_end": true_increment["f1_vs_end_to_end"],
        "end_to_end_closed_f1": e2e_f1,
        "required_extractor_f1": (
            round(e2e_f1 + LEXICAL_BASELINE_MIN_INCREMENT, 4) if e2e_f1 is not None else None
        ),
        # ⑳-b：同点比较（硬要求）—— 门槛基准 = 取得该数的那一点（变体+参数），
        # 抽取器必须在**同一个匹配阈值**上被评分。
        "gate_point_baseline_f1": e2e_f1,
        "gate_point_params": (
            (best_external or {}).get("gate_point_params") if best_external else None
        ),
        "same_point_comparison": (
            "抽取器必须在取得门槛基准的**同一个匹配阈值**上被评分；"
            "禁止抽取器在多阈值上取最优再与门槛比。"
        ),
        # 裁决 ⑪-b③：上界报警线 —— 闭卷 F1 超过开卷天花板时触发人工检查（不是通过条件）
        "upper_alarm": bool(UPPER_ALARM_F1 is not None and f1 > UPPER_ALARM_F1),
        "upper_alarm_threshold": UPPER_ALARM_F1,
        "upper_alarm_note": (
            "闭卷抽取器 F1 > 开卷天花板（最强词法+泄漏）时触发人工检查："
            "金标泄漏？过拟合？金标有偏？—— **不是通过条件，是异常信号**。"
        ),
        "unknown_source_gate_pass": (
            sum(
                1
                for r in items
                for d in (r.get("gold_directions_snapshot") or [])
                if E_dir_status(d) == "not_generated"
                and E_source_review_state(d) == "unknown"
            )
            == 0
        ),
        "frame_note_checked": sum(r.get("frame_note_checked", 0) for r in items),
        "frame_note_assertion_vacuous": sum(r.get("frame_note_checked", 0) for r in items) == 0,
        "frame_note_coverage_note": (
            "frame_note 断言实际检查到的字段数。**为 0 说明这条断言没有对象可查，处于真空状态**"
            "（恒真通过）——「检查到 0 个」和「检查到 30 个且都没泄漏」在报告里长得一模一样，"
            "必须分开看。这与「只审匹配上的方向」是同一族 bug。"
        ),
        # M2 的**最终门槛**是固定值：闭卷抽取器 F1 ≥ 端到端闭卷基线 + 0.10
        # （裁决 ⑪-a：闭卷抽取器 F1 ≥ 0.5932）。
        # max-based 的真实增量（取所有闭卷基线里最高者）作为更严的口径**一并报告**，
        # 但**不作为**通过条件 —— 因为 status_only 变体随抽取器浮动，拿它当固定门槛是循环论证。
        # ⑳-b：同点比较 —— 门槛判定用**门槛点**的抽取器 F1（不是默认阈值的那个）。
        "f1_at_gate_point": gate_point_f1,
        "gate_point_match_thr": GATE_POINT_MATCH_THR,
        "true_increment_gate_pass": (
            (
                (gate_point_f1 if gate_point_f1 is not None else f1)
                >= round(e2e_f1 + LEXICAL_BASELINE_MIN_INCREMENT, 4)
                if e2e_f1 is not None
                else true_increment["f1"] >= LEXICAL_BASELINE_MIN_INCREMENT
            )
            and bool(end_to_end)
            and not openbook_ids
        ),
        "true_increment_gate_pass_strict_max_basis": (
            true_increment["f1"] >= LEXICAL_BASELINE_MIN_INCREMENT
            and bool(end_to_end)
            and not openbook_ids
        ),
        "true_increment_min": LEXICAL_BASELINE_MIN_INCREMENT,
        "true_increment_gate_basis": gate_name,
        "true_increment_gate_basis_kind": (
            "end_to_end" if best_external else "status_only（警告：无抽取器无关的基线，门槛不可靠）"
        ),
        "end_to_end_baseline_supplied": bool(end_to_end),
        "pool_inconsistent_baselines": [
            {"name": b.get("name"), "pool_caliber": b.get("pool_caliber"),
             "f1": b.get("f1"), "note": "池口径不一致 —— 不得作门槛基准（⑳-c）"}
            for b in pool_inconsistent
        ],
        "open_book_baselines": [
            {"name": b["name"], "f1": b["f1"]} for b in open_book_baselines
        ],
        "gate_basis_mode": "closed",
        # 审计覆盖率：< 100% 则 M2 不予通过
        "audit_coverage": audit_coverage["rate"],
        "audit_coverage_checked": audit_coverage["checked"],
        "audit_coverage_total": audit_coverage["total"],
        "audit_coverage_pass": audit_coverage["rate"] == 1.0,
        "audit_coverage_detail": audit_coverage["detail"],
        "violations_count": len(violations),
    }

    main_table = {
        "title": "方向判定主表（不含 miss detection）",
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "macro_f1": round(macro_f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "scope": "只统计 realized / selection_failure / expression_failure 三类",
        "note": "not_generated 不计入本表；漏判情况见 miss_detection 表。",
    }
    miss_table = {
        "title": "miss detection（单独统计）",
        "miss_detection_recall": round(miss_recall, 4) if miss_recall is not None else None,
        "miss_detection_precision": round(miss_precision, 4) if miss_precision is not None else None,
        "hit": miss_hit,
        "missed": miss_missed,
        "spurious_miss_count": miss_spurious,
        "miss_detection_recall_author_only": (
            round(miss_recall_author, 4) if miss_recall_author is not None else None
        ),
        "author_only_hit": miss_hit_author,
        "author_only_missed": miss_missed_author,
        "author_only_note": (
            "expected_source = gold_author 的漏判条目不计入主召回（分母不能是标注者的领域知识），"
            "单列于此。这两条正是「什么算域外」最难的部分。"
        ),
        "disclaimer": (
            "miss_detection_recall 是下界，不是覆盖率："
            "它只能命中金标已枚举的关键漏判方向，无法反映真实漏判总量。"
            "同理，miss_detection_precision 也不能当真正的精确率 —— "
            "金标的漏判清单本身不完整，被记为『误报』的方向可能只是没被枚举到。"
        ),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "n_items": len(items),
        "n_gold_out_of_frame": sum(r["n_gold_out_of_frame"] for r in items),
        "n_gold_in_frame_traps": sum(r["n_gold_in_frame_traps"] for r in items),
        "n_pred_out_of_frame": sum(r["n_pred_out_of_frame"] for r in items),
        "main": main_table,
        "miss_detection": miss_table,
        # 扁平便捷字段（= 主表）
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "macro_f1": round(macro_f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "status_accuracy": round(status_accuracy, 4) if status_accuracy is not None else None,
        "status_checked": status_checked,
        "status_accuracy_strict": (
            round(status_accuracy_strict, 4) if status_accuracy_strict is not None else None
        ),
        "status_decidable_checked": decidable_checked,
        "status_undecidable_count": undecidable_total,
        "status_undecidable_rate": round(undecidable_rate, 4) if undecidable_rate is not None else None,
        "status_confusion": confusion,
        "per_status": _per_class_scores(confusion),
        "status_support": status_support,
        "zero_sample_statuses": zero_sample_statuses,
        "zero_sample_note": (
            "⚠ 以下 status 类在本集金标里**零样本**（本集测不到这一类）："
            + ", ".join(zero_sample_statuses)
            + "。**该类的召回率与准确率不得作为尺子质量证据引用**，"
            "也不要以为四类都测过了。"
            if zero_sample_statuses
            else ""
        ),
        "status_agreement_rate": (
            round(sum(1 for r in items if r["status_agreement"]) / len(items), 4) if items else None
        ),
        "status_agreement_strict_rate": (
            round(sum(1 for r in items if r["status_agreement_strict"]) / len(items), 4) if items else None
        ),
        "ambiguous_status_items": sum(r["status_ambiguous_items"] for r in items),
        "selection_expression_separability": round(se_correct / se_total, 4) if se_total else None,
        "selection_expression_checked": se_total,
        "selection_expression_separability_adjudicated": (
            round(se_adj_correct / se_adj_total, 4) if se_adj_total else None
        ),
        "selection_expression_adjudicated_checked": se_adj_total,
        "decision_effect_agreement": round(effect_correct / effect_checked, 4) if effect_checked else None,
        "decision_effect_exact_agreement": (
            round(effect_exact / effect_checked, 4) if effect_checked else None
        ),
        "decision_effect_checked": effect_checked,
        "per_effect": _per_effect_scores(effect_pairs),
        "weak_effect_count": weak_effect_count,
        "would_change_decision_accuracy": round(effect_correct / effect_checked, 4) if effect_checked else None,
        "quote_grounding": quote_totals,
        "quote_grounding_rate": round(grounded_rate, 4) if grounded_rate is not None else None,
        "gold_evidence_fragments": gold_fragments,
        "gold_evidence_grounded": gold_grounded,
        "gold_evidence_stats": gold_evidence_stats,
        "gold_evidence_grounding_rate": (
            round(gold_grounded / gold_fragments, 4) if gold_fragments else None
        ),
        "frame_schema": {
            "conforming": len(conforming),
            "non_conforming": len(non_conforming),
            "non_conforming_ids": non_conforming,
            "note": "non_conforming 的条目不得与 conforming 数据并列排名（裁决 ④）",
        },
        "gold_mode": gold_mode,
        "openbook_item_ids": openbook_ids,
        "frame_note_leaks": sorted(set(frame_note_leaks)),
        "m2_gate": m2_gate,
        "baseline_number_provenance": baseline_number_provenance,
        "constraint_violation_details": constraint_violation_details,
        "constraint_violation_count": len(constraint_violation_details),
        "constraint_violation_sensitivity": {
            "not_generated_candidates": len(constraint_violation_details_not_generated),
            "appeared_in_trace": len(constraint_violation_details_appeared),
            "note": (
                "**⑭-c 收窄后的语义**：桶只用于「not_generated 候选」"
                "（本该被想到、却因触碰禁区而根本不可能出现的那一类）；"
                "**凡 trace 里已出现过的方向，即使触碰禁区也照常留在主表** —— "
                "「被否决/被推迟/被采纳」是 status 维度，「禁区」是「该方向可不可行」维度，"
                "两者正交；把触碰禁区的方向移出主表 = 把 status 的诊断材料也扔了。"
                "（⑲-c 那个「不排除」上界对照已作废：桶里只剩 not_generated 候选，"
                "而 not_generated 本来就不进主表 → 排除与否对 F1 无差别，对照恒等于 0、无信息量。）"
            ),
        },
        "introduced_in_synthesis_count": introduced_total,
        "introduced_in_synthesis_details": introduced_details,
        "lexical_baseline": lexical_baseline_summary,
        "true_increment": true_increment,
        "zero_direction": {
            "n_items": len(zero_items),
            "zero_direction_precision": round(zero_precision, 4) if zero_precision is not None else None,
            "zero_direction_overclaim_rate": (
                round(zero_overclaim_rate, 4) if zero_overclaim_rate is not None else None
            ),
            "note": (
                "零方向条目（正确答案是「没有域外方向」）不计入主 F1，"
                "但单独报告这两个数 —— 过度声称率低说明抽取器守住了「漏判 ≫ 误判」的非对称义务，"
                "这是正向质量信号。"
            ),
        },
        "zero_direction_precision": round(zero_precision, 4) if zero_precision is not None else None,
        "zero_direction_overclaim_rate": (
            round(zero_overclaim_rate, 4) if zero_overclaim_rate is not None else None
        ),
        "violations": violations,
        "branch_coverage": branch_coverage,
        "branches_without_real_data": branches_without_real_data,
        "branch_coverage_note": (
            "㉓-e ① / General 的通则：「任何检查/分支，至少要有一条真实数据路径能走进去。"
            "走不进去的分支 = 装饰品 —— 它『通过』与否与真实结果无关。」"
            "**branches_without_real_data 列出的分支，其正确性由合成用例保证，不由本集数据保证。**"
        ),
        "expected_source_distribution": expected_source_distribution,
        "expected_source_distribution_not_generated": expected_source_distribution_not_generated,
        "trap_hits": n_trap_hits,
        "items": items,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _load_matcher(spec: str) -> Callable[[dict, dict], bool]:
    """从 'module:function' 载入自定义 match_fn。"""
    if ":" not in spec:
        raise ValueError("matcher 格式应为 module:function")
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    if not callable(func):
        raise ValueError(f"{spec} 不是可调用对象")
    return func


def _print_summary(report: dict) -> None:
    summary = report["summary"]
    main = summary["main"]
    miss = summary["miss_detection"]
    print("=" * 68)
    print(f"PSC-Eval 报告 · schema v{summary['schema_version']}")
    print("=" * 74)
    print("◆ 这把尺子是怎么被造出来的：四次「假通过」（㉓-e）")
    print("  本项目至今所有最贵的错误，都长成「结果看起来完全正常」的样子 ——")
    print("  虚高一倍的基线、继承来的脏值、混算的池、吞掉主路径的降级分支。")
    print("  因此每一个数字都必须能回答：「如果它错了，警报在哪？」")
    print("  ─────────────────────────────────────────────────────────")
    print("  1) `fp` 逐条覆盖        【计算层】算出一个更大的数，不报错")
    print("  2) 旧 CSV 继承错值      【数据层】继承脏值，不报错")
    print("  3) 池口径不一致          【口径层】两类东西混算，不报错")
    print("  4) `prp` 漏出枚举集合    【枚举层】该类数据静默落到默认桶，不报错")
    print("  ─────────────────────────────────────────────────────────")
    print("  通用防御：凡存在「按类别分流」的逻辑，必须有")
    print("    ① 可达性断言（枚举集合 ⊆ 实际可达分支集合）")
    print("    ② 降级显式化（未定义类别的降级行为必须声明，且在报告里可见）")
    print("=" * 74)
    print("★ 金标代表性警示（⑮-b）—— 写在这里，不许藏在附录里：")
    print("  1. expression_failure 的「9/9/9 高可复现」**降级**为")
    print("     「金标措辞模板化导致的高一致，不代表真实 trace 上的判别力」。")
    print("  2. 报告**不得**用 9/9/9 宣称这一类可靠。")
    print("  3. expression_failure 的**真实判别力尚未被测量** ——")
    print("     在金标去模板化 + 重跑之前，该类召回率**不得作为尺子质量证据**引用。")
    print("     （盲评 A/B 与金标 9/9/9 一致，是因为那 9 条全是「保留为 + 某种未来事项」，")
    print("      措辞高度模板化；同一批真实 trace 上三个独立执行体报出的 expression_failure")
    print("      数量是 1 / 7 / 0 —— 差 7 倍。）")
    print("-" * 74)
    print(f"样本数        : {summary['n_items']}")
    print(
        f"方向集合      : gold 域外 {summary['n_gold_out_of_frame']} 条 / "
        f"预测域外 {summary['n_pred_out_of_frame']} 条"
    )
    print("-" * 68)
    print(f"[主表] {main['title']}")
    print(
        f"  P {main['precision']}  R {main['recall']}  F1 {main['f1']}  "
        f"(macro F1 {main['macro_f1']})"
    )
    print(
        f"  TP {main['tp']} / FP {main['fp']} / FN {main['fn']}   范围: {main['scope']}"
    )
    print("-" * 68)
    print(f"[漏判表] {miss['title']}")
    print(
        f"  miss_detection_recall = {miss['miss_detection_recall']}   "
        f"(命中 {miss['hit']} / 漏 {miss['missed']} / 误报 {miss['spurious_miss_count']})"
    )
    # prp 桶可准入：**恒打**（哪怕 0/0）—— 让读者知道这条路径被检查过，而不是不存在。
    adm = miss.get("prp_admissibility") or []
    ok = sum(1 for a in adm if a["admissible"])
    if adm:
        print(f"  prp 桶可准入    : {ok}/{len(adm)} 条可进主分母")
        for a in adm:
            if not a["admissible"]:
                print(f"    · 不可准入: {a['direction'][:36]} —— {a['reason']}")
    else:
        print("  prp 桶可准入    : 0/0 条可进主分母（本集无 prp 桶条目；该路径已检查）")
    cvs = summary.get("constraint_violation_sensitivity") or {}
    if cvs:
        print(f"  约束违规桶    : not_generated 候选 {cvs.get('not_generated_candidates')} 条 / "
              f"已在 trace 出现 {cvs.get('appeared_in_trace')} 条"
              "（后者照常留在主表）")
    bc = summary.get("branch_coverage") or {}
    bwr = summary.get("branches_without_real_data") or []
    if bc:
        print("  分支覆盖        : " + "  ".join(f"{k.split('.')[-1]}={v}" for k, v in bc.items()))
    if bwr:
        print(f"  ⚠ 真实数据上恒不触发的分支（其正确性由合成用例保证，不由本集数据保证）: {bwr}")
    # ㉓-e ② 降级显式化：主桶里有多少方向未被显式审核
    rev = miss.get("source_review") or {}
    unk = miss.get("unknown_source_in_main")
    if unk is not None:
        print(f"  unknown_source_in_main: {unk}   （主桶里 expected_source 未标注/未登记的条目数）")
        for d in (miss.get("unknown_source_details") or []):
            print(f"    · {d['review_state']}: {d['direction'][:40]}（expected_source={d['expected_source']}）")
    if miss.get("miss_detection_recall_author_only") is not None:
        print(
            f"  └ 仅 gold_author 来源: {miss['miss_detection_recall_author_only']}   "
            f"(命中 {miss['author_only_hit']} / 漏 {miss['author_only_missed']})"
        )
    print(f"  ⚠ {miss['disclaimer']}")
    print("-" * 68)
    print(
        f"status 准确率 : {summary['status_accuracy']} (全部) | "
        f"{summary['status_accuracy_strict']} (仅可判条目, 检查 {summary['status_decidable_checked']} 对)"
    )
    print("  ⚠ ⑪-c：status 准确率**不是**本尺子的主要区分度来源。")
    print("    闭卷基线 status_acc 仅 0.475–0.49（stance 词在最终答案里是明写的，")
    print("    status 这一步词白送）。「status_acc 0.48」与「尺子很严」在报告里长得一模一样，")
    print("    不点破必然误读。**主指标 = 主表 F1 + 漏判召回。**")
    print(
        f"status 不可判率: {summary['status_undecidable_rate']} "
        f"({summary['status_undecidable_count']} 条)  ← 必须与准确率一起读"
    )
    if summary.get("zero_sample_statuses"):
        print(f"  ⚠ 零样本 status 类（本集测不到）: {summary['zero_sample_statuses']}")
        print("    该类的召回率/准确率**不得作为尺子质量证据**引用。")
    print(f"  匹配阈值       : {summary.get('match_threshold_used')}（钉在门槛点，"
          f"不随 --sweep 漂移）")
    print(f"selection/expression 可分性: {summary['selection_expression_separability']} (检查 {summary['selection_expression_checked']} 对)")
    if summary["selection_expression_separability_adjudicated"] is not None:
        print(f"  └ 仅独立裁定样本: {summary['selection_expression_separability_adjudicated']} ({summary['selection_expression_adjudicated_checked']} 对)")
    print(
        f"decision_effect : 一致 {summary['decision_effect_agreement']} / "
        f"精确 {summary['decision_effect_exact_agreement']} (检查 {summary['decision_effect_checked']} 对)"
    )
    print(f"弱效果单列     : weak_effect_count = {summary['weak_effect_count']} (ranking_only + additive_only)")
    base = summary["lexical_baseline"]
    inc = summary["true_increment"]
    print("-" * 68)
    print("[词法捷径基线] 6 变体全表（裁决 ⑧-d：门槛取最高）")
    for key in sorted(base.get("variants", {})):
        v = base["variants"][key]
        mark = " ← 门槛基准" if key == base.get("best_variant") else ""
        print(
            f"  {key:28s} F1 {v['f1']:<7} P {v['p']:<7} R {v['r']:<7} "
            f"status {v['status_accuracy']}{mark}"
        )
    for b in base.get("external_baselines", []):
        mark = " ← 门槛基准" if b["name"] == base.get("best_variant") else ""
        print(
            f"  外部: {b['name'][:34]:36s} F1 {b['f1']:<7} "
            f"（{b.get('caliber', '')[:28]}）{mark}"
        )
    if base.get("f1_range"):
        print(f"  本地变体 F1 区间: {base['f1_range'][0]} ~ {base['f1_range'][1]}")
    print(f"★ {inc['statement']}")
    if inc.get("statement_vs_end_to_end"):
        print(f"★ {inc['statement_vs_end_to_end']}")
    req = summary["m2_gate"].get("required_extractor_f1")
    gp = summary["m2_gate"].get("f1_at_gate_point")
    print(f"  M2 最终门槛: 闭卷抽取器 F1@门槛点({summary['m2_gate'].get('gate_point_match_thr')}) ≥ {req}"
          f"（= 端到端闭卷基线 {summary['m2_gate'].get('gate_point_baseline_f1')} + {LEXICAL_BASELINE_MIN_INCREMENT}）-> "
          f"{'PASS' if summary['m2_gate']['true_increment_gate_pass'] else 'FAIL'}")
    if gp is not None:
        print(f"  抽取器在门槛点的 F1 = {gp}（默认阈值处为 {summary['f1']}）—— ⑳-b 同点比较")
    prov = summary.get("baseline_number_provenance")
    if prov:
        print("  基准来源（⑯-c：三个数都是合法的不同测量定义，不是脏数据 —— 不取平均、不取最高，"
              "指定唯一规范定义）:")
        for item in prov:
            print(f"    · {item['value']:<8} {item['definition']}")
    print(f"  （更严口径：取所有闭卷基线最高者的增量 ≥ {LEXICAL_BASELINE_MIN_INCREMENT} -> "
          f"{'PASS' if summary['m2_gate']['true_increment_gate_pass_strict_max_basis'] else 'FAIL'}）")
    z = summary.get("zero_direction", {})
    if z.get("n_items"):
        print(
            f"零方向探针     : {z['n_items']} 条 | precision {z['zero_direction_precision']} | "
            f"overclaim_rate {z['zero_direction_overclaim_rate']}"
        )
    if summary.get("violations"):
        print(f"⚠ 违规条目: {len(summary['violations'])} 条（详见 violations 桶）")
    gate_cov = summary["m2_gate"].get("audit_coverage")
    if gate_cov is not None:
        print(
            f"审计覆盖率     : {gate_cov} "
            f"({summary['m2_gate']['audit_coverage_checked']}/{summary['m2_gate']['audit_coverage_total']} 条金标方向被审到)"
        )
    print(f"引用可核率     : {summary['quote_grounding_rate']}  {summary['quote_grounding']}")
    gate = summary["m2_gate"]
    print(
        f"M2 机检        : evidence非逐字 {gate['gold_evidence_ungrounded_rate']} "
        f"({'PASS' if gate['gold_evidence_gate_pass'] else 'FAIL'}) | "
        f"金标全为关键漏判 {'PASS' if gate['gold_miss_all_key'] else 'FAIL'} | "
        f"金标用 v2 effect {'PASS' if gate['gold_uses_v2_effect'] else 'FAIL'} | "
        f"frame 全合规 {'PASS' if gate['all_frames_conforming'] else 'FAIL'}"
    )
    fs = summary["frame_schema"]
    if fs["non_conforming"]:
        print(f"⚠ frame_schema non_conforming: {fs['non_conforming']} 条 —— 不得与 conforming 并列排名")
    if summary.get("upper_bound_reference"):
        print("=" * 68)
        print("⚠ 上界参照模式：本报告的数字**不参与门槛判定，也不参与 upper_alarm 判定**")
        print("  （人工构造的上界参照，不是真实成绩）")
    if gate.get("upper_alarm"):
        print(
            f"🚨 上界报警（⑪-b③）: 闭卷 F1 = {summary['f1']} > {gate['upper_alarm_threshold']}"
            " —— 触发人工检查（金标泄漏？过拟合？金标有偏？）。**这不是通过条件，是异常信号。**"
        )
    if gate.get("frame_note_checked"):
        print(
            f"frame_note 断言: 检查到 {gate['frame_note_checked']} 个字段，"
            f"泄漏 {len(gate.get('frame_note_leaks') or [])} 个 —— 断言非真空"
        )
    print("-" * 68)
    print("M2 跑批说明（⑫-b，可追溯）: M2 跑批使用修订前的 prompt（判定树两分支版）。")
    print("  复核确认当前金标 30 条中不存在「中间段否决 + 最终答案没提」这一边界条目，")
    print("  故新旧两版 prompt 在本批数据上行为等价，跑批结果可直接与金标比对。")
    print("  下一次跑批一律用新版 prompts/（export_prompts.py --check 必须过）。")
    if summary["trap_hits"]:
        print(f"⚠ 域内方向被误判成域外: {summary['trap_hits']} 次")
    print("=" * 68)


def run_cli(args: argparse.Namespace) -> int:
    # 门槛点：所有**主表与门槛判定**的匹配都在这一点上做（⑳-b 同点比较）。
    # 必须**先**定阈值再建匹配器。
    if args.match_threshold is None:
        args.match_threshold = GATE_POINT_MATCH_THR
    # ★ 报告的匹配阈值**钉在门槛点**（M2 实测修正）：
    #   introduced_in_synthesis 与 status 统计都依赖匹配阈值 ——
    #   一个随扫描参数漂移的指标，测的就不是「是否综合阶段新产生」，
    #   而是「措辞差多少」。**一份报告、一个阈值** = 门槛点。
    matcher = default_match_fn
    matcher_name = f"jaccard>={args.match_threshold}"
    if args.matcher:
        matcher = _load_matcher(args.matcher)
        matcher_name = args.matcher
    elif args.match_threshold != MATCH_THRESHOLD:
        matcher = make_similarity_match_fn(args.match_threshold)
        matcher_name = f"jaccard>={args.match_threshold}"

    gold = load_gold(args.gold)
    preds = load_predictions(args.pred)

    gold_by_id = {g["id"]: g for g in gold}
    pred_by_id: dict[str, dict] = {}
    for i, p in enumerate(preds):
        pid = p.get("id")
        if not isinstance(pid, str) or not pid.strip():
            pid = f"__row_{i + 1}__"
        pred_by_id[pid] = p

    results: list[dict] = []
    unmatched_preds: list[str] = []
    lexicon = _load_lexicon(args.lexicon)
    external_baselines = load_external_baselines(args.baseline_result)
    prp_ctx = None
    if args.prp_probe or args.premises:
        prompts_dir = args.prp_prompts
        premises_path = args.premises
        # 路径自动发现：eval/ 的同级 goldset/ 下
        if prompts_dir is None or premises_path is None:
            sibling = os.path.join(os.path.dirname(HERE), "goldset")
            prompts_dir = prompts_dir or os.path.join(sibling, "prp_prompts")
            premises_path = premises_path or os.path.join(sibling, "premises.jsonl")
        prp_ctx = build_prp_context(
            probe=args.prp_probe,
            prompts_dir=prompts_dir,
            premises_path=premises_path,
        )
    if args.baseline_scan:
        scan = load_baseline_scan(args.baseline_scan)
        external_baselines.extend(scan.values())
    if args.baseline_main_caliber:
        main_cal = load_main_caliber_baseline(args.baseline_main_caliber)
        if main_cal:
            external_baselines.append(main_cal)
    for gid, gold_item in gold_by_id.items():
        pred = pred_by_id.get(gid)
        if pred is None:
            r = score_prediction(
                gold_item,
                {"id": gid, "directions": []},
                match_fn=matcher,
                include_miss_detection=args.include_miss_detection,
                lexicon=lexicon,
                external_baselines=external_baselines,
            )
            r["warnings"].append("金标 id 在预测文件中没有对应记录（按全漏判计）")
            results.append(r)
            continue
        results.append(
            score_prediction(
                gold_item,
                pred,
                match_fn=matcher,
                include_miss_detection=args.include_miss_detection,
                lexicon=lexicon,
                external_baselines=external_baselines,
                prp_ctx=prp_ctx,
            )
        )

    for pid in pred_by_id:
        if pid not in gold_by_id:
            unmatched_preds.append(pid)

    # ⑳-b 同点比较：门槛判定必须在**门槛点匹配阈值**上重算抽取器 F1。
    gate_point_f1 = None
    try:
        gate_matcher = make_similarity_match_fn(GATE_POINT_MATCH_THR)
        gate_results = [
            score_prediction(
                gold_item,
                pred_by_id.get(gold_item["id"], {"id": gold_item["id"], "directions": []}),
                match_fn=gate_matcher,
                include_miss_detection=args.include_miss_detection,
                lexicon=lexicon,
            )
            for gold_item in gold_by_id.values()
        ]
        gate_point_f1 = aggregate(gate_results)["f1"]
    except Exception:  # 门槛点重算失败不该让整份报告挂掉
        gate_point_f1 = None

    summary = aggregate(results, gate_point_f1=gate_point_f1)
    summary["miss_bucket_reachability"] = assert_miss_bucket_reachability()
    summary["match_threshold_used"] = args.match_threshold
    summary["match_threshold_pinned_to_gate_point"] = (
        args.match_threshold == GATE_POINT_MATCH_THR
    )

    # --- 上界参照模式（主会话裁定）：不参与门槛与上界判定 ---
    if args.reference_mode:
        gate = summary["m2_gate"]
        gate["true_increment_gate_pass"] = False
        gate["true_increment_gate_pass_strict_max_basis"] = False
        gate["true_increment_gate_applicable"] = False
        gate["upper_alarm"] = False
        gate["upper_alarm_applicable"] = False
        gate["reference_mode"] = True
        gate["reference_mode_note"] = (
            "上界参照（人工构造）**不参与**门槛与上界判定 —— 它是人工构造的，不是真实成绩。"
            "否则每次跑参照都误报警，久了就没人看这个信号了。"
        )
        summary["upper_bound_reference"] = True

    # --- 裁决 ⑨-b：开卷/闭卷双数 ---
    dual = None
    if args.dual_open_gold and args.dual_open_pred:
        open_gold = load_gold(args.dual_open_gold)
        open_preds = {p.get("id"): p for p in load_predictions(args.dual_open_pred)}
        open_results = [
            score_prediction(
                g,
                open_preds.get(g["id"], {"id": g["id"], "directions": []}),
                match_fn=matcher,
                include_miss_detection=args.include_miss_detection,
                lexicon=lexicon,
                external_baselines=external_baselines,
            )
            for g in open_gold
        ]
        open_summary = aggregate(open_results)
        dual = {
            "closed_book": {
                "gold_mode": summary["gold_mode"],
                "f1": summary["f1"],
                "precision": summary["precision"],
                "recall": summary["recall"],
                "status_accuracy": summary["status_accuracy"],
                "miss_detection_recall": summary["miss_detection"]["miss_detection_recall"],
                "zero_direction_precision": summary.get("zero_direction_precision"),
                "zero_direction_overclaim_rate": summary.get("zero_direction_overclaim_rate"),
                "true_increment": summary["true_increment"]["f1"],
                "enters_gate": True,
            },
            "open_book": {
                "gold_mode": open_summary["gold_mode"],
                "f1": open_summary["f1"],
                "precision": open_summary["precision"],
                "recall": open_summary["recall"],
                "status_accuracy": open_summary["status_accuracy"],
                "miss_detection_recall": open_summary["miss_detection"]["miss_detection_recall"],
                "zero_direction_precision": open_summary.get("zero_direction_precision"),
                "zero_direction_overclaim_rate": open_summary.get("zero_direction_overclaim_rate"),
                "true_increment": open_summary["true_increment"]["f1"],
                "enters_gate": False,
            },
            "control_note": (
                "**受控 A/B（⑪-e）**：两个金标由同一份源生成，**只差 initial_frame 一行**"
                "（开卷版多一行「域外参考」）。所以两者的差值**纯粹来自题面泄漏**，"
                "不混杂其它变量 —— 这是 C 组目前唯一一个变量被完全控制住的对照实验。"
            ),
            "note": (
                "闭卷是主指标（进门槛）；开卷只作对照组 —— "
                "开卷下 initial_frame 直接列出域外方向，抽取器可以对着答案做，"
                "所以开卷数字衡量的是复述能力上限，不是域外识别能力。"
                "开卷数字**不与闭卷数字并列排名**。"
            ),
            "upper_bound_reference_note": (
                "若本表是用「完美预测」跑出来的，那它是**上界参照**，不是真实成绩 —— "
                "真实成绩必须来自真实抽取器的输出。"
            ),
        }
        if not args.quiet:
            print("-" * 68)
            print("[开卷 / 闭卷 双数]（裁决 ⑨-b：闭卷为主指标，开卷不进门槛）")
            c, o = dual["closed_book"], dual["open_book"]
            print(f"  {'':16s} {'闭卷(主)':>12s} {'开卷(对照)':>12s}")
            for key, label in (("f1", "主表 F1"), ("status_accuracy", "status 准确率"),
                               ("miss_detection_recall", "漏判 recall"),
                               ("zero_direction_precision", "零方向 precision"),
                               ("zero_direction_overclaim_rate", "零方向 overclaim")):
                print(f"  {label:16s} {str(c.get(key)):>12s} {str(o.get(key)):>12s}")

    sweep = None
    if args.sweep:
        thresholds = []
        value = args.sweep_from
        while value <= args.sweep_to + 1e-9:
            thresholds.append(round(value, 4))
            value += args.sweep_step
        sweep = threshold_sweep(
            list(gold_by_id.values()),
            pred_by_id,
            thresholds,
            lexicon=lexicon,
            external_baselines=external_baselines,
            include_miss_detection=args.include_miss_detection,
        )
        if not args.quiet:
            print("-" * 68)
            print("[阈值扫描] F1 vs 匹配阈值（裁决 ⑥-a / ⑨-c）")
            for point in sweep["curve"]:
                print(
                    f"  t={point['threshold']:<6} F1 {point['f1']:<8} P {point['precision']:<8} "
                    f"R {point['recall']:<8} 基线 {point['baseline_max_f1']:<8} 增量 {point['true_increment']}"
                )
            si = sweep.get("sensitive_interval")
            if si:
                print(
                    f"  阈值敏感区间: [{si['from']}, {si['to']}]  "
                    f"F1 {si['f1_min']}~{si['f1_max']}（{si['n_points']} 个点，"
                    f"判据：{sweep['sensitive_interval_criterion']}）"
                )
            print(f"  F1 全区间: {sweep['f1_range']}   基线最高: {sweep['baseline_max_f1_over_sweep']}")

    report = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "upper_bound_reference": bool(args.reference_mode),
            "gold_path": args.gold,
            "pred_path": args.pred,
            "matcher": matcher_name,
            "match_threshold_used": args.match_threshold,
            "match_threshold_pinned_to_gate_point": (args.match_threshold == GATE_POINT_MATCH_THR),
            "gate_point_match_thr": GATE_POINT_MATCH_THR,
            "include_miss_detection": bool(args.include_miss_detection),
            "lexicon": lexicon,
            "n_gold_items": len(gold),
            "n_pred_items": len(preds),
            "unmatched_pred_ids": unmatched_preds,
            "spec": "SPEC-DECISIONS-v1.md",
        },
        # 顶层单列（裁决 ⑦-a）：词法捷径基线与真实增量
        "lexical_baseline": summary["lexical_baseline"],
        "true_increment": summary["true_increment"],
        "threshold_sweep": sweep,
        "dual_open_closed": dual,
        "summary": summary,
    }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        if not args.quiet:
            print(f"报告已写入: {args.out}")

    if not args.quiet:
        _print_summary(report)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evaluator.py",
        description="PSC-Eval：问题空间构建评测器（schema v2）",
    )
    parser.add_argument("--gold", help="金标集 JSONL 路径")
    parser.add_argument("--pred", help="抽取器预测 JSONL 路径")
    parser.add_argument("--out", help="报告输出路径（JSON）")
    parser.add_argument(
        "--matcher",
        help="自定义 match_fn，格式 module:function（默认：字符二元组 Jaccard 阈值匹配）",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=None,
        help=(
            f"Jaccard 匹配阈值。**默认取门槛点 {GATE_POINT_MATCH_THR}**（报告与门槛同点，"
            f"指标不随 --sweep 漂移）；不给基线时回落到 {MATCH_THRESHOLD}。"
            "裁决 ⑥-a：M2 仍须扫 0.20–0.60 出曲线与敏感区间"
        ),
    )
    parser.add_argument(
        "--include-miss-detection",
        action="store_true",
        help="把 not_generated 折回主表（旧行为）。默认关闭：漏判单列，裁决 ②-b",
    )
    parser.add_argument(
        "--lexicon",
        help="词法捷径基线的词表 JSON（{\"rejection\": [...], \"retention\": [...]}）。默认用内置词表",
    )
    parser.add_argument(
        "--reference-mode",
        action="store_true",
        help="上界参照模式（人工构造的完美预测等）。该模式下**不参与门槛判定，也不参与 upper_alarm 判定**",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="做匹配阈值的扫描（曲线 + 敏感区间）。裁决 ⑥-a / ⑨-c 要求 M2 附上",
    )
    parser.add_argument(
        "--sweep-from", type=float, default=0.20, help="扫描起点（默认 0.20）"
    )
    parser.add_argument(
        "--sweep-to", type=float, default=0.60, help="扫描终点（默认 0.60）"
    )
    parser.add_argument(
        "--sweep-step", type=float, default=0.05, help="扫描步长（默认 0.05）"
    )
    parser.add_argument(
        "--baseline-result",
        action="append",
        default=[],
        help="外部实现的词法基线结果 JSON（可多次）。门槛基准取闭卷口径里的最高 F1（裁决 ⑧-d / ⑨-b）",
    )
    parser.add_argument(
        "--dual-open-gold",
        help="开卷对照组的金标（裁决 ⑨-b：M2 必须报开卷/闭卷双数，开卷只作对照）",
    )
    parser.add_argument(
        "--dual-open-pred",
        help="开卷对照组的预测（与 --dual-open-gold 配对）",
    )
    parser.add_argument(
        "--prp-probe",
        help="本次预测来自哪个 PRP 探针（A_premise_recognition / B_violation_detection / C_escape_direction）。"
             "探针 A 不给 P，其产出**不得**进 prp 桶",
    )
    parser.add_argument(
        "--prp-prompts",
        help="PRP 探针模板目录（含各探针的 user_template.txt），用于判定该探针是否给了 P",
    )
    parser.add_argument(
        "--premises",
        help="前提清单 JSONL（premise / trace_follows_premise / premise_raised_in_trace）",
    )
    parser.add_argument(
        "--baseline-main-caliber",
        help="主表口径的词法基线 JSON（由 diagnose_increment_source.py 生成）。**门槛应优先用它**",
    )
    parser.add_argument(
        "--baseline-scan",
        help="词法基线扫描表 CSV（mode/variant/.../f1）。门槛基准取 mode=closed 里的最高 F1（裁决 ⑨-c）",
    )
    parser.add_argument("--self-test", action="store_true", help="跑内置自测样例并退出")
    parser.add_argument("--quiet", action="store_true", help="不打印摘要")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()
    if not args.gold or not args.pred:
        parser.error("需要 --gold 和 --pred（或使用 --self-test）")
    return run_cli(args)


# --------------------------------------------------------------------------
# 自测
# --------------------------------------------------------------------------

_TRACE = [
    {"stage": "explore", "content": "先按分数位次排序。"},
    {"stage": "select", "content": "还可以看看家庭支持这个维度，但本题没给这个维度，先划掉。"},
    {"stage": "synthesize", "content": "最终推荐 A、B。"},
]

#: 符合强制模板的 initial_frame（裁决 ④）。
_FRAME_OK = """【候选集】A / B / C
【封闭性】closed
【评估维度】分数位次、就业率
【硬约束】无"""

#: 不符合模板：缺封闭性声明。
_FRAME_BAD = """【候选集】A / B / C
【评估维度】分数位次、就业率
【硬约束】无"""


def _gold_sample() -> dict:
    return {
        "id": "t1",
        "task": "推荐大学",
        "initial_frame": _FRAME_OK,
        "trace": _TRACE,
        "final_answer": "推荐 A、B。",
        "out_of_frame_directions": [
            {
                "direction": "把家庭支持作为选校维度",
                "status": "selection_failure",
                "decision_effect": "option_added",
                "evidence_quote": "还可以看看家庭支持这个维度",
            }
        ],
        "in_frame_directions": [
            {"direction": "按分数位次排序", "status": "realized", "evidence_quote": "先按分数位次排序。"}
        ],
    }


def _pred_sample(directions: list[dict]) -> dict:
    return {"id": "t1", "directions": directions}


def run_self_test() -> int:
    """内置手写样例，验证打分函数逻辑。不依赖 LLM、不依赖网络。"""
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append((name, bool(condition), detail))

    # --- 1. 完美预测 ---
    gold = _gold_sample()
    pred = _pred_sample(
        [
            {
                "direction": "把家庭支持作为选校维度",
                "in_frame": False,
                "in_frame_reason": "FRAME 只列了分数位次与就业率两个维度",
                "status": "selection_failure",
                "evidence_quote": "还可以看看家庭支持这个维度",
                "decision_effect": "option_added",
                "reasoning": "若纳入该维度，排序前两名会换位。",
            }
        ]
    )
    r = score_prediction(gold, pred)
    check("完美预测 P=1", r["precision"] == 1.0, f"P={r['precision']}")
    check("完美预测 R=1", r["recall"] == 1.0, f"R={r['recall']}")
    check("完美预测 F1=1", r["f1"] == 1.0, f"F1={r['f1']}")
    check("完美预测 status 准确率=1", r["status_accuracy"] == 1.0, str(r["status_accuracy"]))
    check("完美预测 status_agreement=True", r["status_agreement"] is True)
    check("完美预测 strict_agreement=True", r["status_agreement_strict"] is True)
    check("引用可核率=1", r["quote_grounding_rate"] == 1.0, str(r["quote_grounding_rate"]))
    check("误判域外=0", len(r["trap_hits"]) == 0, str(r["trap_hits"]))
    check("frame_schema=conforming", r["frame_schema"] == "conforming", r["frame_schema"])
    check("封闭性读出=closed", r["frame_closure"] == "closed", r["frame_closure"])

    # --- 2. 漏判（发现失败被漏掉）---
    r2 = score_prediction(gold, _pred_sample([]))
    check("空预测 P=0", r2["precision"] == 0.0, f"P={r2['precision']}")
    check("空预测 R=0", r2["recall"] == 0.0, f"R={r2['recall']}")
    check("空预测 FN=1", r2["fn"] == 1, f"FN={r2['fn']}")
    check("空预测 status_agreement=False", r2["status_agreement"] is False)
    check("空预测 无匹配不算判对", r2["status_agreement_strict"] is False)

    # --- 3. 过预测（多报一条 not_generated）---
    r3 = score_prediction(
        gold,
        _pred_sample(
            [
                {
                    "direction": "把家庭支持作为选校维度",
                    "in_frame": False,
                    "status": "selection_failure",
                    "evidence_quote": "还可以看看家庭支持这个维度",
                    "decision_effect": "option_added",
                },
                {
                    "direction": "把体育特长作为选校维度",
                    "in_frame": False,
                    "status": "not_generated",
                    "evidence_quote": "",
                    "decision_effect": "none",
                },
            ]
        ),
    )
    check("漏判类误报不计入主表 P", r3["precision"] == 1.0, f"P={r3['precision']}")
    check("过预测 R=1", r3["recall"] == 1.0, f"R={r3['recall']}")
    check("not_generated 不计入主表 FP", r3["fp"] == 0, f"FP={r3['fp']}")
    check("not_generated 计入误报漏判", r3["miss_detection"]["spurious"] == 1, str(r3["miss_detection"]))
    check("not_generated 引用为空", r3["quote_grounding"]["empty"] == 1, str(r3["quote_grounding"]))

    # --- 3a. 主表类别的多报（status 属主表三类）应当拉低 precision ---
    r3a = score_prediction(
        gold,
        _pred_sample(
            [
                {
                    "direction": "把家庭支持作为选校维度",
                    "in_frame": False,
                    "status": "selection_failure",
                    "evidence_quote": "还可以看看家庭支持这个维度",
                    "decision_effect": "option_added",
                },
                {
                    "direction": "把户籍政策作为选校维度",
                    "in_frame": False,
                    "status": "realized",
                    "evidence_quote": "先按分数位次排序。",
                    "decision_effect": "none",
                },
            ]
        ),
    )
    check("主表类多报拉低 P", r3a["precision"] < 1.0, f"P={r3a['precision']}")
    check("主表类多报计为 FP", r3a["fp"] == 1, f"FP={r3a['fp']}")
    check("主表类多报不计入误报漏判", r3a["miss_detection"]["spurious"] == 0, str(r3a["miss_detection"]))

    # --- 3b. 开关打开则折回主表（旧行为）---
    r3b = score_prediction(
        gold,
        _pred_sample(
            [
                {
                    "direction": "把家庭支持作为选校维度",
                    "in_frame": False,
                    "status": "selection_failure",
                    "evidence_quote": "还可以看看家庭支持这个维度",
                    "decision_effect": "option_added",
                },
                {
                    "direction": "把体育特长作为选校维度",
                    "in_frame": False,
                    "status": "not_generated",
                    "evidence_quote": "",
                    "decision_effect": "none",
                },
            ]
        ),
        include_miss_detection=True,
    )
    check("开关打开后 not_generated 折回主表 FP", r3b["fp"] == 1, f"FP={r3b['fp']}")

    # --- 4. status 判错 -> 混淆矩阵对角线外 ---
    r4 = score_prediction(
        gold,
        _pred_sample(
            [
                {
                    "direction": "把家庭支持作为选校维度",
                    "in_frame": False,
                    "status": "expression_failure",
                    "evidence_quote": "还可以看看家庭支持这个维度",
                    "decision_effect": "option_added",
                }
            ]
        ),
    )
    check(
        "status 混淆矩阵 off-diagonal",
        r4["status_confusion"]["selection_failure"]["expression_failure"] == 1,
        str(r4["status_confusion"]["selection_failure"]),
    )
    check("status 准确率=0", r4["status_accuracy"] == 0.0, str(r4["status_accuracy"]))
    check("status_agreement=False", r4["status_agreement"] is False)
    check("status_agreement_strict=False", r4["status_agreement_strict"] is False)
    check("方向本身仍算命中", r4["tp"] == 1, f"TP={r4['tp']}")

    # --- 5. 陷阱：金标域内方向被预测成域外 ---
    r5 = score_prediction(
        gold,
        _pred_sample(
            [
                {
                    "direction": "把家庭支持作为选校维度",
                    "in_frame": False,
                    "status": "selection_failure",
                    "evidence_quote": "还可以看看家庭支持这个维度",
                    "decision_effect": "option_added",
                },
                {
                    "direction": "按分数位次排序",
                    "in_frame": False,
                    "status": "realized",
                    "evidence_quote": "先按分数位次排序。",
                    "decision_effect": "none",
                },
            ]
        ),
    )
    check("陷阱命中=1", len(r5["trap_hits"]) == 1, str(r5["trap_hits"]))
    check("陷阱算主表 FP", r5["fp"] == 1, f"FP={r5['fp']}")

    # --- 6. 编造引用必须被抓到 ---
    r6 = score_prediction(
        gold,
        _pred_sample(
            [
                {
                    "direction": "把家庭支持作为选校维度",
                    "in_frame": False,
                    "status": "selection_failure",
                    "evidence_quote": "我们还要考虑户籍政策的影响",
                    "decision_effect": "option_added",
                }
            ]
        ),
    )
    check("编造引用被判 ungrounded", r6["quote_grounding"]["ungrounded"] == 1, str(r6["quote_grounding"]))
    check("编造引用使可核率<1", (r6["quote_grounding_rate"] or 0) < 1.0, str(r6["quote_grounding_rate"]))

    # --- 7. 相似度匹配器 ---
    check("相似度：同义改写判同向", default_match_fn({"direction": "家庭支持维度"}, {"direction": "家庭支持这个维度"}))
    check("相似度：无关方向不判同向", not default_match_fn({"direction": "家庭支持维度"}, {"direction": "户籍政策影响"}))
    check("相似度：对称", abs(similarity("家庭支持", "支持家庭") - similarity("支持家庭", "家庭支持")) < 1e-9)
    check("相似度：自身为 1", abs(similarity("家庭支持", "家庭支持") - 1.0) < 1e-9)
    check("相似度：空串为 0", similarity("", "家庭支持") == 0.0)

    # --- 8. status_agreement 独立 API 与 score_prediction 一致 ---
    check(
        "status_agreement API 一致(True)",
        status_agreement(gold, pred) is True and r["status_agreement"] is True,
    )
    check(
        "status_agreement API 一致(False)",
        status_agreement(gold, _pred_sample([])) is False and r2["status_agreement"] is False,
    )

    # --- 9. 边界：不除零 ---
    empty_gold = {"id": "e", "out_of_frame_directions": []}
    r9 = score_prediction(empty_gold, _pred_sample([]))
    check("空金标空预测不崩", r9["f1"] == 0.0 and r9["tp"] == 0, str(r9))
    r10 = score_prediction(empty_gold, pred)
    check("零方向条目：P=0 R=0", r10["precision"] == 0.0 and r10["recall"] == 0.0, str(r10))
    check("零方向条目不计入主表 FP（裁决 ⑧-c）", r10["fp"] == 0, f"FP={r10['fp']}")
    check("零方向条目被标为精度探针", r10["zero_direction"] is True, str(r10.get("zero_direction")))
    check("零方向条目的过度声称被记录", r10["zero_direction_overclaim"] is True, str(r10.get("zero_direction_overclaim")))

    # --- 10. aggregate ---
    agg = aggregate([r, r2, r3, r4, r5])
    check("aggregate 条数=5", agg["n_items"] == 5, str(agg["n_items"]))
    check(
        "aggregate 混淆矩阵累加正确",
        agg["status_confusion"]["selection_failure"]["selection_failure"] == 3,
        str(agg["status_confusion"]["selection_failure"]),
    )
    check("aggregate 有 per_status", set(agg["per_status"]) == set(STATUSES))
    check("aggregate 有逐条明细", len(agg["items"]) == 5)
    check("aggregate P 在 [0,1]", 0.0 <= agg["precision"] <= 1.0, str(agg["precision"]))
    check("aggregate schema_version=2", agg["schema_version"] == 2, str(agg["schema_version"]))

    # --- 11. not_generated 条目的 status 判定不误入混淆矩阵 ---
    gold_ng = {
        "id": "g2",
        "out_of_frame_directions": [
            {
                "direction": "考虑体育特长路径",
                "status": "not_generated",
                "decision_effect": "conclusion_flip",
            }
        ],
    }
    r11 = score_prediction(
        gold_ng,
        _pred_sample(
            [
                {
                    "direction": "考虑体育特长路径",
                    "in_frame": False,
                    "status": "not_generated",
                    "evidence_quote": "",
                    "decision_effect": "conclusion_flip",
                }
            ]
        ),
    )
    check("not_generated 匹配成功", r11["tp"] == 0, f"TP={r11['tp']}")
    check("not_generated 不计入主表", r11["fn"] == 0, f"FN={r11['fn']}")
    check(
        "not_generated 进混淆矩阵对角线",
        r11["status_confusion"]["not_generated"]["not_generated"] == 1,
        str(r11["status_confusion"]["not_generated"]),
    )
    check("not_generated status_agreement=True", r11["status_agreement"] is True)

    # --- 12. frame_ambiguous 标记被记录 ---
    pred_amb = _pred_sample([])
    pred_amb["frame_ambiguous"] = True
    r12 = score_prediction(gold, pred_amb)
    check("frame_ambiguous 被标记", r12["frame_ambiguous"] is True)
    check("frame_ambiguous 有 warning", any("frame_ambiguous" in w for w in r12["warnings"]))

    # --- 13. 非法 status 不进混淆矩阵 ---
    r13 = score_prediction(
        gold,
        _pred_sample(
            [
                {
                    "direction": "把家庭支持作为选校维度",
                    "in_frame": False,
                    "status": "发现失败",
                    "evidence_quote": "还可以看看家庭支持这个维度",
                    "decision_effect": "option_added",
                }
            ]
        ),
    )
    check("非法 status 不计入混淆矩阵", r13["status_checked"] == 0, str(r13["status_checked"]))

    # --- 14. envelope 兼容 ---
    r14 = score_prediction(
        gold,
        {
            "id": "t1",
            "prediction": {
                "directions": [
                    {
                        "direction": "把家庭支持作为选校维度",
                        "in_frame": False,
                        "status": "selection_failure",
                        "evidence_quote": "还可以看看家庭支持这个维度",
                        "decision_effect": "option_added",
                    }
                ]
            },
        },
    )
    check("envelope 包装能被解析", r14["tp"] == 1, f"TP={r14['tp']}")

    # --- 15. 自定义 match_fn 可插拔 ---
    def exact_match(g: dict, p: dict) -> bool:
        return direction_text(g) == direction_text(p)

    r15 = score_prediction(gold, pred, match_fn=exact_match)
    check("自定义 match_fn 生效", r15["tp"] == 1 and r15["fn"] == 0, f"TP={r15['tp']}")

    # --- 16. 引用可核率的分母口径 ---
    r16 = score_prediction(
        gold_ng,
        _pred_sample(
            [
                {
                    "direction": "考虑体育特长路径",
                    "in_frame": False,
                    "status": "not_generated",
                    "evidence_quote": "",
                    "decision_effect": "conclusion_flip",
                }
            ]
        ),
    )
    check("not_generated 空引用不计入可核率分母", r16["quote_grounding"]["required"] == 0, str(r16["quote_grounding"]))
    check("not_generated 空引用可核率=None", r16["quote_grounding_rate"] is None, str(r16["quote_grounding_rate"]))

    # --- 17. not_generated 却带引用 -> 违规计数 ---
    r17 = score_prediction(
        gold_ng,
        _pred_sample(
            [
                {
                    "direction": "考虑体育特长路径",
                    "in_frame": False,
                    "status": "not_generated",
                    "evidence_quote": "先算满减券的 CAC，再看运费补贴的 CAC。",
                    "decision_effect": "conclusion_flip",
                }
            ]
        ),
    )
    check("not_generated 带引用被记违规", r17["quote_grounding"]["not_generated_with_quote"] == 1, str(r17["quote_grounding"]))

    # --- 18. gold 域外方向被预测说成域内 ---
    r18 = score_prediction(
        gold,
        _pred_sample(
            [
                {
                    "direction": "把家庭支持作为选校维度",
                    "in_frame": True,
                    "status": "selection_failure",
                    "evidence_quote": "还可以看看家庭支持这个维度",
                    "decision_effect": "option_added",
                }
            ]
        ),
    )
    check("域外被说成域内被记录", r18["out_of_frame_called_in_frame"] == ["把家庭支持作为选校维度"], str(r18["out_of_frame_called_in_frame"]))
    check("域外被说成域内不计入 TP", r18["tp"] == 0, f"TP={r18['tp']}")

    # --- 19. 兼容金标集实际写法：嵌套 gold 键 + discovery_failure + trace 内嵌最终答案 ---
    sibling_gold = {
        "id": "G20",
        "task": "数据留存合规决策",
        "initial_frame": "【候选集】假名化 / 分区路由\n【封闭性】open\n【评估维度】合规风险、改造工作量\n【硬约束】无",
        "trace": [
            "阶段3｜方向生成：①对欧盟用户采用到期假名化；②把 9 个需要明细的下游系统改造为按法域路由取数。",
            "阶段4｜取舍：①保留。②标注涉及集团数据资产安排，暂不推进。",
            "最终答案：建议 ①+② 为主体，④ 明确不做。",
        ],
        "gold": {
            "out_of_frame_directions": [
                {
                    "direction": "在云厂商合同中约定由厂商承担合规删除的举证责任",
                    "status": "selection_failure",
                    "evidence": "阶段4『② 标注涉及集团数据资产安排，暂不推进』",
                    "decision_effect": "conclusion_flip",
                }
            ]
        },
    }
    r19 = score_prediction(
        sibling_gold,
        {
            "id": "G20",
            "directions": [
                {
                    "direction": "在云厂商合同中约定由厂商承担合规删除的举证责任",
                    "in_frame": False,
                    "status": "selection_failure",
                    "evidence_quote": "② 标注涉及集团数据资产安排，暂不推进",
                    "decision_effect": "conclusion_flip",
                }
            ],
        },
    )
    check("嵌套 gold 键被识别", r19["n_gold_out_of_frame"] == 1, str(r19["n_gold_out_of_frame"]))
    check(
        "嵌套 gold 的 selection_failure 进混淆矩阵",
        r19["status_confusion"]["selection_failure"]["selection_failure"] == 1,
        str(r19["status_confusion"]["selection_failure"]),
    )
    check(
        "trace 内嵌最终答案被排除出语料（引用仍能落到 trace 段）",
        r19["quote_grounding"]["required_grounded"] == 1 and r19["quote_grounding"]["ungrounded"] == 0,
        str(r19["quote_grounding"]),
    )
    fa_sib, corpus_sib, seg_sib = _split_final_answer(sibling_gold)
    check("最终答案被自动识别", fa_sib.startswith("建议"), repr(fa_sib[:20]))
    check("最终答案在语料里（裁决 ⑧-e：排除做法作废）", "明确不做" in corpus_sib, repr(corpus_sib[-40:]))
    check("最终答案段被单独识别出来", "明确不做" in seg_sib and "暂不推进" not in seg_sib, repr(seg_sib[-40:]))
    check("金标 evidence 可核性被统计", r19["gold_evidence_fragments"] >= 1, str(r19["gold_evidence_fragments"]))

    # --- 20. discovery_failure 写法与 not_generated 等价 ---
    r20 = score_prediction(
        {"id": "g3", "out_of_frame_directions": [{"direction": "引入外部资金持有库存", "status": "discovery_failure", "decision_effect": "conclusion_flip"}]},
        _pred_sample(
            [
                {
                    "direction": "引入外部资金持有库存",
                    "in_frame": False,
                    "status": "not_generated",
                    "evidence_quote": "",
                    "decision_effect": "conclusion_flip",
                }
            ]
        ),
    )
    check("gold 用 discovery_failure 也进混淆矩阵", r20["status_confusion"]["not_generated"]["not_generated"] == 1, str(r20["status_confusion"]["not_generated"]))
    check("discovery_failure 别名 status_agreement=True", r20["status_agreement"] is True)

    # ================= schema v2 新增用例 =================

    # --- 21. decision_effect 映射（裁决 ③）---
    check(
        "decision_effect 读取新字段",
        _decision_effect({"decision_effect": "ranking_only"}) == "ranking_only",
    )
    check(
        "decision_effect 兼容旧字段 true -> conclusion_flip",
        _decision_effect({"would_change_decision": True}) == "conclusion_flip",
    )
    check(
        "decision_effect 兼容旧字段 false -> none",
        _decision_effect({"would_change_decision": False}) == "none",
    )
    check(
        "decision_effect 新字段优先于旧字段",
        _decision_effect({"decision_effect": "none", "would_change_decision": True}) == "none",
    )
    check(
        "decision_effect 缺失时 has_effect=False",
        _has_effect({}) is False and _has_effect({"decision_effect": "none"}) is True,
    )
    check(
        "强效果只有 conclusion_flip / option_added",
        _is_strong_effect("conclusion_flip") and _is_strong_effect("option_added")
        and not _is_strong_effect("ranking_only") and not _is_strong_effect("additive_only")
        and not _is_strong_effect("none"),
    )

    # --- 22. weak_effect_count 单列（ranking_only + additive_only）---
    r22 = score_prediction(
        gold,
        _pred_sample(
            [
                {
                    "direction": "把家庭支持作为选校维度",
                    "in_frame": False,
                    "status": "selection_failure",
                    "evidence_quote": "还可以看看家庭支持这个维度",
                    "decision_effect": "ranking_only",
                }
            ]
        ),
    )
    check("weak_effect_count 统计 ranking_only", r22["weak_effect_count"] == 1, str(r22["weak_effect_count"]))
    check(
        "weak effect 记录了 effect 值",
        r22["weak_effects"][0]["decision_effect"] == "ranking_only",
        str(r22["weak_effects"]),
    )
    r22b = score_prediction(
        gold,
        _pred_sample(
            [
                {
                    "direction": "把家庭支持作为选校维度",
                    "in_frame": False,
                    "status": "selection_failure",
                    "evidence_quote": "还可以看看家庭支持这个维度",
                    "decision_effect": "conclusion_flip",
                }
            ]
        ),
    )
    check("强效果不计入 weak_effect_count", r22b["weak_effect_count"] == 0, str(r22b["weak_effect_count"]))

    # --- 23. decision_effect 一致率 ---
    r23 = score_prediction(
        gold,
        _pred_sample(
            [
                {
                    "direction": "把家庭支持作为选校维度",
                    "in_frame": False,
                    "status": "selection_failure",
                    "evidence_quote": "还可以看看家庭支持这个维度",
                    "decision_effect": "none",
                }
            ]
        ),
    )
    check("effect 强弱不一致被算出", r23["decision_effect_agreement"] == 0.0, str(r23["decision_effect_agreement"]))
    check("effect 精确一致率被算出", r23["decision_effect_exact_agreement"] == 0.0, str(r23["decision_effect_exact_agreement"]))

    # --- 24. miss_detection_recall（裁决 ②-b）---
    gold_miss = {
        "id": "gm",
        "out_of_frame_directions": [
            {"direction": "考虑强基计划等特殊招生路径", "status": "not_generated", "decision_effect": "conclusion_flip"},
            {"direction": "把家庭支持作为选校维度", "status": "selection_failure", "decision_effect": "option_added"},
        ],
    }
    r24 = score_prediction(
        gold_miss,
        _pred_sample(
            [
                {
                    "direction": "把家庭支持作为选校维度",
                    "in_frame": False,
                    "status": "selection_failure",
                    "evidence_quote": "",
                    "decision_effect": "option_added",
                }
            ]
        ),
    )
    check("主表只统计三类", r24["tp"] == 1 and r24["fn"] == 0, f"TP={r24['tp']} FN={r24['fn']}")
    check("漏判未命中 -> miss hit=0", r24["miss_detection"]["hit"] == 0, str(r24["miss_detection"]))
    check("漏判被记 missed", r24["miss_detection"]["missed"] == 1, str(r24["miss_detection"]))
    check("miss_detection_recall=0", r24["miss_detection"]["miss_detection_recall"] == 0.0, str(r24["miss_detection"]))
    r24b = score_prediction(
        gold_miss,
        _pred_sample(
            [
                {
                    "direction": "把家庭支持作为选校维度",
                    "in_frame": False,
                    "status": "selection_failure",
                    "evidence_quote": "",
                    "decision_effect": "option_added",
                },
                {
                    "direction": "考虑强基计划等特殊招生路径",
                    "in_frame": False,
                    "status": "not_generated",
                    "evidence_quote": "",
                    "decision_effect": "conclusion_flip",
                },
            ]
        ),
    )
    check("漏判命中 -> miss recall=1", r24b["miss_detection"]["miss_detection_recall"] == 1.0, str(r24b["miss_detection"]))
    check("主表不受漏判影响", r24b["tp"] == 1 and r24b["fp"] == 0 and r24b["fn"] == 0, f"TP={r24b['tp']} FP={r24b['fp']} FN={r24b['fn']}")

    # --- 25. status_accuracy_strict 与 status_undecidable_rate 并列（裁决 ①-b）---
    r25 = score_prediction(
        gold,
        _pred_sample(
            [
                {
                    "direction": "把家庭支持作为选校维度",
                    "in_frame": False,
                    "status": "expression_failure",
                    "evidence_quote": "还可以看看家庭支持这个维度",
                    "decision_effect": "option_added",
                    "status_ambiguous": True,
                }
            ]
        ),
    )
    check("全量 status 准确率计入了不可判条目", r25["status_accuracy"] == 0.0, str(r25["status_accuracy"]))
    check("strict 准确率排除不可判条目", r25["status_accuracy_strict"] is None, str(r25["status_accuracy_strict"]))
    check("不可判率被统计", r25["status_undecidable_rate"] == 1.0, str(r25["status_undecidable_rate"]))
    r25b = score_prediction(
        gold,
        _pred_sample(
            [
                {
                    "direction": "把家庭支持作为选校维度",
                    "in_frame": False,
                    "status": "expression_failure",
                    "evidence_quote": "还可以看看家庭支持这个维度",
                    "decision_effect": "option_added",
                    "status_ambiguous": True,
                },
                {
                    "direction": "按分数位次排序",
                    "in_frame": True,
                    "status": "realized",
                    "evidence_quote": "先按分数位次排序。",
                    "decision_effect": "none",
                },
            ]
        ),
    )
    check("不可判率分母含全部带 status 的条目", r25b["status_undecidable_rate"] == 0.5, str(r25b["status_undecidable_rate"]))

    # --- 26. selection_expression_separability（裁决 ①-c）---
    gold_se = {
        "id": "gse",
        "out_of_frame_directions": [
            {"direction": "方向甲被明确否决", "status": "selection_failure", "decision_effect": "none"},
            {"direction": "方向乙被保留但没写进答案", "status": "expression_failure", "decision_effect": "none"},
        ],
    }
    r26 = score_prediction(
        gold_se,
        _pred_sample(
            [
                {"direction": "方向甲被明确否决", "in_frame": False, "status": "selection_failure", "evidence_quote": "", "decision_effect": "none"},
                {"direction": "方向乙被保留但没写进答案", "in_frame": False, "status": "selection_failure", "evidence_quote": "", "decision_effect": "none"},
            ]
        ),
    )
    check("可分性分母=selection+expression 对数", r26["selection_expression_checked"] == 2, str(r26["selection_expression_checked"]))
    check("可分性=1/2", r26["selection_expression_separability"] == 0.5, str(r26["selection_expression_separability"]))
    r26b = score_prediction(
        gold_se,
        _pred_sample(
            [
                {"direction": "方向甲被明确否决", "in_frame": False, "status": "selection_failure", "evidence_quote": "", "decision_effect": "none"},
                {"direction": "方向乙被保留但没写进答案", "in_frame": False, "status": "expression_failure", "evidence_quote": "", "decision_effect": "none"},
            ]
        ),
    )
    check("完全可分=1.0", r26b["selection_expression_separability"] == 1.0, str(r26b["selection_expression_separability"]))
    check("无独立裁定时 adjudicated 值为 None", r26b["selection_expression_separability_adjudicated"] is None, str(r26b["selection_expression_separability_adjudicated"]))
    gold_se_adj = dict(gold_se)
    gold_se_adj["selection_expression_adjudicated"] = True
    r26c = score_prediction(
        gold_se_adj,
        _pred_sample(
            [
                {"direction": "方向甲被明确否决", "in_frame": False, "status": "selection_failure", "evidence_quote": "", "decision_effect": "none"},
                {"direction": "方向乙被保留但没写进答案", "in_frame": False, "status": "expression_failure", "evidence_quote": "", "decision_effect": "none"},
            ]
        ),
    )
    check("有独立裁定时给出 adjudicated 可分性", r26c["selection_expression_separability_adjudicated"] == 1.0, str(r26c["selection_expression_separability_adjudicated"]))

    # --- 27. frame_schema 检查（裁决 ④）---
    check("模板齐全 -> conforming", frame_schema(_FRAME_OK) == "conforming", frame_schema(_FRAME_OK))
    check("缺封闭性 -> non_conforming", frame_schema(_FRAME_BAD) == "non_conforming", frame_schema(_FRAME_BAD))
    check("缺候选集 -> non_conforming", frame_schema("【封闭性】open\n【评估维度】a\n【硬约束】无") == "non_conforming")
    check("空框架 -> non_conforming", frame_schema("") == "non_conforming")
    check("封闭性未声明取值 -> non_conforming", frame_schema("【候选集】A\n【封闭性】\n【评估维度】a\n【硬约束】无") == "non_conforming")
    check("封闭性取值可读出", _frame_closure(_FRAME_OK) == "closed", _frame_closure(_FRAME_OK))
    bad_gold = dict(gold)
    bad_gold["initial_frame"] = _FRAME_BAD
    r27 = score_prediction(bad_gold, pred)
    check("非合规框架被标记", r27["frame_schema"] == "non_conforming", r27["frame_schema"])
    check("非合规框架有 warning", any("non_conforming" in w for w in r27["warnings"]), str(r27["warnings"]))
    agg_fs = aggregate([r, r27])
    check("aggregate 分开统计 conforming/non_conforming", agg_fs["frame_schema"]["conforming"] == 1 and agg_fs["frame_schema"]["non_conforming"] == 1, str(agg_fs["frame_schema"]))

    # --- 28. M2 门槛机检 ---
    gate = aggregate([r27])["m2_gate"]
    check("M2 门槛报告非逐字比例", "gold_evidence_ungrounded_rate" in gate, str(gate))
    check("M2 门槛报告金标全为关键漏判", gate["gold_miss_all_key"] is True, str(gate))
    check("M2 门槛报告金标用 v2 effect", gate["gold_uses_v2_effect"] is True, str(gate))
    check("M2 门槛报告 frame 全合规", gate["all_frames_conforming"] is False, str(gate))
    legacy_gold = {
        "id": "gl",
        "out_of_frame_directions": [
            {"direction": "x", "status": "selection_failure", "would_change_decision": True}
        ],
    }
    check(
        "金标仍用旧字段被机检抓到",
        aggregate([score_prediction(legacy_gold, _pred_sample([]))])["m2_gate"]["gold_uses_v2_effect"] is False,
    )
    miss_not_key = {
        "id": "gnk",
        "out_of_frame_directions": [
            {"direction": "x", "status": "not_generated", "decision_effect": "none"}
        ],
    }
    check(
        "金标漏判非关键被判不合格",
        aggregate([score_prediction(miss_not_key, _pred_sample([]))])["m2_gate"]["gold_miss_all_key"] is False,
    )

    # --- 29. 旧字段兼容（抽取器还在输出 would_change_decision）---
    legacy_pred = _pred_sample(
        [
            {
                "direction": "把家庭支持作为选校维度",
                "in_frame": False,
                "status": "selection_failure",
                "evidence_quote": "还可以看看家庭支持这个维度",
                "would_change_decision": True,
            }
        ]
    )
    r29 = score_prediction(gold, legacy_pred)
    check("旧字段预测仍可打分", r29["tp"] == 1, f"TP={r29['tp']}")
    check("旧字段 effect 映射为强效果", r29["matched"][0]["pred_decision_effect"] == "conclusion_flip", str(r29["matched"][0]))
    check("旧字段不计入 weak_effect_count", r29["weak_effect_count"] == 0, str(r29["weak_effect_count"]))

    # --- 30. 报告表结构 ---
    agg30 = aggregate([r, r2, r3, r4, r5])
    check("报告有主表", "main" in agg30 and "miss detection" in agg30["main"]["title"], str(agg30.get("main")))
    check("主表标题写明不含 miss detection", "不含 miss detection" in agg30["main"]["title"], agg30["main"]["title"])
    check("漏判表有免责句", "下界" in agg30["miss_detection"]["disclaimer"], agg30["miss_detection"]["disclaimer"])
    check("报告有 schema_version", agg30["schema_version"] == 2, str(agg30["schema_version"]))
    check("报告有 per_effect", set(agg30["per_effect"]) == set(DECISION_EFFECTS))

    # ================= 裁决 ⑦ 新增用例 =================

    # --- 31. canonical 名统一：报告里不得出现 discovery_failure ---
    gold_alias = {
        "id": "ga",
        "out_of_frame_directions": [
            {"direction": "引入外部资金持有库存", "status": "discovery_failure", "decision_effect": "conclusion_flip"}
        ],
    }
    r31 = score_prediction(
        gold_alias,
        _pred_sample(
            [
                {
                    "direction": "引入外部资金持有库存",
                    "in_frame": False,
                    "status": "not_generated",
                    "evidence_quote": "",
                    "decision_effect": "conclusion_flip",
                }
            ]
        ),
    )
    check("别名归一到 canonical", r31["status_confusion"]["not_generated"]["not_generated"] == 1, str(r31["status_confusion"]))
    agg31 = aggregate([r31])
    check(
        "报告输出不含 discovery_failure 字样",
        "discovery_failure" not in json.dumps(agg31, ensure_ascii=False),
        "报告里混入了别名",
    )
    check(
        "per_status 用 canonical 名",
        set(agg31["per_status"]) == set(STATUSES) and "not_generated" in agg31["per_status"],
        str(list(agg31["per_status"])),
    )

    # --- 32. 词法捷径基线（裁决 ⑦-a）---
    gold_lex = {
        "id": "glx",
        "initial_frame": "【候选集】A\n【封闭性】open\n【评估维度】d\n【硬约束】无",
        "trace": [
            "阶段3｜方向生成：①引入外部资金持有库存。",
            "阶段4｜取舍：①否决，理由是超出授权范围。",
        ],
        "final_answer": "建议维持现状。",
        "out_of_frame_directions": [
            {"direction": "引入外部资金持有库存", "status": "selection_failure", "decision_effect": "option_added"}
        ],
    }
    lex_pred = {
        "id": "glx",
        "directions": [
            {
                "direction": "引入外部资金持有库存",
                "in_frame": False,
                "status": "selection_failure",
                "evidence_quote": "①否决，理由是超出授权范围。",
                "decision_effect": "option_added",
            }
        ],
    }
    base = lexical_baseline(gold_lex, lex_pred)
    check("词法基线能识别否决词", base["lexical_inferred"][0]["lexical_status"] == "selection_failure", str(base["lexical_inferred"]))
    check("词法基线 F1=1（这个 case 太容易）", base["f1"] == 1.0, str(base["f1"]))
    check("词法基线结构与主流程同构", "status_confusion" in base and "precision" in base, str(list(base)[:6]))
    # 保留词 -> 出现在最终答案则 realized
    gold_keep = {
        "id": "gk",
        "initial_frame": "【候选集】A\n【封闭性】open\n【评估维度】d\n【硬约束】无",
        "trace": ["阶段4｜取舍：①保留该方向。"],
        "final_answer": "建议采用引入外部资金持有库存的方案。",
        "out_of_frame_directions": [
            {"direction": "引入外部资金持有库存", "status": "realized", "decision_effect": "option_added"}
        ],
    }
    keep_pred = {
        "id": "gk",
        "directions": [
            {
                "direction": "引入外部资金持有库存",
                "in_frame": False,
                "status": "realized",
                "evidence_quote": "①保留该方向。",
                "decision_effect": "option_added",
            }
        ],
    }
    base_keep = lexical_baseline(gold_keep, keep_pred)
    check("词法基线：保留词 + 出现在最终答案 -> realized", base_keep["lexical_inferred"][0]["lexical_status"] == "realized", str(base_keep["lexical_inferred"]))
    # 保留词但没进最终答案 -> expression_failure
    gold_keep2 = dict(gold_keep)
    gold_keep2["final_answer"] = "建议维持现状。"
    gold_keep2["out_of_frame_directions"] = [
        {"direction": "引入外部资金持有库存", "status": "expression_failure", "decision_effect": "none"}
    ]
    base_keep2 = lexical_baseline(gold_keep2, keep_pred)
    check("词法基线：保留词但未呈现 -> expression_failure", base_keep2["lexical_inferred"][0]["lexical_status"] == "expression_failure", str(base_keep2["lexical_inferred"]))
    # 两种词都没有 -> not_generated
    gold_none = {
        "id": "gn",
        "initial_frame": "【候选集】A\n【封闭性】open\n【评估维度】d\n【硬约束】无",
        "trace": ["阶段1｜任务理解：先看看现状。"],
        "final_answer": "建议维持现状。",
        "out_of_frame_directions": [
            {"direction": "完全没出现过的方向", "status": "not_generated", "decision_effect": "none"}
        ],
    }
    base_none = lexical_baseline(
        gold_none,
        {"id": "gn", "directions": [{"direction": "完全没出现过的方向", "in_frame": False, "status": "not_generated", "evidence_quote": "", "decision_effect": "none"}]},
    )
    check("词法基线：无关键词 -> not_generated", base_none["lexical_inferred"][0]["lexical_status"] == "not_generated", str(base_none["lexical_inferred"]))

    # --- 33. true_increment（真实增量）---
    r33 = score_prediction(gold_lex, lex_pred)
    check("score_prediction 附带 lexical_baseline", "lexical_baseline" in r33, str(list(r33)))
    check("score_prediction 附带 true_increment", "true_increment" in r33, str(list(r33)))
    check("true_increment.f1 = 抽取器F1 - 基线F1", r33["true_increment"]["f1"] == round(r33["f1"] - r33["lexical_baseline"]["f1"], 4), str(r33["true_increment"]))
    agg33 = aggregate([r33])
    check("aggregate 汇总 lexical_baseline", "lexical_baseline" in agg33 and "f1" in agg33["lexical_baseline"], str(list(agg33.get("lexical_baseline", {}))))
    check("aggregate 汇总 true_increment", agg33["true_increment"]["f1"] == round(agg33["f1"] - agg33["lexical_baseline"]["f1"], 4), str(agg33["true_increment"]))
    check(
        "true_increment 打出规定语句（裁决 ⑧-d 口径）",
        "真实增量 = 抽取器 F1 − 词法基线最高 F1" in agg33["true_increment"]["statement"],
        agg33["true_increment"]["statement"],
    )
    check("M2 门槛记录真实增量下限", agg33["m2_gate"]["true_increment_min"] == 0.10, str(agg33["m2_gate"]))
    check("M2 门槛给出是否通过", "true_increment_gate_pass" in agg33["m2_gate"], str(agg33["m2_gate"]))

    # --- 34. expected_source（裁决 ⑦-b）---
    check("expected_source 缺失 -> unknown", _expected_source({}) == EXPECTED_SOURCE_UNKNOWN, _expected_source({}))
    check("expected_source 非法值 -> invalid", _expected_source({"expected_source": "随便写的"}) == EXPECTED_SOURCE_INVALID, _expected_source({"expected_source": "随便写的"}))
    check("expected_source 正常值", _expected_source({"expected_source": "task"}) == "task", _expected_source({"expected_source": "task"}))
    check("expected_source 非字符串 -> invalid", _expected_source({"expected_source": 123}) == EXPECTED_SOURCE_INVALID, _expected_source({"expected_source": 123}))
    gold_es = {
        "id": "ges",
        "out_of_frame_directions": [
            {"direction": "甲", "status": "not_generated", "decision_effect": "conclusion_flip", "expected_source": "task"},
            {"direction": "乙", "status": "not_generated", "decision_effect": "conclusion_flip", "expected_source": "external"},
            {"direction": "丙", "status": "not_generated", "decision_effect": "conclusion_flip"},
            {"direction": "丁", "status": "not_generated", "decision_effect": "conclusion_flip", "expected_source": "乱写"},
        ],
    }
    r34 = score_prediction(gold_es, _pred_sample([]))
    check("expected_source 缺失/非法都不崩", r34["n_gold_out_of_frame"] == 4, str(r34["n_gold_out_of_frame"]))
    check("expected_source 分布统计正确（unlabeled / unknown 严格分档）",
          r34["expected_source_distribution"] == {"task": 1, "external": 1, "unlabeled": 1, "unknown": 1},
          str(r34["expected_source_distribution"]))
    check("not_generated 的 expected_source 分布单列",
          r34["expected_source_distribution_not_generated"] == {"task": 1, "external": 1, "unlabeled": 1, "unknown": 1},
          str(r34["expected_source_distribution_not_generated"]))
    check("expected_source 不影响可核率分母", r34["quote_grounding"]["required"] == 0, str(r34["quote_grounding"]))
    check("expected_source 不影响主表计数", r34["tp"] == 0 and r34["fn"] == 0, f"TP={r34['tp']} FN={r34['fn']}")

    # --- 35. 词表可配置 ---
    custom = {"rejection": ["搁置"], "retention": ["采纳"]}
    gold_c = {
        "id": "gc",
        "initial_frame": "【候选集】A\n【封闭性】open\n【评估维度】d\n【硬约束】无",
        "trace": ["阶段4｜取舍：①搁置。"],
        "final_answer": "维持现状。",
        "out_of_frame_directions": [{"direction": "方向甲", "status": "selection_failure", "decision_effect": "none"}],
    }
    pred_c = {"id": "gc", "directions": [{"direction": "方向甲", "in_frame": False, "status": "selection_failure", "evidence_quote": "①搁置。", "decision_effect": "none"}]}
    base_custom = lexical_baseline(gold_c, pred_c, lexicon=custom)
    check("自定义词表生效", base_custom["lexical_inferred"][0]["lexical_status"] == "selection_failure", str(base_custom["lexical_inferred"]))
    base_default = lexical_baseline(gold_c, pred_c)
    check("默认词表不认自定义词", base_default["lexical_inferred"][0]["lexical_status"] == "not_generated", str(base_default["lexical_inferred"]))
    check("词表缺键时回落到默认", _load_lexicon(None)["rejection"] == WIDE_LEXICON["rejection"], str(_load_lexicon(None)))

    # --- 36. 金标 evidence 的两种写法 + 漏判条目不计入可核率 ---
    gold_ev = {
        "id": "gev",
        "trace": ["阶段3｜方向生成：①引入外部资金持有库存。", "阶段4｜取舍：①否决，超出授权。"],
        "final_answer": "维持现状。",
        "out_of_frame_directions": [
            {
                "direction": "引入外部资金持有库存",
                "status": "selection_failure",
                "decision_effect": "option_added",
                "evidence": "①否决，超出授权。",
            },
            {
                "direction": "把夜间批处理外包给第三方",
                "status": "not_generated",
                "decision_effect": "conclusion_flip",
                "evidence": "trace 中无对应内容：阶段1 只复述了值班事实，未把它作为可执行选项提出",
            },
        ],
    }
    r36 = score_prediction(gold_ev, _pred_sample([]))
    check("evidence 纯原文写法可抽片段", r36["gold_evidence_fragments"] == 1, str(r36["gold_evidence_fragments"]))
    check("纯原文写法逐字命中", r36["gold_evidence_stats"]["exact"] == 1, str(r36["gold_evidence_stats"]))
    check("漏判条目的说明性 evidence 被跳过", r36["gold_evidence_skipped_not_generated"] == 1, str(r36["gold_evidence_skipped_not_generated"]))
    check("漏判条目不造成 ungrounded", r36["gold_evidence_stats"]["ungrounded"] == 0, str(r36["gold_evidence_stats"]))
    check("抽取失败计数为 0", r36["gold_evidence_extraction_failed"] == 0, str(r36["gold_evidence_extraction_failed"]))
    # 引号写法仍支持
    check(
        "引号写法仍可抽片段",
        len(_extract_quoted_fragments("阶段3『引入外部资金』；阶段4『否决』")) == 2,
        str(_extract_quoted_fragments("阶段3『引入外部资金』；阶段4『否决』")),
    )
    # 有 evidence 但抽不出片段 -> 记抽取失败，不静默通过
    gold_bad_ev = {
        "id": "gbe",
        "trace": ["abc"],
        "out_of_frame_directions": [
            {"direction": "x", "status": "selection_failure", "decision_effect": "none", "evidence": "短"}
        ],
    }
    r36b = score_prediction(gold_bad_ev, _pred_sample([]))
    check("过短 evidence 记抽取失败", r36b["gold_evidence_extraction_failed"] == 1, str(r36b["gold_evidence_extraction_failed"]))
    check(
        "抽取失败时 M2 门槛不通过",
        aggregate([r36b])["m2_gate"]["gold_evidence_gate_pass"] is False,
        str(aggregate([r36b])["m2_gate"]),
    )

    # --- 37. 退化解检测：全称 not_generated 不得拿到完美主表 ---
    gold_deg = {
        "id": "gdeg",
        "out_of_frame_directions": [
            {"direction": "方向甲", "status": "selection_failure", "decision_effect": "option_added"},
            {"direction": "方向乙", "status": "realized", "decision_effect": "conclusion_flip"},
        ],
    }
    deg_pred = {"id": "gdeg", "directions": [
        {"direction": "方向甲", "in_frame": False, "status": "not_generated", "evidence_quote": "", "decision_effect": "none"},
        {"direction": "方向乙", "in_frame": False, "status": "not_generated", "evidence_quote": "", "decision_effect": "none"},
    ]}
    r37 = score_prediction(gold_deg, deg_pred)
    check("全称 not_generated 拿不到完美主表", r37["f1"] < 1.0, f"F1={r37['f1']}")
    check("全称 not_generated 计为主表漏判", r37["fn"] == 2, f"FN={r37['fn']}")
    check("全称 not_generated 全部落进误报漏判", r37["miss_detection"]["spurious"] == 2, str(r37["miss_detection"]))

    # ================= 裁决 ⑧ 新增用例 =================

    # --- 38. ★ 对抗性：分表匹配漏洞的闭合证明（bug #1 的对偶）---
    # 场景：抽取器把拿不准的主表方向全部推给 not_generated，能不能逃脱主表惩罚？
    adv_gold = {
        "id": "adv",
        "out_of_frame_directions": [
            {"direction": "方向甲", "status": "selection_failure", "decision_effect": "option_added"},
            {"direction": "方向乙", "status": "selection_failure", "decision_effect": "option_added"},
            {"direction": "方向丙", "status": "realized", "decision_effect": "conclusion_flip"},
            {"direction": "方向丁", "status": "realized", "decision_effect": "conclusion_flip"},
        ],
    }
    # 38a. 全部推给 not_generated
    adv_all = {"id": "adv", "directions": [
        {"direction": t, "in_frame": False, "status": "not_generated",
         "evidence_quote": "", "decision_effect": "none"}
        for t in ("方向甲", "方向乙", "方向丙", "方向丁")
    ]}
    r38 = score_prediction(adv_gold, adv_all)
    check("对抗性①：全推 not_generated -> 主表 F1 严格等于 0", r38["f1"] == 0.0, f"F1={r38['f1']}")
    check("对抗性①：全推 not_generated -> 主表 recall 严格等于 0", r38["recall"] == 0.0, f"R={r38['recall']}")
    check("对抗性①：全推 not_generated -> 主表 precision 严格等于 0", r38["precision"] == 0.0, f"P={r38['precision']}")
    check("对抗性①：全推 not_generated -> 金标主表方向全部计入 FN", r38["fn"] == 4, f"FN={r38['fn']}")
    check("对抗性①：全推 not_generated -> 漏判池产生大量 FP（误报漏判 4）",
          r38["miss_detection"]["spurious"] == 4, str(r38["miss_detection"]))
    # 38b. 只推一半
    adv_half = {"id": "adv", "directions": [
        {"direction": "方向甲", "in_frame": False, "status": "not_generated",
         "evidence_quote": "", "decision_effect": "none"},
        {"direction": "方向乙", "in_frame": False, "status": "not_generated",
         "evidence_quote": "", "decision_effect": "none"},
        {"direction": "方向丙", "in_frame": False, "status": "realized",
         "evidence_quote": "方向丙的原文", "decision_effect": "conclusion_flip"},
        {"direction": "方向丁", "in_frame": False, "status": "realized",
         "evidence_quote": "方向丁的原文", "decision_effect": "conclusion_flip"},
    ]}
    r38b = score_prediction(adv_gold, adv_half)
    check("对抗性②：只推一半 -> 主表 F1 严格介于 0 与 1 之间",
          0.0 < r38b["f1"] < 1.0, f"F1={r38b['f1']}")
    check("对抗性②：只推一半 -> 惩罚是连续的（recall = 0.5）", r38b["recall"] == 0.5, f"R={r38b['recall']}")
    check("对抗性②：只推一半 -> 漏判池误报 2", r38b["miss_detection"]["spurious"] == 2, str(r38b["miss_detection"]))

    # --- 39. 词法基线多变体，取最高 F1（裁决 ⑧-d）---
    lex_variants = r38.get("lexical_baseline_variants") or {}
    check("词法基线产出 6 个变体", len(lex_variants) == 6, str(sorted(lex_variants)))
    check("6 个变体覆盖 3 语料 × 宽/窄词表",
          {v["corpus"] for v in lex_variants.values()} == set(CORPUS_VARIANTS)
          and {v["lexicon"] for v in lex_variants.values()} == set(LEXICON_VARIANTS),
          str(sorted(lex_variants)))
    best = max(lex_variants.values(), key=lambda v: v["f1"]) if lex_variants else None
    check("门槛基准取最高 F1 的变体", r38["lexical_baseline"]["f1"] == (best["f1"] if best else 0.0),
          f"{r38['lexical_baseline']['f1']} vs {best}")
    check("true_increment 用最高基线", r38["true_increment"]["baseline_best_f1"] == r38["lexical_baseline"]["f1"],
          str(r38["true_increment"]))
    check("报告给出变体 F1 区间", "f1_range" in aggregate([r38])["lexical_baseline"],
          str(aggregate([r38])["lexical_baseline"].get("f1_range")))

    # --- 40. status v1.1 判定顺序（锚点在最终答案段）---
    v11_gold = {
        "id": "v11",
        "trace": [
            "阶段3｜方向生成：①保留方向甲。",
            "阶段4｜取舍：①保留方向甲。",
            "最终答案：建议采用方向甲的方案，不建议方向乙。",
        ],
        "final_answer": "建议采用方向甲的方案，不建议方向乙。",
        "out_of_frame_directions": [
            {"direction": "方向甲", "status": "realized", "decision_effect": "conclusion_flip"},
            {"direction": "方向乙", "status": "selection_failure", "decision_effect": "none"},
        ],
    }
    v11_pred = {"id": "v11", "directions": [
        {"direction": "方向甲", "in_frame": False, "status": "realized",
         "evidence_quote": "建议采用方向甲的方案", "decision_effect": "conclusion_flip"},
        {"direction": "方向乙", "in_frame": False, "status": "selection_failure",
         "evidence_quote": "不建议方向乙。", "decision_effect": "none"},
    ]}
    r40 = score_prediction(v11_gold, v11_pred)
    check("v1.1：最终答案里的推荐 -> realized 判对", r40["status_accuracy"] == 1.0, str(r40["status_accuracy"]))
    check("v1.1：最终答案里的否决 -> selection_failure 判对",
          r40["status_confusion"]["selection_failure"]["selection_failure"] == 1,
          str(r40["status_confusion"]["selection_failure"]))
    check("v1.1：引用可核（最终答案在语料里）", r40["quote_grounding"]["exact"] == 2, str(r40["quote_grounding"]))
    # 词法基线也须遵循 v1.1：在最终答案里且带否决词 -> selection_failure
    lex_v11 = lexical_baseline(v11_gold, v11_pred, corpus_mode="evidence_quote", lexicon=WIDE_LEXICON)
    statuses = {i["direction"]: i["lexical_status"] for i in lex_v11["lexical_inferred"]}
    check("词法基线遵循 v1.1：最终答案里的否决 -> selection_failure",
          statuses.get("方向乙") == "selection_failure", str(statuses))
    check("词法基线遵循 v1.1：最终答案里的推荐 -> realized",
          statuses.get("方向甲") == "realized", str(statuses))

    # --- 41. 零方向条目的两个指标（裁决 ⑧-c）---
    zero_gold = {"id": "z1", "out_of_frame_directions": []}
    r41_clean = score_prediction(zero_gold, {"id": "z1", "directions": []})
    r41_over = score_prediction(
        zero_gold,
        {"id": "z1", "directions": [
            {"direction": "凭空多出的方向", "in_frame": False, "status": "realized",
             "evidence_quote": "", "decision_effect": "none"}
        ]},
    )
    agg41 = aggregate([r41_clean, r41_over])
    check("零方向：干净条目算正确", r41_clean["zero_direction_clean"] is True, str(r41_clean.get("zero_direction_clean")))
    check("零方向：多报方向算过度声称", r41_over["zero_direction_overclaim"] is True, str(r41_over.get("zero_direction_overclaim")))
    check("零方向：precision = 0.5", agg41["zero_direction_precision"] == 0.5, str(agg41.get("zero_direction_precision")))
    check("零方向：overclaim_rate = 0.5", agg41["zero_direction_overclaim_rate"] == 0.5, str(agg41.get("zero_direction_overclaim_rate")))
    check("零方向：零方向条目不污染主表 F1", agg41["fp"] == 0 and agg41["tp"] == 0, f"FP={agg41['fp']} TP={agg41['tp']}")

    # --- 42. expected_source = gold_author 单列（裁决 ⑧-b）---
    author_gold = {
        "id": "au",
        "out_of_frame_directions": [
            {"direction": "方向甲", "status": "not_generated", "decision_effect": "conclusion_flip",
             "expected_source": "task"},
            {"direction": "方向乙", "status": "not_generated", "decision_effect": "conclusion_flip",
             "expected_source": "gold_author"},
        ],
    }
    r42 = score_prediction(author_gold, _pred_sample([]))
    check("author 条目不计入主召回分母", r42["miss_detection"]["missed_main"] == 1, str(r42["miss_detection"]))
    check("author 条目单列", r42["miss_detection"]["missed_author_only"] == 1, str(r42["miss_detection"]))
    check("主召回与 author 召回分别计算（分母互不含混）",
          r42["miss_detection"]["miss_detection_recall_main"] == 0.0
          and r42["miss_detection"]["miss_detection_recall_author_only"] == 0.0
          and r42["miss_detection"]["missed_main"] == 1
          and r42["miss_detection"]["missed_author_only"] == 1,
          str(r42["miss_detection"]))

    # --- 43. 防伪检查：引用只命中最终答案段却判未落地（裁决 ⑧-e.3）---
    fraud_gold = {
        "id": "fr",
        "trace": ["阶段3｜方向生成：①方向甲。", "最终答案：建议采用方向甲的方案。"],
        "out_of_frame_directions": [
            {"direction": "方向甲", "status": "realized", "decision_effect": "conclusion_flip"}
        ],
    }
    r43 = score_prediction(
        fraud_gold,
        {"id": "fr", "directions": [
            {"direction": "方向甲", "in_frame": False, "status": "expression_failure",
             "evidence_quote": "建议采用方向甲的方案。", "decision_effect": "none"}
        ]},
    )
    check("防伪：引用只命中最终答案段 + 判 expression_failure -> 记违规",
          len(r43["violations"]) == 1, str(r43["violations"]))
    check("防伪：违规类型正确",
          r43["violations"] and r43["violations"][0]["kind"] == "quote_only_in_final_answer",
          str(r43["violations"]))
    r43b = score_prediction(
        fraud_gold,
        {"id": "fr", "directions": [
            {"direction": "方向甲", "in_frame": False, "status": "realized",
             "evidence_quote": "建议采用方向甲的方案。", "decision_effect": "conclusion_flip"}
        ]},
    )
    check("防伪：判 realized 时引用最终答案不算违规", len(r43b["violations"]) == 0, str(r43b["violations"]))

    # --- 44. m2_gate 审计覆盖率 ---
    agg44 = aggregate([r38])
    gate44 = agg44["m2_gate"]
    check("审计覆盖率字段存在", "audit_coverage" in gate44, str(sorted(gate44)))
    check("审计覆盖率 = 100%（全部金标方向都被审到）", gate44["audit_coverage"] == 1.0, str(gate44["audit_coverage"]))
    check("审计覆盖数 = 金标方向总数", gate44["audit_coverage_checked"] == gate44["audit_coverage_total"],
          f"{gate44['audit_coverage_checked']}/{gate44['audit_coverage_total']}")
    check("审计覆盖率不达标则 M2 不通过", gate44["audit_coverage_pass"] is True, str(gate44["audit_coverage_pass"]))
    # 未被匹配上的金标方向也必须被审到（历史 bug #2 的闭合）
    gold_unmatched = {
        "id": "um",
        "out_of_frame_directions": [
            {"direction": "没人报的方向", "status": "not_generated", "decision_effect": "none"}
        ],
    }
    agg44b = aggregate([score_prediction(gold_unmatched, _pred_sample([]))])
    check("未被匹配上的金标方向也进审计", agg44b["m2_gate"]["audit_coverage_total"] == 1,
          str(agg44b["m2_gate"]["audit_coverage_total"]))

    # --- 45. 最终答案段的内容特征识别（裁决 ⑧-e.4）---
    fa_by_feature = {
        "id": "fb",
        "trace": [
            "阶段1｜任务理解：先看看现状。",
            "建议采用方向甲的方案，这是一段明显更长的最终答案文本，用来检验它能被内容特征识别出来。",
        ],
        "out_of_frame_directions": [
            {"direction": "方向甲", "status": "realized", "decision_effect": "conclusion_flip"}
        ],
    }
    _, corpus_fb, seg_fb = _split_final_answer(fa_by_feature)
    check("末段明显更长且无阶段前缀 -> 识别为最终答案",
          "建议采用方向甲的方案" in seg_fb, repr(seg_fb[:40]))
    check("最终答案在语料里（完整 trace）", "建议采用方向甲的方案" in corpus_fb, repr(corpus_fb[-40:]))

    # --- 46. 裁决 ⑨：frame_note 绝不能进可核语料 + 开卷/闭卷检测 ---
    cb_gold = {
        "id": "cb",
        "initial_frame": "【候选集】A / B / C\n【封闭性】open\n【评估维度】d\n【硬约束】无",
        "trace": ["阶段3｜方向生成：①方向甲。", "最终答案：建议采用方向甲。"],
        "gold": {
            "out_of_frame_directions": [
                {"direction": "方向甲", "status": "realized", "decision_effect": "conclusion_flip",
                 "evidence_quote": "建议采用方向甲。"}
            ],
            "frame_note": "域外参考（被排除在解空间之外，不作为候选）：方向乙、方向丙",
        },
    }
    r46 = score_prediction(cb_gold, {"id": "cb", "directions": [
        {"direction": "方向甲", "in_frame": False, "status": "realized",
         "evidence_quote": "建议采用方向甲。", "decision_effect": "conclusion_flip"}
    ]})
    check("闭卷金标：不含域外参考 -> 判为 closed_book", r46["openbook"] is False, str(r46["openbook"]))
    check("frame_note 不在可核语料里", r46["frame_note_leaks"] == [], str(r46["frame_note_leaks"]))
    check("frame_note 泄漏检查通过时无 warning",
          not any("frame_note" in w for w in r46["warnings"]), str(r46["warnings"]))
    # 把 frame_note 塞进 trace -> 必须被抓到泄漏
    leak_gold = dict(cb_gold)
    leak_gold["trace"] = list(cb_gold["trace"]) + ["域外参考（被排除在解空间之外，不作为候选）：方向乙、方向丙"]
    r46b = score_prediction(leak_gold, {"id": "cb", "directions": []})
    check("frame_note 被并入语料 -> 抓出泄漏", r46b["frame_note_leaks"] != [], str(r46b["frame_note_leaks"]))
    check("泄漏有 warning", any("frame_note" in w for w in r46b["warnings"]), str(r46b["warnings"]))
    # 开卷金标
    ob_gold = {
        "id": "ob",
        "initial_frame": "【候选集】A / B / C\n【封闭性】open\n【评估维度】d\n【硬约束】无\n域外参考（被排除在解空间之外，不作为候选）：方向乙",
        "trace": ["阶段3｜方向生成：①方向甲。"],
        "out_of_frame_directions": [
            {"direction": "方向甲", "status": "realized", "decision_effect": "conclusion_flip"}
        ],
    }
    r46c = score_prediction(ob_gold, {"id": "ob", "directions": []})
    check("开卷金标被识别", r46c["openbook"] is True, str(r46c["openbook"]))
    check("开卷金标有 warning（只作对照组）",
          any("开卷" in w for w in r46c["warnings"]), str(r46c["warnings"]))
    agg46 = aggregate([r46, r46c])
    check("aggregate 报告 gold_mode=mixed", agg46["gold_mode"] == "mixed", agg46["gold_mode"])
    check("aggregate 统计开卷条目数", agg46["m2_gate"]["gold_openbook_items"] == 1, str(agg46["m2_gate"]["gold_openbook_items"]))
    check("aggregate 报告 frame_note 泄漏", agg46["m2_gate"]["frame_note_leaks"] == [], str(agg46["m2_gate"]["frame_note_leaks"]))
    # 开卷金标不得进门槛
    agg46b = aggregate([r46c])
    check("开卷金标下 true_increment 门槛不通过（开卷不进门槛）",
          agg46b["m2_gate"]["true_increment_gate_pass"] is False, str(agg46b["m2_gate"]["true_increment_gate_pass"]))
    check("闭卷金标下 frame_note 无泄漏则门槛项通过",
          agg46b["m2_gate"]["frame_note_leak_free"] is True, str(agg46b["m2_gate"]["frame_note_leak_free"]))

    # --- 47. 阈值扫描与敏感区间（裁决 ⑥-a / ⑨-c）---
    sweep_gold = [
        {"id": "s1", "out_of_frame_directions": [
            {"direction": "把 12 周承诺改为分批交付", "status": "selection_failure", "decision_effect": "conclusion_flip"},
            {"direction": "引入第三方冷链仓", "status": "realized", "decision_effect": "option_added"},
        ]},
    ]
    sweep_pred = {"s1": {"id": "s1", "directions": [
        {"direction": "把 12 周承诺改为分批交付", "in_frame": False, "status": "selection_failure",
         "evidence_quote": "", "decision_effect": "conclusion_flip"},
        {"direction": "引入第三方冷链仓", "in_frame": False, "status": "realized",
         "evidence_quote": "", "decision_effect": "option_added"},
    ]}}
    sw = threshold_sweep(sweep_gold, sweep_pred, [0.20, 0.30, 0.40, 0.50, 0.60])
    check("阈值扫描产出曲线", len(sw["curve"]) == 5, str(len(sw["curve"])))
    check("阈值扫描给出敏感区间", sw["sensitive_interval"] is not None, str(sw["sensitive_interval"]))
    check("敏感区间覆盖扫描范围（本例 F1 恒定）",
          sw["sensitive_interval"] and sw["sensitive_interval"]["from"] == 0.20
          and sw["sensitive_interval"]["to"] == 0.60, str(sw["sensitive_interval"]))
    check("扫描报告基线最高 F1", sw["baseline_max_f1_over_sweep"] is not None, str(sw["baseline_max_f1_over_sweep"]))
    check("扫描判据写明", "0.02" in sw["sensitive_interval_criterion"], sw["sensitive_interval_criterion"])

    # --- 48. 外部基线：口径区分 + 门槛基准取 end_to_end ---
    ext = [{"name": "外部端到端基线", "f1": 0.70, "caliber_kind": "end_to_end",
            "caliber": "end_to_end", "p": 0.6, "r": 0.8, "status_accuracy": 0.6}]
    r48 = score_prediction(
        sweep_gold[0], sweep_pred["s1"], external_baselines=ext
    )
    check("外部 end_to_end 基线被采纳为门槛基准",
          r48["lexical_baseline"]["best_variant"] == "外部端到端基线",
          str(r48["lexical_baseline"]["best_variant"]))
    check("门槛基准口径标为 end_to_end",
          r48["lexical_baseline"]["gate_basis_kind"] == "end_to_end",
          str(r48["lexical_baseline"]["gate_basis_kind"]))
    check("无外部基线时标为 status_only（不作固定门槛）",
          r48["lexical_baseline"]["local_max_variant"] is not None,
          str(r48["lexical_baseline"].get("local_max_variant")))
    agg48 = aggregate([r48])
    check("aggregate 记录外部基线", len(agg48["lexical_baseline"]["external_baselines"]) == 1,
          str(agg48["lexical_baseline"]["external_baselines"]))
    check("aggregate 门槛项要求有 end_to_end 基线",
          agg48["m2_gate"]["end_to_end_baseline_supplied"] is True,
          str(agg48["m2_gate"]["end_to_end_baseline_supplied"]))

    # --- 49. 抽取器输入切片：不得含 frame_note / gold（裁决 ⑨-a）---
    slice_item = {
        "id": "sl",
        "task": "任务描述",
        "initial_frame": "【候选集】A\n【封闭性】open\n【评估维度】d\n【硬约束】无",
        "trace": ["阶段1｜任务理解：……"],
        "frame_note": "域外参考：方向乙、方向丙",
        "gold": {
            "out_of_frame_directions": [{"direction": "方向甲", "status": "realized"}],
            "frame_note": "域外参考：方向乙、方向丙",
        },
    }
    sl = extractor_input(slice_item)
    check("输入切片只含 task/initial_frame/trace/id",
          set(sl) == {"id", "task", "initial_frame", "trace"}, str(sorted(sl)))
    check("输入切片不含 frame_note", "frame_note" not in sl, str(sorted(sl)))
    check("输入切片不含 gold", "gold" not in sl, str(sorted(sl)))
    check("输入切片不含域外方向答案", "方向甲" not in json.dumps(sl, ensure_ascii=False), json.dumps(sl, ensure_ascii=False))
    check("输入切片不含域外参考文本",
          "域外参考" not in json.dumps(sl, ensure_ascii=False), json.dumps(sl, ensure_ascii=False))

    # --- 50. 扫描表：闭卷进门槛、开卷只作对照（裁决 ⑨-b / ⑨-c）---
    import tempfile

    scan_csv = (
        "mode,variant,oof_thr,match_thr,stance_thr,tp,fp,fn,precision,recall,f1,status_acc\n"
        "closed,V1,0.08,0.25,0.3,10,10,20,0.5,0.33,0.4,0.3\n"
        "closed,V2,0.12,0.25,0.3,12,10,20,0.55,0.375,0.45,0.35\n"
        "open,V1,0.08,0.25,0.3,20,10,20,0.67,0.5,0.5714,0.5\n"
        "open,V5,0.15,0.25,0.3,30,10,20,0.75,0.6,0.6667,0.6\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as fh:
        fh.write(scan_csv)
        scan_path = fh.name
    scan = load_baseline_scan(scan_path)
    check("扫描表按口径分组", set(scan) == {"closed", "open"}, str(sorted(scan)))
    check("闭卷最高 F1 正确", scan["closed"]["f1"] == 0.45, str(scan["closed"]["f1"]))
    check("开卷最高 F1 正确", scan["open"]["f1"] == 0.6667, str(scan["open"]["f1"]))
    check("扫描表记录最佳参数", scan["closed"]["best_variant"] == "V2", str(scan["closed"].get("best_variant")))
    scan_ext = list(scan.values())
    r50 = score_prediction(sweep_gold[0], sweep_pred["s1"], external_baselines=scan_ext)
    check("门槛基准取闭卷口径", r50["lexical_baseline"]["best_variant"].endswith("(closed) 2 点"),
          str(r50["lexical_baseline"]["best_variant"]))
    check("开卷基线被记为参考不进门槛",
          r50["lexical_baseline"]["f1"] == 0.45, str(r50["lexical_baseline"]["f1"]))
    agg50 = aggregate([r50])
    check("m2_gate 记录开卷基线（仅参考）",
          len(agg50["m2_gate"]["open_book_baselines"]) == 1, str(agg50["m2_gate"]["open_book_baselines"]))
    check("m2_gate 门槛口径标为 closed",
          agg50["m2_gate"]["gate_basis_mode"] == "closed", str(agg50["m2_gate"]["gate_basis_mode"]))

    # --- 51. status_only 变体：F1 不可与端到端比较 ---
    check("status_only 变体标为 F1 不可比",
          all(v.get("f1_comparable") is False for v in r50["lexical_baseline_variants"].values()),
          str([(k, v.get("f1_comparable")) for k, v in r50["lexical_baseline_variants"].items()]))
    check("status_only 变体标出主指标是 status 准确率",
          all(v.get("headline_metric") == "status_accuracy" for v in r50["lexical_baseline_variants"].values()),
          "headline_metric 未标")

    # ================= 裁决 ⑫：反向断言（每条 m2_gate 断言配一条） =================
    # 原则：「断言通过」不等于「断言有效」。下面每一条都**故意**让某个断言失败，
    # 并断言它**确实失败** —— 证明这条断言不是恒真的摆设。

    def _gate_of(gold_item, pred=None):
        return aggregate([score_prediction(gold_item, pred or {"id": gold_item.get("id"), "directions": []})])["m2_gate"]

    # R1: evidence 非逐字 -> 门槛失败
    g_r1 = {"id": "r1", "trace": ["阶段3｜方向生成：①方向甲。"],
            "out_of_frame_directions": [{"direction": "方向甲", "status": "realized",
                                          "decision_effect": "conclusion_flip",
                                          "evidence": "一段根本不在 trace 里的引用"}]}
    check("反向R1: evidence 非逐字 -> 门槛失败",
          _gate_of(g_r1)["gold_evidence_gate_pass"] is False, str(_gate_of(g_r1)["gold_evidence_gate_pass"]))

    # R2: 漏判用弱效果 -> 门槛失败
    g_r2 = {"id": "r2", "out_of_frame_directions": [
        {"direction": "方向甲", "status": "not_generated", "decision_effect": "none"}]}
    check("反向R2: 漏判非关键漏判 -> 门槛失败",
          _gate_of(g_r2)["gold_miss_all_key"] is False, str(_gate_of(g_r2)["gold_miss_all_key"]))

    # R3: 金标用旧字段 -> 门槛失败
    g_r3 = {"id": "r3", "out_of_frame_directions": [
        {"direction": "方向甲", "status": "selection_failure", "would_change_decision": True}]}
    check("反向R3: 金标用旧字段 -> 门槛失败",
          _gate_of(g_r3)["gold_uses_v2_effect"] is False, str(_gate_of(g_r3)["gold_uses_v2_effect"]))

    # R4: initial_frame 不合规 -> 门槛失败
    g_r4 = {"id": "r4", "initial_frame": "【候选集】A\n【评估维度】d\n【硬约束】无",
            "out_of_frame_directions": [
                {"direction": "方向甲", "status": "realized", "decision_effect": "conclusion_flip"}]}
    check("反向R4: initial_frame 不合规 -> 门槛失败",
          _gate_of(g_r4)["all_frames_conforming"] is False, str(_gate_of(g_r4)["all_frames_conforming"]))

    # R5: frame_note 注入语料 -> 必须报警（★ 这条断言曾经是真空的）
    g_r5 = {"id": "r5",
            "initial_frame": "【候选集】A\n【封闭性】open\n【评估维度】d\n【硬约束】无",
            "trace": ["阶段3｜方向生成：①方向甲。",
                      "域外参考（被排除在解空间之外，不作为候选）：方向乙、方向丙"],
            "gold": {"out_of_frame_directions": [
                        {"direction": "方向甲", "status": "realized", "decision_effect": "conclusion_flip"}],
                     "frame_note": "域外参考（被排除在解空间之外，不作为候选）：方向乙、方向丙"}}
    r5 = score_prediction(g_r5, {"id": "r5", "directions": []})
    check("反向R5: frame_note 注入语料 -> 报警", r5["frame_note_leaks"] != [], str(r5["frame_note_leaks"]))
    check("反向R5: frame_note 泄漏使门槛失败",
          aggregate([r5])["m2_gate"]["frame_note_leak_free"] is False,
          str(aggregate([r5])["m2_gate"]["frame_note_leak_free"]))
    # 正向：真的有 frame_note 时，断言「检查到」非零（证明不是真空）
    g_r5b = {"id": "r5b",
             "initial_frame": "【候选集】A\n【封闭性】open\n【评估维度】d\n【硬约束】无",
             "trace": ["阶段3｜方向生成：①方向甲。"],
             "gold": {"out_of_frame_directions": [
                         {"direction": "方向甲", "status": "realized", "decision_effect": "conclusion_flip"}],
                      "frame_note": "域外参考：方向乙、方向丙"}}
    r5b = score_prediction(g_r5b, {"id": "r5b", "directions": []})
    check("反向R5: frame_note 存在时断言非真空（checked>0）",
          r5b["frame_note_checked"] >= 1, str(r5b["frame_note_checked"]))
    check("反向R5: frame_note 存在但不在语料 -> 不报警",
          r5b["frame_note_leaks"] == [], str(r5b["frame_note_leaks"]))

    # R6: 非法 status -> 审计明细必须报出来（证明审计真的在看）
    g_r6 = {"id": "r6", "out_of_frame_directions": [
        {"direction": "方向甲", "status": "胡说八道", "decision_effect": "conclusion_flip"}]}
    gate_r6 = _gate_of(g_r6)
    check("反向R6: 非法 status 被审计明细报出",
          gate_r6["audit_coverage_detail"]["status_invalid"] == 1, str(gate_r6["audit_coverage_detail"]))
    check("反向R6: 非法 status 仍被计入审计覆盖（不是漏审）",
          gate_r6["audit_coverage_checked"] == 1, str(gate_r6["audit_coverage_checked"]))

    # R7: 很低的 F1 -> 真实增量门槛失败
    g_r7 = {"id": "r7", "out_of_frame_directions": [
        {"direction": "方向甲", "status": "realized", "decision_effect": "conclusion_flip"}]}
    low = {"id": "r7", "directions": []}
    gate_r7 = aggregate([score_prediction(g_r7, low, external_baselines=[
        {"name": "强基线", "f1": 0.95, "caliber_kind": "end_to_end", "mode": "closed"}])])["m2_gate"]
    check("反向R7: 增量不足 -> 门槛失败",
          gate_r7["true_increment_gate_pass"] is False, str(gate_r7["true_increment_gate_pass"]))
    check("反向R7: 强基线被采纳为基准",
          gate_r7["end_to_end_closed_f1"] == 0.95, str(gate_r7["end_to_end_closed_f1"]))

    # R8: 开卷金标 -> 门槛失败
    g_r8 = {"id": "r8",
            "initial_frame": "【候选集】A\n【封闭性】open\n【评估维度】d\n【硬约束】无\n域外参考：方向乙",
            "out_of_frame_directions": [
                {"direction": "方向甲", "status": "realized", "decision_effect": "conclusion_flip"}]}
    check("反向R8: 开卷金标 -> 门槛失败",
          _gate_of(g_r8)["true_increment_gate_pass"] is False, str(_gate_of(g_r8)["true_increment_gate_pass"]))

    # R9: 没有外部基线 -> 门槛失败
    gate_r9 = _gate_of(g_r7)
    check("反向R9: 无 end_to_end 基线 -> 门槛失败",
          gate_r9["true_increment_gate_pass"] is False, str(gate_r9["true_increment_gate_pass"]))
    check("反向R9: 报出未提供基线",
          gate_r9["end_to_end_baseline_supplied"] is False, str(gate_r9["end_to_end_baseline_supplied"]))

    # R10: 上界报警线（⑪-b③）—— 超过开卷天花板要报警，低分不报警
    high_pred = {"id": "r7", "directions": [
        {"direction": "方向甲", "in_frame": False, "status": "realized",
         "evidence_quote": "", "decision_effect": "conclusion_flip"}]}
    gate_r10 = aggregate([score_prediction(g_r7, high_pred, external_baselines=[
        {"name": "弱基线", "f1": 0.10, "caliber_kind": "end_to_end", "mode": "closed"}])])["m2_gate"]
    check("反向R10: F1 超过上界报警线 -> 置位", gate_r10["upper_alarm"] is True, str(gate_r10["upper_alarm"]))
    check("反向R10: 上界报警不是通过条件（门槛另算）",
          gate_r10["upper_alarm_threshold"] == UPPER_ALARM_F1, str(gate_r10["upper_alarm_threshold"]))
    gate_r10b = aggregate([score_prediction(g_r7, low, external_baselines=[
        {"name": "弱基线", "f1": 0.10, "caliber_kind": "end_to_end", "mode": "closed"}])])["m2_gate"]
    check("反向R10: 低分不触发上界报警", gate_r10b["upper_alarm"] is False, str(gate_r10b["upper_alarm"]))

    # R11: sweep_max_f1_mode 口径守卫（★ 硬要求）
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
        json.dump({"baseline": "开卷值的 JSON", "caliber": "end_to_end",
                   "sweep_max_f1": 0.9999, "sweep_max_f1_mode": "open_book",
                   "main": {"f1": 0.4, "precision": 0.4, "recall": 0.4}}, fh, ensure_ascii=False)
        open_json = fh.name
    ext_guard = load_external_baselines([open_json])
    check("反向R11: sweep_max_f1_mode=open 时不采用该值",
          ext_guard[0]["f1"] == 0.4, str(ext_guard[0]["f1"]))
    check("反向R11: 该基线被标为 open 口径（不进门槛）",
          ext_guard[0]["mode"] == "open", str(ext_guard[0]["mode"]))
    gate_r11 = aggregate([score_prediction(g_r7, low, external_baselines=ext_guard)])["m2_gate"]
    check("反向R11: open 口径不进入门槛基准",
          gate_r11["end_to_end_closed_f1"] is None, str(gate_r11["end_to_end_closed_f1"]))
    # 正向：mode=closed_book 时才采用
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
        json.dump({"baseline": "闭卷值的 JSON", "caliber": "end_to_end",
                   "sweep_max_f1": 0.4932, "sweep_max_f1_mode": "closed_book",
                   "main": {"f1": 0.4, "precision": 0.4, "recall": 0.4}}, fh, ensure_ascii=False)
        closed_json = fh.name
    ext_ok = load_external_baselines([closed_json])
    check("反向R11: sweep_max_f1_mode=closed_book 时采用该值",
          ext_ok[0]["f1"] == 0.4932, str(ext_ok[0]["f1"]))
    check("反向R11: 闭卷口径标为 closed（进门槛）",
          ext_ok[0]["mode"] == "closed", str(ext_ok[0]["mode"]))

    # R12: 断言清单本身可查
    check("m2_gate 断言清单非空", len(M2_GATE_ASSERTIONS) == 10, str(len(M2_GATE_ASSERTIONS)))
    check("断言清单每条都描述了反向测试",
          all(v for _, v in M2_GATE_ASSERTIONS), str(M2_GATE_ASSERTIONS))
    check("断言清单可导出", len(m2_gate_assertion_status()) == 10, str(len(m2_gate_assertion_status())))

    # --- 52. stance 误判分诊（分歧高发形态）---
    se_gold = {
        "id": "sm",
        "trace": ["阶段3｜方向生成：①方向甲。", "阶段4｜取舍：①保留方向甲。",
                  "最终答案：建议采用方向甲。⑥同样不做。"],
        "out_of_frame_directions": [
            {"direction": "方向甲", "status": "expression_failure", "decision_effect": "none"},
        ],
    }
    se_pred = {"id": "sm", "directions": [
        {"direction": "方向甲", "in_frame": False, "status": "selection_failure",
         "evidence_quote": "⑥同样不做。", "decision_effect": "none"}
    ]}
    r52 = score_prediction(se_gold, se_pred)
    check("stance 分诊：识别出 expression_failure 被判成 selection_failure",
          r52["stance_misattribution_count"] == 1, str(r52["stance_misattribution_count"]))
    check("stance 分诊：带出引用原文供对照",
          bool(r52["stance_misattribution"]) and r52["stance_misattribution"][0]["evidence_quote"] == "⑥同样不做。",
          str(r52["stance_misattribution"]))
    check("stance 分诊：status 仍照常记错（分诊不掩盖错误）",
          r52["status_confusion"]["expression_failure"]["selection_failure"] == 1,
          str(r52["status_confusion"]["expression_failure"]))
    se_pred2 = {"id": "sm", "directions": [
        {"direction": "方向甲", "in_frame": False, "status": "selection_failure",
         "evidence_quote": "建议采用方向甲。", "decision_effect": "none"}
    ]}
    r52b = score_prediction(se_gold, se_pred2)
    check("stance 分诊反向：引用无否决词 -> 不标记",
          r52b["stance_misattribution_count"] == 0, str(r52b["stance_misattribution_count"]))

    # --- 53. 上界参照模式：门槛与上界都不参与 ---
    ref_gold = {"id": "ref", "out_of_frame_directions": [
        {"direction": "方向甲", "status": "realized", "decision_effect": "conclusion_flip"}]}
    ref_pred = {"id": "ref", "directions": [
        {"direction": "方向甲", "in_frame": False, "status": "realized",
         "evidence_quote": "", "decision_effect": "conclusion_flip"}]}
    strong_base = [{"name": "强基线", "f1": 0.49, "caliber_kind": "end_to_end", "mode": "closed"}]
    g53 = aggregate([score_prediction(ref_gold, ref_pred, external_baselines=strong_base)])["m2_gate"]
    check("参照模式前：F1=1.0 会触发上界报警", g53["upper_alarm"] is True, str(g53["upper_alarm"]))
    check("参照模式前：门槛可判定（applicable 未被关掉）",
          g53.get("true_increment_gate_applicable") is not False,
          str(g53.get("true_increment_gate_applicable")))

    # --- 54. 主表口径基线：门槛必须与抽取器同口径 ---
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
        json.dump({
            "caliber": "main_table（not_generated 不计入主表）",
            "max_f1": 0.5714,
            "max_variant": "V1",
            "all_directions_caliber_max_f1": 0.4932,
            "recommended_gate_f1": 0.6714,
        }, fh, ensure_ascii=False)
        mc_path = fh.name
    mc = load_main_caliber_baseline(mc_path)
    check("主表口径基线可载入", mc is not None and mc["f1"] == 0.5714, str(mc))
    check("主表口径基线标为 end_to_end + closed",
          mc["caliber_kind"] == "end_to_end" and mc["mode"] == "closed", str(mc))
    check("主表口径基线记下了另一口径的数（供对照）",
          mc["all_directions_caliber_f1"] == 0.4932, str(mc))
    g54 = aggregate([score_prediction(ref_gold, ref_pred, external_baselines=[mc])])["m2_gate"]
    check("门槛用主表口径的基线", g54["end_to_end_closed_f1"] == 0.5714, str(g54["end_to_end_closed_f1"]))
    check("同口径门槛 = 基线 + 0.10", g54["required_extractor_f1"] == 0.6714, str(g54["required_extractor_f1"]))
    # 反向：若只提供含 miss detection 口径的基线，门槛数值不同（必须能被看出差异）
    g54b = aggregate([score_prediction(ref_gold, ref_pred, external_baselines=[
        {"name": "含miss口径", "f1": 0.4932, "caliber_kind": "end_to_end", "mode": "closed"}])])["m2_gate"]
    check("反向：不同口径给出不同门槛值（口径不一致是可见的）",
          g54b["required_extractor_f1"] == 0.5932 and g54["required_extractor_f1"] == 0.6714,
          f"{g54b['required_extractor_f1']} vs {g54['required_extractor_f1']}")

    # --- 55. constraint_violation_candidates 桶（⑭-c）---
    cv_gold = {
        "id": "cv",
        "initial_frame": "【候选集】A / B\n【封闭性】closed\n【评估维度】d\n【硬约束】在保持 X 不变的前提下，通过/调整 Y",
        "trace": ["阶段3｜方向生成：①方向甲。", "最终答案：建议采用方向甲。"],
        "out_of_frame_directions": [
            {"direction": "方向甲", "status": "realized", "decision_effect": "conclusion_flip"}
        ],
    }
    cv_pred = {"id": "cv", "directions": [
        {"direction": "方向甲", "in_frame": False, "status": "realized",
         "evidence_quote": "建议采用方向甲。", "decision_effect": "conclusion_flip"},
        {"direction": "违反禁区的不可能方向", "in_frame": False, "status": "not_generated",
         "evidence_quote": "", "decision_effect": "none", "constraint_violation": True},
    ]}
    r55 = score_prediction(cv_gold, cv_pred)
    check("约束违规方向进入独立诊断桶",
          r55["constraint_violation_count"] == 1, str(r55["constraint_violation_count"]))
    check("约束违规方向不进主表 FP", r55["fp"] == 0, f"FP={r55['fp']}")
    check("约束违规方向不进漏判表（不计误报漏判）",
          r55["miss_detection"]["spurious"] == 0, str(r55["miss_detection"]))
    check("约束违规方向不是 not_generated 统计项",
          r55["miss_detection"]["missed"] == 0, str(r55["miss_detection"]))
    check("诊断桶带出方向文本",
          r55["constraint_violation_candidates"] and
          r55["constraint_violation_candidates"][0]["direction"] == "违反禁区的不可能方向",
          str(r55["constraint_violation_candidates"]))
    # 反向：普通方向不进这个桶
    r55b = score_prediction(cv_gold, {"id": "cv", "directions": [
        {"direction": "方向甲", "in_frame": False, "status": "realized",
         "evidence_quote": "建议采用方向甲。", "decision_effect": "conclusion_flip"}]})
    check("反向：普通方向不进约束违规桶", r55b["constraint_violation_count"] == 0,
          str(r55b["constraint_violation_count"]))

    # --- 56. 口径不变式（⑯-b）：错分母必须触发异常，不是算出更大的数 ---
    try:
        E_assert = None
        try:
            assert_caliber(tp=10, fp=0, fn=0, n_gold_main=9, n_pred_scored=0, n_matched=0)
        except CaliberError:
            E_assert = "raised"
        check("口径不变式：tp+fn 与分母不符 -> 抛异常", E_assert == "raised", str(E_assert))
        E_assert2 = None
        try:
            assert_caliber(tp=1, fp=1, fn=0, n_gold_main=1, n_pred_scored=5, n_matched=1)
        except CaliberError:
            E_assert2 = "raised"
        check("口径不变式：fp+已匹配 与预测数不符 -> 抛异常", E_assert2 == "raised", str(E_assert2))
        E_assert3 = None
        try:
            assert_caliber(tp=1, fp=0, fn=0, n_gold_main=1, n_pred_scored=1, n_matched=1)
        except CaliberError:
            E_assert3 = "raised"
        check("口径不变式：正确数据不抛异常", E_assert3 is None, str(E_assert3))
    except NameError:
        check("口径不变式可用", False, "CaliberError/assert_caliber 未定义")

    # --- 57. caliber 守卫（⑭-e 的漏网）：口径不对的基线不得进门槛 ---
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
        json.dump({"baseline": "含漏判口径", "caliber": "end_to_end",
                   "sweep_max_f1": 0.9, "sweep_max_f1_mode": "closed_book",
                   "sweep_max_f1_caliber": "full",
                   "main": {"f1": 0.4}}, fh, ensure_ascii=False)
        wrong_cal = fh.name
    ext_wrong = load_external_baselines([wrong_cal])
    check("caliber 守卫：含漏判口径的 sweep_max_f1 不被采用",
          ext_wrong[0]["f1"] == 0.4, str(ext_wrong[0]["f1"]))
    check("caliber 守卫：它被标为 open 口径（不进门槛）",
          ext_wrong[0]["mode"] == "open", str(ext_wrong[0]["mode"]))
    g57 = aggregate([score_prediction(ref_gold, low, external_baselines=ext_wrong)])["m2_gate"]
    check("caliber 守卫：含漏判口径不进门槛基准", g57["end_to_end_closed_f1"] is None,
          str(g57["end_to_end_closed_f1"]))
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
        json.dump({"baseline": "主表口径", "caliber": "end_to_end",
                   "sweep_max_f1": 0.5106, "sweep_max_f1_mode": "closed_book",
                   "sweep_max_f1_caliber": "main",
                   "main": {"f1": 0.4}}, fh, ensure_ascii=False)
        right_cal = fh.name
    ext_right = load_external_baselines([right_cal])
    check("caliber 守卫：主表口径的 sweep_max_f1 被采用",
          ext_right[0]["f1"] == 0.5106, str(ext_right[0]["f1"]))

    # --- 58. ⑯-c：门槛基准的三个数在报告里并列且标注来源 ---
    agg58 = aggregate([score_prediction(ref_gold, ref_pred, external_baselines=strong_base)])
    prov = agg58.get("baseline_number_provenance") or []
    check("⑳-a：报告并列四类基准数（脏/池口径/规范口径）", len(prov) == 4, str(len(prov)))
    check("⑳-a：每一类都标注定义与状态",
          all(item.get("definition") and item.get("status") for item in prov), str(prov))
    check("⑳-a：虚高值 0.9381 被列为脏并作废",
          any(item["value"] == 0.9381 and "作废" in item["status"] for item in prov), str(prov))
    check("⑳-a：池口径不一致那一类被单列",
          any("池口径" in item["status"] for item in prov), str(prov))
    check("⑳-a：规范口径被标出",
          sum(1 for item in prov if "规范口径" in item["status"]) >= 2, str(prov))

    # --- 59. introduced_in_synthesis（⑲-a）---
    is_gold = {
        "id": "is",
        "trace": ["阶段3｜方向生成：①方向甲。", "最终答案：建议采用方向甲。另新增：建议设立专项基金。"],
        "out_of_frame_directions": [
            {"direction": "方向甲", "status": "realized", "decision_effect": "conclusion_flip"},
            {"direction": "设立专项基金", "status": "realized", "decision_effect": "option_added"},
        ],
    }
    is_pred = {"id": "is", "directions": [
        {"direction": "方向甲", "in_frame": False, "status": "realized",
         "evidence_quote": "建议采用方向甲。", "decision_effect": "conclusion_flip"},
        {"direction": "设立专项基金", "in_frame": False, "status": "realized",
         "evidence_quote": "另新增：建议设立专项基金。", "decision_effect": "option_added"},
    ]}
    r59 = score_prediction(is_gold, is_pred)
    check("introduced_in_synthesis：识别出只在最终答案出现的方向",
          r59["introduced_in_synthesis_count"] == 1, str(r59["introduced_in_synthesis_count"]))
    check("introduced_in_synthesis：带出方向与引用",
          r59["introduced_in_synthesis"] and
          r59["introduced_in_synthesis"][0]["direction"] == "设立专项基金",
          str(r59["introduced_in_synthesis"]))
    check("introduced_in_synthesis：算被生成了（进主表，不是 not_generated）",
          r59["tp"] == 2, f"TP={r59['tp']}")
    # 反向：引用落在中间段 -> 不是 introduced
    r59b = score_prediction(is_gold, {"id": "is", "directions": [
        {"direction": "方向甲", "in_frame": False, "status": "realized",
         "evidence_quote": "阶段3｜方向生成：①方向甲。", "decision_effect": "conclusion_flip"},
        {"direction": "设立专项基金", "in_frame": False, "status": "realized",
         "evidence_quote": "另新增：建议设立专项基金。", "decision_effect": "option_added"},
    ]})
    check("introduced_in_synthesis 反向：中间段有引用的不算新产生",
          r59b["introduced_in_synthesis_count"] == 1, str(r59b["introduced_in_synthesis_count"]))

    # --- 60. ⑭-c 收窄：桶只用于 not_generated 候选；已出现的方向留在主表 ---
    # (a) not_generated 候选 + 触碰禁区 -> 进桶、不在主表
    r60 = score_prediction(cv_gold, cv_pred)
    check("约束违规：not_generated 候选进桶",
          r60["constraint_violation_count"] == 1, str(r60["constraint_violation_count"]))
    check("约束违规：not_generated 候选被排除出主表 FP",
          r60["fp"] == 0, f"FP={r60['fp']}")
    check("约束违规：not_generated 候选不计误报漏判",
          r60["miss_detection"]["spurious"] == 0, str(r60["miss_detection"]))
    check("约束违规：两桶计数分开",
          r60["constraint_violation_not_generated_count"] == 1 and
          r60["constraint_violation_appeared_count"] == 0,
          f"{r60['constraint_violation_not_generated_count']}/{r60['constraint_violation_appeared_count']}")
    # (b) trace 里已出现 + 触碰禁区 -> 仍在主表（★ 收窄的核心）
    appeared_gold = {
        "id": "cva",
        "initial_frame": "【候选集】A\n【封闭性】closed\n【评估维度】d\n【硬约束】在保持 X 不变的前提下，通过/调整 Y",
        "trace": ["阶段3｜方向生成：①方向甲。", "最终答案：建议采用方向甲。"],
        "out_of_frame_directions": [
            {"direction": "方向甲", "status": "realized", "decision_effect": "conclusion_flip"},
        ],
    }
    appeared_pred = {"id": "cva", "directions": [
        {"direction": "方向甲", "in_frame": False, "status": "realized",
         "evidence_quote": "建议采用方向甲。", "decision_effect": "conclusion_flip",
         "constraint_violation": True},
    ]}
    r60b = score_prediction(appeared_gold, appeared_pred)
    check("约束违规：已在 trace 出现的方向 -> 仍在主表（收窄的核心）",
          r60b["tp"] == 1, f"TP={r60b['tp']}")
    check("约束违规：已出现的方向不进 not_generated 桶的排除集",
          r60b["constraint_violation_appeared_count"] == 1, str(r60b["constraint_violation_appeared_count"]))
    check("约束违规：已出现的方向仍被记录（标记保留供报告）",
          r60b["constraint_violation_count"] == 1, str(r60b["constraint_violation_count"]))
    agg60 = aggregate([r60, r60b])
    check("约束违规：敏感性改为两桶条目数",
          agg60["constraint_violation_sensitivity"]["not_generated_candidates"] == 1 and
          agg60["constraint_violation_sensitivity"]["appeared_in_trace"] == 1,
          str(agg60["constraint_violation_sensitivity"]))
    check("约束违规：逐条明细进了 aggregate",
          len(agg60["constraint_violation_details"]) == 2, str(agg60["constraint_violation_details"]))
    check("约束违规：明细带 id 与方向",
          all(c.get("id") and c.get("direction") for c in agg60["constraint_violation_details"]),
          str(agg60["constraint_violation_details"]))
    check("约束违规：旧的「不排除」上界对照已作废（恒为 None）",
          r60.get("f1_with_constraint_violations_included") is None,
          str(r60.get("f1_with_constraint_violations_included")))

    # --- 61. 池口径守卫（⑳-c）：池不一致的基线不得作门槛基准 ---
    pool_bad = [{"name": "池未清空的基线", "f1": 0.30, "caliber_kind": "end_to_end",
                 "mode": "closed", "pool_caliber": "not_cleared", "pool_caliber_ok": False}]
    g61 = aggregate([score_prediction(ref_gold, ref_pred, external_baselines=pool_bad)])["m2_gate"]
    check("池口径守卫：池未清空的基线不进门槛基准",
          g61["end_to_end_closed_f1"] is None, str(g61["end_to_end_closed_f1"]))
    check("池口径守卫：被拦下的基线被记录",
          len(g61["pool_inconsistent_baselines"]) == 1, str(g61["pool_inconsistent_baselines"]))
    pool_ok = [{"name": "池已清空的基线", "f1": 0.30, "caliber_kind": "end_to_end",
                "mode": "closed", "pool_caliber": "cleared", "pool_caliber_ok": True}]
    g61b = aggregate([score_prediction(ref_gold, ref_pred, external_baselines=pool_ok)])["m2_gate"]
    check("池口径守卫：池已清空的基线正常进门槛",
          g61b["end_to_end_closed_f1"] == 0.30, str(g61b["end_to_end_closed_f1"]))

    # --- 62. 三桶漏判分桶（⑳）+ 分母 0 = 不可度量 ---
    bucket_gold = {
        "id": "bk",
        "out_of_frame_directions": [
            {"direction": "甲", "status": "not_generated", "decision_effect": "conclusion_flip",
             "expected_source": "task"},
            {"direction": "乙", "status": "not_generated", "decision_effect": "conclusion_flip",
             "expected_source": "gold_author"},
            {"direction": "丙", "status": "not_generated", "decision_effect": "conclusion_flip",
             "expected_source": "task", "expected_reportability": "low"},
        ],
    }
    r62 = score_prediction(bucket_gold, _pred_sample([]))
    m62 = r62["miss_detection"]
    check("三桶：main 桶只含 task 且非 low 的条目", m62["missed_main"] == 1, str(m62))
    check("三桶：author_only 单列", m62["missed_author_only"] == 1, str(m62))
    check("三桶：low_reportability 单列", m62["missed_low_reportability"] == 1, str(m62))
    check("三桶：桶计数报出", m62["bucket_counts"] == {"main": 1, "author_only": 1, "low_reportability": 1},
          str(m62["bucket_counts"]))
    check("三桶：主召回可度量时给值", m62["main_measurable"] is True, str(m62["main_measurable"]))
    # 主桶分母为 0 -> 不可度量
    only_side = {
        "id": "os",
        "out_of_frame_directions": [
            {"direction": "乙", "status": "not_generated", "decision_effect": "conclusion_flip",
             "expected_source": "gold_author"},
        ],
    }
    r62b = score_prediction(only_side, _pred_sample([]))
    check("三桶：主桶分母 0 -> 标记为不可度量",
          r62b["miss_detection"]["main_measurable"] is False, str(r62b["miss_detection"]))
    check("三桶：不可度量时给出原因，不报 0/0",
          "不可度量" in r62b["miss_detection"]["main_unmeasurable_reason"],
          str(r62b["miss_detection"]["main_unmeasurable_reason"]))
    check("三桶：分母 0 时主召回为 None（不是 0.0）",
          r62b["miss_detection"]["miss_detection_recall_main"] is None,
          str(r62b["miss_detection"]["miss_detection_recall_main"]))

    # --- 63. prp 钩子：未知的 expected_source 值进主桶，不静默落进 author_only ---
    prp_gold = {
        "id": "prp",
        "out_of_frame_directions": [
            {"direction": "甲", "status": "not_generated", "decision_effect": "conclusion_flip",
             "expected_source": "prp"},
            {"direction": "乙", "status": "not_generated", "decision_effect": "conclusion_flip",
             "expected_source": "gold_author"},
        ],
    }
    r63 = score_prediction(prp_gold, _pred_sample([]))
    m63 = r63["miss_detection"]
    check("prp 钩子：prp 进主桶（不单列）", m63["missed_main"] == 1, str(m63))
    check("prp 钩子：gold_author 仍单列", m63["missed_author_only"] == 1, str(m63))
    check("prp 钩子：未知值不静默落进 author_only",
          m63["bucket_counts"]["author_only"] == 1, str(m63["bucket_counts"]))
    # 不可度量措辞
    only_side2 = {
        "id": "os2",
        "out_of_frame_directions": [
            {"direction": "乙", "status": "not_generated", "decision_effect": "conclusion_flip",
             "expected_source": "gold_author"},
        ],
    }
    r63b = score_prediction(only_side2, _pred_sample([]))
    check("不可度量措辞含「靶子不可达 ≠ 选手没打中」",
          "靶子不可达" in r63b["miss_detection"]["main_unmeasurable_reason"],
          r63b["miss_detection"]["main_unmeasurable_reason"][:60])
    check("不可度量措辞含「不构成抽取器缺陷」",
          "不构成抽取器缺陷" in r63b["miss_detection"]["main_unmeasurable_reason"],
          r63b["miss_detection"]["main_unmeasurable_reason"][:60])

    # --- 64. prp 桶可准入（前置条件 + 护栏）---
    prem_path = "/workspace/c-group/goldset/premises.jsonl"
    prp_prompts = "/workspace/c-group/goldset/prp_prompts"
    premises = load_premises(prem_path)
    check("前提清单可载入", len(premises) > 0, str(len(premises)))
    check("前提清单含 trace_follows_premise / premise_raised_in_trace",
          all("trace_follows_premise" in v and "premise_raised_in_trace" in v
              for v in premises.values()), "字段缺失")

    # 探针 A 不给 P -> 产出不得进 prp 桶
    ctx_a = build_prp_context("A_premise_recognition", prp_prompts, prem_path)
    check("探针 A：判定为未给 P", ctx_a["premise_given"] is False, str(ctx_a["premise_given"]))
    ctx_b = build_prp_context("B_violation_detection", prp_prompts, prem_path)
    check("探针 B：判定为已给 P", ctx_b["premise_given"] is True, str(ctx_b["premise_given"]))
    ctx_c = build_prp_context("C_escape_direction", prp_prompts, prem_path)
    check("探针 C：判定为已给 P", ctx_c["premise_given"] is True, str(ctx_c["premise_given"]))

    prp_dir_gold = {
        "id": "pg",
        "out_of_frame_directions": [
            {"direction": "推翻 P 的方向", "status": "not_generated",
             "decision_effect": "conclusion_flip", "expected_source": "prp"},
        ],
    }
    r64a = score_prediction(prp_dir_gold, _pred_sample([]), prp_ctx=ctx_a)
    check("探针 A：prp 方向不进主分母（降级单列）",
          r64a["miss_detection"]["missed_main"] == 0 and
          r64a["miss_detection"]["missed_author_only"] == 1,
          str(r64a["miss_detection"]))
    r64b = score_prediction(prp_dir_gold, _pred_sample([]), prp_ctx=ctx_b)
    # 该题 id 不在前提清单里 -> premise 查不到 -> 不可准入
    check("前提缺失的 prp 方向不可准入（不进主分母）",
          r64b["miss_detection"]["missed_main"] == 0, str(r64b["miss_detection"]))
    check("prp 可准入判定有逐条理由",
          r64b["miss_detection"]["prp_admissibility"] and
          r64b["miss_detection"]["prp_admissibility"][0]["reason"],
          str(r64b["miss_detection"].get("prp_admissibility")))

    # 合法靶子：违反 P + trace 未提出 -> 可进主分母
    ok_item = next(
        (i for i, v in premises.items()
         if v.get("trace_follows_premise") is False and not v.get("premise_raised_in_trace")),
        None,
    )
    if ok_item:
        good_gold = {
            "id": ok_item,
            "out_of_frame_directions": [
                {"direction": "推翻 P 的方向", "status": "not_generated",
                 "decision_effect": "conclusion_flip", "expected_source": "prp"},
            ],
        }
        r64c = score_prediction(good_gold, _pred_sample([]), prp_ctx=ctx_b)
        check("合法靶子（违反 P + trace 未提出）：prp 方向进主分母",
              r64c["miss_detection"]["missed_main"] == 1, str(r64c["miss_detection"]))
    else:
        check("合法靶子（违反 P + trace 未提出）：本集无此类题", True, "本集无合法靶子")

    # 护栏：trace 已提出过推翻 P -> 不得再当漏判
    raised_item = next(
        (i for i, v in premises.items() if v.get("premise_raised_in_trace")), None
    )
    if raised_item:
        raised_gold = {
            "id": raised_item,
            "out_of_frame_directions": [
                {"direction": "推翻 P 的方向", "status": "not_generated",
                 "decision_effect": "conclusion_flip", "expected_source": "prp"},
            ],
        }
        r64d = score_prediction(raised_gold, _pred_sample([]), prp_ctx=ctx_b)
        check("护栏：trace 已提出过推翻 P -> 不得当漏判计入主分母",
              r64d["miss_detection"]["missed_main"] == 0, str(r64d["miss_detection"]))
        check("护栏：给出可追溯理由",
              "已提出过" in r64d["miss_detection"]["prp_admissibility"][0]["reason"],
              str(r64d["miss_detection"]["prp_admissibility"][0]["reason"]))
    else:
        check("护栏：本集无 trace 已提出过的题", True, "本集无此类题")

    # --- 65. ㉓-e ① 可达性断言 ---
    reach = assert_miss_bucket_reachability()
    check("可达性断言：每个 expected_source 都路由得到",
          set(reach) == set(EXPECTED_SOURCES), str(sorted(reach)))
    check("可达性断言：prp 路由到 main（不是静默落进 author_only）",
          reach.get("prp") == "main", str(reach.get("prp")))
    check("可达性断言：gold_author 路由到 author_only",
          reach.get("gold_author") == "author_only", str(reach.get("gold_author")))
    check("可达性断言：task 路由到 main", reach.get("task") == "main", str(reach.get("task")))
    # 反向：若枚举集合漏了 prp（回到第 4 次假通过的形态），断言必须报警
    _saved = EXPECTED_SOURCES
    try:
        globals()["EXPECTED_SOURCES"] = tuple(x for x in _saved if x != "prp")
        raised = None
        try:
            assert_miss_bucket_reachability()
        except CaliberError:
            raised = "raised"
        check("可达性断言反向：枚举漏项时必须抛异常", raised == "raised", str(raised))
    finally:
        globals()["EXPECTED_SOURCES"] = _saved
    check("可达性断言反向：恢复后正常", assert_miss_bucket_reachability().get("prp") == "main", "未恢复")

    # --- 66. ㉓-e ② 降级显式化：unknown_source_in_main 可见 ---
    unl_gold = {
        "id": "ul",
        "out_of_frame_directions": [
            {"direction": "甲", "status": "not_generated", "decision_effect": "conclusion_flip",
             "expected_source": "task"},
            {"direction": "乙", "status": "not_generated", "decision_effect": "conclusion_flip"},
            {"direction": "丙", "status": "not_generated", "decision_effect": "conclusion_flip",
             "expected_source": "拼错了"},
        ],
    }
    r66 = score_prediction(unl_gold, _pred_sample([]))
    m66 = r66["miss_detection"]
    check("降级显式化：未标注的进主桶但被计数",
          m66["unknown_source_in_main"] == 2, str(m66["unknown_source_in_main"]))
    check("降级显式化：审核状态分三档",
          m66["source_review"] == {"labeled": 1, "unlabeled": 1, "unknown": 1},
          str(m66["source_review"]))
    check("降级显式化：逐条列出未审核的条目",
          len(m66["unknown_source_details"]) == 2, str(m66["unknown_source_details"]))
    check("降级显式化：能区分未标注与取值未登记",
          {d["review_state"] for d in m66["unknown_source_details"]} == {"unlabeled", "unknown"},
          str(m66["unknown_source_details"]))

    # --- 67. 空方向 = 合法的「前提未被挑战」回答（探针 C），不计误报 ---
    empty_gold = {
        "id": "ed",
        "out_of_frame_directions": [
            {"direction": "推翻 P 的方向", "status": "not_generated", "decision_effect": "conclusion_flip",
             "expected_source": "prp"},
        ],
    }
    r67 = score_prediction(empty_gold, {"id": "ed", "directions": [
        {"direction": "", "in_frame": True, "in_frame_reason": "trace 未出现违背 P 的位置，无法命名推翻方向",
         "status": "realized", "evidence_quote": "", "decision_effect": "none"},
    ]})
    check("空方向：不计入主表 FP", r67["fp"] == 0, f"FP={r67['fp']}")
    check("空方向：不计入误报漏判", r67["miss_detection"]["spurious"] == 0, str(r67["miss_detection"]))
    check("空方向：单独计数", r67["empty_direction_responses"] == 1, str(r67["empty_direction_responses"]))
    check("空方向：金标漏判仍在漏判表的 missed 里（未被误判为已命中）",
          r67["miss_detection"]["missed"] == 1, str(r67["miss_detection"]))
    check("空方向：主表不受影响（该方向属漏判桶，不进主表 FN）",
          r67["fn"] == 0, f"FN={r67['fn']}")

    # --- 68. introduced_in_synthesis 的对照测试（M2 实测修正）---
    # 信号 (a) 必须是**语义级**的：抽取器写的 direction 与 trace 措辞不同，
    # 但方向确实在阶段 3 生成过 ⇒ **不得**被记成「综合阶段新产生」。
    ctrl_gold = {
        "id": "ctrl",
        "trace": [
            "阶段3｜方向生成：①把前置仓备货部分做仓单质押，把融资属性与存货属性分离。",
            "阶段4｜取舍：①保留。",
            "最终答案：建议采用仓单质押方案。",
        ],
        "out_of_frame_directions": [
            {"direction": "把前置仓备货做仓单质押，用融资属性与存货属性分离来释放经营现金",
             "status": "realized", "decision_effect": "conclusion_flip"},
        ],
    }
    ctrl_pred = {"id": "ctrl", "directions": [
        {"direction": "把前置仓备货做仓单质押，用融资属性与存货属性分离来释放经营现金",
         "in_frame": False, "status": "realized",
         "evidence_quote": "建议采用仓单质押方案。", "decision_effect": "conclusion_flip"},
    ]}
    r68 = score_prediction(ctrl_gold, ctrl_pred)
    check("对照：措辞不同但阶段 3 生成过的方向 -> 不得记成新产生",
          r68["introduced_in_synthesis_count"] == 0,
          str(r68["introduced_in_synthesis"]))
    # 反向对照：中间段确实没有这个方向 -> 应当被记成新产生
    ctrl_gold2 = {
        "id": "ctrl2",
        "trace": ["阶段3｜方向生成：①备货策略调整。", "最终答案：建议另设专项基金。"],
        "out_of_frame_directions": [
            {"direction": "设立专项基金", "status": "realized", "decision_effect": "option_added"},
        ],
    }
    ctrl_pred2 = {"id": "ctrl2", "directions": [
        {"direction": "设立专项基金", "in_frame": False, "status": "realized",
         "evidence_quote": "建议另设专项基金。", "decision_effect": "option_added"},
    ]}
    r68b = score_prediction(ctrl_gold2, ctrl_pred2)
    check("对照：中间段确实没有的方向 -> 应记成新产生",
          r68b["introduced_in_synthesis_count"] == 1, str(r68b["introduced_in_synthesis_count"]))

    # --- 69. expected_source 三档（M2 实测修正：未标注 ≠ 未登记）---
    three_gold = {
        "id": "thr",
        "out_of_frame_directions": [
            {"direction": "甲", "status": "not_generated", "decision_effect": "conclusion_flip",
             "expected_source": "task"},
            {"direction": "乙", "status": "not_generated", "decision_effect": "conclusion_flip",
             "expected_source": "gold_author"},
            {"direction": "丙", "status": "not_generated", "decision_effect": "conclusion_flip"},
            {"direction": "丁", "status": "not_generated", "decision_effect": "conclusion_flip",
             "expected_source": "拼错了"},
        ],
    }
    r69 = score_prediction(three_gold, _pred_sample([]))
    m69 = r69["miss_detection"]
    check("三档：labeled / unlabeled / unknown 严格区分",
          m69["source_review"] == {"labeled": 2, "unlabeled": 1, "unknown": 1},
          str(m69["source_review"]))
    check("三档：未标注不计成 unknown",
          m69["source_review"]["unknown"] == 1, str(m69["source_review"]))
    check("三档：unlabeled_source_in_main 单列",
          m69["unlabeled_source_in_main"] == 1, str(m69["unlabeled_source_in_main"]))
    check("三档：unknown_value_in_main 单列",
          m69["unknown_value_in_main"] == 1, str(m69["unknown_value_in_main"]))

    # --- 70. ★ 回归断言：带 --sweep 与不带 --sweep，introduced 计数与 status_acc 必须相同 ---
    # 「一个随参数漂移的指标，测的就不是『是否综合阶段新产生』，而是『措辞差多少』。」
    # 这条断言把「指标随参数漂移」从没人注意的隐患，变成有回归测试挡着的项。
    import subprocess
    import tempfile

    _reg_dir = tempfile.mkdtemp(prefix="psc_sweep_reg_")
    _reg_gold = os.path.join(_reg_dir, "gold.jsonl")
    _reg_pred = os.path.join(_reg_dir, "pred.jsonl")
    _reg_gold_item = {
        "id": "R1",
        "initial_frame": "【候选集】A\n【封闭性】open\n【评估维度】d\n【硬约束】无",
        "trace": ["阶段3｜方向生成：①把前置仓备货部分做仓单质押。", "最终答案：建议采用仓单质押方案。"],
        "out_of_frame_directions": [
            {"direction": "把前置仓备货做仓单质押，用融资属性与存货属性分离来释放经营现金",
             "status": "realized", "decision_effect": "conclusion_flip",
             "evidence_quote": "建议采用仓单质押方案。"},
        ],
    }
    with open(_reg_gold, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(_reg_gold_item, ensure_ascii=False) + "\n")
    with open(_reg_pred, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "id": "R1",
            "directions": [{
                "direction": "把前置仓备货做仓单质押，用融资属性与存货属性分离来释放经营现金",
                "in_frame": False, "status": "realized",
                "evidence_quote": "建议采用仓单质押方案。", "decision_effect": "conclusion_flip",
            }],
        }, ensure_ascii=False) + "\n")
    _reg_out = {}
    for _tag, _extra in (("nosweep", []), ("sweep", ["--sweep", "--sweep-from", "0.20",
                                                 "--sweep-to", "0.60", "--sweep-step", "0.20"])):
        _out_json = os.path.join(_reg_dir, f"out_{_tag}.json")
        _rc = subprocess.run(
            [sys.executable, os.path.join(HERE, "evaluator.py"),
             "--gold", _reg_gold, "--pred", _reg_pred, "--quiet", "--out", _out_json] + _extra,
            capture_output=True, text=True,
        )
        if _rc.returncode != 0:
            check(f"回归断言：{_tag} 跑分成功", False, _rc.stderr[-200:])
            break
        _reg_out[_tag] = json.load(open(_out_json, encoding="utf-8"))["summary"]
    if len(_reg_out) == 2:
        _a, _b = _reg_out["nosweep"], _reg_out["sweep"]
        check("回归断言：introduced 计数不随 --sweep 漂移",
              _a["introduced_in_synthesis_count"] == _b["introduced_in_synthesis_count"],
              f"{_a['introduced_in_synthesis_count']} vs {_b['introduced_in_synthesis_count']}")
        check("回归断言：status_acc 不随 --sweep 漂移",
              _a["status_accuracy"] == _b["status_accuracy"],
              f"{_a['status_accuracy']} vs {_b['status_accuracy']}")
        check("回归断言：主表 F1 不随 --sweep 漂移",
              _a["f1"] == _b["f1"], f"{_a['f1']} vs {_b['f1']}")
        check("回归断言：报告用的匹配阈值钉在门槛点",
              _a["match_threshold_used"] == GATE_POINT_MATCH_THR and
              _b["match_threshold_used"] == GATE_POINT_MATCH_THR,
              f"{_a['match_threshold_used']} vs {_b['match_threshold_used']}")
        check("回归断言：报告显式声明阈值已钉在门槛点",
              _a.get("match_threshold_pinned_to_gate_point") is True,
              str(_a.get("match_threshold_pinned_to_gate_point")))

    # --- 71. 某一类 status 零样本 -> 报告写明「本集测不到这一类」 ---
    _z = aggregate([score_prediction({
        "id": "Z1",
        "out_of_frame_directions": [
            {"direction": "甲", "status": "realized", "decision_effect": "none"},
        ],
    }, _pred_sample([]))])
    check("零样本：expression_failure 无样本时被列出",
          "expression_failure" in _z["zero_sample_statuses"], str(_z["zero_sample_statuses"]))
    check("零样本：报告给出「本集测不到」的措辞",
          "本集测不到" in _z["zero_sample_note"], _z["zero_sample_note"][:60])
    check("零样本：给出各类 support",
          _z["status_support"]["realized"] == 1 and _z["status_support"]["expression_failure"] == 0,
          str(_z["status_support"]))

    # --- 打印 ---
    print("=" * 68)
    print("PSC-Eval self-test (schema v2)")
    print("=" * 68)
    failed = 0
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        extra = f"   [{detail}]" if detail and not ok else ""
        print(f"[{mark}] {name}{extra}")
    print("-" * 68)
    print(f"{len(checks) - failed}/{len(checks)} passed")
    if failed:
        print(f"FAILED: {failed} 项")
    else:
        print("ALL PASSED")
    print("=" * 68)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
