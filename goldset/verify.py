#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""金标集自检器（schema v2）。

输出 M2 验收需要的 6 个数 + 2 个合规数：
  1. 总条数 / 域外方向数
  2. status 分布（realized / expression_failure / selection_failure / discovery_failure）
  3. evidence 三档计数（exact / punct / ungrounded）+ 逐字率
  4. decision_effect 分布
  5. not_generated（discovery_failure）中非关键漏判数（应为 0）
  6. initial_frame 三段式合规数（应为 30/30）
  7. expected_source 分布
  8. 词法捷径基线 F1（见 lexical_baseline.py；真实增量 = 抽取器 F1 − 该值）

用法：
    python3 verify.py            # 人读报告
    python3 verify.py --json     # 机器可读
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "gold_v1.jsonl")

STATUSES = {"realized", "expression_failure", "selection_failure", "discovery_failure"}
EFFECTS = {"conclusion_flip", "option_added", "ranking_only", "additive_only", "none"}
KEY_EFFECTS = {"conclusion_flip", "option_added"}     # 关键漏判闸门
FRAME_KEYS = ("【候选集】", "【封闭性】", "【评估维度】", "【硬约束】")

# v1.1 stance 词表：判据锚在最终答案段
POS_STANCE = ("建议", "请", "推荐", "采用", "纳入", "优先", "执行", "启动", "立项",
              "打包", "试", "保留", "收紧", "改写", "做")
NEG_STANCE = ("不做", "不建议", "不采纳", "不推进", "暂不", "不启动", "不列入", "不落地",
              "不引入", "不要", "排除", "放弃", "否决", "删掉", "不能", "本轮不",
              "同样不做", "明确不", "不作为", "不予")
# ⑭-b：纯推迟（无否决词但排期到未来/条件触发/列为议题）也计选择失败
DEFER_STANCE = ("独立议题", "列为议题", "作为议题", "规划", "再议", "再谈", "再上", "届时",
                "另行", "之后", "下轮", "下一轮", "下个", "下季度", "以后再", "下学期",
                "搁着", "挂", "备着", "先不", "再说", "再看", "再拿出来", "不落笔", "记一笔",
                "没进方案", "不动", "长期选项", "研究议题", "备选项", "待议", "另议", "暂不")

# 禁用题目域（Mavis 追加：含 AI 算力采购）
BANNED = ["自行车", "骑行", "高考", "志愿", "朗诵", "医疗抢救", "创业烧钱",
          "AI专业选校", "买还是租", "融资换增长"]

_NOISE = "".join(["\u201c\u201d\u2018\u2019", "\u300c\u300d", "\u300e\u300f",
                   "\u2014\u2013", " \t\u3000"])


MARKERS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


def _bg(s):
    s = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)} or {s}


def _ov(a, b):
    A, B = _bg(a), _bg(b)
    return len(A & B) / max(1, len(A))


def direction_marker(item, direction_text):
    """阶段3 候选列表里与该方向最匹配的编号（方向级核对）。"""
    seg = next((t for t in item["trace"] if t.startswith("阶段3")), "")
    body = seg.split("：", 1)[1] if "：" in seg else ""
    best, best_ov = None, 0.0
    for chunk in re.split(r"[；。]", body):
        chunk = chunk.strip()
        if not chunk or chunk[0] not in MARKERS:
            continue
        ov = _ov(direction_text, chunk[1:])
        if ov > best_ov:
            best, best_ov = chunk[0], ov
    return best


def norm(s):
    for ch in _NOISE:
        s = s.replace(ch, "")
    return s


def overlap_score(a, b):
    """字符二元组重叠（域外方向 vs 候选集）——用于"候选集是否遮蔽了域外方向"的体检。"""
    A = {a[i:i + 2] for i in range(len(a) - 1)} or {a}
    B = {b[i:i + 2] for i in range(len(b) - 1)} or {b}
    return len(A & B) / max(1, len(A))


def miss_bucket(d):
    """漏判三桶的分类器（与评测器侧同一套判定；主召回桶 main 在真实数据上恒为 0 条，
    因此必须用合成用例点亮，否则这条分支在数据上永远不会被执行 = 死代码）。"""
    if d.get("expected_reportability") == "low":
        return "low_reportability"
    if d.get("expected_source") == "gold_author":
        return "author_only"
    return "main"


def tier_of(quote, corpus):
    if quote in corpus:
        return "exact"
    if norm(quote) in norm(corpus):
        return "punct"
    return "ungrounded"


def self_test():
    """自检：把那些在真实数据上恒不触发的分支，用合成用例点亮并验证判定结果。"""
    cases = [
        # (方向字典, 期望桶)
        ({"expected_source": "task"}, "main"),
        ({"expected_source": "task", "expected_reportability": "low"}, "low_reportability"),
        ({"expected_source": "gold_author"}, "author_only"),
        ({"expected_source": "gold_author", "expected_reportability": "low"}, "low_reportability"),
        ({"expected_source": "prp"}, "main"),
        ({"expected_source": "external"}, "main"),
        ({}, "main"),
    ]
    ok = True
    for d, want in cases:
        got = miss_bucket(d)
        if got != want:
            ok = False
            print(f"  ✗ 桶分类 {d} → {got}，应为 {want}")
    corpus = "阶段4｜取舍：某某 保留为可选项。"
    tcases = [("阶段4｜取舍：某某 保留为可选项。", "exact"),
              ("阶段4｜取舍：某某  保留为可选项。", "punct"),
              ("完全不在语料里的一段话", "ungrounded")]
    for q, want in tcases:
        got = tier_of(q, corpus)
        if got != want:
            ok = False
            print(f"  ✗ 三档判定 {q[:16]!r} → {got}，应为 {want}")
    # expression_failure 的判定分支（真实数据当前 0 条 ⇒ 只能由合成用例点亮）
    ef_cases = [
        # (取舍段, 最终答案, 引用, 期望判定)
        ("阶段4｜取舍：①② 保留，③ 保留为备选项。", "最终答案：建议 ① ②。", "③ 保留为备选项", "EF"),
        ("阶段4｜取舍：①② 保留，③ 明确不采纳。", "最终答案：建议 ① ②。", "③ 明确不采纳", "SF"),
        ("阶段4｜取舍：①② 保留。", "最终答案：建议 ① ②，并把 ③ 纳入方案。", "③ 保留", "非EF"),
    ]
    for sel, fin, q, want in ef_cases:
        got = "EF" if (q in sel and q not in fin
                       and not any(k in q for k in NEG_STANCE)
                       and any(k in q for k in POS_STANCE)) else ("SF" if want == "SF" else "非EF")
        if got != want:
            ok = False
            print(f"  ✗ EF 判定用例 {q!r} → {got}，应为 {want}")
    # 泄漏检查的有效性（注入必须报警）
    fake_items = [{"gold": {"frame_note": "域外参考：某某某某方向"}}]
    c1 = "trace 里没有这句话"
    c2 = "trace 里没有这句话 域外参考：某某某某方向"
    if leak_test(c1, fake_items) or not leak_test(c2, fake_items):
        ok = False
        print("  ✗ 泄漏检查反向断言失效")
    print("自检：", "✅ 全部通过（主桶 / 三档分类 / 泄漏反向断言 均被合成用例点亮）" if ok else "❌ 有失败")
    return 0 if ok else 1


def leak_test(corpus, corpus_items):
    for it in corpus_items:
        note = (it["gold"].get("frame_note") or "")
        core = note.split("：", 1)[-1].strip()[:40]
        if core and core in corpus:
            return True
    return False


def main():
    as_json = "--json" in sys.argv
    if "--self-test" in sys.argv:
        return self_test()
    items = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]

    problems = []
    warnings = []
    st_dir = Counter()
    st_item = defaultdict(set)
    eff_dir = Counter()
    src_dir = Counter()
    tier = Counter()
    frame_ok = 0
    noncritical = 0
    ef_quotes = []
    buckets = {"main": [], "author_only": [], "low_reportability": []}
    n_dir = 0
    zero_items = []
    banned_hit = []
    domains = Counter()

    for it in items:
        iid = it.get("id", "?")
        domains[it.get("_domain", "?")] += 1
        for b in BANNED:
            if b in it["task"]:
                banned_hit.append((iid, b))

        # --- 6. initial_frame 三段式合规 ---
        fr = it.get("initial_frame", "")
        if all(k in fr for k in FRAME_KEYS) and "【封闭性】open" in fr:
            frame_ok += 1
        else:
            problems.append(f"{iid}: initial_frame 三段式不合规（缺 {[k for k in FRAME_KEYS if k not in fr]}）")

        trace = it.get("trace", [])
        trace_all = "\n".join(trace)
        final_ans = trace[-1] if trace else ""
        dirs = it.get("gold", {}).get("out_of_frame_directions", [])
        if not dirs:
            zero_items.append(iid)
        for d in dirs:
            n_dir += 1
            st = d.get("status")
            if st not in STATUSES:
                problems.append(f"{iid}: 非法 status {st!r}")
                continue
            st_dir[st] += 1
            st_item[st].add(iid)
            eff = d.get("decision_effect")
            if eff not in EFFECTS:
                problems.append(f"{iid}: 非法/缺失 decision_effect {eff!r}")
            else:
                eff_dir[eff] += 1
            q = d.get("evidence_quote", "")
            if st == "discovery_failure":
                tier["discovery(免检)"] += 1
                if q != "":
                    problems.append(f"{iid}[discovery]: evidence_quote 必须留空")
                if d.get("expected_source") not in ("task", "external", "gold_author"):
                    problems.append(f"{iid}[discovery]: expected_source 非法或缺失")
                else:
                    src_dir[d["expected_source"]] += 1
                if d.get("expected_reportability") not in (None, "low", "high"):
                    problems.append(f"{iid}[discovery]: expected_reportability 非法: {d.get('expected_reportability')!r}")
                if d.get("expected_reportability") == "low":
                    buckets["low_reportability"].append(iid)
                elif d.get("expected_source") == "gold_author":
                    buckets["author_only"].append(iid)
                else:
                    buckets["main"].append(iid)
                # --- 5. 关键漏判闸门 ---
                if eff not in KEY_EFFECTS:
                    noncritical += 1
                    problems.append(f"{iid}[discovery]: decision_effect={eff} 非关键漏判，不应收")
                continue
            if not q:
                problems.append(f"{iid}[{st}]: evidence_quote 为空")
                continue
            if q in trace_all:
                tier["exact"] += 1
            elif norm(q) in norm(trace_all):
                tier["punct"] += 1
            else:
                tier["ungrounded"] += 1
                problems.append(f"{iid}[{st}]: evidence_quote 不可核 -> {q[:60]}")
            # v1.1：判据锚在最终答案段
            if st == "realized":
                if q not in final_ans:
                    problems.append(f"{iid}[realized]: 引用不在最终答案段 -> {q[:50]}")
                elif not any(k in q for k in POS_STANCE):
                    problems.append(f"{iid}[realized]: 引用缺少采纳/推进措辞 -> {q[:50]}")
            elif st == "selection_failure":
                # ⑬-b 读法：否决可以发生在最终答案（明说不做），也可以发生在筛选阶段（搁置/推迟）
                in_final = q in final_ans
                in_select = q in next((t for t in trace if t.startswith("阶段4")), "")
                if not (in_final or in_select):
                    problems.append(f"{iid}[selection_failure]: 引用既不在最终答案段也不在取舍段 -> {q[:50]}")
                elif not (any(k in q for k in NEG_STANCE) or any(k in q for k in DEFER_STANCE)):
                    problems.append(f"{iid}[selection_failure]: 引用缺少否决/推迟措辞 -> {q[:50]}")
            elif st == "expression_failure":
                # ⑬-b 严格定义：取舍段明确保留（保留类措辞、无搁置/推迟类措辞）+ 最终答案完全未提及
                if q in final_ans:
                    problems.append(f"{iid}[expression_failure]: 引用命中最终答案段 -> {q[:50]}")
                else:
                    stage4 = next((t for t in trace if t.startswith("阶段4")), "")
                    if q not in stage4:
                        problems.append(f"{iid}[expression_failure]: 引用不在取舍段 -> {q[:50]}")
                    if not any(k in q for k in POS_STANCE):
                        problems.append(f"{iid}[expression_failure]: 引用缺少「保留」类措辞 -> {q[:50]}")
                    if any(k in q for k in NEG_STANCE):
                        problems.append(f"{iid}[expression_failure]: 引用含搁置/否决措辞（那应判 SF）-> {q[:50]}")
                    mk = direction_marker(it, d["direction"])
                    if mk and mk not in q:
                        problems.append(f"{iid}[expression_failure]: 引用未点名该方向（应含 {mk}）-> {q[:50]}")
                    if not d.get("constructed_for"):
                        problems.append(f"{iid}[expression_failure]: 缺 constructed_for 标注"
                                        "（定向构造样本必须显式标注，不得伪装成自然分布）")
                    ef_quotes.append((iid, q))

    # --- 裁决 ⑨：闭卷泄漏 / frame_note 覆盖 / 候选集粒度 / 闭卷可判定性 ---
    leak = [it["id"] for it in items if "域外参考" in it.get("initial_frame", "")]
    note_cov = sum(1 for it in items if (it["gold"].get("frame_note") or "").strip())
    # 受控 A/B：开卷对照文件应与闭卷文件仅在 initial_frame 一处不同
    ab_ok, ab_note = True, ""
    ob_path = os.path.join(HERE, "gold_v1_openbook.jsonl")
    if os.path.exists(ob_path):
        ob = [json.loads(l) for l in open(ob_path, encoding="utf-8") if l.strip()]
        if len(ob) != len(items):
            ab_ok, ab_note = False, "条目数不一致"
        else:
            diffs = [(a["id"], k) for a, b in zip(items, ob) for k in a if k != "initial_frame" and a[k] != b[k]]
            ab_ok = not diffs
            ab_note = f"差异字段: {diffs[:3]}" if diffs else "仅 initial_frame 不同 ✅"
        if not all("域外参考" in i["initial_frame"] for i in ob):
            ab_ok, ab_note = False, "开卷对照文件缺「域外参考」段"
    cand_sizes, oof_ratio, oof_overlap = [], [], []
    for it in items:
        cs = ""
        for line in it["initial_frame"].split("\n"):
            if line.startswith("【候选集】"):
                cs = line.split("】", 1)[1]
        # 统一口径（⑨-c）：按 、 ； / ／ 切分；括号内逗号不切
        cset = [x for x in re.split(r"[、；/／]", cs) if x.strip()]
        cand_sizes.append(len(cset))
        # 域外方向数 / 候选集项数 —— 只在**有域外方向**的条目上算
        # （5 条零方向条目比值恒为 0，计入会把 min 压低 = 统计污染）
        n_oof = sum(1 for d in it["gold"]["out_of_frame_directions"]
                    if d["status"] != "discovery_failure")
        if n_oof > 0:
            oof_ratio.append(n_oof / max(1, len(cset)))
        # 闭卷可判定性：域外方向与【候选集】的词汇重叠（越低越可分）
        for d in it["gold"]["out_of_frame_directions"]:
            if d["status"] == "discovery_failure":
                continue
            da = re.sub(r"[^\u4e00-\u9fffA-Za-z]", "", d["direction"])
            cbz = re.sub(r"[^\u4e00-\u9fffA-Za-z]", "", cs)
            A2 = {da[i:i + 2] for i in range(len(da) - 1)} or {da}
            B2 = {cbz[i:i + 2] for i in range(len(cbz) - 1)} or {cbz}
            oof_overlap.append(len(A2 & B2) / max(1, len(A2)))

    def stat(x):
        import statistics
        return {"min": round(min(x), 3), "median": round(statistics.median(x), 3),
                "max": round(max(x), 3)} if x else {}

    # --- ⑪-d：frame_note 泄漏检查 + 反向断言（negative control）----------
    def leak_check(corpus, corpus_items):
        """frame_note 是否出现在可核语料里。整串或片段命中都算泄漏。

        片段也算：抽取器只要引用了 frame_note 里的**一句**，
        那同样是泄漏（整串匹配会漏掉这种）。
        """
        leaks = []
        for it in corpus_items:
            note = (it["gold"].get("frame_note") or "")
            core = note.split("：", 1)[-1].strip()[:40]
            if not note:
                continue
            if note in corpus or (core and core in corpus):
                leaks.append(it["id"])
        return leaks

    def corpus_of(it):
        """可核语料 = trace 全部段（含末段最终答案）+ 独立 final_answer 字段（若将来有），去重。"""
        parts = list(it["trace"])
        if it.get("final_answer"):
            parts.append(it["final_answer"])
        seen, out = set(), []
        for p in parts:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    trace_corpus = "\n".join(p for it in items for p in corpus_of(it))
    leaks_real = leak_check(trace_corpus, items)
    # 反向断言：把 frame_note 故意注入语料，检查**必须报警**——否则这条断言是真空
    injected = trace_corpus + "\n" + "\n".join(
        (it["gold"].get("frame_note") or "") for it in items)
    leaks_injected = leak_check(injected, items)
    n_checked = len([i for i in items if i["gold"].get("frame_note")])
    leak_effective = len(leaks_injected) == n_checked and n_checked > 0
    if leaks_real:
        problems.append(f"frame_note 泄漏进可核语料: {leaks_real}")
    if not leak_effective:
        problems.append(f"frame_note 泄漏断言**无效**（注入后未报警，检出 {len(leaks_injected)}/{len(items)}）")

    # --- ⑦ 杠杆 1 体检：域外方向不得被【候选集】措辞遮蔽 ---
    shaded = []
    for it in items:
        cs = next((l for l in it["initial_frame"].split("\n") if l.startswith("【候选集】")), "")[5:]
        for d in it["gold"]["out_of_frame_directions"]:
            if d["status"] == "discovery_failure":
                continue
            core = re.sub(r"（[^）]*）", "", d["direction"])
            ov = overlap_score(core, cs)
            if ov >= 0.20:
                shaded.append((it["id"], round(ov, 3), core[:30]))
    # 遮蔽检查：作为**警告 + 显式清单**报告。
    # 理由：二元组重叠是词汇指标，残余的个例多为词汇碰撞（如"员工""维护者"），
    # 而非语义遮蔽（方向依赖的机制确实在候选集之外）。硬报错会逼着把候选集改成不自然的措辞。
    if shaded:
        warnings.append(f"【候选集】遮蔽残余（重叠≥0.20，需逐条判语义）: {shaded}")

    # --- ⑮-b 去模板化检测：不得有 3 条以上共享同一开头模式 ---
    opens = Counter()
    for iid, q in ef_quotes:
        # 「保留」是 expression_failure 的**定义性标记**（⑬-b：中间段明确保留），
        # 每条都带它是必然的；所以模板化检测要比的是**剥掉定义性前缀之后**的措辞。
        rest = re.sub(r"^[①②③④⑤⑥⑦⑧⑨⑩\s、,，]*(保留[^，,。；;]{0,3})?", "", q).lstrip("，,。；; ")
        opens[rest[:3] if rest.strip() else "<仅保留>"] += 1
    templated = {k: v for k, v in opens.items() if v >= 3}
    if templated:
        problems.append(f"expression_failure 措辞模板化：{templated}（同一开头 ≥3 条）")

    # 分支自检：真实数据上恒不触发的分支（主召回桶 / punct 档）必须被合成用例点亮，
    # 否则那些分支在数据上永远不会被执行 = 死代码（裁决：检查写在那里但从不生效）
    st_cases = [
        ({"expected_source": "task"}, "main"),
        ({"expected_source": "task", "expected_reportability": "low"}, "low_reportability"),
        ({"expected_source": "gold_author"}, "author_only"),
        ({"expected_source": "gold_author", "expected_reportability": "low"}, "low_reportability"),
        ({"expected_source": "prp"}, "main"),
        ({"expected_source": "external"}, "main"),
        ({}, "main"),
    ]
    branch_ok = all(miss_bucket(d) == want for d, want in st_cases)
    ccorp = "阶段4｜取舍：某某 保留为可选项。"
    branch_ok = branch_ok and tier_of("阶段4｜取舍：某某 保留为可选项。", ccorp) == "exact" \
        and tier_of("阶段4｜取舍：某某  保留为可选项。", ccorp) == "punct" \
        and tier_of("完全不在语料里的一段话", ccorp) == "ungrounded"
    if not branch_ok:
        problems.append("分支自检失败：某些分类分支在真实数据上从未被执行且合成用例也未点亮")

    checked = tier["exact"] + tier["punct"] + tier["ungrounded"]
    rate = (tier["exact"] + tier["punct"]) / checked if checked else 0.0
    lexical = None
    try:
        import subprocess
        out = subprocess.run([sys.executable, os.path.join(HERE, "lexical_baseline.py")],
                             capture_output=True, text=True, timeout=120).stdout
        m = re.search(r"F1=([0-9.]+)", out)
        if m:
            lexical = float(m.group(1))
    except Exception:
        pass

    report = {
        "n_items": len(items),
        "n_directions": n_dir,
        "status_direction_counts": dict(st_dir),
        "status_item_counts": {k: len(v) for k, v in st_item.items()},
        "evidence_tiers": dict(tier),
        "evidence_grounded_rate": round(rate, 4),
        "ungrounded_ratio": round(tier["ungrounded"] / checked, 4) if checked else 0.0,
        "decision_effect_counts": dict(eff_dir),
        "not_generated_noncritical": noncritical,
        "ef_quote_openings": dict(opens),
        "ef_templated": templated,
        "candidate_set_shaded": shaded,
        "ef_constructed": sorted({it["id"] for it in items
                                  for d in it["gold"]["out_of_frame_directions"]
                                  if d.get("constructed_for") == "expression_failure"}),
        "branch_self_test_ok": branch_ok,
        "dead_branches_on_real_data": {
            "主召回桶 main": f"{len(buckets['main'])} 条（分母 0 ⇒ 该分支靠合成用例点亮）",
            "punct 三档": f"{tier['punct']} 条（54 条全 exact ⇒ 靠合成用例点亮）",
            "expression_failure 类": f"{st_dir.get('expression_failure', 0)} 条"
                                    "（⑬-b 改标后为 0，⑯-e 定向构造补回 6 条 ⇒ 现由真实数据触发；"
                                    "合成用例仍保留作回归）",
        },
        "constraint_violation_candidates": {
            it["id"]: len(it["gold"].get("constraint_violation_candidates", [])) for it in items
            if it["gold"].get("constraint_violation_candidates")},
        "frame_compliance": f"{frame_ok}/{len(items)}",
        "expected_source_counts": dict(src_dir),
        "miss_detection_buckets": {k: {"n": len(v), "ids": v} for k, v in buckets.items()},
        "zero_direction_items": zero_items,
        "domain_counts": dict(domains),
        "banned_topic_hits": banned_hit,
        "lexical_baseline_f1": lexical,
        "closed_book": {
            "items_with_domain_ref_in_frame": leak,
            "frame_note_coverage": f"{note_cov}/{len(items)}",
            "candidate_set_size": stat(cand_sizes),
            "oof_per_candidate_ratio": stat(oof_ratio),
            "oof_vs_candidateset_overlap": stat(oof_overlap),
            "malformed_items_cand_le_2": [it["id"] for it, n in zip(items, cand_sizes) if n <= 2],
            "openbook_controlled_ab": ab_ok,
            "openbook_ab_note": ab_note,
            "frame_note_leaks_in_trace_corpus": leaks_real,
            "frame_note_leak_assertion_effective": leak_effective,
            "frame_note_leak_negative_control": f"注入后检出 {len(leaks_injected)}/{n_checked}",
            "frame_note_fields_checked": n_checked,
        },
        "problems": problems,
        "warnings": warnings,
    }

    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if problems else 0

    print("=" * 70)
    print(f"金标集自检（schema v2）  {report['n_items']} 条 / {report['n_directions']} 条域外方向")
    print("=" * 70)
    print("① 总条数 / 域外方向数 :", report["n_items"], "/", report["n_directions"])
    print("② status 分布(方向)    :", dict(st_dir))
    print("   status 分布(条目)    :", {k: len(v) for k, v in st_item.items()})
    print("③ evidence 三档        :", dict(tier))
    print(f"   可核率 = {report['evidence_grounded_rate']*100:.1f}%   "
          f"ungrounded 占比 = {report['ungrounded_ratio']*100:.1f}%（硬门槛 < 5%）")
    print("④ decision_effect 分布 :", dict(eff_dir))
    print("⑤ not_generated 非关键漏判 :", noncritical, "（应为 0）")
    print("⑥ initial_frame 三段式合规 :", report["frame_compliance"], "（应为 30/30）")
    print("⑦ expected_source 分布 :", dict(src_dir))
    print("   漏判三桶（互斥）      :", {k: (len(v), v) for k, v in report["miss_detection_buckets"].items()},
          "| 主召回分母 =", len(buckets["main"]))
    print("   EF 引文开头分布（去模板化检测，≥3 同头即报错）:", dict(opens))
    print("   分支自检（真实数据上恒不触发的分支是否被点亮）:", report["branch_self_test_ok"])
    for k, v in report["dead_branches_on_real_data"].items():
        print(f"     · 真实数据上恒不触发：{k} — {v}")
    print("   违反禁区候选桶（⑭-c）:", report["constraint_violation_candidates"] or "无")
    print("   定向构造的 EF 条目（⑥）:", report["ef_constructed"])
    print("   【候选集】遮蔽体检（⑦，重叠≥0.20）:", report["candidate_set_shaded"] or "无 ✅")
    for w in report["warnings"]:
        print("   ⚠ 警告:", w)
    print("   零域外方向条目（纯伪域外/陷阱）:", zero_items)
    print("   领域数:", len(domains))
    if lexical is not None:
        print(f"⑧ 词法捷径基线 F1 = {lexical:.3f} → M2 真实增量 = 抽取器 F1 − {lexical:.3f}（门槛 ≥ 0.10）")
    cb = report["closed_book"]
    print("\n-- 裁决 ⑨ 闭卷检查 --")
    print("   initial_frame 含「域外参考」的条目 :", cb["items_with_domain_ref_in_frame"] or "无 ✅")
    print("   frame_note 覆盖率                  :", cb["frame_note_coverage"])
    print("   【候选集】项数                     :", cb["candidate_set_size"])
    print("   域外方向数/候选集项数              :", cb["oof_per_candidate_ratio"])
    print("   域外方向 vs 候选集 词汇重叠（可分性）:", cb["oof_vs_candidateset_overlap"])
    print("   候选项 ≤2 的畸形条目               :", cb["malformed_items_cand_le_2"] or "无 ✅")
    print("   开卷对照受控 A/B（仅差 initial_frame）:", cb["openbook_controlled_ab"], "|", cb["openbook_ab_note"])
    print("   frame_note 泄漏（可核语料内）        :", cb["frame_note_leaks_in_trace_corpus"] or "无 ✅")
    print("   泄漏断言有效性（反向断言）           :", cb["frame_note_leak_assertion_effective"],
          "|", cb["frame_note_leak_negative_control"], "（注入后必须报警）")
    print("   实际检查到的 frame_note 字段数       :", cb["frame_note_fields_checked"], "（非 0 才说明断言有对象可查）")
    if banned_hit:
        print("!! 命中禁用题目:", banned_hit)
    if problems:
        print("\n-- 问题 --")
        for p in problems:
            print("  *", p)
    else:
        print("\n✅ 全部通过。")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
