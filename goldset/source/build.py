# -*- coding: utf-8 -*-
"""构建 gold_v1.jsonl（schema v2）。

流程：
  1. 载入 parts 里的 30 条（任务/框架/trace）
  2. 用 item_g11.py 整条替换 G11（去创业烧钱外壳，换成纯分配机制轴）
  3. 应用 edits.TRACE_EDITS（统一取舍段措辞，使 status 可机械判定）
  4. 统一引号：『→「 / 』→」（使 evidence_quote 可以是干净的连续原文）
  5. initial_frame 重排为三段式（【候选集】/【封闭性】/【评估维度】/【硬约束】+ 域外参考）
  6. 用 gold_final.GOLD 覆盖 gold 段，补 evidence / notes / decision_effect / expected_source
  7. 硬校验：evidence_quote 必须是 trace 的连续逐字子串；status 与「最终答案是否出现」自洽
  8. 写 gold_v1.jsonl
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)

import part1, part2, part3          # noqa: E402
from edits import TRACE_EDITS, DIMS  # noqa: E402
from edits2 import FA_EDITS, GOLD_PATCH  # noqa: E402
from edits3 import TRACE_EDITS3, GOLD_PATCH3, CONSTRAINT_VIOLATION  # noqa: E402
from edits4 import GOLD_PATCH4  # noqa: E402
from edits5 import GOLD_PATCH5  # noqa: E402
from edits5b import GOLD_PATCH5B  # noqa: E402
from edits6 import FA_EDITS3, FRAME_EDITS, GOLD_PATCH6  # noqa: E402
from gold_final import GOLD         # noqa: E402
from item_g11 import G11            # noqa: E402

DOMAIN = {
 "G01": "供应链/库存决策", "G02": "跨文化团队协作/中德合资研发", "G03": "老系统重构", "G04": "学术审稿",
 "G05": "法务合规/数据跨境", "G06": "招聘标准", "G07": "定价", "G08": "供应链/逆向物流",
 "G09": "教育制度设计", "G10": "公共政策/路权", "G11": "资源分配/算力", "G12": "老系统重构/责任划分",
 "G13": "并购整合", "G14": "学术审稿/制度", "G15": "开源项目治理", "G16": "会议与协作流程",
 "G17": "跨文化团队协作", "G18": "教育制度设计/导师制", "G19": "法务合规/员工数据",
 "G20": "数据治理/留存合规", "G21": "老系统重构/运维", "G22": "招聘标准/知情同意",
 "G23": "开源项目治理/许可", "G24": "知识产权与授权", "G25": "定价/演出票务",
 "G26": "教育制度设计/班级管理", "G27": "公共政策/旧改加装电梯", "G28": "资源分配/预算",
 "G29": "并购整合/制造业", "G30": "跨文化团队协作/远程交付",
}

# expression_failure 的方向必须**完全不出现在最终答案里**——用哨兵词做硬校验。
SENTINELS = {
 ("G05", 2): ["本地部署", "放弃云订阅", "license"],
 ("G11", 2): ["算力委员会", "轮值代表"],
 ("G14", 2): ["快速通道", "主编直管"],
 ("G15", 2): ["商业化", "付费", "订阅"],
 ("G16", 2): ["组长", "项目制"],
 ("G17", 2): ["迁至", "搬迁", "签证"],
 ("G18", 2): ["名额"],
 ("G19", 1): ["员工手册", "评优", "年终"],
 ("G20", 2): ["本地化", "存算"],
}

EFFECT_GATE = {"conclusion_flip", "option_added"}

# v1.1 stance 词表：最终答案段里对方向的表态
#   采纳/推进（realized） / 排除·否决·推迟（selection_failure）
POS_STANCE = ("建议", "请", "推荐", "采用", "纳入", "优先", "执行", "启动", "立项",
              "打包", "试", "保留", "收紧", "改写", "做三件事", "做")
NEG_STANCE = ("不做", "不建议", "不采纳", "不推进", "暂不", "不启动", "不列入", "不落地",
              "不引入", "不要", "排除", "放弃", "否决", "删掉", "不能", "本轮不",
              "同样不做", "明确不", "不作为", "不予")
# ⑭-b 结构性判据的机械对应：纯推迟（排期到未来 / 条件触发 / 列为议题）也算 selection_failure，
# 即使句子里没有显式否决词（G11-① 就是这一类）。
DEFER_STANCE = ("独立议题", "列为议题", "作为议题", "规划", "再议", "再谈", "再上", "届时",
                "另行", "之后", "下轮", "下一轮", "下个", "下季度", "以后再", "下学期",
                # ⑬-b 读法：筛选阶段的"搁置/挂起/先放着/再议"也算被否决（推迟）
                "搁着", "挂", "备着", "先不", "再说", "再看", "再拿出来", "不落笔", "记一笔",
                "没进方案", "不动", "长期选项", "研究议题", "备选项", "待议", "另议", "暂不")


MARKERS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


def _bigrams(s):
    s = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)} or {s}


def _ov(a, b):
    A, B = _bigrams(a), _bigrams(b)
    return len(A & B) / max(1, len(A))


def direction_marker(item, direction_text):
    """从阶段3 的候选列表里找出与该方向最匹配的编号（方向级核对用）。"""
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


def normalize_quotes(s):
    return s.replace("『", "「").replace("』", "」")


def restructure_frame(item):
    """把『本题初始框架：『杠杆』…\n域内：…\n域外：…』重排为四段式。

    裁决 ⑨：**「域外参考」必须移出 initial_frame**——它等于把答案印在题面上，
    闭卷下抽取器只做字串比对就能判域外。域外参考改存金标专有字段 frame_note
    （仅评测器可见，不进可核语料、不是抽取器输入）。
    """
    fr = item["initial_frame"]
    if fr.startswith("【候选集】"):
        # G11 是重写过的题，直接用显式给出的 frame_note
        return fr, item.get("frame_note", "")
    m = re.search(r"本题初始框架：(.*?)\n域内：(.*?)\n域外：(.*)$", fr, re.S)
    if not m:
        raise SystemExit(f"{item['id']}: initial_frame 解析失败\n{fr[:200]}")
    lever, inner, outer = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    frame = (
        f"【候选集】{inner}\n"
        f"【封闭性】open\n"
        f"【评估维度】{DIMS[item['id']]}。\n"
        f"【硬约束】本题的初始边界是：{lever.rstrip(chr(12290))}。"
    )
    note = f"域外参考（被排除在解空间之外，不作为候选）：{outer}"
    return frame, note


def main():
    items = []
    for src in (part1.ITEMS, part2.ITEMS, part3.ITEMS):
        items.extend(src)
    by_id = {it["id"]: it for it in items}
    assert len(items) == 30 and len(by_id) == 30, f"expected 30 unique items, got {len(items)}/{len(by_id)}"

    # --- 2. G11 整条替换 -------------------------------------------------
    for k in ("task", "initial_frame", "trace", "frame_note"):
        by_id["G11"][k] = G11[k]

    # --- 3. trace 措辞统一 ----------------------------------------------
    for iid, edits in TRACE_EDITS.items():
        it = by_id[iid]
        blob = "\n".join(it["trace"])
        for old, new in edits:
            cnt = blob.count(old)
            if cnt != 1:
                raise SystemExit(f"{iid}: 期望 old 唯一出现，实际 {cnt} 次\n---\n{old}\n---")
            blob = blob.replace(old, new)
        it["trace"] = blob.split("\n")

    # --- 4. 引号统一 -----------------------------------------------------
    for it in items:
        it["task"] = normalize_quotes(it["task"])
        it["initial_frame"] = normalize_quotes(it["initial_frame"])
        it["trace"] = [normalize_quotes(t) for t in it["trace"]]

    # --- 4b. 最终答案改写（v1.1：让最终答案承担 status 判据）------------
    for iid, edits in FA_EDITS.items():
        it = by_id[iid]
        blob = "\n".join(it["trace"])
        for old_s, new_s in edits:
            cnt = blob.count(old_s)
            if cnt != 1:
                raise SystemExit(f"{iid}[FA]: 期望 old 唯一出现，实际 {cnt} 次\n---\n{old_s}\n---")
            blob = blob.replace(old_s, new_s)
        it["trace"] = blob.split("\n")

    # --- 4b2. v6 trace 改写（⑥ 定向构造真 EF）---------------------------
    for iid, edits in FA_EDITS3.items():
        it = by_id[iid]
        blob = "\n".join(it["trace"])
        for old_s, new_s in edits:
            cnt = blob.count(old_s)
            if cnt != 1:
                raise SystemExit(f"{iid}[v6]: 期望 old 唯一出现，实际 {cnt} 次\n---\n{old_s}\n---")
            blob = blob.replace(old_s, new_s)
        it["trace"] = blob.split("\n")

    # --- 4c. v2 trace 措辞（⑮-b 去模板化 / ⑭-b G16-② 缺陷修复）----------
    for iid, edits in TRACE_EDITS3.items():
        it = by_id[iid]
        blob = "\n".join(it["trace"])
        for old_s, new_s in edits:
            cnt = blob.count(old_s)
            if cnt != 1:
                raise SystemExit(f"{iid}[v2]: 期望 old 唯一出现，实际 {cnt} 次\n---\n{old_s}\n---")
            blob = blob.replace(old_s, new_s)
        it["trace"] = blob.split("\n")

    # --- 5. 框架四段式（域外参考移入 frame_note，闭卷）------------------
    for it in items:
        it["initial_frame"], it["frame_note"] = restructure_frame(it)
        if "域外参考" in it["initial_frame"]:
            raise SystemExit(f"{it['id']}: initial_frame 仍含「域外参考」——开卷泄漏")

    # --- 6. gold 段覆盖 --------------------------------------------------
    for it in items:
        iid = it["id"]
        dirs = []
        for idx, g in enumerate(GOLD[iid]):
            rec = {
                "direction": g["direction"],
                "status": g["status"],
                "evidence_quote": g["evidence_quote"],
                "evidence": g["evidence_quote"] if g["status"] != "discovery_failure" else g["evidence"],
                "notes": g["notes"],
                "decision_effect": g["decision_effect"],
            }
            # 不再保留 would_change_decision 布尔位：它与 decision_effect 语义重叠，
            # 一旦两者不一致会静默走错分支。章程坑 3 的硬闸门改由
            # decision_effect ∈ {conclusion_flip, option_added} 承载（见 README §0）。
            if g.get("expected_source"):
                rec["expected_source"] = g["expected_source"]
            dirs.append(rec)
        it["gold"] = {"out_of_frame_directions": dirs, "frame_note": it["frame_note"]}

    # --- 5b. ⑦ 杠杆 1：候选集措辞去遮蔽（只改【候选集】）-----------------
    for iid, edits in FRAME_EDITS.items():
        it = by_id[iid]
        fr = it["initial_frame"]
        for old_s, new_s in edits:
            cnt = fr.count(old_s)
            if cnt != 1:
                raise SystemExit(f"{iid}[杠杆1]: 期望 old 唯一出现，实际 {cnt} 次\n{old_s}")
            fr = fr.replace(old_s, new_s)
        it["initial_frame"] = fr

    # --- 6b. v1.1 改标补丁 ------------------------------------------------
    for (iid, idx), patch in GOLD_PATCH.items():
        d = by_id[iid]["gold"]["out_of_frame_directions"][idx]
        for k in ("status", "evidence_quote", "notes"):
            if k in patch:
                d[k] = patch[k]
        d["evidence"] = d["evidence_quote"] if d["status"] != "discovery_failure" else d["evidence"]

    # --- 6b. v3 改标（⑯ §⑥ 逐条裁定）-------------------------------------
    for (iid, idx), patch in GOLD_PATCH4.items():
        d = by_id[iid]["gold"]["out_of_frame_directions"][idx]
        for k in ("status", "evidence_quote", "notes", "decision_effect"):
            if k in patch:
                d[k] = patch[k]
        d["evidence"] = d["evidence_quote"] if d["status"] != "discovery_failure" else d["evidence"]
        d["would_change_decision"] = d["decision_effect"] in EFFECT_GATE

    # --- 6b2. v4 改标（⑳：可报性弱的方向移出主召回分母）-----------------
    for (iid, idx), patch in GOLD_PATCH5.items():
        d = by_id[iid]["gold"]["out_of_frame_directions"][idx]
        for k in ("expected_reportability", "notes"):
            if k in patch:
                d[k] = patch[k]

    # --- 6c. v2 改标（去模板化引文）+ ⑭-c 违反禁区桶 ---------------------
    for (iid, idx), patch in GOLD_PATCH3.items():
        d = by_id[iid]["gold"]["out_of_frame_directions"][idx]
        for k in ("status", "evidence_quote", "notes"):
            if k in patch:
                d[k] = patch[k]
        d["evidence"] = d["evidence_quote"] if d["status"] != "discovery_failure" else d["evidence"]
    viol = []
    for (iid, idx), spec in CONSTRAINT_VIOLATION.items():
        d = by_id[iid]["gold"]["out_of_frame_directions"][idx]
        viol.append({"direction": d["direction"], "violates": spec["violates"],
                     "why_not_not_generated": spec["why_not_not_generated"],
                     "decision_effect": spec["decision_effect"]})
        by_id[iid]["gold"]["out_of_frame_directions"].pop(idx)   # 从主金标移出
        for k, dd in enumerate(by_id[iid]["gold"]["out_of_frame_directions"]):
            pass
    for it in items:
        it["gold"]["constraint_violation_candidates"] = [
            v for v in viol if v["direction"] in
            [d["direction"] for d in it["gold"]["out_of_frame_directions"]] ] or (
            viol if it["id"] == "G21" else [])

    # --- 6d. v5 改标（⑬-b：筛选阶段的推迟/搁置 = 被否决 ⇒ selection_failure）----
    # 放在 6c 之后：edits3 会重写这 9 条的 notes（去模板化那轮）；
    # 若本块先跑，notes 会被 6c 覆盖回旧文案 → notes 与 status 自相矛盾。
    for (iid, idx), patch in GOLD_PATCH5B.items():
        d = by_id[iid]["gold"]["out_of_frame_directions"][idx]
        d["status"] = patch["status"]
        d["notes"] = patch["notes"]
        d["evidence"] = d["evidence_quote"] if d["status"] != "discovery_failure" else d["evidence"]

    # --- 6d2. v6 改标（⑥ 定向构造真 EF）-----------------------------------
    for (iid, idx), patch in GOLD_PATCH6.items():
        d = by_id[iid]["gold"]["out_of_frame_directions"][idx]
        for k in ("status", "evidence_quote", "notes", "constructed_for"):
            if k in patch:
                d[k] = patch[k]
        d["evidence"] = d["evidence_quote"] if d["status"] != "discovery_failure" else d["evidence"]

    # --- 7. 硬校验 -------------------------------------------------------
    errors = []
    for it in items:
        iid = it["id"]
        trace_all = "\n".join(it["trace"])
        final_ans = it["trace"][-1]
        for idx, d in enumerate(it["gold"]["out_of_frame_directions"]):
            st, q = d["status"], d["evidence_quote"]
            if st == "discovery_failure":
                if q != "":
                    errors.append(f"{iid}[{idx}] discovery_failure 必须留空 evidence_quote")
                if "expected_source" not in d:
                    errors.append(f"{iid}[{idx}] discovery_failure 缺 expected_source")
                continue
            if not q:
                errors.append(f"{iid}[{idx}] {st} 缺 evidence_quote")
                continue
            if q not in trace_all:
                errors.append(f"{iid}[{idx}] {st} evidence_quote 不是 trace 的连续逐字子串: {q[:60]}")
            # v1.1：判据锚在最终答案段
            if st == "realized":
                if q not in final_ans:
                    errors.append(f"{iid}[{idx}] realized 的引用不在最终答案段: {q[:60]}")
                elif not any(k in q for k in POS_STANCE):
                    errors.append(f"{iid}[{idx}] realized 的引用缺少采纳/推进措辞: {q[:60]}")
            elif st == "selection_failure":
                in_final = q in final_ans
                in_select = q in next((t for t in it["trace"] if t.startswith("阶段4")), "")
                if not (in_final or in_select):
                    errors.append(f"{iid}[{idx}] selection_failure 的引用既不在最终答案段也不在取舍段: {q[:60]}")
                elif not (any(k in q for k in NEG_STANCE) or any(k in q for k in DEFER_STANCE)):
                    errors.append(f"{iid}[{idx}] selection_failure 的引用缺少否决/推迟措辞: {q[:60]}")
            elif st == "expression_failure":
                # ⑬-b 严格定义：中间段**明确保留**（有保留类措辞、无搁置/推迟类措辞）+ 最终答案完全未提及
                stage4_all = next((t for t in it["trace"] if t.startswith("阶段4")), "")
                if q not in stage4_all:
                    errors.append(f"{iid}[{idx}] expression_failure 的引用不在取舍段: {q[:60]}")
                if not any(k in q for k in POS_STANCE):
                    errors.append(f"{iid}[{idx}] expression_failure 的引用缺少「保留」类措辞: {q[:60]}")
                if any(k in q for k in NEG_STANCE):
                    errors.append(f"{iid}[{idx}] expression_failure 的引用含搁置/否决措辞（那应判 SF）: {q[:60]}")
                if q in final_ans:
                    errors.append(f"{iid}[{idx}] expression_failure 的引用出现在最终答案段: {q[:60]}")
                if not d.get("constructed_for"):
                    errors.append(f"{iid}[{idx}] expression_failure 缺少 constructed_for 标注"
                                  "（定向构造的样本必须显式标注，不得伪装成自然分布）")
                if q in final_ans:
                    errors.append(f"{iid}[{idx}] expression_failure 的引用命中了最终答案段: {q[:60]}")
                for w in SENTINELS.get((iid, idx), []):
                    if w in final_ans:
                        errors.append(f"{iid}[{idx}] expression_failure 哨兵词「{w}」出现在最终答案里")
                # ⑮-b 去模板化后不再靠"保留/建议"这类关键词判定，改为结构化核对：
                #   引用必须在取舍段、不得含否决词、必须点名该方向的编号
                stage4 = next((t for t in it["trace"] if t.startswith("阶段4")), "")
                if q not in stage4:
                    errors.append(f"{iid}[{idx}] expression_failure 的引用不在取舍段: {q[:60]}")
                if any(k in q for k in NEG_STANCE):
                    errors.append(f"{iid}[{idx}] expression_failure 的引用含否决词（应为推迟而非否决）: {q[:60]}")
                mk = direction_marker(it, d["direction"])
                if mk and mk not in q:
                    errors.append(f"{iid}[{idx}] expression_failure 的引用未点名该方向（应含 {mk}）: {q[:60]}")
    if errors:
        print("构建失败：")
        for e in errors:
            print("  *", e)
        raise SystemExit(1)

    # --- 8. 写出 ---------------------------------------------------------
    openbook = "--openbook" in sys.argv
    if openbook:
        for it in items:
            it["initial_frame"] = it["initial_frame"] + "\n" + it["frame_note"]
    out = os.path.join(HERE, "gold_v1_openbook.jsonl" if openbook else "gold_v1.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            rec = {
                "id": it["id"],
                "task": it["task"],
                "initial_frame": it["initial_frame"],
                "trace": it["trace"],
                "gold": it["gold"],          # 含 frame_note（金标专有，抽取器不可见）
                "_domain": DOMAIN[it["id"]],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {out}  ({len(items)} lines, {os.path.getsize(out)} bytes)")


if __name__ == "__main__":
    main()
