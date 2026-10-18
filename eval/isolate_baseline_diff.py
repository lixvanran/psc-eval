"""隔离三个基线数（0.5106 / 0.5397 / 0.5714）的差异来源。

⑯-c：同一份数据经三条独立路径得出三个数时，不要取平均、不要取最高 ——
直接去找出哪一个的输入是脏的。

假设：差异来自**匹配函数**（非对称 bigram 覆盖 vs 对称 Jaccard）与**贪心顺序**，
而不是来自脏数据（我的数用他们的 predict() + 我的 scorer，不含他们的 score()）。
"""
import importlib.util
import json
import os
import sys

HERE = "/workspace/c-group/eval"
GOLDSET = "/workspace/c-group/goldset"
sys.path.insert(0, HERE)
import evaluator as E

spec = importlib.util.spec_from_file_location("lb", os.path.join(GOLDSET, "lexical_baseline.py"))
lb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lb)

gold = [json.loads(l) for l in open(os.path.join(GOLDSET, "gold_v1.jsonl"), encoding="utf-8") if l.strip()]
MATCH_THR = 0.25


def overlap_match(gold_dir, pred_dir):
    """金标构造方的匹配：非对称 bigram 覆盖，预测为分母。"""
    a = lb.bigrams(E.direction_text(pred_dir))
    b = lb.bigrams(E.direction_text(gold_dir))
    if not a:
        return False
    return len(a & b) / max(1, len(a)) >= MATCH_THR


def jaccard_match(gold_dir, pred_dir):
    """我的匹配：对称 Jaccard。"""
    return E.similarity(E.direction_text(gold_dir), E.direction_text(pred_dir)) >= MATCH_THR


def my_scorer(gold_item, preds, match_fn):
    pred_obj = {"id": gold_item["id"], "directions": [
        {"direction": p["direction"], "in_frame": False, "status": p["status"]} for p in preds]}
    return E.score_prediction(gold_item, pred_obj, match_fn=match_fn)


def their_caliber(gold_items, preds_by_id, match_fn):
    """他们的口径（分母 = 非 not_generated 的金标方向），但用可注入的匹配函数。"""
    tp = fp = fn = 0
    for it in gold_items:
        golds = it["gold"]["out_of_frame_directions"]
        preds = preds_by_id[it["id"]]
        used = set()
        matched = {}
        for gi, g in enumerate(golds):
            hit = None
            for i, p in enumerate(preds):
                if i not in used and match_fn(g, p):
                    hit = i
                    break
            if hit is not None:
                used.add(hit)
                matched[gi] = hit
        for gi, g in enumerate(golds):
            if g["status"] == "not_generated":
                continue
            if gi in matched:
                tp += 1
            else:
                fn += 1
        fp += len(preds) - len(matched)
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-9, p + r)
    return tp, fp, fn, round(p, 4), round(r, 4), round(f1, 4)


print("=" * 74)
print("三个基线数的来源隔离（闭卷 · 主表口径 · 同批预测 V1 oof=0.08 match=0.25）")
print("=" * 74)
preds_by_id = {it["id"]: lb.predict(it, "V1", 0.08, 0.3) for it in gold}

theirs = lb.score(gold, "V1", 0.08, MATCH_THR, 0.3)
print(f"A) 他们的 score() + 他们的非对称匹配（gold 序 first-fit）")
print(f"   tp={theirs['tp']} fp={theirs['fp']} fn={theirs['fn']} "
      f"P={theirs['precision']} R={theirs['recall']} F1={theirs['f1']}")

agg = E.aggregate([my_scorer(it, preds_by_id[it["id"]], E.default_match_fn) for it in gold])
print(f"B) 我的 scorer + 我的 Jaccard 匹配（相似度降序贪心）")
print(f"   tp={agg['tp']} fp={agg['fp']} fn={agg['fn']} "
      f"P={agg['precision']} R={agg['recall']} F1={agg['f1']}")

agg_c = E.aggregate([my_scorer(it, preds_by_id[it["id"]], overlap_match) for it in gold])
print(f"C) 我的 scorer + 他们的非对称匹配")
print(f"   tp={agg_c['tp']} fp={agg_c['fp']} fn={agg_c['fn']} "
      f"P={agg_c['precision']} R={agg_c['recall']} F1={agg_c['f1']}")

tp, fp, fn, p, r, f1 = their_caliber(gold, preds_by_id, jaccard_match)
print(f"D) 他们的口径 + 我的 Jaccard 匹配（gold 序 first-fit）")
print(f"   tp={tp} fp={fp} fn={fn} P={p} R={r} F1={f1}")

print("-" * 74)
print("结论：")
if abs(agg_c["f1"] - theirs["f1"]) < 0.01 and abs(f1 - agg["f1"]) < 0.01:
    print("  ★ 差异完全来自**匹配函数 + 贪心顺序**，与数据脏不脏无关：")
    print("    他们的非对称覆盖匹配（预测为分母）比对称 Jaccard **更严**（更少匹配）。")
    print("    等等 —— 看上面 tp 的差异方向即可判断。")
else:
    print("  ★ 差异不是纯匹配函数造成的，需要进一步定位。")
print(f"    我的 tp={agg['tp']} vs 他们 tp={theirs['tp']}  （匹配到的数量差异）")
print(f"    我的 fp={agg['fp']} vs 他们 fp={theirs['fp']}")
print("=" * 74)
