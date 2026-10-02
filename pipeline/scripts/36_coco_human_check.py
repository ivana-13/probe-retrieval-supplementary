"""COCO: does the o3 assessor's low recall inflate the hardness of mined negatives?

On the 12 + 12 queries that the expert labelled, the matched-negative conditions are computed twice on the same queries
and the same pooled candidates: with the o3 labels and with the expert's labels. A correct item that o3 marked incorrect
is a negative under o3 and is removed under the human labels, so the difference between the two columns is the effect of
the assessor's false negatives. Output: results/coco_human_check.json"""
import json
import numpy as np
from svo_eval.paths import RESULTS, DIRECTIONS
from svo_eval.coco import load_coco
from svo_eval.coco_scoring import CocoScorer, available_models
from svo_eval import metrics as M

coll = load_coco()
pool = json.load(open(RESULTS / "coco_pool_candidates.json", encoding="utf-8"))
lab = {j: json.load(open(RESULTS / f"coco_pool_labels_{j}.json", encoding="utf-8")) for j in ["o3", "human"]}
models = available_models()
out = {}
for m in models:
    s = CocoScorer(m)
    out[m] = {}
    for d in DIRECTIONS:
        conv = (lambda c: int(c)) if d == "t2i" else (lambda c: c)
        rows = {j: {"self": [], "other": [], "all": [], "n_self": 0, "n_other": 0} for j in lab}
        moved = {"o3_negative_human_correct": 0, "o3_correct_human_negative": 0, "judged_by_both": 0}
        for q in lab["human"].get(d, {}):
            cands = pool[d][q]
            qq = q if d == "t2i" else int(q)
            gold = coll.gold(d, qq)
            g = float(np.max(s.score(d, qq, list(gold))))
            for c in cands:
                a, b = lab["o3"][d].get(q, {}).get(c), lab["human"][d][q].get(c)
                if a and b and conv(c) not in gold:
                    moved["judged_by_both"] += 1
                    moved["o3_negative_human_correct"] += int(a != "correct" and b == "correct")
                    moved["o3_correct_human_negative"] += int(a == "correct" and b != "correct")
            for j in lab:
                L = lab[j][d].get(q, {})
                negs = [c for c in cands if L.get(c) and L[c] != "correct" and conv(c) not in gold]
                self_n = [conv(c) for c in negs if m in cands[c]["retrieved_by"]]
                other_n = [conv(c) for c in negs if m not in cands[c]["retrieved_by"]]
                acc = lambda ns: float(np.mean(s.score(d, qq, ns) < g)) if ns else None
                rows[j]["self"].append(acc(self_n))
                rows[j]["other"].append(acc(other_n))
                rows[j]["all"].append(acc([conv(c) for c in negs]))
                rows[j]["n_self"] += len(self_n)
                rows[j]["n_other"] += len(other_n)
        res = {"n_queries": len(lab["human"].get(d, {})), **moved}
        for j in lab:
            for k in ["self", "other", "all"]:
                v = [x for x in rows[j][k] if x is not None]
                res[f"{j}_{k}"] = 100 * float(np.mean(v)) if v else None
            idx = [i for i in range(len(rows[j]["self"])) if rows[j]["self"][i] is not None and rows[j]["other"][i] is not None]
            dd = np.array([rows[j]["other"][i] - rows[j]["self"][i] for i in idx])
            res[f"{j}_other_minus_self"] = 100 * float(dd.mean()) if len(dd) else None
            res[f"{j}_p_randomisation"] = M.randomisation_p(dd) if len(dd) else None
            res[f"{j}_n_self"], res[f"{j}_n_other"] = rows[j]["n_self"], rows[j]["n_other"]
        out[m][d] = res
        print(f"{m:8s} {d}: n {res['n_queries']} | self o3 {res['o3_self']:.1f} human {res['human_self']:.1f} | other o3 {res['o3_other']:.1f} "
              f"human {res['human_other']:.1f} | other-self o3 {res['o3_other_minus_self']:.1f} (p {res['o3_p_randomisation']:.3f}) "
              f"human {res['human_other_minus_self']:.1f} (p {res['human_p_randomisation']:.3f}) | self negatives o3 {res['o3_n_self']} human {res['human_n_self']}")
m0 = models[0]
for d in DIRECTIONS:
    r = out[m0][d]
    print(d, "pairs judged by both", r["judged_by_both"], "| o3 incorrect but expert correct", r["o3_negative_human_correct"],
          "| o3 correct but expert incorrect", r["o3_correct_human_negative"])
json.dump(out, open(RESULTS / "coco_human_check.json", "w"), indent=1)
rng = lambda k: f"{min(out[m][d][k] for m in models for d in DIRECTIONS):.0f}--{max(out[m][d][k] for m in models for d in DIRECTIONS):.0f}"
print("self-mined: o3", rng("o3_self"), "human", rng("human_self"), "| other-self gap: o3", rng("o3_other_minus_self"), "human", rng("human_other_minus_self"),
      "| cells with p<0.05 under human labels:", sum(out[m][d]["human_p_randomisation"] < 0.05 for m in models for d in DIRECTIONS), "of 12")
print("mean change in self-mined accuracy (human - o3): %.1f" % np.mean([out[m][d]["human_self"] - out[m][d]["o3_self"] for m in models for d in DIRECTIONS]))
