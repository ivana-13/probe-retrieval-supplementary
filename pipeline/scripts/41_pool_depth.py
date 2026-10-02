"""Sensitivity of the matched-negative comparison to the pool depth, from the existing labels.

A pool of depth D keeps the judged candidates that at least one model ranked within its top D. For the scored model,
self-mined negatives are the judged-incorrect items of its own top D and other-mined negatives the judged-incorrect
items that only other models ranked within their top D. Depth 10 is the pool of the paper; depth 5 and depth 3 are
shallower pools drawn from the same labels (deeper pools would need new labels).
The benchmark and random conditions do not depend on the depth; their per-query values are read from
results/svo_analyses_perquery.json (SVO-Probes) and results/coco_matched_perquery.json (COCO).
Output: results/pool_depth.json"""
import json
import numpy as np
from svo_eval.paths import RESULTS, MODELS, DIRECTIONS
from svo_eval.collection import load_collection
from svo_eval.embeddings import Scorer
from svo_eval.coco import load_coco
from svo_eval.coco_scoring import CocoScorer
from svo_eval import pools, metrics as M

DEPTHS = [3, 5, 10]


def summ(v):
    v = [x for x in v if x is not None]
    return M.summary(v) if v else None


def paired(a, b):
    idx = [i for i in range(len(a)) if a[i] is not None and b[i] is not None]
    if not idx:
        return None
    d = np.array([a[i] - b[i] for i in idx])
    return {"n": len(idx), "diff": 100 * float(d.mean()), "p_randomisation": M.randomisation_p(d)}


def analyse(queries, per_query_ref, negatives_of, score_fn, gold_score_fn, m):
    """queries: list of query ids; negatives_of(q) -> {candidate: {model: rank}} of judged-incorrect non-gold items."""
    res = {}
    cache = {}
    for q in queries:
        negs = negatives_of(q)
        cands = list(negs)
        cache[q] = (negs, dict(zip(cands, score_fn(q, cands))) if cands else {}, gold_score_fn(q))
    for D in DEPTHS:
        other, self_, n_other, n_self = [], [], 0, 0
        for q in queries:
            negs, sc, g = cache[q]
            in_pool = [c for c, r in negs.items() if min(r.values()) <= D]
            s_n = [c for c in in_pool if negs[c].get(m, 99) <= D]
            o_n = [c for c in in_pool if negs[c].get(m, 99) > D]
            other.append(float(np.mean([sc[c] < g for c in o_n])) if o_n else None)
            self_.append(float(np.mean([sc[c] < g for c in s_n])) if s_n else None)
            n_other += len(o_n)
            n_self += len(s_n)
        bench = [per_query_ref[q].get("benchmark") for q in queries]
        rand = [per_query_ref[q].get("random") for q in queries]
        res[str(D)] = {"pool_other": summ(other), "pool_self": summ(self_), "benchmark": summ(bench), "random": summ(rand),
                       "n_other": n_other, "n_self": n_self,
                       "benchmark_vs_other": paired(bench, other), "random_vs_other": paired(rand, other),
                       "other_vs_self": paired(other, self_)}
    return res


out = {"svo": {}, "coco": {}}

# ---------------------------------------------------------------------------------------------- SVO-Probes (human labels)
coll = load_collection()
fp = pools.build_fixed_pool()
perq = json.load(open(RESULTS / "svo_analyses_perquery.json"))
for m in MODELS:
    s = Scorer(m)
    out["svo"][m] = {}
    for d in DIRECTIONS:
        ref = {r["q"]: r for r in perq[m][d]}
        queries = pools.query_ids(d)

        def negatives_of(q, d=d):
            gold = coll.gold(d, q)
            return {c: e["retrieved_by"] for c, e in fp[d][q].items() if not e["correct"] and c not in gold}

        out["svo"][m][d] = analyse(queries, {q: ref[str(q)] for q in queries}, negatives_of,
                                   lambda q, c, d=d: s.score(d, q, c),
                                   lambda q, d=d: float(np.max(s.score(d, q, list(coll.gold(d, q))))), m)

# ---------------------------------------------------------------------------------------------- COCO (o3 labels)
ccoll = load_coco()
cpool = json.load(open(RESULTS / "coco_pool_candidates.json", encoding="utf-8"))
clab = json.load(open(RESULTS / "coco_pool_labels_o3.json", encoding="utf-8"))
cref = json.load(open(RESULTS / "coco_matched_perquery.json", encoding="utf-8"))
cnew = json.load(open(RESULTS / "coco_analyses.json"))
for m in MODELS:
    s = CocoScorer(m)
    out["coco"][m] = {}
    for d in DIRECTIONS:
        conv = (lambda c: int(c)) if d == "t2i" else (lambda c: c)
        queries = list(cpool[d])
        # the benchmark value per query (image queries only); the random value of the paper is not stored per query,
        # so the random condition is left out for COCO here
        ref = {q: {"benchmark": cref.get(m, {}).get(d, {}).get("benchmark", {}).get(str(q)), "random": None} for q in queries}

        def negatives_of(q, d=d, conv=conv):
            qq = q if d == "t2i" else int(q)
            gold = ccoll.gold(d, qq)
            lab = clab.get(d, {}).get(q, {})
            return {c: e["retrieved_by"] for c, e in cpool[d][q].items()
                    if lab.get(c) and lab[c] != "correct" and conv(c) not in gold}

        out["coco"][m][d] = analyse(queries, ref, negatives_of,
                                    lambda q, c, d=d, conv=conv: s.score(d, q if d == "t2i" else int(q), [conv(x) for x in c]),
                                    lambda q, d=d: float(np.max(s.score(d, q if d == "t2i" else int(q),
                                                                        list(ccoll.gold(d, q if d == "t2i" else int(q)))))), m)

# Holm correction over the cells in which a comparison exists, per benchmark and depth
for name in out:
    for D in DEPTHS:
        for comp in ["benchmark_vs_other", "random_vs_other", "other_vs_self"]:
            cells = {(m, d): out[name][m][d][str(D)][comp]["p_randomisation"] for m in MODELS for d in DIRECTIONS
                     if out[name][m][d][str(D)][comp]}
            for (m, d), p in M.holm(cells).items():
                out[name][m][d][str(D)][comp]["p_randomisation_holm"] = p
json.dump(out, open(RESULTS / "pool_depth.json", "w"), indent=1)

rng = lambda v: f"{min(v):.1f}--{max(v):.1f}"
for name in out:
    for D in DEPTHS:
        cells = [(m, d) for m in MODELS for d in DIRECTIONS]
        R = lambda k: [out[name][m][d][str(D)][k]["mean"] for m, d in cells if out[name][m][d][str(D)][k]]
        t = [out[name][m][d][str(D)]["benchmark_vs_other"] for m, d in cells if out[name][m][d][str(D)]["benchmark_vs_other"]]
        ts = [out[name][m][d][str(D)]["other_vs_self"] for m, d in cells]
        print(f"{name} depth {D:2d}: other {rng(R('pool_other'))} self {rng(R('pool_self'))} | benchmark - other {rng([x['diff'] for x in t])} "
              f"p<0.05 in {sum(x['p_randomisation_holm'] < 0.05 for x in t)}/{len(t)}, p<0.01 in {sum(x['p_randomisation_holm'] < 0.01 for x in t)} "
              f"| other - self {rng([x['diff'] for x in ts])} p<0.05 in {sum(x['p_randomisation_holm'] < 0.05 for x in ts)}/{len(ts)} "
              f"| other-mined negatives in total {sum(out[name][m][d][str(D)]['n_other'] for m, d in cells)}")
print("saved", RESULTS / "pool_depth.json")
