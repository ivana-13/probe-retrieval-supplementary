"""Scoring-rule check for the matched-negative comparison on SVO-Probes.

CLIP, FLAVA and SigLIP2 produced their judged lists under the dot product of unnormalised embeddings. This script scores
the same judged negatives with cosine similarity (L2-normalised embeddings) and repeats the comparison of handcrafted and
other-mined negatives on the 100 pool queries per direction:
  other        the paper's other-mined negatives (judged incorrect, not in the model's judged top 10), cosine scores
  other_strict the same without the items that the model ranks in its top 10 under cosine
  outside      the queries whose first relevant item is outside the model's top 10 under cosine
Paired randomisation tests, Holm-corrected over the six model-and-direction cells.
Output: results/cosine_check.json"""
import json
import numpy as np
from svo_eval import embeddings as E
from svo_eval.paths import RESULTS, DIRECTIONS
from svo_eval.collection import load_collection
from svo_eval import pools, metrics as M

COSINE = ["CLIP", "FLAVA", "SigLIP2"]
for m in COSINE:
    E.NORMALISE[m] = True
E.load_embeddings.cache_clear()
coll = load_collection()
fp = pools.build_fixed_pool()
full = json.load(open(RESULTS / "full_metrics.json"))
cos = json.load(open(RESULTS / "full_metrics_cosine.json"))


def paired(a, b):
    idx = [i for i in range(len(a)) if a[i] is not None and b[i] is not None]
    d = np.array([a[i] - b[i] for i in idx])
    mean, lo, hi = M.bootstrap_ci(d)
    return {"n": len(idx), "diff": 100 * mean, "lo": 100 * lo, "hi": 100 * hi, "p_randomisation": M.randomisation_p(d)}


def summ(v):
    v = [x for x in v if x is not None]
    return M.summary(v) if v else None


out = {}
for m in COSINE:
    s = E.Scorer(m)
    assert abs(float(np.linalg.norm(s.e.cap[0])) - 1.0) < 1e-4
    out[m] = {"full": {d: {"S@1_dot": full[m][d]["S@1"]["mean"], "S@1_cosine": cos[m][d]["S@1"]["mean"]} for d in DIRECTIONS},
              "pairwise_dot": full[m]["pairwise"]["overall"]["mean"], "pairwise_cosine": cos[m]["pairwise"]["overall"]["mean"]}
    for d in DIRECTIONS:
        rows = []
        for q in pools.query_ids(d):
            gold = coll.gold(d, q)
            scores = s.all_scores(d, q)
            gidx = [s.index_of(d, g) for g in gold]
            g = float(scores[gidx].max())
            mask = np.ones(len(scores), bool)
            mask[gidx] = False
            rank = int((scores[mask] > g).sum()) + 1
            top10 = set(s.topk(q, d, 10))
            entries = fp[d][q]
            neg_other = [c for c, e in entries.items() if not e["correct"] and c not in gold and m not in e["retrieved_by"]]
            neg_strict = [c for c in neg_other if c not in top10]
            bench = list(coll.cap2negimgs[q]) if d == "t2i" else list(coll.img2negcaps.get(q, []))
            acc = lambda negs: float(np.mean(s.score(d, q, list(negs)) < g)) if len(negs) else None
            rows.append({"rank": rank, "benchmark": acc(bench), "other": acc(neg_other), "other_strict": acc(neg_strict),
                         "n_other": len(neg_other), "n_in_cosine_top10": len(neg_other) - len(neg_strict)})
        col = lambda k, rs=rows: [r[k] for r in rs]
        outside = [r for r in rows if r["rank"] > 10]
        out[m][d] = {"benchmark": summ(col("benchmark")), "other": summ(col("other")), "other_strict": summ(col("other_strict")),
                     "benchmark_vs_other": paired(col("benchmark"), col("other")),
                     "benchmark_vs_other_strict": paired(col("benchmark"), col("other_strict")),
                     "share_of_other_in_cosine_top10": 100 * sum(col("n_in_cosine_top10")) / max(1, sum(col("n_other"))),
                     "S@1_pool_strict": 100 * float(np.mean([r["rank"] == 1 for r in rows])),
                     "outside": {"n_queries": len(outside), "benchmark": summ(col("benchmark", outside)), "other": summ(col("other", outside)),
                                 "test": paired(col("benchmark", outside), col("other", outside))}}
for comp in ["benchmark_vs_other", "benchmark_vs_other_strict"]:
    for (m, d), p in M.holm({(m, d): out[m][d][comp]["p_randomisation"] for m in COSINE for d in DIRECTIONS}).items():
        out[m][d][comp]["p_randomisation_holm"] = p
for (m, d), p in M.holm({(m, d): out[m][d]["outside"]["test"]["p_randomisation"] for m in COSINE for d in DIRECTIONS}).items():
    out[m][d]["outside"]["test"]["p_randomisation_holm"] = p
json.dump(out, open(RESULTS / "cosine_check.json", "w"), indent=1)
f1 = lambda c: "--" if c is None else f"{c['mean']:.1f}"
for m in COSINE:
    print(m, "pairwise dot %.1f cosine %.1f" % (out[m]["pairwise_dot"], out[m]["pairwise_cosine"]))
    for d in DIRECTIONS:
        r, o = out[m][d], out[m][d]["outside"]
        print(f"  {d}: S@1 dot {out[m]['full'][d]['S@1_dot']:.1f} cosine {out[m]['full'][d]['S@1_cosine']:.1f} | handcrafted {f1(r['benchmark'])} "
              f"other {f1(r['other'])} gap {r['benchmark_vs_other']['diff']:.1f} (Holm p {r['benchmark_vs_other']['p_randomisation_holm']:.4f}) | "
              f"other w/o cosine top 10 {f1(r['other_strict'])} gap {r['benchmark_vs_other_strict']['diff']:.1f} "
              f"(Holm p {r['benchmark_vs_other_strict']['p_randomisation_holm']:.4f}), {r['share_of_other_in_cosine_top10']:.1f}% removed | "
              f"outside top 10: n {o['n_queries']} handcrafted {f1(o['benchmark'])} other {f1(o['other'])} gap {o['test']['diff']:.1f} "
              f"(Holm p {o['test']['p_randomisation_holm']:.4f})")
print("saved", RESULTS / "cosine_check.json")
