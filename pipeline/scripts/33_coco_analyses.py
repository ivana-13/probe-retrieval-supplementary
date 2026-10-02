"""COCO val2017 + SugarCrepe: the analyses of scripts/16_svo_analyses.py (long mode) on the o3-labelled pool.

  A. matched negatives per query (benchmark, random, other-mined, self-mined, all pooled) with paired randomisation
     tests and Holm correction over the model-and-direction cells of each comparison; random negatives are ten items,
     averaged over 20 draws shared by all models (seeded per query), as on SVO-Probes
  B. the queries whose positive lies outside the model's own top 10 (elsewhere other-mined accuracy is 1 by construction)
  C. strict Success@1/10 and o3-judged Success@1 on the 100 pool queries
  D. n-way curves with other-mined negatives only (nested prefixes of 50 random orders) and with random items, both on
     the queries that have at least ten other-mined negatives; number of random negatives equivalent to ten mined ones
Benchmark (SugarCrepe) per-query accuracies come from results/coco_matched_perquery.json (scripts/21); SugarCrepe
negatives are captions, so the benchmark condition exists for image queries only.
Usage: python scripts/33_coco_analyses.py [o3|human]   -> results/coco_analyses.json"""
import hashlib
import json
import sys
import numpy as np
from svo_eval.paths import RESULTS, DIRECTIONS
from svo_eval.coco import load_coco
from svo_eval.coco_scoring import CocoScorer, available_models
from svo_eval import metrics as M

judge = sys.argv[1] if len(sys.argv) > 1 else "o3"
coll = load_coco()
pool = json.load(open(RESULTS / "coco_pool_candidates.json", encoding="utf-8"))
labels = json.load(open(RESULTS / f"coco_pool_labels_{judge}.json", encoding="utf-8"))
bench_perq = json.load(open(RESULTS / "coco_matched_perquery.json", encoding="utf-8"))
models = available_models()
N_RANDOM, RANDOM_DRAWS, NWAY_PERMS = 10, 20, 50
NWAY_KS = [1, 2, 5, 10]
RAND_KS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, None]


def seed_for(*parts):
    return int(hashlib.md5(str(parts).encode()).hexdigest(), 16) % (2 ** 32)


def summ(vals):
    vals = [v for v in vals if v is not None]
    return M.summary(vals) if vals else None


def paired(a, b):
    idx = [i for i in range(len(a)) if a[i] is not None and b[i] is not None]
    if not idx:
        return None
    d = np.array([a[i] - b[i] for i in idx])
    mean, lo, hi = M.bootstrap_ci(d)
    return {"n": len(idx), "diff": 100 * mean, "lo": 100 * lo, "hi": 100 * hi, "p_randomisation": M.randomisation_p(d)}


out = {"matched": {}, "tests": {}, "reference": {}, "nway": {}}
universe_sorted = {}
probe_by_image = {}
for item in coll.probe:                                  # SugarCrepe items of an image: (caption, edited caption)
    probe_by_image.setdefault(int(item["image_id"]), []).append((item["caption"], item["negative_caption"]))
for m in models:
    s = CocoScorer(m)
    for k in out:
        out[k][m] = {}
    for d in DIRECTIONS:
        conv = (lambda c: int(c)) if d == "t2i" else (lambda c: c)
        if d not in universe_sorted:                       # one candidate order for every model, so draws are shared
            universe_sorted[d] = sorted(int(x) for x in s.candidates(d)) if d == "t2i" else sorted(str(x) for x in s.candidates(d))
        rows = []
        for q, cands in pool[d].items():
            qq = q if d == "t2i" else int(q)
            lab = labels.get(d, {}).get(q, {})
            gold = coll.gold(d, qq)
            scores = s.all_scores(d, qq)
            gidx = [s.img_index[int(x)] for x in gold] if d == "t2i" else [s.cap_index[x] for x in gold]
            g = float(scores[gidx].max())
            mask = np.ones(len(scores), bool)
            mask[gidx] = False
            neg_scores = scores[mask]
            rank = int((neg_scores > g).sum()) + 1
            negs = [c for c in cands if lab.get(c) and lab[c] != "correct" and conv(c) not in gold]
            neg_self = [conv(c) for c in negs if m in cands[c]["retrieved_by"]]
            neg_other = [conv(c) for c in negs if m not in cands[c]["retrieved_by"]]

            def acc(ns):
                return float(np.mean(s.score(d, qq, ns) < g)) if ns else None

            universe = [c for c in universe_sorted[d] if c not in gold]
            rng = np.random.default_rng(seed_for(d, q, "random"))
            rand = float(np.mean([acc([universe[i] for i in rng.choice(len(universe), N_RANDOM, replace=False)])
                                  for _ in range(RANDOM_DRAWS)]))
            own = sorted([c for c in cands if m in cands[c]["retrieved_by"]], key=lambda c: cands[c]["retrieved_by"][m])
            top1 = lab.get(own[0]) if own else None
            # the probe's own decision: each edited caption against the caption it was built from (image queries only);
            # "benchmark" scores the same edited captions against the best relevant caption, like the other conditions
            # "*_own" for the mined conditions applies the probe's rule to them: the negatives are scored against the
            # caption each SugarCrepe item was built from, so handcrafted and mined negatives face the same positive
            probe_own, own_rule = None, {"other": None, "self": None, "all": None}
            if d == "i2t" and probe_by_image.get(qq):
                dec = []
                for cap, neg in probe_by_image[qq]:
                    a, b = s.score(d, qq, [cap, neg])
                    dec.append(float(a > b))
                probe_own = float(np.mean(dec))
                src = s.score(d, qq, [cap for cap, _ in probe_by_image[qq]])
                for name, ns in [("other", neg_other), ("self", neg_self), ("all", [conv(c) for c in negs])]:
                    if ns:
                        sc = s.score(d, qq, ns)
                        own_rule[name] = float(np.mean([np.mean(sc < x) for x in src]))
            rows.append({"q": str(q), "rank": rank, "benchmark": bench_perq.get(m, {}).get(d, {}).get("benchmark", {}).get(str(q)),
                         "benchmark_own": probe_own, "pool_other_own": own_rule["other"],
                         "pool_self_own": own_rule["self"], "pool_all_own": own_rule["all"],
                         "random": rand, "pool_other": acc(neg_other), "pool_self": acc(neg_self),
                         "pool_all": acc([conv(c) for c in negs]), "n_other": len(neg_other),
                         "other_beating": int(np.sum(s.score(d, qq, neg_other) >= g)) if neg_other else 0,
                         "top1_judged": (top1 == "correct") if top1 else None,
                         "_g": g, "_neg_scores": neg_scores,
                         "_other_scores": s.score(d, qq, neg_other) if neg_other else np.array([])})
        col = lambda k: [r[k] for r in rows]
        res = {k: summ(col(k)) for k in ["benchmark", "benchmark_own", "random", "pool_other", "pool_self", "pool_all",
                                         "pool_other_own", "pool_self_own", "pool_all_own"]}
        res["violations_of_bound"] = int(sum(r["other_beating"] > 0 for r in rows if r["rank"] <= 10))
        out["matched"][m][d] = res
        outside = [r for r in rows if r["rank"] > 10]
        out["tests"][m][d] = {
            "benchmark_vs_other": paired(col("benchmark"), col("pool_other")),
            "benchmark_vs_all": paired(col("benchmark"), col("pool_all")),
            "random_vs_other": paired(col("random"), col("pool_other")),
            "random_vs_all": paired(col("random"), col("pool_all")),
            "random_vs_benchmark": paired(col("random"), col("benchmark")),
            "benchmark_vs_benchmark_own": paired(col("benchmark"), col("benchmark_own")),
            "all_vs_benchmark_own": paired(col("pool_all"), col("benchmark_own")),
            "self_vs_benchmark_own": paired(col("pool_self"), col("benchmark_own")),
            "benchmark_own_vs_other_own": paired(col("benchmark_own"), col("pool_other_own")),
            "benchmark_own_vs_all_own": paired(col("benchmark_own"), col("pool_all_own")),
            "benchmark_own_vs_self_own": paired(col("benchmark_own"), col("pool_self_own")),
            "other_vs_self": paired(col("pool_other"), col("pool_self")),
            "outside_top10": {"n_queries": len(outside), "benchmark": summ([r["benchmark"] for r in outside]),
                              "pool_other": summ([r["pool_other"] for r in outside]),
                              "random": summ([r["random"] for r in outside])}}
        ranks = np.array(col("rank"))
        judged = [x for x in col("top1_judged") if x is not None]
        out["reference"][m][d] = {"S@1": M.summary(ranks <= 1), "S@10": M.summary(ranks <= 10),
                                  "judged_S@1": M.summary(np.array(judged)) if judged else None, "n_judged": len(judged)}

        # n-way: other-mined negatives only, nested prefixes, fixed query set; random items on the same queries
        fixed = [r for r in rows if r["n_other"] >= max(NWAY_KS)]
        nw = {"k": NWAY_KS, "n_queries_fixed": len(fixed), "other_fixed": [], "rand_k": [k if k else "all" for k in RAND_KS],
              "random_fixed_full": []}
        for k in NWAY_KS:
            vals = []
            for r in fixed:
                rng = np.random.default_rng(seed_for(d, r["q"], "nway"))
                osc = r["_other_scores"]
                vals.append(float(np.mean([np.all(osc[rng.permutation(len(osc))[:k]] < r["_g"]) for _ in range(NWAY_PERMS)])))
            nw["other_fixed"].append(M.summary(vals) if vals else None)
        for k in RAND_KS:
            vals = []
            for r in fixed:
                ns = r["_neg_scores"]
                if k is None or k >= len(ns):
                    vals.append(float(np.all(ns < r["_g"])))
                else:
                    rng = np.random.default_rng(seed_for(d, r["q"], k, "nway-random"))
                    vals.append(float(np.mean([np.all(ns[rng.choice(len(ns), k, replace=False)] < r["_g"])
                                               for _ in range(NWAY_PERMS)])))
            nw["random_fixed_full"].append(M.summary(vals) if vals else None)
        nw["random_fixed"] = [nw["random_fixed_full"][RAND_KS.index(k)] for k in NWAY_KS]
        n_all = (len(s.image_ids) if d == "t2i" else len(s.captions)) - 1
        eq = None
        if fixed:
            target = nw["other_fixed"][-1]["mean"]
            ks = [k if k else n_all for k in RAND_KS]
            ys = [c["mean"] for c in nw["random_fixed_full"]]
            if target >= ys[0]:
                eq = 1.0
            for i in range(len(ks) - 1):
                if ys[i] >= target >= ys[i + 1] and ys[i] != ys[i + 1]:
                    t = (ys[i] - target) / (ys[i] - ys[i + 1])
                    eq = float(np.exp(np.log(ks[i]) + t * (np.log(ks[i + 1]) - np.log(ks[i]))))
                    break
        nw["equivalent_random_k_for_10_other_fixed"] = eq
        out["nway"][m][d] = nw

# Holm correction over the cells in which each comparison exists
for comp in ["benchmark_vs_other", "benchmark_vs_all", "random_vs_other", "random_vs_all", "random_vs_benchmark", "other_vs_self",
             "benchmark_vs_benchmark_own", "all_vs_benchmark_own", "self_vs_benchmark_own",
             "benchmark_own_vs_other_own", "benchmark_own_vs_all_own", "benchmark_own_vs_self_own"]:
    cells = {(m, d): out["tests"][m][d][comp]["p_randomisation"] for m in models for d in DIRECTIONS if out["tests"][m][d][comp]}
    for (m, d), p in M.holm(cells).items():
        out["tests"][m][d][comp]["p_randomisation_holm"] = p
        out["tests"][m][d][comp]["holm_cells"] = len(cells)

json.dump(out, open(RESULTS / "coco_analyses.json", "w"), indent=1)
f1 = lambda c: "--" if c is None else f"{c['mean']:.1f}"
for m in models:
    for d in DIRECTIONS:
        a, t, r, n = out["matched"][m][d], out["tests"][m][d], out["reference"][m][d], out["nway"][m][d]
        ref = "benchmark" if t["benchmark_vs_other"] else "random"
        o = t["outside_top10"]
        print(f"{m:8s} {d}: bench {f1(a['benchmark'])} random {f1(a['random'])} other {f1(a['pool_other'])} self {f1(a['pool_self'])} "
              f"all {f1(a['pool_all'])} | {ref}-other {t[ref + '_vs_other']['diff']:.1f} p {t[ref + '_vs_other']['p_randomisation_holm']:.4f} "
              f"{ref}-all {t[ref + '_vs_all']['diff']:.1f} p {t[ref + '_vs_all']['p_randomisation_holm']:.4f} "
              f"other-self {t['other_vs_self']['diff']:.1f} p {t['other_vs_self']['p_randomisation_holm']:.4f} | "
              f"S@1 {r['S@1']['mean']:.0f} S@10 {r['S@10']['mean']:.0f} judged S@1 {f1(r['judged_S@1'])} (n {r['n_judged']}) | "
              f"outside top 10: n {o['n_queries']} bench {f1(o['benchmark'])} other {f1(o['pool_other'])} | "
              f"n-way n {n['n_queries_fixed']} 10-other {f1(n['other_fixed'][-1])} 10-random {f1(n['random_fixed'][-1])} "
              f"equiv k {n['equivalent_random_k_for_10_other_fixed']} | violations {a['violations_of_bound']}")
print("saved", RESULTS / "coco_analyses.json")
