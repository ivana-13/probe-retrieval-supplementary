"""Matched-negative analyses on SVO-Probes (100 queries per direction).

Everything is computed on the existing judged pool and embeddings; nothing is labelled or re-extracted.
  A. matched negatives per query with paired randomisation and Wilcoxon tests, Holm-corrected over the eight
     model-and-direction cells; benchmark condition against the best relevant item and against the triplet's own positive;
     random condition averaged over several draws shared by all models
  B. label-noise and pool-composition checks on the other-mined condition (unanimous annotators, no Qwen-only items,
     no pairs with conflicting labels)
  C. the informative subset: queries whose positive lies outside the model's own top 10
  D. reference points on the sampled queries (pairwise accuracy, strict Success@K)
  E. n-way curves with other-mined negatives only, nested, on a fixed query set
  F. probe pass/fail against judged top-1 (full two-by-two table) and the error types of the pass-but-fail queries
  G. role overlap between query and negative from the benchmark's own triplets (not from the error labels)
  H. caption-clustered bootstrap intervals for the pairwise accuracy of Table 1

Usage: python scripts/16_svo_analyses.py          four dual encoders -> results/svo_analyses_dual_encoders.json
       python scripts/16_svo_analyses.py long     all six models     -> results/svo_analyses.json
       (Holm correction over the 8 or 12 model-and-direction cells respectively)
"""
import hashlib
import json
import sys
import numpy as np
from scipy.stats import wilcoxon, fisher_exact
from svo_eval.paths import RESULTS, DUAL_ENCODERS, MODELS, DIRECTIONS
from svo_eval.collection import load_collection
from svo_eval.embeddings import Scorer
from svo_eval import pools, metrics as M

LONG = len(sys.argv) > 1 and sys.argv[1] == "long"
SYSTEMS = MODELS if LONG else DUAL_ENCODERS
OUT = "svo_analyses" if LONG else "svo_analyses_dual_encoders"

coll = load_collection()
fp = pools.build_fixed_pool()
N_RANDOM, RANDOM_DRAWS = 10, 20
NWAY_KS = [1, 2, 5, 10]
NWAY_PERMS = 50
RAND_KS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, None]
ROLES = ["subject", "verb", "object"]
QWEN = {"Qwen25", "Qwen3"}


def seed_for(*parts):
    return int(hashlib.md5(str(parts).encode()).hexdigest(), 16) % (2 ** 32)


randomisation_p, holm = M.randomisation_p, M.holm


def wilcoxon_p(diff):
    d = np.asarray(diff, float)
    if np.all(d == 0):
        return 1.0
    return float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)


def summ(vals):
    vals = [v for v in vals if v is not None]
    return M.summary(vals) if vals else None


def paired(a, b):
    idx = [i for i in range(len(a)) if a[i] is not None and b[i] is not None]
    d = np.array([a[i] - b[i] for i in idx])
    mean, lo, hi = M.bootstrap_ci(d)
    return {"n": len(idx), "diff": 100 * mean, "lo": 100 * lo, "hi": 100 * hi,
            "p_randomisation": randomisation_p(d), "p_wilcoxon": wilcoxon_p(d),
            "n_positive": int((d > 0).sum()), "n_negative": int((d < 0).sum())}


def shared_roles(d, q, c):
    """Largest number of roles (0-3) that a triplet of the query shares with a triplet of the candidate;
    None when the candidate image has no caption in the collection."""
    qt = {coll.cap2trip[q]} if d == "t2i" else coll.img2trips[int(q)]
    ct = coll.img2trips.get(int(c), set()) if d == "t2i" else {coll.cap2trip[c]}
    if not ct:
        return None
    return max(sum(a == b for a, b in zip(t, u)) for t in qt for u in ct)


def unanimous_incorrect(entry):
    """True when every three-annotator label set of the pair says incorrect three times; None when the pair
    has no three-annotator labels (retrieved by the Qwen embedders only, or a text-to-image pair)."""
    votes = [lab["votes"] for lab in entry["labels"].values() if isinstance(lab, dict) and len(lab["votes"]) == 3]
    if not votes:
        return None
    return all(v == [-1, -1, -1] for v in votes)


# ---------------------------------------------------------------------------------------------------------------
out = {"matched": {}, "tests": {}, "nway": {}, "joint": {}, "errors": {}, "roles": {}, "reference": {}, "pool": {}}
perq = {}

# pool composition
for d in DIRECTIONS:
    n = sum(len(v) for v in fp[d].values())
    qwen_only = sum(1 for v in fp[d].values() for e in v.values() if set(e["retrieved_by"]) <= QWEN)
    conflicts = sum(1 for v in fp[d].values() for e in v.values() if e["labels"].get("disagreement"))
    sizes = sorted(len(v) for v in fp[d].values())
    out["pool"][d] = {"pairs": n, "qwen_only": qwen_only, "conflicting_labels": conflicts,
                      "median_per_query": sizes[len(sizes) // 2],
                      "correct": int(sum(bool(e["correct"]) for v in fp[d].values() for e in v.values()))}

for m in SYSTEMS:
    s = Scorer(m)
    out["matched"][m], out["nway"][m], out["joint"][m], out["errors"][m] = {}, {}, {}, {}
    out["roles"][m], out["reference"][m], perq[m] = {}, {}, {}
    pool_m = {d: pools.load_pool(m, d) for d in DIRECTIONS}
    for d in DIRECTIONS:
        universe_all = list(coll.images) if d == "t2i" else list(coll.captions)
        qs = pools.query_ids(d)
        rows = []
        for q in qs:
            gold = coll.gold(d, q)
            scores = s.all_scores(d, q)
            gidx = [s.index_of(d, g) for g in gold]
            g = float(scores[gidx].max())
            mask = np.ones(len(scores), bool)
            mask[gidx] = False
            neg_scores = scores[mask]
            rank = int((neg_scores > g).sum()) + 1                  # rank of the first relevant item
            entries = fp[d][q]
            neg_all = [c for c, e in entries.items() if not e["correct"] and c not in gold]
            neg_self = [c for c in neg_all if m in entries[c]["retrieved_by"]]
            neg_other = [c for c in neg_all if m not in entries[c]["retrieved_by"]]
            neg_other_dual = [c for c in neg_other if not set(entries[c]["retrieved_by"]) <= QWEN]
            neg_other_unan = [c for c in neg_other if unanimous_incorrect(entries[c])]
            neg_other_consistent = [c for c in neg_other if not entries[c]["labels"].get("disagreement")]

            def acc(negs):
                return float(np.mean(s.score(d, q, list(negs)) < g)) if len(negs) else None

            # benchmark negatives: against the best relevant item (as the pooled conditions) and against the
            # triplet's own positive (the probe's actual decision)
            if d == "t2i":
                bench = list(coll.cap2negimgs[q])
                trip = coll.df[coll.df["sentence"] == q]
                own = [float(a > b) for a, b in
                       (s.score_t2i(q, [int(r.pos_image_id), int(r.neg_image_id)]) for r in trip.itertuples(index=False))]
                bench_types = sorted(set(coll.cap2negtype[q].values()))
            else:
                bench = list(coll.img2negcaps.get(q, []))
                own = []
                trip = coll.df[coll.df["pos_image_id"] == int(q)]
                seen = set()
                for r in trip.itertuples(index=False):
                    for c2 in coll.img2caps.get(int(r.neg_image_id), set()):
                        if int(q) in coll.cap2imgs.get(c2, set()) or (r.sentence, c2) in seen:
                            continue
                        seen.add((r.sentence, c2))
                        a, b = s.score_i2t(q, [r.sentence, c2])
                        own.append(float(a > b))
                bench_types = sorted({("subject" if r.subj_neg else "verb" if r.verb_neg else "object")
                                      for r in trip.itertuples(index=False)})
            # random negatives: the same draws for every model, averaged over RANDOM_DRAWS draws of N_RANDOM items
            universe = [c for c in universe_all if c not in gold]
            rng = np.random.default_rng(seed_for(d, q, "random"))
            rand = float(np.mean([acc([universe[i] for i in rng.choice(len(universe), N_RANDOM, replace=False)])
                                  for _ in range(RANDOM_DRAWS)]))
            top1 = pool_m[d][q][0]
            # o3 label of the top-ranked item: from the model's own judged list, else from another model's list
            # that contains the same pair; None when the assessor returned no label for the pair
            o3_labels = [top1["o3"]] + [lab.get("o3") for lab in entries.get(top1["cand"], {"labels": {}})["labels"].values()
                                        if isinstance(lab, dict)]
            o3_labels = [x for x in o3_labels if x]
            row = {"q": str(q), "rank": rank, "top1_o3_correct": (o3_labels[0] == "correct") if o3_labels else None, "benchmark": acc(bench), "benchmark_own": float(np.mean(own)) if own else None,
                   "own_all": bool(all(own)) if own else None, "n_own": len(own), "own_hits": float(np.sum(own)),
                   "random": rand, "pool_other": acc(neg_other), "pool_self": acc(neg_self), "pool_all": acc(neg_all),
                   "pool_other_dual": acc(neg_other_dual), "pool_other_unanimous": acc(neg_other_unan),
                   "pool_other_consistent": acc(neg_other_consistent),
                   "n_other": len(neg_other), "n_other_qwen_only": len(neg_other) - len(neg_other_dual),
                   "n_other_unanimous": len(neg_other_unan), "n_self": len(neg_self), "n_bench": len(bench),
                   "other_beating": int(np.sum(s.score(d, q, neg_other) >= g)) if neg_other else 0,
                   "top1_correct": bool(pools.is_correct(top1)), "top1_error": pools.error_type(top1),
                   "bench_types": bench_types,
                   "roles_other": [shared_roles(d, q, c) for c in neg_other],
                   "roles_self": [shared_roles(d, q, c) for c in neg_self],
                   "roles_bench": [shared_roles(d, q, c) for c in bench],
                   "roles_random": [shared_roles(d, q, c) for c in
                                    [universe[i] for i in np.random.default_rng(seed_for(d, q, "roles")).choice(
                                        len(universe), 20, replace=False)]],
                   "_g": g, "_neg_scores": neg_scores,
                   "_other_scores": s.score(d, q, neg_other) if neg_other else np.array([])}
            rows.append(row)

        col = lambda k: [r[k] for r in rows]
        res = {k: summ(col(k)) for k in ["benchmark", "benchmark_own", "random", "pool_other", "pool_self", "pool_all",
                                         "pool_other_dual", "pool_other_unanimous", "pool_other_consistent"]}
        res["n_other_total"] = int(sum(col("n_other")))
        res["qwen_only_share_of_other"] = 100 * sum(col("n_other_qwen_only")) / max(1, sum(col("n_other")))
        res["unanimous_share_of_other"] = 100 * sum(col("n_other_unanimous")) / max(1, sum(col("n_other")))
        res["median_other_per_query"] = float(np.median(col("n_other")))
        res["median_bench_per_query"] = float(np.median(col("n_bench")))
        # consistency of the lower-bound argument: positive inside the top 10 => no other-mined negative above it
        inside = [r for r in rows if r["rank"] <= 10]
        res["violations_of_bound"] = int(sum(r["other_beating"] > 0 for r in inside))
        out["matched"][m][d] = res

        tests = {"benchmark_vs_other": paired(col("benchmark"), col("pool_other")),
                 "benchmark_own_vs_other": paired(col("benchmark_own"), col("pool_other")),
                 "random_vs_benchmark": paired(col("random"), col("benchmark")),
                 "random_vs_benchmark_own": paired(col("random"), col("benchmark_own")),
                 "other_vs_self": paired(col("pool_other"), col("pool_self")),
                 "benchmark_vs_all": paired(col("benchmark"), col("pool_all")),
                 "benchmark_vs_other_dual": paired(col("benchmark_own"), col("pool_other_dual")),
                 "benchmark_vs_other_unanimous": paired(col("benchmark_own"), col("pool_other_unanimous")),
                 "benchmark_vs_other_consistent": paired(col("benchmark_own"), col("pool_other_consistent"))}
        # C. informative subset: positive outside the model's top 10 (otherwise other-mined accuracy is 1 by construction)
        outside = [r for r in rows if r["rank"] > 10]
        tests["outside_top10"] = {
            "n_queries": len(outside),
            "benchmark": summ([r["benchmark"] for r in outside]),
            "benchmark_own": summ([r["benchmark_own"] for r in outside]),
            "pool_other": summ([r["pool_other"] for r in outside]),
            "random": summ([r["random"] for r in outside]),
            "test": paired([r["benchmark"] for r in outside], [r["pool_other"] for r in outside]),
            "test_own": paired([r["benchmark_own"] for r in outside], [r["pool_other"] for r in outside])}
        out["tests"].setdefault(m, {})[d] = tests

        # D. reference points on the sampled queries
        ranks = np.array(col("rank"))
        ref = {"S@1": M.summary(ranks <= 1), "S@5": M.summary(ranks <= 5), "S@10": M.summary(ranks <= 10),
               "pairwise_own_per_query": summ(col("benchmark_own")),
               "pairwise_own_over_decisions": 100 * sum(col("own_hits")) / max(1, sum(col("n_own"))),
               "n_decisions": int(sum(col("n_own"))),
               "judged_S@1": M.summary(np.array(col("top1_correct")))}
        o3 = [x for x in col("top1_o3_correct") if x is not None]
        ref["judged_S@1_o3"] = M.summary(np.array(o3)) if o3 else None      # over the queries whose top-1 has an o3 label
        ref["judged_S@1_o3_own_list_labelled"] = bool(any(j["o3"] for items in pool_m[d].values() for j in items))
        out["reference"][m][d] = ref

        # E. n-way with other-mined negatives only: nested prefixes of random orders, fixed query set
        fixed = [r for r in rows if r["n_other"] >= max(NWAY_KS)]
        nw = {"k": NWAY_KS, "n_queries_fixed": len(fixed), "other_fixed": [], "random_fixed": [],
              "other_capped_all_queries": [], "rand_k": [k if k else "all" for k in RAND_KS], "random_all_queries": [],
              "random_fixed_full": []}
        for k in NWAY_KS:
            of, rf, oc = [], [], []
            for r in rows:
                rng = np.random.default_rng(seed_for(d, r["q"], "nway"))
                osc = r["_other_scores"]
                wins = []
                for _ in range(NWAY_PERMS):
                    perm = rng.permutation(len(osc))[:k]
                    wins.append(float(np.all(osc[perm] < r["_g"])) if len(osc) else 1.0)
                oc.append(float(np.mean(wins)))
                if r["n_other"] >= max(NWAY_KS):
                    of.append(float(np.mean(wins)))
            nw["other_fixed"].append(M.summary(of))
            nw["other_capped_all_queries"].append(M.summary(oc))
        for k in RAND_KS:
            ra, rf = [], []
            for r in rows:
                ns = r["_neg_scores"]
                if k is None:
                    v = float(np.all(ns < r["_g"]))
                else:
                    rng = np.random.default_rng(seed_for(d, r["q"], k, "nway-random"))
                    v = float(np.mean([np.all(ns[rng.choice(len(ns), k, replace=False)] < r["_g"])
                                       for _ in range(NWAY_PERMS)]))
                ra.append(v)
                if r["n_other"] >= max(NWAY_KS):
                    rf.append(v)
            nw["random_all_queries"].append(M.summary(ra))
            nw["random_fixed_full"].append(M.summary(rf))
        nw["random_fixed"] = [nw["random_fixed_full"][RAND_KS.index(k)] for k in NWAY_KS]

        def equivalent_random(target, curve):
            """Number of random negatives at which the random curve falls to `target` (log-linear interpolation)."""
            n_all = (len(coll.images) if d == "t2i" else len(coll.captions)) - 1
            ks = [k if k else n_all for k in RAND_KS]
            ys = [c["mean"] for c in curve]
            for i in range(len(ks) - 1):
                if ys[i] >= target >= ys[i + 1]:
                    if ys[i] == ys[i + 1]:
                        return float(ks[i])
                    t = (ys[i] - target) / (ys[i] - ys[i + 1])
                    return float(np.exp(np.log(ks[i]) + t * (np.log(ks[i + 1]) - np.log(ks[i]))))
            return None

        nw["equivalent_random_k_for_10_other_fixed"] = equivalent_random(nw["other_fixed"][-1]["mean"], nw["random_fixed_full"])
        nw["equivalent_random_k_for_10_other_capped"] = equivalent_random(nw["other_capped_all_queries"][-1]["mean"],
                                                                         nw["random_all_queries"])
        out["nway"][m][d] = nw

        # F. probe outcome (own positive above every handcrafted negative of the query) against judged top-1
        joint = {"pass_ok": 0, "pass_fail": 0, "fail_ok": 0, "fail_fail": 0}
        err_pf = {t: 0 for t in ROLES + ["unlabelled"]}
        tested_role = {"in_tested_role": 0, "n": 0}
        for r in rows:
            p = "pass" if r["own_all"] else "fail"
            joint[f"{p}_{'ok' if r['top1_correct'] else 'fail'}"] += 1
            if r["own_all"] and not r["top1_correct"]:
                e = r["top1_error"] or "unlabelled"
                err_pf[e] += 1
                if e in ROLES:
                    tested_role["n"] += 1
                    tested_role["in_tested_role"] += int(e in r["bench_types"])
        joint["fisher_p"] = float(fisher_exact([[joint["pass_ok"], joint["pass_fail"]],
                                                [joint["fail_ok"], joint["fail_fail"]]])[1])
        joint["ok_given_pass"] = 100 * joint["pass_ok"] / max(1, joint["pass_ok"] + joint["pass_fail"])
        joint["ok_given_fail"] = 100 * joint["fail_ok"] / max(1, joint["fail_ok"] + joint["fail_fail"])
        joint["other_negatives_above_positive"] = int(sum(r["other_beating"] for r in rows))
        joint["queries_with_several_relevant"] = int(sum(len(coll.gold(d, q)) > 1 for q in qs))
        out["joint"][m][d] = joint
        # error types: all judged-incorrect top-10 items of the model's own list, and the pass-but-fail top-1 items
        all_err = {t: 0 for t in ROLES + ["unlabelled"]}
        for items in pool_m[d].values():
            for j in items:
                if not pools.is_correct(j):
                    all_err[pools.error_type(j) or "unlabelled"] += 1
        bench_mix = {t: 0 for t in ROLES}
        for r in rows:
            for t in r["bench_types"]:
                bench_mix[t] += 1
        out["errors"][m][d] = {"top10_incorrect": all_err, "pass_fail_top1": err_pf, "pass_fail_in_tested_role": tested_role,
                               "queries_testing_role": bench_mix}

        # G. role overlap from the benchmark triplets
        def role_dist(key):
            vals = [v for r in rows for v in r[key] if v is not None]
            n = len(vals)
            return {"n": n, "missing": sum(1 for r in rows for v in r[key] if v is None),
                    "share_ge2": 100 * sum(v >= 2 for v in vals) / max(1, n),
                    "share_eq2": 100 * sum(v == 2 for v in vals) / max(1, n),
                    "share_3": 100 * sum(v == 3 for v in vals) / max(1, n),
                    "share_le1": 100 * sum(v <= 1 for v in vals) / max(1, n)}
        out["roles"][m][d] = {k: role_dist("roles_" + k) for k in ["other", "self", "bench", "random"]}

        perq[m][d] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]

# Holm correction over the eight model-and-direction cells, per comparison and per test
for comp in ["benchmark_vs_other", "benchmark_own_vs_other", "random_vs_benchmark", "random_vs_benchmark_own",
             "other_vs_self", "benchmark_vs_all",
             "benchmark_vs_other_dual", "benchmark_vs_other_unanimous", "benchmark_vs_other_consistent"]:
    for test in ["p_randomisation", "p_wilcoxon"]:
        adj = holm({(m, d): out["tests"][m][d][comp][test] for m in SYSTEMS for d in DIRECTIONS})
        for (m, d), p in adj.items():
            out["tests"][m][d][comp][test + "_holm"] = p
for key in ["test", "test_own"]:
    adj = holm({(m, d): out["tests"][m][d]["outside_top10"][key]["p_randomisation"] for m in SYSTEMS for d in DIRECTIONS})
    for (m, d), p in adj.items():
        out["tests"][m][d]["outside_top10"][key]["p_randomisation_holm"] = p
adj = holm({(m, d): out["joint"][m][d]["fisher_p"] for m in SYSTEMS for d in DIRECTIONS})
for (m, d), p in adj.items():
    out["joint"][m][d]["fisher_p_holm"] = p
out["benchmark_type_share"] = {t: 100 * float(coll.df[c].mean()) for t, c in
                               [("subject", "subj_neg"), ("verb", "verb_neg"), ("object", "obj_neg")]}

# H. caption-clustered bootstrap for pairwise accuracy (triplets that share a caption are resampled together)
out["pairwise_clustered"] = {}
df = coll.df
cap_id = df["sentence"].map(coll.cap_index).to_numpy()
types = {"overall": np.ones(len(df), bool), "subject": df["subj_neg"].to_numpy(bool),
         "verb": df["verb_neg"].to_numpy(bool), "object": df["obj_neg"].to_numpy(bool)}
for m in SYSTEMS:
    s = Scorer(m)
    e = s.e
    ci = np.array([e.cap_index[c] for c in df["sentence"]])
    pi = np.array([e.img_index[int(i)] for i in df["pos_image_id"]])
    ni = np.array([e.img_index[int(i)] for i in df["neg_image_id"]])
    corr = (np.einsum("ij,ij->i", e.cap[ci], e.img[pi]) > np.einsum("ij,ij->i", e.cap[ci], e.img[ni])).astype(float)
    out["pairwise_clustered"][m] = {}
    for name, sel in types.items():
        hits = np.bincount(cap_id[sel], weights=corr[sel], minlength=len(coll.captions))
        cnt = np.bincount(cap_id[sel], minlength=len(coll.captions)).astype(float)
        keep = cnt > 0
        hits, cnt = hits[keep], cnt[keep]
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(cnt), len(cnt))
            boots.append(hits[idx].sum() / cnt[idx].sum())
        out["pairwise_clustered"][m][name] = {"mean": 100 * hits.sum() / cnt.sum(), "lo": 100 * float(np.quantile(boots, 0.025)),
                                              "hi": 100 * float(np.quantile(boots, 0.975)), "n_triplets": int(cnt.sum()),
                                              "n_captions": int(len(cnt))}

json.dump(out, open(RESULTS / f"{OUT}.json", "w"), indent=1)
json.dump(perq, open(RESULTS / f"{OUT}_perquery.json", "w"))

# ---------------------------------------------------------------------------------------------------------------
r1 = lambda x: "--" if x is None else f"{x['mean']:.1f}"
print("pool", out["pool"])
for m in SYSTEMS:
    for d in DIRECTIONS:
        a, t, ref = out["matched"][m][d], out["tests"][m][d], out["reference"][m][d]
        print(f"\n{m} {d}: bench {r1(a['benchmark'])} (own {r1(a['benchmark_own'])}) random {r1(a['random'])} "
              f"other {r1(a['pool_other'])} self {r1(a['pool_self'])} all {r1(a['pool_all'])} | other w/o Qwen-only "
              f"{r1(a['pool_other_dual'])} unanimous {r1(a['pool_other_unanimous'])} consistent {r1(a['pool_other_consistent'])}"
              f" | Qwen-only share {a['qwen_only_share_of_other']:.0f}% unanimous share {a['unanimous_share_of_other']:.0f}%"
              f" | bound violations {a['violations_of_bound']}")
        for comp in ["benchmark_vs_other", "benchmark_own_vs_other", "random_vs_benchmark", "other_vs_self",
                     "benchmark_vs_other_dual", "benchmark_vs_other_unanimous"]:
            c = t[comp]
            print(f"   {comp:30s} diff {c['diff']:5.1f} [{c['lo']:5.1f},{c['hi']:5.1f}] n {c['n']:3d} "
                  f"rand p {c['p_randomisation']:.5f} (Holm {c['p_randomisation_holm']:.5f}) "
                  f"wilcoxon {c['p_wilcoxon']:.5f} (Holm {c['p_wilcoxon_holm']:.5f})")
        o = t["outside_top10"]
        print(f"   outside top 10 (own positive): bench {r1(o['benchmark_own'])} diff {o['test_own']['diff']:.1f} "
              f"p {o['test_own']['p_randomisation']:.5f} (Holm {o['test_own']['p_randomisation_holm']:.5f})")
        print(f"   outside top 10: n {o['n_queries']} bench {r1(o['benchmark'])} other {r1(o['pool_other'])} random {r1(o['random'])} "
              f"diff {o['test']['diff']:.1f} p {o['test']['p_randomisation']:.5f} (Holm {o['test']['p_randomisation_holm']:.5f})")
        print(f"   sampled queries: S@1 {r1(ref['S@1'])} S@10 {r1(ref['S@10'])} judged S@1 {r1(ref['judged_S@1'])} "
              f"pairwise(own) per-query {r1(ref['pairwise_own_per_query'])} over decisions {ref['pairwise_own_over_decisions']:.1f} (n {ref['n_decisions']})")
        nw = out["nway"][m][d]
        print(f"   n-way other-mined fixed (n={nw['n_queries_fixed']}):", [round(x['mean'], 1) for x in nw["other_fixed"]],
              "random same queries:", [round(x['mean'], 1) for x in nw["random_fixed"]],
              "| capped all queries:", [round(x['mean'], 1) for x in nw["other_capped_all_queries"]],
              "| 10 other-mined = random k", nw["equivalent_random_k_for_10_other_fixed"], nw["equivalent_random_k_for_10_other_capped"])
        print("   random all queries:", [round(x['mean'], 1) for x in nw["random_all_queries"]])
        print("   joint", out["joint"][m][d], "errors", out["errors"][m][d])
        print("   roles", {k: (round(v['share_ge2'], 1), v['n'], v['missing']) for k, v in out["roles"][m][d].items()})
    print(m, "pairwise clustered", {k: (round(v['mean'], 1), round((v['hi'] - v['lo']) / 2, 2)) for k, v in out["pairwise_clustered"][m].items()})
print("benchmark negative types (share of triplets)", out["benchmark_type_share"])
print("saved")
