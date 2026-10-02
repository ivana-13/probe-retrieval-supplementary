"""Two like-for-like checks for the paper.

1. Assessor against individual annotators. On the three-annotator lists (image-to-text lists of the four dual encoders)
   Cohen's kappa of o3 with each annotator separately, next to the annotator-annotator kappas; agreement with a majority
   vote is systematically higher than agreement with an individual, so this is the comparison on equal terms.
2. Probe order against retrieval order under human labels. Kendall's tau between pairwise accuracy and the human-judged
   Success@1 on the 100 pool queries (six embedding models), and the paired difference between FLAVA and SigLIP2 in
   human-judged Success@1 on the same queries.
Output: results/assessor_and_order_checks.json"""
import json
import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import cohen_kappa_score, f1_score
from svo_eval.paths import RESULTS, MODELS, DUAL_ENCODERS, DIRECTIONS
from svo_eval import pools, metrics as M

out = {}
votes, o3 = [], []
for m in DUAL_ENCODERS:
    for items in pools.load_pool(m, "i2t").values():
        for j in items:
            if j["o3"] is None:
                continue
            votes.append(j["votes"])
            o3.append(j["o3"] == "correct")
votes, o3 = np.array(votes), np.array(o3)
yes = votes == 1
out["assessor"] = {
    "n_pairs": int(len(o3)),
    "o3_vs_each_annotator": [float(cohen_kappa_score(yes[:, k], o3)) for k in range(3)],
    "annotator_vs_annotator": [float(cohen_kappa_score(yes[:, a], yes[:, b])) for a, b in [(0, 1), (0, 2), (1, 2)]],
    "o3_vs_majority": float(cohen_kappa_score(yes.sum(1) >= 2, o3)),
}
print("o3 vs annotators 1-3:", [round(x, 3) for x in out["assessor"]["o3_vs_each_annotator"]],
      "| annotator pairs:", [round(x, 3) for x in out["assessor"]["annotator_vs_annotator"]],
      "| o3 vs majority:", round(out["assessor"]["o3_vs_majority"], 3), "| n", len(o3))

A = json.load(open(RESULTS / "svo_analyses.json"))
order = json.load(open(RESULTS / "probe_vs_retrieval_order.json"))
perq = json.load(open(RESULTS / "svo_analyses_perquery.json"))
out["order"] = {}
for d in DIRECTIONS:
    pw = [order["svo"]["pairwise"][m] for m in MODELS]
    judged = [A["reference"][m][d]["judged_S@1"]["mean"] for m in MODELS]
    strict_pool = [A["reference"][m][d]["S@1"]["mean"] for m in MODELS]
    tau, p = kendalltau(pw, judged)
    tau_s, p_s = kendalltau(pw, [order["svo"]["S@1"][d][m] for m in MODELS])
    a = np.array([r["top1_correct"] for r in perq["FLAVA"][d]], float)
    b = np.array([r["top1_correct"] for r in perq["SigLIP2"][d]], float)
    assert [r["q"] for r in perq["FLAVA"][d]] == [r["q"] for r in perq["SigLIP2"][d]]
    diff = b - a
    out["order"][d] = {"tau_pairwise_vs_human_judged_S@1": float(tau), "p": float(p),
                       "tau_pairwise_vs_strict_S@1_full": float(tau_s), "p_strict": float(p_s),
                       "human_judged_S@1": dict(zip(MODELS, judged)), "strict_S@1_pool": dict(zip(MODELS, strict_pool)),
                       "siglip2_minus_flava_human_judged": 100 * float(diff.mean()),
                       "siglip2_minus_flava_p_randomisation": M.randomisation_p(diff)}
    print(d, "tau (human-judged S@1) %.2f p %.2f | tau (strict, full) %.2f p %.2f | SigLIP2 - FLAVA human-judged %.0f points, p %.3f"
          % (tau, p, tau_s, p_s, 100 * diff.mean(), out["order"][d]["siglip2_minus_flava_p_randomisation"]),
          "| judged", dict(zip(MODELS, [round(x) for x in judged])))


# 3. Query-clustered intervals for the assessor-agreement table. A pair retrieved by several models appears in several
#    lists and the pairs of one query are not independent, so whole queries are resampled (2,000 resamples).
def clustered(groups, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g for g in groups if len(g[0])]
    f1s, ks = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(groups), len(groups))
        t = np.concatenate([groups[i][0] for i in idx])
        p = np.concatenate([groups[i][1] for i in idx])
        f1s.append(f1_score(t, p))
        ks.append(cohen_kappa_score(t, p))
    q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    return {"n_queries": len(groups), "f1_ci": q(f1s), "kappa_ci": q(ks)}


out["clustered_ci"] = {}
for judge in ["o3", "gpt4o"]:
    for d in DIRECTIONS:
        by_q = {}
        for m in DUAL_ENCODERS:
            for q, items in pools.load_pool(m, d).items():
                for j in items:
                    if j[judge] is None:
                        continue
                    g = by_q.setdefault(q, ([], []))
                    g[0].append(pools.is_correct(j))
                    g[1].append(j[judge] == "correct")
        out["clustered_ci"][f"{judge}_{d}"] = clustered([(np.array(a, bool), np.array(b, bool)) for a, b in by_q.values()])
# o3 against the expert on the lists of the two Qwen embedders (single-annotator lists, not part of the table)
out["o3_vs_expert_qwen_lists"] = {}
for m in [x for x in MODELS if x not in DUAL_ENCODERS]:
    for d in DIRECTIONS:
        t, p = [], []
        for items in pools.load_pool(m, d).values():
            for j in items:
                if j["o3"] is not None:
                    t.append(pools.is_correct(j))
                    p.append(j["o3"] == "correct")
        out["o3_vs_expert_qwen_lists"][f"{m}_{d}"] = {"kappa": float(cohen_kappa_score(t, p)), "n": len(t)}
print("o3 vs expert on the Qwen lists:", {k: round(v["kappa"], 3) for k, v in out["o3_vs_expert_qwen_lists"].items()})
# share of list entries without a parseable assessor label (dual encoders' lists)
out["unparseable_share"] = {}
for judge in ["o3", "gpt4o"]:
    for d in DIRECTIONS:
        flags = [j[judge] is None for m in DUAL_ENCODERS for items in pools.load_pool(m, d).values() for j in items]
        out["unparseable_share"][f"{judge}_{d}"] = 100 * float(np.mean(flags))
print("unparseable (%):", {k: round(v, 1) for k, v in out["unparseable_share"].items()})
human = json.load(open(RESULTS / "coco_pool_labels_human.json", encoding="utf-8"))
llm = json.load(open(RESULTS / "coco_pool_labels_o3.json", encoding="utf-8"))
sample = json.load(open(RESULTS / "coco_human_sample.json", encoding="utf-8"))
VALID = {"correct", "incorrect", "subject incorrect", "verb incorrect", "object incorrect"}
groups = []
for d in DIRECTIONS:
    for q, s in sample[d].items():
        t, p = [], []
        for c in s["to_label"]:
            h, l = human.get(d, {}).get(q, {}).get(c, ""), llm.get(d, {}).get(q, {}).get(c, "")
            if h in VALID and l in VALID:
                t.append(h == "correct")
                p.append(l == "correct")
        groups.append((np.array(t, bool), np.array(p, bool)))
out["clustered_ci"]["o3_coco"] = clustered(groups)
for k, v in out["clustered_ci"].items():
    print(k, "queries", v["n_queries"], "F1 [%.2f, %.2f] kappa [%.2f, %.2f]" % (*v["f1_ci"], *v["kappa_ci"]))
json.dump(out, open(RESULTS / "assessor_and_order_checks.json", "w"), indent=1)
