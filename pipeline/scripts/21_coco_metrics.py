"""COCO val2017 + SugarCrepe: strict retrieval metrics, SugarCrepe pairwise accuracy per subset, the benchmark and
random matched-negative conditions, and the top-10 pool of 100 queries per direction to be judged.

Writes results/coco_metrics.json and results/coco_pool_candidates.json (pairs to judge, with the systems that
retrieved each). The mined-negative conditions are computed by scripts/22_coco_pool_analysis.py once labels exist."""
import json
import numpy as np
from tqdm import tqdm
from svo_eval.paths import RESULTS, DIRECTIONS
from svo_eval.coco import load_coco, SUBSETS
from svo_eval.coco_scoring import CocoScorer, available_models
from svo_eval import metrics as M

coll = load_coco()
models = available_models()
print("models with COCO embeddings:", models)
rng = np.random.default_rng(0)
probe_images = sorted({p["image_id"] for p in coll.probe if p["image_id"] in coll.img_index})
image_queries = [int(x) for x in rng.choice(probe_images, 100, replace=False)]
caption_pool = sorted({c for i in image_queries for c in coll.img2caps[i]})       # captions of probe images
caption_queries = [str(x) for x in rng.choice(caption_pool, 100, replace=False)]
queries = {"t2i": caption_queries, "i2t": image_queries}
out, pool = {}, {"t2i": {}, "i2t": {}}
items_pairwise = {}
perq = {}                                  # per-query accuracies of the random and benchmark conditions (for paired tests)
for m in models:
    s = CocoScorer(m)
    out[m] = {}
    for d in DIRECTIONS:
        ranks = [M.first_gold_rank(s.topk(q, d, 20), coll.gold(d, q)) for q in tqdm(coll.evaluable_queries(d), desc=f"{m} {d}")]
        out[m][d] = {f"S@{k}": M.summary(M.success_vector(ranks, k)) for k in [1, 5, 10, 20]}
        out[m][d]["MRR@20"] = M.summary(M.rr_vector(ranks, 20))
        for q in queries[d]:
            top = s.topk(q, d, 10)
            bucket = pool[d].setdefault(str(q), {})
            for r, c in enumerate(top):
                bucket.setdefault(str(c), {"retrieved_by": {}})["retrieved_by"][m] = r + 1
    # SugarCrepe pairwise: image scores its caption above the edited negative caption
    corr, subs = [], []
    for p in coll.probe:
        if p["image_id"] not in s.img_index:
            continue
        sc = s.score("i2t", p["image_id"], [p["caption"], p["negative_caption"]])
        corr.append(bool(sc[0] > sc[1])); subs.append(p["subset"])
    corr, subs = np.array(corr), np.array(subs)
    items_pairwise[m] = [bool(x) for x in corr]           # aligned to coll.probe (probe images present in the embeddings)
    out[m]["pairwise"] = {"overall": M.summary(corr), **{sub: M.summary(corr[subs == sub]) for sub in SUBSETS if (subs == sub).any()}}
    # matched negatives, benchmark (SugarCrepe negatives of the image) and random, I->T; random only for T->I
    bench_acc, rand_acc = {"t2i": [], "i2t": []}, {"t2i": [], "i2t": []}
    perq[m] = {"t2i": {"random": {}, "benchmark": {}}, "i2t": {"random": {}, "benchmark": {}}}
    negs_by_img = {}
    for p in coll.probe:
        negs_by_img.setdefault(p["image_id"], set()).add(p["negative_caption"])
    for d in DIRECTIONS:
        universe = list(s.candidates(d))
        for q in queries[d]:
            gold = coll.gold(d, q)
            g = float(np.max(s.score(d, q, list(gold))))
            pool_u = [c for c in universe if c not in gold]
            rand = [pool_u[i] for i in rng.choice(len(pool_u), 10, replace=False)]
            rand_acc[d].append(float(np.mean(s.score(d, q, rand) < g)))
            perq[m][d]["random"][str(q)] = rand_acc[d][-1]
            if d == "i2t":
                bench = list(negs_by_img.get(int(q), []))
                if bench:
                    bench_acc[d].append(float(np.mean(s.score(d, q, bench) < g)))
                    perq[m][d]["benchmark"][str(q)] = bench_acc[d][-1]
    out[m]["matched"] = {d: {"random": M.summary(rand_acc[d]), "benchmark": M.summary(bench_acc[d]) if bench_acc[d] else None}
                         for d in DIRECTIONS}
    print(m, {d: round(out[m][d]["S@1"]["mean"], 2) for d in DIRECTIONS}, "pairwise %.2f" % out[m]["pairwise"]["overall"]["mean"],
          {d: (round(out[m]["matched"][d]["random"]["mean"], 1), None if out[m]["matched"][d]["benchmark"] is None else round(out[m]["matched"][d]["benchmark"]["mean"], 1)) for d in DIRECTIONS})

n_pairs = {d: sum(len(v) for v in pool[d].values()) for d in DIRECTIONS}
out["pool"] = {"queries": queries, "unique_pairs": n_pairs, "models": models}
json.dump(out, open(RESULTS / "coco_metrics.json", "w"), indent=1)
json.dump(pool, open(RESULTS / "coco_pool_candidates.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
json.dump(perq, open(RESULTS / "coco_matched_perquery.json", "w", encoding="utf-8"), ensure_ascii=False)
json.dump(items_pairwise, open(RESULTS / "coco_pairwise_items.json", "w"))
print("pool sizes (unique pairs):", n_pairs, "saved")
