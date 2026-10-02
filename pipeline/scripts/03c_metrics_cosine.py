"""Robustness check: strict Success@K and pairwise accuracy of CLIP, FLAVA and SigLIP2 under cosine similarity
(L2-normalised embeddings) instead of the raw dot product that produced the judged lists. Full-collection metrics only:
pool-based conditions are not recomputed because the judged negatives were mined under the dot-product rankings."""
import json
import numpy as np
from tqdm import tqdm
from svo_eval import embeddings as E
from svo_eval.paths import RESULTS, DIRECTIONS
from svo_eval.collection import load_collection
from svo_eval import metrics as M

for m in ["CLIP", "FLAVA", "SigLIP2"]:
    E.NORMALISE[m] = True
E.load_embeddings.cache_clear()
coll = load_collection()
out = {}
for m in ["CLIP", "FLAVA", "SigLIP2"]:
    s = E.Scorer(m)
    assert abs(float(np.linalg.norm(s.e.cap[0])) - 1.0) < 1e-4
    out[m] = {}
    for d in DIRECTIONS:
        ranks = [M.first_gold_rank(s.topk(q, d, 20), coll.gold(d, q)) for q in tqdm(coll.evaluable_queries(d), desc=f"{m} {d} cosine")]
        out[m][d] = {f"S@{k}": M.summary(M.success_vector(ranks, k)) for k in [1, 5, 10, 20]}
        out[m][d]["MRR@20"] = M.summary(M.rr_vector(ranks, 20))
    corr, types = [], []
    for r in coll.df.itertuples(index=False):
        sc = s.score_t2i(r.sentence, [int(r.pos_image_id), int(r.neg_image_id)])
        corr.append(sc[0] > sc[1]); types.append([r.subj_neg, r.verb_neg, r.obj_neg])
    corr, types = np.array(corr), np.array(types, bool)
    out[m]["pairwise"] = {"overall": M.summary(corr), "subject": M.summary(corr[types[:, 0]]),
                          "verb": M.summary(corr[types[:, 1]]), "object": M.summary(corr[types[:, 2]])}
    print(m, "cosine: S@1 t2i %.2f i2t %.2f, pairwise %.2f" % (out[m]["t2i"]["S@1"]["mean"], out[m]["i2t"]["S@1"]["mean"], out[m]["pairwise"]["overall"]["mean"]))
json.dump(out, open(RESULTS / "full_metrics_cosine.json", "w"), indent=1)
print("saved")
