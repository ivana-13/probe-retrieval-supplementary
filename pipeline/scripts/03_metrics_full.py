"""Full-collection retrieval metrics with bootstrap intervals and pairwise accuracy.

Pairwise accuracy is the cosine decision for the five embedding systems; for Qwen2.5 it is the A/B answer of the
zero-shot Qwen2.5-VL-7B-Instruct model recorded on the VM, over the answered triplets only (svo_eval.qwen_csv).
ITM accuracy on the benchmark lives in scripts/03b_itm_benchmark.py.
"""
import json
import numpy as np
from tqdm import tqdm
from svo_eval.paths import RESULTS, MODELS, DIRECTIONS
from svo_eval.collection import load_collection
from svo_eval.embeddings import Scorer
from svo_eval.qwen_csv import load_qwen25_pairwise
from svo_eval import metrics as M

coll = load_collection()
out = {}
KS = [1, 5, 10, 20]
for m in MODELS:
    s = Scorer(m)
    out[m] = {}
    for d in DIRECTIONS:
        queries = coll.evaluable_queries(d)
        ranks, role_hits = [], {"subject": [], "verb": [], "object": []}
        for q in tqdm(queries, desc=f"{m} {d}"):
            gold = coll.gold(d, q)
            top = s.topk(q, d, 20)
            ranks.append(M.first_gold_rank(top, gold))
            qtrips = {coll.cap2trip[q]} if d == "t2i" else coll.img2trips[q]
            for r, name in enumerate(["subject", "verb", "object"]):
                qvals = {t[r] for t in qtrips}
                hits = []
                for c in top[:10]:
                    ctrips = coll.img2trips.get(c, set()) if d == "t2i" else {coll.cap2trip[c]}
                    hits.append(any(t[r] in qvals for t in ctrips))
                role_hits[name].append(float(np.mean(hits)))
        res = {f"S@{k}": M.summary(M.success_vector(ranks, k)) for k in KS}
        res["MRR@20"] = M.summary(M.rr_vector(ranks, 20))
        res["role_hits@10"] = {k: M.summary(v) for k, v in role_hits.items()}
        out[m][d] = res
        print(m, d, {k: round(v["mean"], 2) for k, v in res.items() if k.startswith("S@") or k == "MRR@20"})
    meta = {}
    if m == "Qwen25":
        pw = load_qwen25_pairwise()
        corr, types = pw["correct"], pw["types"]
        meta = {"source": "VM A/B prompt, zero-shot Qwen2.5-VL-7B-Instruct", "n_rows": pw["n_rows"],
                "n_unanswered": pw["n_unanswered"]}
    else:
        corr, types = [], []
        for r in coll.df.itertuples(index=False):
            sc = s.score_t2i(r.sentence, [int(r.pos_image_id), int(r.neg_image_id)])
            corr.append(sc[0] > sc[1])
            types.append([r.subj_neg, r.verb_neg, r.obj_neg])
        corr = np.array(corr)
        types = np.array(types, bool)
    out[m]["pairwise"] = {"overall": M.summary(corr), "subject": M.summary(corr[types[:, 0]]),
                          "verb": M.summary(corr[types[:, 1]]), "object": M.summary(corr[types[:, 2]]), "meta": meta}
    print(m, "pairwise", round(out[m]["pairwise"]["overall"]["mean"], 2), meta)
    if m == "Qwen25":
        # the fine-tuned embedder's own pairwise decision (cosine), comparable to the other embedders
        corr, types = [], []
        for r in coll.df.itertuples(index=False):
            sc = s.score_t2i(r.sentence, [int(r.pos_image_id), int(r.neg_image_id)])
            corr.append(sc[0] > sc[1])
            types.append([r.subj_neg, r.verb_neg, r.obj_neg])
        corr, types = np.array(corr), np.array(types, bool)
        out[m]["pairwise_embedding"] = {"overall": M.summary(corr), "subject": M.summary(corr[types[:, 0]]),
                                        "verb": M.summary(corr[types[:, 1]]), "object": M.summary(corr[types[:, 2]])}
        print(m, "pairwise (fine-tuned embedder, cosine)", round(out[m]["pairwise_embedding"]["overall"]["mean"], 2))

json.dump(out, open(RESULTS / "full_metrics.json", "w"), indent=1)
print("saved", RESULTS / "full_metrics.json")
