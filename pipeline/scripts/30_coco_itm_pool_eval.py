"""ITM decisions on the labelled COCO pool against the assessor's labels, for every system with predictions, in the
form of scripts/07_itm_pools.py: accuracy on all labelled pairs, on the pairs the system itself retrieved (self) and
on the others, TPR on labelled-correct and TNR on labelled-incorrect pairs, share predicted "match".

Predictions: results/coco_itm_preds_{Qwen3,BLIP2,FLAVA}_pool.csv (pred_yes) and results/Qwen25_itm_pool.csv
(model_pred yes/no from the GPU server; unanswered rows excluded and counted). Labels: results/coco_pool_labels_<judge>.json.
Usage: python scripts/30_coco_itm_pool_eval.py [o3|human]   -> results/coco_itm_pools_<judge>.json"""
import json
import sys
import numpy as np
import pandas as pd
from svo_eval.paths import RESULTS
from svo_eval import metrics as M

judge = sys.argv[1] if len(sys.argv) > 1 else "o3"
labels = json.load(open(RESULTS / f"coco_pool_labels_{judge}.json", encoding="utf-8"))
pool = json.load(open(RESULTS / "coco_pool_candidates.json", encoding="utf-8"))
SOURCES = {"BLIP2": ("coco_itm_preds_BLIP2_pool.csv", "pred_yes"), "FLAVA": ("coco_itm_preds_FLAVA_pool.csv", "pred_yes"),
           "Qwen25": ("Qwen25_itm_pool.csv", "model_pred"), "Qwen3": ("coco_itm_preds_Qwen3_pool.csv", "pred_yes")}
out = {}
for model, (fname, col) in SOURCES.items():
    path = RESULTS / fname
    if not path.exists():
        print(model, "no predictions yet"); continue
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    out[model] = {}
    for d in ("t2i", "i2t"):
        rows, unanswered, unlabelled = [], 0, 0
        for r in df[df["direction"] == d].itertuples(index=False):
            q, c = str(r.query), str(r.candidate)
            lab = labels.get(d, {}).get(q, {}).get(c, "")
            if not lab:
                unlabelled += 1; continue
            v = str(getattr(r, col)).strip().lower()
            if col == "pred_yes":
                pred = None if v == "" else v in ("1", "true")
            else:
                pred = {"yes": True, "no": False}.get(v)
            if pred is None:
                unanswered += 1; continue
            entry = pool[d].get(q, {}).get(c, {})
            rows.append((lab == "correct", pred, model in entry.get("retrieved_by", {})))
        if not rows:
            out[model][d] = None; continue
        y = np.array([a for a, _, _ in rows]); p = np.array([b for _, b, _ in rows]); selfm = np.array([s for _, _, s in rows])
        ok = y == p
        res = {"pool_all": M.summary(ok), "pool_self": M.summary(ok[selfm]) if selfm.any() else None,
               "pool_other": M.summary(ok[~selfm]) if (~selfm).any() else None,
               "positives_recall": M.summary(p[y]) if y.any() else None, "negatives_recall": M.summary(~p[~y]) if (~y).any() else None,
               "share_predicted_yes": M.summary(p), "n_pairs": int(len(rows)), "n_unanswered": int(unanswered), "n_unlabelled": int(unlabelled)}
        out[model][d] = res
        print(model, d, "all %.1f self %s other %s TPR %.1f TNR %.1f yes %.1f n=%d (unanswered %d, unlabelled %d)" % (
            res["pool_all"]["mean"], None if res["pool_self"] is None else round(res["pool_self"]["mean"], 1),
            None if res["pool_other"] is None else round(res["pool_other"]["mean"], 1),
            res["positives_recall"]["mean"] if res["positives_recall"] else float("nan"),
            res["negatives_recall"]["mean"] if res["negatives_recall"] else float("nan"), res["share_predicted_yes"]["mean"], len(rows), unanswered, unlabelled))
json.dump(out, open(RESULTS / f"coco_itm_pools_{judge}.json", "w"), indent=1)
print("saved", RESULTS / f"coco_itm_pools_{judge}.json")
