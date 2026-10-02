"""Threshold-free check of the two matching heads: AUC of the match probability on the benchmark pairs and on the judged
pools of both benchmarks, and the share of incorrect pairs in each pool (the accuracy of answering "no match" always).
Output: results/itm_head_auc.json"""
import json
import pandas as pd
from sklearn.metrics import roc_auc_score
from svo_eval.paths import RESULTS, DIRECTIONS

labels = json.load(open(RESULTS / "coco_pool_labels_o3.json", encoding="utf-8"))
out = {}
for m in ["BLIP2", "FLAVA"]:
    out[m] = {}
    # SVO-Probes pool (human labels)
    df = pd.read_csv(RESULTS / f"itm_pool_preds_{m}.csv")
    df = df.dropna(subset=["p_match"])
    for d in DIRECTIONS:
        x = df[df["direction"] == d]
        y = x["human_correct"].astype(str).str.lower().eq("true")
        out[m][f"svo_pool_{d}"] = {"auc": float(roc_auc_score(y, x["p_match"])), "share_incorrect": 100 * float(1 - y.mean()), "n": int(len(x))}
    # SugarCrepe pairs
    df = pd.read_csv(RESULTS / f"coco_itm_preds_{m}_sugarcrepe.csv")
    y = df["is_positive"].astype(str).str.lower().eq("true")
    out[m]["sugarcrepe"] = {"auc": float(roc_auc_score(y, df["p_match"])), "n": int(len(df))}
    # COCO pool (o3 labels)
    df = pd.read_csv(RESULTS / f"coco_itm_preds_{m}_pool.csv", dtype={"query": str, "candidate": str})
    for d in DIRECTIONS:
        x = df[df["direction"] == d].copy()
        x["label"] = [labels[d].get(str(q), {}).get(str(c)) for q, c in zip(x["query"], x["candidate"])]
        x = x.dropna(subset=["label", "p_match"])
        y = x["label"].eq("correct")
        out[m][f"coco_pool_{d}"] = {"auc": float(roc_auc_score(y, x["p_match"])), "share_incorrect": 100 * float(1 - y.mean()), "n": int(len(x))}
    print(m, {k: (round(v["auc"], 2), round(v.get("share_incorrect", float("nan")), 1), v["n"]) for k, v in out[m].items()})
json.dump(out, open(RESULTS / "itm_head_auc.json", "w"), indent=1)
