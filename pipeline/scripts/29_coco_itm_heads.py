"""BLIP-2 (LAVIS) and FLAVA matching heads on the second collection, with the heads and threshold of SVO-Probes
(svo_eval/itm_heads.py): the SugarCrepe pairs (benchmark ITM: (image, caption) -> match, (image, negative) -> no
match) and the six-system COCO pool (scored against labels later by scripts/30_coco_itm_pool_eval.py).

Usage: python scripts/29_coco_itm_heads.py BLIP2|FLAVA sugarcrepe|pool      (BLIP2 runs in lavis-env)
Outputs results/coco_itm_preds_<model>_<mode>.csv with p_match and pred_yes per pair (empty where the image is
unusable) and, for sugarcrepe, the summary in results/coco_itm.json[<model>]["sugarcrepe"]. The CSV is written at the
end of the run (the heads take a few minutes on these sizes)."""
import json
import sys
from functools import lru_cache
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from svo_eval.paths import RESULTS
from svo_eval.coco import load_coco, SUBSETS
from svo_eval.itm_heads import MatchingHead, THRESHOLD, MIN_SIDE
from svo_eval import metrics as M

MODEL, MODE = sys.argv[1], sys.argv[2]
coll = load_coco()


@lru_cache(maxsize=256)
def coco_image(image_id):
    try:
        im = Image.open(coll.image_path(image_id)).convert("RGB")
    except Exception:
        return None
    return None if min(im.size) < MIN_SIDE else im


if MODE == "sugarcrepe":
    jobs = []
    for k, p in enumerate(coll.probe):
        jobs.append((k, p["subset"], p["image_id"], p["caption"], True))
        jobs.append((k, p["subset"], p["image_id"], p["negative_caption"], False))
    pairs = [(j[2], j[3]) for j in jobs]
    header = ["pair_index", "subset", "image_id", "text", "is_positive"]
elif MODE == "pool":
    pp = json.load(open(RESULTS / "coco_pool_pairs.json", encoding="utf-8"))
    jobs = []
    for d in ("t2i", "i2t"):
        for q, c in pp[d]:
            img, cap = (int(c), str(q)) if d == "t2i" else (int(q), str(c))
            jobs.append((d, q, c, img, cap))
    pairs = [(j[3], j[4]) for j in jobs]
    header = ["direction", "query", "candidate", "image_id", "caption"]
else:
    raise SystemExit("mode: sugarcrepe | pool")

head = MatchingHead(MODEL)
probs = head.match_probs(pairs, progress=lambda it, total: tqdm(it, total=total, desc=f"{MODEL} {MODE}"), image_loader=coco_image)
rows = [list(j[:5]) + ["" if p is None else f"{p:.6f}", "" if p is None else int(p > THRESHOLD)] for j, p in zip(jobs, probs)]
out_csv = RESULTS / f"coco_itm_preds_{MODEL}_{MODE}.csv"
pd.DataFrame(rows, columns=header + ["p_match", "pred_yes"]).to_csv(out_csv, index=False)
skipped = sum(1 for p in probs if p is None)
print(MODEL, MODE, len(jobs), "pairs,", skipped, "skipped; saved", out_csv.name)

if MODE == "sugarcrepe":
    ok = np.array([p is not None for p in probs])
    pred = np.array([bool(p > THRESHOLD) if p is not None else False for p in probs])[ok]
    pos = np.array([j[4] for j in jobs])[ok]
    subs = np.array([j[1] for j in jobs])[ok]
    correct = pred == pos
    res = {"overall": M.summary(correct), "tpr": M.summary(pred[pos]), "tnr": M.summary(~pred[~pos]),
           "share_predicted_match": M.summary(pred), "n_pairs": int(ok.sum()), "n_skipped": int(skipped)}
    for sub in SUBSETS:
        if (subs == sub).any():
            res[sub] = M.summary(correct[subs == sub])
    path = RESULTS / "coco_itm.json"
    existing = json.load(open(path)) if path.exists() else {}
    existing.setdefault(MODEL, {})["sugarcrepe"] = res
    json.dump(existing, open(path, "w"), indent=1)
    print(MODEL, "SugarCrepe ITM %.2f  TPR %.1f  TNR %.1f  n=%d" % (res["overall"]["mean"], res["tpr"]["mean"], res["tnr"]["mean"], res["n_pairs"]))
