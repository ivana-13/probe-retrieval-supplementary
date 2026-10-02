"""Error-type distribution among judged top-10 items, contrasted with pairwise accuracy by negative type."""
import json
import numpy as np
from svo_eval.paths import RESULTS, MODELS, DIRECTIONS
from svo_eval import pools, metrics as M

full = json.load(open(RESULTS / "full_metrics.json"))
out = {}
for m in MODELS:
    out[m] = {}
    for d in DIRECTIONS:
        per_q = {"correct": [], "subject": [], "verb": [], "object": []}
        for items in pools.load_pool(m, d).values():
            n = len(items)
            per_q["correct"].append(sum(pools.is_correct(j) for j in items) / n)
            for t in ["subject", "verb", "object"]:
                per_q[t].append(sum((not pools.is_correct(j)) and pools.error_type(j) == t for j in items) / n)
        res = {k: M.summary(v) for k, v in per_q.items()}
        inc = sum(np.mean(per_q[t]) for t in ["subject", "verb", "object"])
        res["among_incorrect"] = {t: 100 * float(np.mean(per_q[t])) / inc for t in ["subject", "verb", "object"]}
        res["pairwise_by_type"] = full[m]["pairwise"]
        out[m][d] = res
        print(m, d, {k: round(v["mean"], 1) for k, v in res.items() if isinstance(v, dict) and "mean" in v},
              {k: round(v, 1) for k, v in res["among_incorrect"].items()})
json.dump(out, open(RESULTS / "errors.json", "w"), indent=1)
print("saved")
