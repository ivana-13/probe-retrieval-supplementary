"""Cross-model matrix: pairwise accuracy of each scoring system on judged negatives mined by each system."""
import json
import numpy as np
from svo_eval.paths import RESULTS, MODELS, DIRECTIONS
from svo_eval.collection import load_collection
from svo_eval.embeddings import Scorer
from svo_eval import pools, metrics as M

coll = load_collection()
fp = pools.build_fixed_pool()
matched = json.load(open(RESULTS / "matched_negatives.json"))
out = {}
for d in DIRECTIONS:
    out[d] = {}
    qs = [q for q in pools.query_ids(d) if d == "t2i" or coll.img2caps.get(q)]
    for scorer in MODELS:
        s = Scorer(scorer)
        out[d][scorer] = {}
        for miner in MODELS:
            accs = []
            for q in qs:
                gold = coll.gold(d, q)
                negs = [c for c, e in fp[d][q].items()
                        if not e["correct"] and miner in e["retrieved_by"] and c not in gold]
                if not negs:
                    continue
                g = float(np.max(s.score(d, q, list(gold))))
                accs.append(float(np.mean(s.score(d, q, negs) < g)))
            out[d][scorer][miner] = M.summary(accs)
        out[d][scorer]["benchmark"] = matched[scorer][d]["benchmark"]
        out[d][scorer]["random"] = matched[scorer][d]["random"]
        print(d, scorer, {k: round(v["mean"], 1) for k, v in out[d][scorer].items() if v})
    means = {miner: float(np.mean([out[d][sc][miner]["mean"] for sc in MODELS])) for miner in MODELS}
    out[d]["hardness_rank"] = sorted(means, key=means.get)
    out[d]["miner_mean"] = means
    diag_lowest = sum(min(out[d][sc][mi]["mean"] for mi in MODELS) == out[d][sc][sc]["mean"] for sc in MODELS)
    out[d]["rows_where_self_is_hardest"] = diag_lowest
    print(d, "hardness rank (hardest first):", out[d]["hardness_rank"], "rows where self-mined is hardest:", diag_lowest)
json.dump(out, open(RESULTS / "cross_model.json", "w"), indent=1)
print("saved")
