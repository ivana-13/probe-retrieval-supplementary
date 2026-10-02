"""Validate o3 and GPT-4o labels against human labels; judge-derived success/precision; cost estimate."""
import json
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score
from svo_eval.paths import RESULTS, DUAL_ENCODERS, DIRECTIONS
from svo_eval import pools, metrics as M
from svo_eval.judges import binary_agreement, multiclass_agreement, bootstrap_stat

LABELS = pools.LABELS
out = {"judges": {}, "judge_success": {}, "cost": {}, "inter_annotator": {}}
SPLITS = {"dev": "i2t", "held_out": "t2i"}

# inter-annotator agreement (i2t, 3 annotators) for reference: pairwise Cohen kappa and Fleiss
rows = []
for m in DUAL_ENCODERS:
    for items in pools.load_pool(m, "i2t").values():
        for j in items:
            rows.append(j["votes"])
rows = np.array(rows)
pair_k = [float(cohen_kappa_score(rows[:, a] == 1, rows[:, b] == 1)) for a, b in [(0, 1), (0, 2), (1, 2)]]
out["inter_annotator"] = {"fleiss_3class": pools.fleiss_kappa_i2t(), "pairwise_cohen_binary": pair_k,
                          "expert_vs_majority_accuracy": float(np.mean((rows[:, 0] == 1) == ((rows == 1).sum(1) >= 2))),
                          "n": int(len(rows))}

for judge in ["o3", "gpt4o"]:
    out["judges"][judge] = {}
    for split, d in SPLITS.items():
        t_all, p_all, te, pe, per_model = [], [], [], [], {}
        for m in DUAL_ENCODERS:
            t, p = [], []
            for items in pools.load_pool(m, d).values():
                for j in items:
                    if j[judge] is None:
                        continue
                    t.append(pools.is_correct(j))
                    p.append(j[judge] == "correct")
                    te.append(j["expert"])
                    pe.append(j[judge])
            t, p = np.array(t, bool), np.array(p, bool)
            per_model[m] = binary_agreement(t, p)
            t_all.append(t)
            p_all.append(p)
        t, p = np.concatenate(t_all), np.concatenate(p_all)
        res = binary_agreement(t, p)
        res["f1_ci"] = bootstrap_stat(lambda a, b: f1_score(a, b), t, p)
        res["kappa_ci"] = bootstrap_stat(lambda a, b: cohen_kappa_score(a, b), t, p)
        res["per_model"] = per_model
        mask = [x in LABELS and y in LABELS for x, y in zip(te, pe)]
        res["error_type"] = multiclass_agreement([x for x, k in zip(te, mask) if k],
                                                 [y for y, k in zip(pe, mask) if k], LABELS)
        inc = [(x, y) for x, y, k in zip(te, pe, mask) if k and x != "correct" and y != "correct"]
        res["error_type_among_incorrect"] = multiclass_agreement([x for x, _ in inc], [y for _, y in inc], LABELS[1:])
        out["judges"][judge][split] = res
        print(judge, split, {k: round(v, 3) for k, v in res.items() if isinstance(v, float)})

# judge-derived metrics: each LLM judge is compared with the human labels on the queries for which that
# judge labelled all 10 candidates ("human" entries are computed on the o3-complete queries; GPT-4o has
# its own, much smaller, complete set in T->I, recorded as n_queries)
for judge in ["human", "o3", "gpt4o"]:
    out["judge_success"][judge] = {}
    ref = "gpt4o" if judge == "gpt4o" else "o3"
    for m in DUAL_ENCODERS:
        out["judge_success"][judge][m] = {}
        for d in DIRECTIONS:
            lab_lists = []
            for items in pools.load_pool(m, d).values():
                if any(j[ref] is None for j in items):
                    continue
                labs = [(pools.is_correct(j) if judge == "human" else (j[judge] == "correct")) for j in items]
                lab_lists.append(labs)
            res = {"n_queries": len(lab_lists)}
            for k in [1, 5, 10]:
                res[f"S@{k}"] = M.summary([any(l[:k]) for l in lab_lists])
                res[f"P@{k}"] = M.summary(M.precision_vector(lab_lists, k))
            out["judge_success"][judge][m][d] = res

# cost estimate: o3 list price (2025): 2 USD per 1M input tokens, 8 USD per 1M output tokens.
# One call judges one query with 10 candidates; images cost about 1,100 tokens each at the sizes used,
# the prompt about 350 tokens, and reasoning plus answer about 1,500 output tokens.
calls = 100 * len(DUAL_ENCODERS) * 2
inp = calls * (10 * 1100 + 350)
outp = calls * 1500
out["cost"] = {"o3_calls": calls, "o3_input_tokens": inp, "o3_output_tokens": outp,
               "o3_usd": inp / 1e6 * 2 + outp / 1e6 * 8,
               "human_pairs_triple_annotated": 4000, "human_pairs_single_annotated": 3719 + 2000}
json.dump(out, open(RESULTS / "judge_validation.json", "w"), indent=1)
print("inter-annotator:", out["inter_annotator"])
print("saved")
