"""Leave-one-out pooling bias (reusability test).

For each system, the pool pairs that only this system contributed are treated as unjudged, which is how a system
outside the pool would be scored: unjudged items count as incorrect. We recompute judged Success@1 and Precision@10 of
the system's own top-10 lists under full labels and under leave-one-out labels, and report the share of its top-10
items that no other system retrieved. The matched-negative "Other" condition of Table 3 is the corresponding
leave-one-out value for pool accuracy (all self-mined negatives removed)."""
import json
import numpy as np
from svo_eval.paths import RESULTS, ECIR, MODELS, DIRECTIONS, DISPLAY, DIRECTION_LABEL
from svo_eval import pools, metrics as M

fp = pools.build_fixed_pool()
out = {}
for m in MODELS:
    out[m] = {}
    for d in DIRECTIONS:
        s1_full, s1_loo, p10_full, p10_loo, unique = [], [], [], [], []
        for q, items in pools.load_pool(m, d).items():
            entries = fp[d].get(q, {})
            full, loo, uniq = [], [], 0
            for j in items:
                e = entries.get(j["cand"])
                if e is None:                       # candidate outside the collection: unjudged for everyone
                    full.append(False); loo.append(False); continue
                lab = bool(e["correct"])
                only_me = set(e["retrieved_by"]) == {m}
                uniq += only_me
                full.append(lab)
                loo.append(lab if not only_me else False)
            s1_full.append(full[0]); s1_loo.append(loo[0])
            p10_full.append(float(np.mean(full))); p10_loo.append(float(np.mean(loo)))
            unique.append(uniq / len(items))
        out[m][d] = {"S@1_full": M.summary(s1_full), "S@1_loo": M.summary(s1_loo),
                     "P@10_full": M.summary(p10_full), "P@10_loo": M.summary(p10_loo),
                     "unique_share": M.summary(unique),
                     "S@1_drop": 100 * float(np.mean(s1_full) - np.mean(s1_loo)),
                     "P@10_drop": 100 * float(np.mean(p10_full) - np.mean(p10_loo)), "n_queries": len(s1_full)}
        r = out[m][d]
        print(m, d, "S@1 %.1f -> %.1f (drop %.1f)  P@10 %.1f -> %.1f (drop %.1f)  unique %.1f%%" % (
            r["S@1_full"]["mean"], r["S@1_loo"]["mean"], r["S@1_drop"], r["P@10_full"]["mean"], r["P@10_loo"]["mean"],
            r["P@10_drop"], r["unique_share"]["mean"]))
json.dump(out, open(RESULTS / "pooling_bias.json", "w"), indent=1)

rows = []
for m in MODELS:
    cells = [DISPLAY[m]]
    for d in DIRECTIONS:
        r = out[m][d]
        cells += [f"{r['unique_share']['mean']:.0f}", f"{r['S@1_full']['mean']:.1f}", f"{r['S@1_loo']['mean']:.1f}",
                  f"{r['P@10_full']['mean']:.1f}", f"{r['P@10_loo']['mean']:.1f}"]
    rows.append(" & ".join(cells) + " \\\\")
head = "Unique (\\%) & \\sat{1} & \\sat{1}$_{-1}$ & P@10 & P@10$_{-1}$"
body = ("\\begin{tabular}{l ccccc ccccc}\n\\toprule\n& \\multicolumn{5}{c}{" + DIRECTION_LABEL["t2i"] + "} & \\multicolumn{5}{c}{"
        + DIRECTION_LABEL["i2t"] + "}\\\\\n\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\nModel & " + head + " & " + head +
        "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
for sub in ["paper", "paper_subset"]:
    (ECIR / sub / "tables" / "tab_pooling_bias.tex").write_text(body, encoding="utf-8")
print("saved")
