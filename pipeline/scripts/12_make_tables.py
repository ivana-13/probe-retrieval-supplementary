"""Render LaTeX table bodies from results/*.json into paper/tables and paper_subset/tables."""
import json
from pathlib import Path
from svo_eval.paths import RESULTS, ECIR, MODELS, DUAL_ENCODERS, DIRECTIONS, DISPLAY, DIRECTION_LABEL

OUT_LONG = ECIR / "paper" / "tables"
OUT_SHORT = ECIR / "paper_subset" / "tables"
OUT_LONG.mkdir(parents=True, exist_ok=True)
OUT_SHORT.mkdir(parents=True, exist_ok=True)


def load(name):
    p = RESULTS / name
    return json.load(open(p)) if p.exists() else None


def f1(x):
    return "--" if x is None else f"{x:.1f}"


def fmt(cell, ci=True):
    if cell is None:
        return "--"
    if cell.get("lo") is None or not ci:
        return f"{cell['mean']:.1f}"
    return f"{cell['mean']:.1f}\\ci{{{(cell['hi'] - cell['lo']) / 2:.1f}}}"


def stars(p):
    return "$^{**}$" if p < 0.01 else ("$^{*}$" if p < 0.05 else "")


def write(name, body, short=False):
    (OUT_LONG / name).write_text(body, encoding="utf-8")
    if short:
        # the dual-encoder subset studies the four dual encoders only: drop the Qwen rows from its copy
        body_short = "\n".join(line for line in body.splitlines() if not line.startswith("Qwen")) + "\n"
        (OUT_SHORT / name).write_text(body_short, encoding="utf-8")
    print("wrote", name)


# The Qwen2.5 probe rows (pairwise A/B and yes/no ITM) come from the prompted zero-shot Instruct model,
# its retrieval rows from the fine-tuned embedder; label the probe rows accordingly.
PROBE_NAME = dict(DISPLAY, Qwen25="Qwen2.5-VL (zero-shot)")

full = load("full_metrics.json")
sem = load("semantic_success.json")
matched = load("matched_negatives.json")
cross = load("cross_model.json")
itm = load("itm_pools.json")
itmb = load("itm_benchmark.json")          # one implementation per system, same as on the pool
judge = load("judge_validation.json")
errors = load("errors.json")
pvr = load("pairwise_vs_retrieval.json")
verify = load("verify_lists.json")
metval = load("metric_validation.json")

# 1. pairwise + ITM on the original benchmark
if full and itmb:
    rows = []
    for m in MODELS:
        pw = full[m]["pairwise"]
        it = itmb.get(m)
        cells = [PROBE_NAME[m]] + [fmt(pw[k]) for k in ["overall", "subject", "verb", "object"]]
        if it:
            cells += [fmt(it[k]) for k in ["overall", "subject", "verb", "object"]]
        else:
            cells += ["--"] * 4
        rows.append(" & ".join(cells) + " \\\\")
        if m == "Qwen25" and "pairwise_embedding" in full[m]:
            pe = full[m]["pairwise_embedding"]
            rows.append(" & ".join([DISPLAY[m] + " (embedding)"] + [fmt(pe[k]) for k in ["overall", "subject", "verb", "object"]]
                                   + ["--"] * 4) + " \\\\")
    body = ("\\begin{tabular}{l cccc cccc}\n\\toprule\n& \\multicolumn{4}{c}{Pairwise ranking accuracy} & "
            "\\multicolumn{4}{c}{ITM accuracy}\\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
            "Model & All & Subj. & Verb & Obj. & All & Subj. & Verb & Obj.\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_pairwise_itm.tex", body, short=True)

# 2. full-collection retrieval
if full and sem:
    rows = []
    for m in MODELS:
        cells = [DISPLAY[m]]
        for d in DIRECTIONS:
            r = full[m][d]
            cells += [fmt(r["S@1"]), f1(r["S@5"]["mean"]), f1(r["S@10"]["mean"]),
                      fmt(sem[m][d]["Ssem@1"]), f1(sem[m][d]["Ssem@10"]["mean"])]
        rows.append(" & ".join(cells) + " \\\\")
    head = " & ".join(["\\sat{1}", "\\sat{5}", "\\sat{10}", "\\sat{1}$_{\\mathrm{sem}}$", "\\sat{10}$_{\\mathrm{sem}}$"])
    body = ("\\begin{tabular}{l ccccc ccccc}\n\\toprule\n& \\multicolumn{5}{c}{" + DIRECTION_LABEL["t2i"] +
            " (10,978 caption queries)} & \\multicolumn{5}{c}{" + DIRECTION_LABEL["i2t"] +
            " (11,455 image queries)}\\\\\n\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\nModel & " + head + " & " + head +
            "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_full_retrieval.tex", body, short=True)
    # dual-encoder subset: strict Success@K only (the semantic rule is not part of the preliminary study)
    rows = []
    for m in DUAL_ENCODERS:
        cells = [DISPLAY[m]]
        for d in DIRECTIONS:
            r = full[m][d]
            cells += [fmt(r["S@1"]), f1(r["S@5"]["mean"]), f1(r["S@10"]["mean"])]
        rows.append(" & ".join(cells) + " \\\\")
    head = " & ".join(["\\sat{1}", "\\sat{5}", "\\sat{10}"])
    body = ("\\begin{tabular}{l ccc ccc}\n\\toprule\n& \\multicolumn{3}{c}{" + DIRECTION_LABEL["t2i"] +
            " (10,978 caption queries)} & \\multicolumn{3}{c}{" + DIRECTION_LABEL["i2t"] +
            " (11,455 image queries)}\\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nModel & " + head + " & " + head +
            "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (OUT_SHORT / "tab_full_retrieval.tex").write_text(body, encoding="utf-8")
    print("wrote tab_full_retrieval.tex (short, strict only)")

# 3. matched negatives
if matched:
    for short, models in [(False, MODELS), (True, DUAL_ENCODERS)]:
        rows = []
        for m in models:
            cells = [DISPLAY[m]]
            for d in DIRECTIONS:
                r = matched[m][d]
                t = r["tests"]
                cells += [fmt(r["benchmark"]), fmt(r["random"]),
                          fmt(r["pool_other"]) + stars(t["benchmark_vs_pool_all"]["mcnemar_p"]) if False else fmt(r["pool_other"]),
                          fmt(r["pool_self"]) + stars(t["pool_other_vs_pool_self"]["mcnemar_p"]),
                          fmt(r["pool_all"]) + stars(t["benchmark_vs_pool_all"]["mcnemar_p"])]
            rows.append(" & ".join(cells) + " \\\\")
        head = "Bench. & Random & Other & Self & All"
        body = ("\\begin{tabular}{l ccccc ccccc}\n\\toprule\n& \\multicolumn{5}{c}{" + DIRECTION_LABEL["t2i"] +
                "} & \\multicolumn{5}{c}{" + DIRECTION_LABEL["i2t"] + "}\\\\\n\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n"
                "Model & " + head + " & " + head + "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
        write("tab_matched_short.tex" if short else "tab_matched.tex", body, short=short)

# 4. cross-model matrices
if cross:
    for d in DIRECTIONS:
        rows = []
        for sc in MODELS:
            cells = [DISPLAY[sc]] + [f1(cross[d][sc][mi]["mean"]) for mi in MODELS] + \
                    [f1(cross[d][sc]["benchmark"]["mean"]), f1(cross[d][sc]["random"]["mean"])]
            rows.append(" & ".join(cells) + " \\\\")
        head = " & ".join(DISPLAY[m] for m in MODELS) + " & Bench. & Random"
        body = ("\\begin{tabular}{l " + "c" * (len(MODELS) + 2) + "}\n\\toprule\nScored $\\backslash$ Mined by & " + head +
                "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
        write(f"tab_cross_{d}.tex", body)

# 5. ITM heads on pools
if itm and itmb:
    rows = []
    for m in ["BLIP2", "FLAVA", "Qwen25", "Qwen3"]:
        if m not in itm:
            continue
        orig = itmb.get(m, {}).get("overall")
        cells = [PROBE_NAME[m], fmt(orig) if orig else "--"]
        for d in DIRECTIONS:
            r = itm[m][d]
            cells += [fmt(r["pool_all"]), fmt(r["pool_self"]), fmt(r["pool_other"]), fmt(r["positives_recall"]),
                      fmt(r["negatives_recall"])]
        rows.append(" & ".join(cells) + " \\\\")
    head = "All & Self & Other & TPR & TNR"
    body = ("\\begin{tabular}{l c ccccc ccccc}\n\\toprule\n& & \\multicolumn{5}{c}{" + DIRECTION_LABEL["t2i"] +
            " pool} & \\multicolumn{5}{c}{" + DIRECTION_LABEL["i2t"] + " pool}\\\\\n\\cmidrule(lr){3-7}\\cmidrule(lr){8-12}\n"
            "Model & Orig. & " + head + " & " + head + "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_itm_pools.tex", body)

# 6. judge validation
if judge:
    # Rows: assessor x label set. "dev" = I->T pools (majority of three annotators), "held_out" = T->I pools (expert).
    rows = []
    names = {"o3": "o3", "gpt4o": "GPT-4o"}
    labs = [("dev", "\\itot{} (majority of 3)"), ("held_out", "\\ttoi{} (expert)")]
    for j in ["o3", "gpt4o"]:
        for split, lab in labs:
            r = judge["judges"][j][split]
            rows.append(" & ".join([names[j], lab, str(r["n"]), f"{100*r['accuracy']:.1f}",
                                    f"{r['precision_correct']:.2f} / {r['recall_correct']:.2f}",
                                    f"{r['f1_correct']:.2f} [{r['f1_ci'][0]:.2f}, {r['f1_ci'][1]:.2f}]",
                                    f"{r['kappa']:.2f} [{r['kappa_ci'][0]:.2f}, {r['kappa_ci'][1]:.2f}]",
                                    f"{r['error_type']['kappa']:.2f}"]) + " \\\\")
    # the COCO validation sample (scripts/28 agreement), one row, once the expert labels exist
    cj = load("coco_judge_validation.json")
    if cj and cj.get("overall"):
        r = cj["overall"]
        et = {"kappa": r.get("kappa_4way")}          # four-way label over both directions (scripts/28 agreement + this file)
        rows.append(" & ".join(["o3", "COCO sample (expert)", str(r["n_pairs"]), f"{100*r['accuracy']:.1f}",
                                f"{r['precision_correct']:.2f} / {r['recall_correct']:.2f}",
                                f"{r['f1_correct']:.2f} [{r['f1_ci'][0]:.2f}, {r['f1_ci'][1]:.2f}]",
                                f"{r['kappa']:.2f} [{r['kappa_ci'][0]:.2f}, {r['kappa_ci'][1]:.2f}]",
                                f"{et['kappa']:.2f}" if et.get("kappa") is not None else "--"]) + " \\\\")
    ia = judge["inter_annotator"]
    body = ("\\begin{tabular}{ll r c c c c c}\n\\toprule\nAssessor & Human labels & $n$ & Acc.\\ (\\%) & P / R (correct) & "
            "F1 (correct) [95\\% CI] & $\\kappa$ [95\\% CI] & $\\kappa$ (4-way)\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\midrule\n\\multicolumn{8}{l}{\\footnotesize Between the three annotators (\\itot{}, 4,000 pairs): pairwise Cohen's $\\kappa$ "
            + ", ".join(f"{k:.2f}" for k in ia["pairwise_cohen_binary"]) + f"; Fleiss' $\\kappa$ {ia['fleiss_3class']:.2f}; "
            f"expert agrees with the majority on {100*ia['expert_vs_majority_accuracy']:.1f}\\% of pairs.}}\\\\\n" + "\\bottomrule\n\\end{tabular}\n")
    write("tab_judge.tex", body)
    rows = []
    for m in DUAL_ENCODERS:
        cells = [DISPLAY[m]]
        for d in DIRECTIONS:
            h = judge["judge_success"]["human"][m][d]
            o = judge["judge_success"]["o3"][m][d]
            cells += [f1(full[m][d]["S@1"]["mean"]), f1(h["S@1"]["mean"]), f1(o["S@1"]["mean"]),
                      f1(h["P@10"]["mean"]), f1(o["P@10"]["mean"]), str(h["n_queries"])]
        rows.append(" & ".join(cells) + " \\\\")
    body = ("\\begin{tabular}{l cccccr cccccr}\n\\toprule\n& \\multicolumn{6}{c}{" + DIRECTION_LABEL["t2i"] +
            "} & \\multicolumn{6}{c}{" + DIRECTION_LABEL["i2t"] + "}\\\\\n\\cmidrule(lr){2-7}\\cmidrule(lr){8-13}\n"
            "& Strict & \\multicolumn{2}{c}{\\sat{1}} & \\multicolumn{2}{c}{P@10} & & Strict & \\multicolumn{2}{c}{\\sat{1}} & "
            "\\multicolumn{2}{c}{P@10} & \\\\\nModel & \\sat{1} & Human & o3 & Human & o3 & $n$ & \\sat{1} & Human & o3 & Human & o3 & $n$\\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_judge_success.tex", body)

# 7. errors
if errors:
    rows = []
    for m in MODELS:
        cells = [DISPLAY[m]]
        for d in DIRECTIONS:
            r = errors[m][d]
            cells += [fmt(r["correct"]), fmt(r["subject"]), fmt(r["verb"]), fmt(r["object"])]
        pw = errors[m]["t2i"]["pairwise_by_type"]
        cells += [f1(pw["subject"]["mean"]), f1(pw["verb"]["mean"]), f1(pw["object"]["mean"])]
        rows.append(" & ".join(cells) + " \\\\")
    head = "Corr. & Subj. & Verb & Obj."
    body = ("\\begin{tabular}{l cccc cccc ccc}\n\\toprule\n& \\multicolumn{4}{c}{" + DIRECTION_LABEL["t2i"] +
            " top-10 share (\\%)} & \\multicolumn{4}{c}{" + DIRECTION_LABEL["i2t"] +
            " top-10 share (\\%)} & \\multicolumn{3}{c}{Pairwise acc.\\ by neg.\\ type}\\\\\n"
            "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-12}\nModel & " + head + " & " + head +
            " & Subj. & Verb & Obj.\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_errors.tex", body, short=True)

# 8. joint outcomes and rank correlations
if pvr:
    rows = []
    for m in MODELS:
        cells = [DISPLAY[m]]
        for d in DIRECTIONS:
            c = pvr["joint"][m][d]
            cells += [str(c["pw_ok_s1_ok"]), str(c["pw_ok_s1_fail"]), str(c["pw_fail_s1_ok"]), str(c["pw_fail_s1_fail"])]
        rows.append(" & ".join(cells) + " \\\\")
    head = "PW$\\checkmark$S$\\checkmark$ & PW$\\checkmark$S$\\times$ & PW$\\times$S$\\checkmark$ & PW$\\times$S$\\times$"
    body = ("\\begin{tabular}{l cccc cccc}\n\\toprule\n& \\multicolumn{4}{c}{" + DIRECTION_LABEL["t2i"] +
            "} & \\multicolumn{4}{c}{" + DIRECTION_LABEL["i2t"] + "}\\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\nModel & "
            + head + " & " + head + "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_joint.tex", body, short=True)
    keys = ["pairwise", "strict_S@1", "MRR@20", "judged_S@1", "pool_all"]
    lab = {"pairwise": "Pairwise", "strict_S@1": "Strict \\sat{1}", "MRR@20": "MRR", "judged_S@1": "Judged \\sat{1}",
           "pool_all": "Pool acc."}
    rows = []
    for d in DIRECTIONS:
        for a in keys:
            rows.append(" & ".join([DIRECTION_LABEL[d], lab[a]] + [f"{pvr['kendall'][d][f'{a}|{b}']:.2f}" for b in keys]) + " \\\\")
    body = ("\\begin{tabular}{ll " + "c" * len(keys) + "}\n\\toprule\n& & " + " & ".join(lab[k] for k in keys) +
            "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_kendall.tex", body, short=True)

# 9. provenance of pools (verify_lists)
if verify:
    rows = []
    for m in MODELS:
        cells = [DISPLAY[m]]
        for d in DIRECTIONS:
            v = verify[m][d]
            cells += [f"{100*v['top1_agreement']:.0f}", f"{100*v['judged_within_10']:.0f}", f"{100*v['judged_within_20']:.0f}",
                      f"{v['judged_median_rank']:.0f}"]
        rows.append(" & ".join(cells) + " \\\\")
    head = "Top-1 & $\\leq$10 & $\\leq$20 & Med."
    body = ("\\begin{tabular}{l cccc cccc}\n\\toprule\n& \\multicolumn{4}{c}{" + DIRECTION_LABEL["t2i"] +
            "} & \\multicolumn{4}{c}{" + DIRECTION_LABEL["i2t"] + "}\\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\nModel & "
            + head + " & " + head + "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_provenance.tex", body)

# 10. metric validation summary
if metval:
    rows = []
    for rule in ["strict", "lenient"]:
        for d in DIRECTIONS:
            for r in metval["curves"][rule][d]:
                if r["tau"] in (0.8, 0.9, 0.95, 1.0):
                    rows.append(" & ".join([rule, DIRECTION_LABEL[d], f"{r['tau']:.2f}", f"{r['precision']:.2f}",
                                            f"{r['recall']:.2f}", f"{r['f1']:.2f}"]) + " \\\\")
    body = ("\\begin{tabular}{llcccc}\n\\toprule\nRule & Direction & $\\tau$ & Precision & Recall & F1\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_metric_validation.tex", body)
print("done")
