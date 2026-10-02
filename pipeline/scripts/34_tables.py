"""Tables and the n-way figure of the paper that depend on the six-model analyses:
  results/svo_analyses.json  (scripts/16_svo_analyses.py long)
  results/coco_analyses.json   (scripts/33_coco_analyses.py)
Writes paper/tables/tab_long_*.tex and figures/fig_nway_other_long.pdf; the other long-paper tables still come from
scripts/12_make_tables.py and scripts/31_coco_tables.py. Prints the ranges quoted in the text."""
import json
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from svo_eval.paths import RESULTS, ECIR, FIGURES, MODELS, DIRECTIONS, DISPLAY, DIRECTION_LABEL

OUT = ECIR / "paper" / "tables"
A = json.load(open(RESULTS / "svo_analyses.json"))
C = json.load(open(RESULTS / "coco_analyses.json"))
full = json.load(open(RESULTS / "full_metrics.json"))
cfull = json.load(open(RESULTS / "coco_metrics.json"))
cells = [(m, d) for m in MODELS for d in DIRECTIONS]


def hw(c):
    return (c["hi"] - c["lo"]) / 2


def fmt(c, ci=True):
    if c is None:
        return "--"
    return f"{c['mean']:.1f}\\ci{{{hw(c):.1f}}}" if ci else f"{c['mean']:.1f}"


def f0(c):
    return "--" if c is None else f"{c['mean']:.0f}"


def stars(t):
    if not t:
        return ""
    p = t["p_randomisation_holm"]
    return "$^{**}$" if p < 0.01 else ("$^{*}$" if p < 0.05 else "")


def write(name, body):
    (OUT / name).write_text(body, encoding="utf-8")
    print("wrote", name)


# 1. SVO-Probes: strict Success@K over the full dataset, and Success@1 on the 100 pool queries under the benchmark's
#    labels, the human labels and the o3 labels
rows = []
for m in MODELS:
    c = [DISPLAY[m]]
    for d in DIRECTIONS:
        r, ref = full[m][d], A["reference"][m][d]
        o3 = ref["judged_S@1_o3"] if ref["judged_S@1_o3_own_list_labelled"] else None
        c += [fmt(r["S@1"]), f"{r['S@5']['mean']:.1f}", f"{r['S@10']['mean']:.1f}", f0(ref["S@1"]), f0(ref["judged_S@1"]), f0(o3)]
    rows.append(" & ".join(c) + " \\\\")
head = "\\sat{1} & \\sat{5} & \\sat{10} & Strict & Human & o3"
body = ("\\begin{tabular}{l ccc ccc ccc ccc}\n\\toprule\n& \\multicolumn{6}{c}{" + DIRECTION_LABEL["t2i"] +
        "} & \\multicolumn{6}{c}{" + DIRECTION_LABEL["i2t"] + "}\\\\\n\\cmidrule(lr){2-7}\\cmidrule(lr){8-13}\n"
        "& \\multicolumn{3}{c}{Full dataset} & \\multicolumn{3}{c}{\\sat{1}, pool queries} & \\multicolumn{3}{c}{Full dataset} & "
        "\\multicolumn{3}{c}{\\sat{1}, pool queries}\\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-13}\n"
        "Model & " + head + " & " + head + "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
write("tab_long_full.tex", body)

# 2. COCO: the same with the o3 labels only
rows = []
for m in MODELS:
    c = [DISPLAY[m]]
    for d in DIRECTIONS:
        r, ref = cfull[m][d], C["reference"][m][d]
        c += [fmt(r["S@1"]), f"{r['S@5']['mean']:.1f}", f"{r['S@10']['mean']:.1f}", f0(ref["S@1"]), f0(ref["judged_S@1"])]
    rows.append(" & ".join(c) + " \\\\")
head = "\\sat{1} & \\sat{5} & \\sat{10} & Strict & o3"
body = ("\\begin{tabular}{l ccc cc ccc cc}\n\\toprule\n& \\multicolumn{5}{c}{" + DIRECTION_LABEL["t2i"] +
        "} & \\multicolumn{5}{c}{" + DIRECTION_LABEL["i2t"] + "}\\\\\n\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n"
        "& \\multicolumn{3}{c}{Full dataset} & \\multicolumn{2}{c}{\\sat{1}, pool queries} & \\multicolumn{3}{c}{Full dataset} & "
        "\\multicolumn{2}{c}{\\sat{1}, pool queries}\\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-11}\n"
        "Model & " + head + " & " + head + "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
write("tab_long_coco_full.tex", body)

# 3. matched negatives, both benchmarks. Stars (paired randomisation test, Holm-corrected): Other against the handcrafted
#    negatives (against Random where no handcrafted negatives exist), Self against Other, All against the handcrafted
#    negatives (or Random).
head = "Bench. & Random & Other & Self & All"
top = ("\\begin{tabular}{l ccccc ccccc}\n\\toprule\n& \\multicolumn{5}{c}{" + DIRECTION_LABEL["t2i"] +
       "} & \\multicolumn{5}{c}{" + DIRECTION_LABEL["i2t"] + "}\\\\\n\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n"
       "Model & " + head + " & " + head + "\\\\\n\\midrule\n")
rows = []
for m in MODELS:
    c = [DISPLAY[m]]
    for d in DIRECTIONS:
        a, t = A["matched"][m][d], A["tests"][m][d]
        c += [fmt(a["benchmark"]), fmt(a["random"]), fmt(a["pool_other"]) + stars(t["benchmark_vs_other"]),
              fmt(a["pool_self"]) + stars(t["other_vs_self"]), fmt(a["pool_all"]) + stars(t["benchmark_vs_all"])]
    rows.append(" & ".join(c) + " \\\\")
write("tab_long_matched.tex", top + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
rows = []
for m in MODELS:
    c = [DISPLAY[m]]
    for d in DIRECTIONS:
        a, t = C["matched"][m][d], C["tests"][m][d]
        ref = "benchmark" if a["benchmark"] else "random"
        c += [fmt(a["benchmark"]), fmt(a["random"]), fmt(a["pool_other"]) + stars(t[ref + "_vs_other"]),
              fmt(a["pool_self"]) + stars(t["other_vs_self"]), fmt(a["pool_all"]) + stars(t[ref + "_vs_all"])]
    rows.append(" & ".join(c) + " \\\\")
write("tab_long_coco_matched.tex", top + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def body_rows(name):
    """the data rows of an existing table file (between \\midrule and \\bottomrule)"""
    lines = (OUT / name).read_text(encoding="utf-8").splitlines()
    return lines[max(i for i, x in enumerate(lines) if x.startswith("\\midrule")) + 1:
                 next(i for i, x in enumerate(lines) if x.startswith("\\bottomrule"))]


def block(title, ncol):
    return "\\multicolumn{" + str(ncol) + "}{l}{\\emph{" + title + "}}\\\\"


# 3b. both benchmarks in one table (saves a float): matched negatives, and the ITM decisions on the pools
write("tab_long_matched_both.tex",
      top + block("SVO-Probes (human labels)", 11) + "\n" + "\n".join(body_rows("tab_long_matched.tex")) + "\n\\midrule\n"
      + block("COCO with SugarCrepe (o3 labels)", 11) + "\n" + "\n".join(body_rows("tab_long_coco_matched.tex"))
      + "\n\\bottomrule\n\\end{tabular}\n")
itm_top = ("\\begin{tabular}{l c ccccc ccccc}\n\\toprule\n& & \\multicolumn{5}{c}{" + DIRECTION_LABEL["t2i"] +
           " pool} & \\multicolumn{5}{c}{" + DIRECTION_LABEL["i2t"] + " pool}\\\\\n\\cmidrule(lr){3-7}\\cmidrule(lr){8-12}\n"
           "Model & Orig. & All & Self & Other & TPR & TNR & All & Self & Other & TPR & TNR\\\\\n\\midrule\n")
write("tab_long_itm_pools_both.tex",
      itm_top + block("SVO-Probes (human labels)", 12) + "\n" + "\n".join(body_rows("tab_itm_pools.tex")) + "\n\\midrule\n"
      + block("COCO with SugarCrepe (o3 labels)", 12) + "\n" + "\n".join(body_rows("tab_coco_itm_pools.tex"))
      + "\n\\bottomrule\n\\end{tabular}\n")

# 3c. assessor-validation table without the full-width footnote row (it forced the table to a tiny font);
#     the inter-annotator figures go into the caption
#     the intervals are replaced by query-clustered ones (scripts/42_assessor_and_order_checks.py), because a pair
#     retrieved by several models occurs in several lists and the pairs of a query are not independent
import re
CL = json.load(open(RESULTS / "assessor_and_order_checks.json"))["clustered_ci"]
lines = (OUT / "tab_judge.tex").read_text(encoding="utf-8").splitlines()
foot = next(i for i, x in enumerate(lines) if "Between the three annotators" in x)
lines = lines[:foot - 1] + lines[foot + 1:]
first = max(i for i, x in enumerate(lines) if x.startswith("\\midrule")) + 1
for i, key in enumerate(["o3_i2t", "o3_t2i", "gpt4o_i2t", "gpt4o_t2i", "o3_coco"]):
    cis = iter([CL[key]["f1_ci"], CL[key]["kappa_ci"]])
    row, n = re.subn(r"\[[0-9.]+, [0-9.]+\]", lambda _: "[%.2f, %.2f]" % tuple(next(cis)), lines[first + i])
    assert n == 2, lines[first + i]
    lines[first + i] = row
write("tab_long_judge.tex", "\n".join(lines) + "\n")

# 3d. one probe table for the six embedding models: pairwise accuracy on SVO-Probes and on SugarCrepe
def probe_rows(name):
    out = {}
    for r in body_rows(name):
        c = [x.strip() for x in r.rstrip("\\ ").split("&")]
        if "zero-shot" in c[0]:
            continue                      # the prompted model does no retrieval and is documented in the repository
        out[c[0].replace(" (embedding)", "")] = c[1:5]
    return out


svo_p, coco_p = probe_rows("tab_pairwise_itm.tex"), probe_rows("tab_coco_pairwise_itm.tex")
assert list(svo_p) == list(coco_p) == [DISPLAY[m] for m in MODELS], (list(svo_p), list(coco_p))
write("tab_long_probe.tex",
      "\\begin{tabular}{l cccc cccc}\n\\toprule\n& \\multicolumn{4}{c}{SVO-Probes} & \\multicolumn{4}{c}{SugarCrepe}\\\\\n"
      "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\nModel & All & Subj. & Verb & Obj. & All & Obj. & Att. & Rel.\\\\\n\\midrule\n"
      + "\n".join(" & ".join([k] + svo_p[k] + coco_p[k]) + " \\\\" for k in svo_p) + "\n\\bottomrule\n\\end{tabular}\n")

# 3e. one retrieval table for both benchmarks (COCO has no human labels on the full pool)
head = "\\sat{1} & \\sat{5} & \\sat{10} & Strict & Human & o3"
rows = {"svo": [], "coco": []}
for m in MODELS:
    c, cc = [DISPLAY[m]], [DISPLAY[m]]
    for d in DIRECTIONS:
        r, ref = full[m][d], A["reference"][m][d]
        c += [fmt(r["S@1"]), f"{r['S@5']['mean']:.1f}", f"{r['S@10']['mean']:.1f}", f0(ref["S@1"]), f0(ref["judged_S@1"]), f0(ref["judged_S@1_o3"])]
        r, ref = cfull[m][d], C["reference"][m][d]
        cc += [fmt(r["S@1"]), f"{r['S@5']['mean']:.1f}", f"{r['S@10']['mean']:.1f}", f0(ref["S@1"]), "--", f0(ref["judged_S@1"])]
    rows["svo"].append(" & ".join(c) + " \\\\")
    rows["coco"].append(" & ".join(cc) + " \\\\")
write("tab_long_full_both.tex",
      "\\begin{tabular}{l ccc ccc ccc ccc}\n\\toprule\n& \\multicolumn{6}{c}{" + DIRECTION_LABEL["t2i"] +
      "} & \\multicolumn{6}{c}{" + DIRECTION_LABEL["i2t"] + "}\\\\\n\\cmidrule(lr){2-7}\\cmidrule(lr){8-13}\n"
      "& \\multicolumn{3}{c}{Full dataset} & \\multicolumn{3}{c}{\\sat{1}, pool queries} & \\multicolumn{3}{c}{Full dataset} & "
      "\\multicolumn{3}{c}{\\sat{1}, pool queries}\\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-13}\n"
      "Model & " + head + " & " + head + "\\\\\n\\midrule\n" + block("SVO-Probes", 13) + "\n" + "\n".join(rows["svo"])
      + "\n\\midrule\n" + block("COCO val2017", 13) + "\n" + "\n".join(rows["coco"]) + "\n\\bottomrule\n\\end{tabular}\n")

# 3f. binary matching decisions: ITM accuracy on the benchmark's own pairs, and accuracy, true-positive and true-negative
#     rate on the judged pools (caption / image queries), with the constant answer "no match" as a reference row
itm_svo = json.load(open(RESULTS / "itm_pools.json"))
itm_coco = json.load(open(RESULTS / "coco_itm_pools_o3.json"))
itm_bench = json.load(open(RESULTS / "itm_benchmark.json"))
itm_sc = json.load(open(RESULTS / "coco_itm.json"))
head_auc = json.load(open(RESULTS / "itm_head_auc.json"))
ITM_ROWS = [("BLIP2", "BLIP-2 head"), ("FLAVA", "FLAVA head"), ("Qwen25", "Qwen2.5-VL prompt"), ("Qwen3", "\\qemb{} yes/no")]


def both(src, m, key):
    return "/".join(f"{src[m][d][key]['mean']:.0f}" for d in DIRECTIONS)


rows = []
for m, name in ITM_ROWS:
    rows.append(" & ".join([name, f"{itm_bench[m]['overall']['mean']:.1f}", both(itm_svo, m, "pool_all"), both(itm_svo, m, "positives_recall"),
                            both(itm_svo, m, "negatives_recall"), f"{itm_sc[m]['sugarcrepe']['overall']['mean']:.1f}", both(itm_coco, m, "pool_all"),
                            both(itm_coco, m, "positives_recall"), both(itm_coco, m, "negatives_recall")]) + " \\\\")
const = lambda key: "/".join(f"{head_auc['BLIP2'][key + d]['share_incorrect']:.0f}" for d in DIRECTIONS)
rows.append("\\midrule\nAlways ``no match'' & 50.0 & " + const("svo_pool_") + " & 0/0 & 100/100 & 50.0 & " + const("coco_pool_") + " & 0/0 & 100/100 \\\\")
write("tab_long_itm_compact.tex",
      "\\begin{tabular}{l cccc cccc}\n\\toprule\n& \\multicolumn{4}{c}{\\svo{} (human labels)} & \\multicolumn{4}{c}{COCO with SugarCrepe (o3 labels)}\\\\\n"
      "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\nDecision & Bench. & Pool & TPR & TNR & Bench. & Pool & TPR & TNR\\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# 4. n-way figure on SVO-Probes: random negatives against other-mined negatives, on the same fixed query set per model
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
DLAB = {"t2i": "T→I (caption query)", "i2t": "I→T (image query)"}
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                     "axes.axisbelow": True, "lines.linewidth": 1.4, "lines.markersize": 4})
N_ALL = {"t2i": 13285 - 1, "i2t": 10978 - 1}
fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.55), sharey=True, gridspec_kw={"wspace": 0.08})
for ax, d in zip(axes, DIRECTIONS):
    for i, m in enumerate(MODELS):
        r = A["nway"][m][d]
        ks = [N_ALL[d] if k == "all" else k for k in r["rand_k"]]
        ax.plot(ks, [v["mean"] for v in r["random_fixed_full"]], color=PALETTE[i], marker=MARKERS[i], linestyle="-",
                markeredgecolor="white", markeredgewidth=0.6, label=DISPLAY[m])
        ax.plot(r["k"], [v["mean"] for v in r["other_fixed"]], color=PALETTE[i], marker=MARKERS[i], linestyle="--",
                markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_xticklabels(["1", "10", "100", "1k", "10k"])
    ax.set_xlim(0.8, 2.2e4)
    ax.set_xlabel("number of negatives $k$")
    ax.set_title(DLAB[d], fontsize=8, color=INK)
    ax.set_ylim(0, 102)
axes[0].set_ylabel("positive outranks all $k$ (%)")
axes[1].tick_params(labelleft=False)
h, l = axes[0].get_legend_handles_labels()
h += [Line2D([0], [0], color=INK2, linestyle="-", linewidth=1.1), Line2D([0], [0], color=INK2, linestyle="--", linewidth=1.1)]
l += ["random negatives", "other-mined negatives"]
fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.36), fontsize=6.3, handlelength=1.8,
           columnspacing=1.0, handletextpad=0.5, labelspacing=0.35)
path = FIGURES / "fig_nway_other_long.pdf"
fig.savefig(path, bbox_inches="tight")
fig.savefig(RESULTS / "fig_nway_other_long.png", bbox_inches="tight", dpi=220)
plt.close(fig)
shutil.copy(path, ECIR / "paper" / "figures" / "fig_nway_other_long.pdf")
print("wrote fig_nway_other_long.pdf")


# 5. ranges for the text
def rng(vals, nd=1):
    vals = [v for v in vals if v is not None]
    return f"{min(vals):.{nd}f}--{max(vals):.{nd}f}"


def by(fn, ds=DIRECTIONS, src=A):
    return [fn(src, m, d) for m in MODELS for d in ds]


print("\n=== SVO-Probes (six models) ===")
for k in ["benchmark", "benchmark_own", "random", "pool_other", "pool_self", "pool_all"]:
    print(f"{k:14s} t2i {rng(by(lambda S, m, d: S['matched'][m][d][k]['mean'], ['t2i']))}  i2t {rng(by(lambda S, m, d: S['matched'][m][d][k]['mean'], ['i2t']))}")
print("bench(max) vs own, max abs diff", f"{max(abs(A['matched'][m][d]['benchmark']['mean'] - A['matched'][m][d]['benchmark_own']['mean']) for m, d in cells):.2f}")
for comp in ["benchmark_vs_other", "benchmark_vs_all", "other_vs_self", "random_vs_benchmark"]:
    ps = [A["tests"][m][d][comp]["p_randomisation_holm"] for m, d in cells]
    print(f"{comp:22s} diff {rng([A['tests'][m][d][comp]['diff'] for m, d in cells])}  Holm p max {max(ps):.4f}  cells p<0.01: {sum(p < 0.01 for p in ps)}/12  p<0.05: {sum(p < 0.05 for p in ps)}/12",
          [f"{DISPLAY[m]} {d} {p:.3f}" for (m, d), p in zip(cells, ps) if p >= 0.01])
o = lambda S, m, d: S["tests"][m][d]["outside_top10"]
print("outside top 10: n", rng(by(lambda S, m, d: o(S, m, d)["n_queries"]), 0), "bench", rng(by(lambda S, m, d: o(S, m, d)["benchmark"]["mean"])),
      "other", rng(by(lambda S, m, d: o(S, m, d)["pool_other"]["mean"])), "diff", rng(by(lambda S, m, d: o(S, m, d)["test"]["diff"])),
      "Holm p max", f"{max(by(lambda S, m, d: o(S, m, d)['test']['p_randomisation_holm'])):.4f}",
      "cells p<0.01:", sum(p < 0.01 for p in by(lambda S, m, d: o(S, m, d)["test"]["p_randomisation_holm"])))
for d in DIRECTIONS:
    R = lambda k: by(lambda S, m, dd: S["reference"][m][dd][k]["mean"], [d])
    print(d, "pool queries: strict S@1", rng(R("S@1"), 0), "S@10", rng(R("S@10"), 0), "human-judged S@1", rng(R("judged_S@1"), 0),
          "o3-judged S@1", rng([A["reference"][m][d]["judged_S@1_o3"]["mean"] for m in MODELS if A["reference"][m][d]["judged_S@1_o3_own_list_labelled"]], 0),
          "n o3", [A["reference"][m][d]["judged_S@1_o3"]["n"] for m in MODELS])
    N = lambda k: by(lambda S, m, dd: S["nway"][m][dd][k], [d])
    print(d, "n-way: n", rng(N("n_queries_fixed"), 0), "10 other", rng([x[-1]["mean"] for x in N("other_fixed")], 0),
          "10 random", rng([x[-1]["mean"] for x in N("random_fixed")], 0),
          "50 random", rng([x[5]["mean"] for x in N("random_fixed_full")], 0),
          "all", rng([x[-1]["mean"] for x in N("random_fixed_full")], 0),
          "equivalent k", rng(N("equivalent_random_k_for_10_other_fixed"), 0))
    J = lambda k: by(lambda S, m, dd: S["joint"][m][dd][k], [d])
    print(d, "joint: pass&fail", rng(J("pass_fail"), 0), "fail&ok", rng(J("fail_ok"), 0), "ok|pass", rng(J("ok_given_pass"), 0),
          "ok|fail", rng(J("ok_given_fail"), 0), "fisher p", [round(x, 2) for x in J("fisher_p")], "Holm min", f"{min(J('fisher_p_holm')):.2f}")
print("violations of the bound:", sum(A["matched"][m][d]["violations_of_bound"] for m, d in cells))
for m in MODELS:
    for d in DIRECTIONS:
        e = A["errors"][m][d]["top10_incorrect"]
        tot = e["subject"] + e["verb"] + e["object"]
        print(f"   errors {m:8s} {d}: subject {100 * e['subject'] / tot:.0f} verb {100 * e['verb'] / tot:.0f} object {100 * e['object'] / tot:.0f}")
print("benchmark triplet types", {k: round(v, 1) for k, v in A["benchmark_type_share"].items()})
print("exact S@1 full i2t:", {m: round(full[m]["i2t"]["S@1"]["mean"], 2) for m in MODELS}, "t2i:", {m: round(full[m]["t2i"]["S@1"]["mean"], 2) for m in MODELS})

print("\n=== COCO with SugarCrepe (six models, o3 labels) ===")
for k in ["benchmark", "random", "pool_other", "pool_self", "pool_all"]:
    v = lambda d: [C["matched"][m][d][k]["mean"] for m in MODELS if C["matched"][m][d][k]]
    print(f"{k:12s} t2i {rng(v('t2i')) if v('t2i') else '--'}  i2t {rng(v('i2t'))}")
for comp, ds in [("benchmark_vs_other", ["i2t"]), ("benchmark_vs_all", ["i2t"]), ("random_vs_other", DIRECTIONS), ("random_vs_all", ["t2i"]), ("other_vs_self", DIRECTIONS)]:
    cc = [(m, d) for m in MODELS for d in ds]
    ps = [C["tests"][m][d][comp]["p_randomisation_holm"] for m, d in cc]
    print(f"{comp:20s} {ds} diff {rng([C['tests'][m][d][comp]['diff'] for m, d in cc])} cells p<0.05: {sum(p < 0.05 for p in ps)}/{len(ps)} p<0.01: {sum(p < 0.01 for p in ps)}",
          [f"{DISPLAY[m]} {d} {p:.3f}" for (m, d), p in zip(cc, ps) if p < 0.05] if comp.startswith("benchmark") else "")
for d in DIRECTIONS:
    R = lambda k: [C["reference"][m][d][k]["mean"] for m in MODELS]
    print(d, "pool queries: strict S@1", rng(R("S@1"), 0), "S@10", rng(R("S@10"), 0), "o3-judged S@1", rng(R("judged_S@1"), 0),
          "| n-way n", rng([C["nway"][m][d]["n_queries_fixed"] for m in MODELS], 0),
          "10 other", rng([C["nway"][m][d]["other_fixed"][-1]["mean"] for m in MODELS], 0),
          "10 random", rng([C["nway"][m][d]["random_fixed"][-1]["mean"] for m in MODELS], 0),
          "equivalent k", rng([C["nway"][m][d]["equivalent_random_k_for_10_other_fixed"] for m in MODELS], 0),
          "| outside top 10 n", rng([C["tests"][m][d]["outside_top10"]["n_queries"] for m in MODELS], 0))
