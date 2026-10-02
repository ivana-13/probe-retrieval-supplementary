"""LaTeX table bodies for the second collection (COCO val2017 + SugarCrepe) into paper/tables/tab_coco_*.tex,
with the same layout, column order and model order as the SVO-Probes tables of scripts/12_make_tables.py:
  tab_coco_pairwise_itm     like tab_pairwise_itm: pairwise and ITM accuracy overall and by edit type
                            (object / attribute / relation groups of the seven SugarCrepe subsets)
  tab_coco_full             like tab_full_retrieval: strict Success@K and MRR (no semantic rule on COCO)
  tab_coco_matched          like tab_matched: benchmark / random / other / self / all with McNemar stars
  tab_coco_judge_success    like tab_judge_success: strict S@1 next to judged S@1 and P@10 under the o3 labels
  tab_coco_itm_pools        like tab_itm_pools: ITM decisions on the pool against the labels
  tab_coco_pairwise_subsets pairwise accuracy by subset (appendix)
  tab_coco_judge            LLM assessor against the human sample (when results/coco_judge_validation.json exists)
Usage: python scripts/31_coco_tables.py [--judge o3]"""
import json
import sys
import numpy as np
import pandas as pd
from svo_eval.paths import RESULTS, ECIR, MODELS, DIRECTIONS, DISPLAY, DIRECTION_LABEL
from svo_eval.coco import load_coco, SUBSETS
from svo_eval import metrics as M

OUT = ECIR / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
judge = sys.argv[sys.argv.index("--judge") + 1] if "--judge" in sys.argv else "o3"
PROBE_NAME = dict(DISPLAY, Qwen25="Qwen2.5-VL (zero-shot)")
GROUPS = {"obj": ["add_obj", "replace_obj", "swap_obj"], "att": ["add_att", "replace_att", "swap_att"], "rel": ["replace_rel"]}
GROUP_LABEL = {"obj": "Obj.", "att": "Att.", "rel": "Rel."}
SUBSET_LABEL = {"add_att": "Add att.", "add_obj": "Add obj.", "replace_att": "Repl. att.", "replace_obj": "Repl. obj.",
                "replace_rel": "Repl. rel.", "swap_att": "Swap att.", "swap_obj": "Swap obj."}


def load(name):
    p = RESULTS / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def fmt(cell, ci=True):
    if cell is None:
        return "--"
    if cell.get("lo") is None or not ci:
        return f"{cell['mean']:.1f}"
    return f"{cell['mean']:.1f}\\ci{{{(cell['hi'] - cell['lo']) / 2:.1f}}}"


def stars(p):
    if p is None:
        return ""
    return "$^{**}$" if p < 0.01 else ("$^{*}$" if p < 0.05 else "")


def write(name, body):
    (OUT / name).write_text(body, encoding="utf-8")
    print("wrote", name)


def grouped(correct, subsets):
    """overall and per edit-type group summaries of a boolean vector aligned with a subset vector."""
    correct, subsets = np.asarray(correct, bool), np.asarray(subsets)
    res = {"all": M.summary(correct)}
    for g, subs in GROUPS.items():
        m = np.isin(subsets, subs)
        res[g] = M.summary(correct[m]) if m.any() else None
    return res


coll = load_coco()
probe_subsets = np.array([p["subset"] for p in coll.probe])
metrics = load("coco_metrics.json") or {}
items = load("coco_pairwise_items.json") or {}
prompted = load("coco_prompted_qwen25.json") or {}
itm = load("coco_itm.json") or {}
pa = load(f"coco_pool_analysis_{judge}.json") or {}
ip = load(f"coco_itm_pools_{judge}.json") or {}
jv = load("coco_judge_validation.json")
models = [m for m in MODELS if m in metrics]


def truthy(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1"]).values


# --- per-item pairwise and ITM correctness by edit type
pairwise_g = {}
for m in models:
    v = items.get(m)
    if v is not None and len(v) == len(probe_subsets):
        pairwise_g[m] = grouped(v, probe_subsets)
pw_csv = RESULTS / "Qwen25_pairwise_sugarcrepe.csv"
if pw_csv.exists():
    df = pd.read_csv(pw_csv, dtype=str, keep_default_na=False)
    ans = df["correct"].str.strip().str.lower()
    ok = ans.isin(["true", "false"]).values
    pairwise_g["Qwen25_zs"] = grouped((ans.values[ok] == "true"), df["subset"].values[ok])
itm_g = {}
for m, (fname, col) in {"BLIP2": ("coco_itm_preds_BLIP2_sugarcrepe.csv", "pred_yes"), "FLAVA": ("coco_itm_preds_FLAVA_sugarcrepe.csv", "pred_yes"),
                        "Qwen3": ("coco_itm_preds_Qwen3_sugarcrepe.csv", "pred_yes"), "Qwen25_zs": ("Qwen25_itm_sugarcrepe.csv", "model_pred")}.items():
    p = RESULTS / fname
    if not p.exists():
        continue
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    v = df[col].str.strip().str.lower()
    ok = v.isin(["1", "true", "0", "false", "yes", "no"]).values
    pred = v.values[ok]
    pred = np.isin(pred, ["1", "true", "yes"])
    itm_g[m] = grouped(pred == truthy(df["is_positive"])[ok], df["subset"].values[ok])

# 1. pairwise + ITM by edit type (layout of tab_pairwise_itm)
rows = []
for m in models:
    if m == "Qwen25":
        pw, it = pairwise_g.get("Qwen25_zs"), itm_g.get("Qwen25_zs")
        rows.append(f"{PROBE_NAME['Qwen25']} & " + " & ".join(fmt(pw[k]) if pw else "--" for k in ["all", "obj", "att", "rel"]) + " & "
                    + " & ".join(fmt(it[k]) if it else "--" for k in ["all", "obj", "att", "rel"]) + " \\\\")
    pw, it = pairwise_g.get(m), itm_g.get(m)
    name = DISPLAY[m] + (" (embedding)" if m == "Qwen25" else "")
    rows.append(f"{name} & " + " & ".join(fmt(pw[k]) if pw else "--" for k in ["all", "obj", "att", "rel"]) + " & "
                + " & ".join(fmt(it[k]) if it else "--" for k in ["all", "obj", "att", "rel"]) + " \\\\")
body = ("\\begin{tabular}{l cccc cccc}\n\\toprule\n"
        "& \\multicolumn{4}{c}{Pairwise ranking accuracy} & \\multicolumn{4}{c}{ITM accuracy}\\\\\n"
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
        "Model & All & Obj. & Att. & Rel. & All & Obj. & Att. & Rel.\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
write("tab_coco_pairwise_itm.tex", body)

# 2. full-collection retrieval (layout of tab_full_retrieval without the semantic columns)
rows = []
for m in models:
    r = metrics[m]
    cells = []
    for d in DIRECTIONS:
        cells += [fmt(r[d]["S@1"]), fmt(r[d]["S@5"], ci=False), fmt(r[d]["S@10"], ci=False)]
    rows.append(f"{DISPLAY[m]} & " + " & ".join(cells) + " \\\\")
n_t2i, n_i2t = len(coll.evaluable_queries("t2i")), len(coll.evaluable_queries("i2t"))
body = ("\\begin{tabular}{l ccc ccc}\n\\toprule\n"
        f"& \\multicolumn{{3}}{{c}}{{{DIRECTION_LABEL['t2i']} ({n_t2i:,} caption queries)}} & \\multicolumn{{3}}{{c}}{{{DIRECTION_LABEL['i2t']} ({n_i2t:,} image queries)}}\\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
        "Model & \\sat{1} & \\sat{5} & \\sat{10} & \\sat{1} & \\sat{5} & \\sat{10}\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
write("tab_coco_full.tex", body)

# 3. matched negatives (layout of tab_matched)
if pa:
    rows = []
    for m in models:
        if m not in pa:
            continue
        cells = []
        for d in DIRECTIONS:
            r = pa[m][d]
            cells += [fmt(r.get("benchmark")) if r.get("benchmark") else "--", fmt(r.get("random")), fmt(r.get("pool_other")),
                      fmt(r.get("pool_self")) + stars(r.get("p_self_vs_other")), fmt(r.get("pool_all")) + stars(r.get("p_all_vs_reference"))]
        rows.append(f"{DISPLAY[m]} & " + " & ".join(cells) + " \\\\")
    body = ("\\begin{tabular}{l ccccc ccccc}\n\\toprule\n"
            "& \\multicolumn{5}{c}{" + DIRECTION_LABEL['t2i'] + "} & \\multicolumn{5}{c}{" + DIRECTION_LABEL['i2t'] + "}\\\\\n"
            "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n"
            "Model & Bench. & Random & Other & Self & All & Bench. & Random & Other & Self & All\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_coco_matched.tex", body)

    # 4. judged success (layout of tab_judge_success, o3 labels; human columns once the sample is labelled)
    rows = []
    for m in models:
        if m not in pa:
            continue
        cells = []
        for d in DIRECTIONS:
            r = pa[m][d]
            cells += [fmt(metrics[m][d]["S@1"], ci=False), fmt(r.get("judged_S@1"), ci=False), fmt(r.get("judged_P@10"), ci=False), str(r.get("n_queries_judged", ""))]
        rows.append(f"{DISPLAY[m]} & " + " & ".join(cells) + " \\\\")
    body = ("\\begin{tabular}{l cccr cccr}\n\\toprule\n"
            "& \\multicolumn{4}{c}{" + DIRECTION_LABEL['t2i'] + "} & \\multicolumn{4}{c}{" + DIRECTION_LABEL['i2t'] + "}\\\\\n"
            "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
            "Model & Strict \\sat{1} & Judged \\sat{1} & P@10 & $n$ & Strict \\sat{1} & Judged \\sat{1} & P@10 & $n$\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_coco_judge_success.tex", body)

# 5. ITM decisions on the pool (layout of tab_itm_pools)
if ip:
    rows = []
    for m in ["BLIP2", "FLAVA", "Qwen25", "Qwen3"]:
        if m not in ip:
            continue
        orig = itm_g.get("Qwen25_zs" if m == "Qwen25" else m)
        cells = [fmt(orig["all"]) if orig else "--"]
        for d in DIRECTIONS:
            r = ip[m].get(d)
            cells += ["--"] * 5 if not r else [fmt(r["pool_all"]), fmt(r["pool_self"]), fmt(r["pool_other"]), fmt(r["positives_recall"]), fmt(r["negatives_recall"])]
        rows.append(f"{PROBE_NAME[m]} & " + " & ".join(cells) + " \\\\")
    body = ("\\begin{tabular}{l c ccccc ccccc}\n\\toprule\n"
            "& & \\multicolumn{5}{c}{" + DIRECTION_LABEL['t2i'] + " pool} & \\multicolumn{5}{c}{" + DIRECTION_LABEL['i2t'] + " pool}\\\\\n"
            "\\cmidrule(lr){3-7}\\cmidrule(lr){8-12}\n"
            "Model & Orig. & All & Self & Other & TPR & TNR & All & Self & Other & TPR & TNR\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_coco_itm_pools.tex", body)

# 6. pairwise by subset (appendix)
rows = []
for m in models:
    r = metrics[m]["pairwise"]
    if m == "Qwen25" and prompted.get("pairwise"):
        r2 = prompted["pairwise"]
        rows.append(f"{PROBE_NAME['Qwen25']}" + "".join(f" & {fmt(r2.get(s), ci=False)}" for s in SUBSETS) + " \\\\")
    rows.append(f"{DISPLAY[m]}" + (" (embedding)" if m == "Qwen25" else "") + "".join(f" & {fmt(r.get(s), ci=False)}" for s in SUBSETS) + " \\\\")
body = ("\\begin{tabular}{l " + "c" * len(SUBSETS) + "}\n\\toprule\nModel & " + " & ".join(SUBSET_LABEL[s] for s in SUBSETS) +
        "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
write("tab_coco_pairwise_subsets.tex", body)

# 7. assessor validation on the human sample
if jv:
    rows = []
    for d, label in [("t2i", DIRECTION_LABEL["t2i"]), ("i2t", DIRECTION_LABEL["i2t"]), ("overall", "Both")]:
        r = jv.get(d)
        if not r:
            continue
        k, f = r.get("kappa_ci") or [None, None], r.get("f1_ci") or [None, None]
        kappa = f"{r['kappa']:.2f}" + (f" [{k[0]:.2f}, {k[1]:.2f}]" if k[0] is not None else "")
        f1s = f"{r['f1_correct']:.2f}" + (f" [{f[0]:.2f}, {f[1]:.2f}]" if f[0] is not None else "")
        rows.append(f"{label} & {r['n_pairs']} & {100 * r['accuracy']:.1f} & {100 * r['precision_correct']:.1f} & {100 * r['recall_correct']:.1f} & {f1s} & {kappa} \\\\")
    body = ("\\begin{tabular}{l r c c c c c}\n\\toprule\nDirection & $n$ & Acc. & P$_{\\mathrm{corr}}$ & R$_{\\mathrm{corr}}$ & F1 (correct) & Cohen's $\\kappa$\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_coco_judge.tex", body)
print("done")
