"""Render PDF figures from results/*.json into figures/ (and copy to both paper folders)."""
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from svo_eval.paths import RESULTS, FIGURES, ECIR, MODELS, DIRECTIONS, DISPLAY, DUAL_ENCODERS

# validated categorical palette (dataviz skill reference instance, slots 1-6, fixed order)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
COLOR = {m: PALETTE[i] for i, m in enumerate(MODELS)}
MARK = {m: MARKERS[i] for i, m in enumerate(MODELS)}
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#3987e5", "#0d366b"])
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
DLAB = {"t2i": "T→I (caption query)", "i2t": "I→T (image query)"}
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                     "axes.axisbelow": True, "lines.linewidth": 1.4, "lines.markersize": 4})


def load(name):
    p = RESULTS / name
    return json.load(open(p)) if p.exists() else None


def save(fig, name):
    path = FIGURES / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    for sub in ["paper", "paper_subset"]:
        d = ECIR / sub / "figures"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, d / name)
    print("wrote", name)


nway = load("nway.json")


def nway_figure(models, name, ncol):
    """n-way curves for the given systems; the paper shows all six, the dual-encoder subset the four dual encoders."""
    # k = "all" is plotted at the size of the candidate pool minus the positives (whole collection)
    N_ALL = {"t2i": 13285 - 1, "i2t": 10978 - 1}
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.55), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, d in zip(axes, DIRECTIONS):
        for m in models:
            r = nway[m][d]
            ks = [N_ALL[d] if k == "all" else k for k in r["k"]]
            rand = [v["mean"] for v in r["random"]]
            ax.plot(ks, rand, color=COLOR[m], marker=MARK[m], linestyle="-", markeredgecolor="white",
                    markeredgewidth=0.6, label=DISPLAY[m])
            px = [k for k, v in zip(ks, r["pool"]) if v is not None]
            py = [v["mean"] for v in r["pool"] if v is not None]
            ax.plot(px, py, color=COLOR[m], marker=MARK[m], linestyle="--", markeredgecolor="white",
                    markeredgewidth=0.6)
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 100, 1000, 10000])
        ax.set_xticklabels(["1", "10", "100", "1k", "10k"])
        ax.set_xlim(0.8, 2.2e4)
        ax.set_xlabel("number of negatives $k$")
        ax.set_title(DLAB[d], fontsize=8, color=INK)
        ax.set_ylim(0, 102)
    axes[0].set_ylabel("positive outranks all $k$ (%)")
    axes[1].tick_params(labelleft=False)
    # one legend below the panels: the six systems, then the two line styles (no text inside the panels)
    from matplotlib.lines import Line2D
    h, l = axes[0].get_legend_handles_labels()
    h += [Line2D([0], [0], color=INK2, linestyle="-", linewidth=1.1), Line2D([0], [0], color=INK2, linestyle="--", linewidth=1.1)]
    l += ["random corpus negatives", "judged-pool negatives"]
    fig.legend(h, l, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.36), fontsize=6.3, handlelength=1.8,
               columnspacing=1.0, handletextpad=0.5, labelspacing=0.35)
    save(fig, name)


if nway:
    nway_figure(MODELS, "fig_nway.pdf", 4)
    nway_figure(DUAL_ENCODERS, "fig_nway_short.pdf", 3)

mv = load("metric_validation.json")
if mv:
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.0), sharey=True)
    for ax, d in zip(axes, [mv["dev_direction"], mv["held_out_direction"]]):
        cs = mv["curves"]["strict"][d]
        taus = [r["tau"] for r in cs]
        for key, col, lab in [("precision", PALETTE[0], "precision"), ("recall", PALETTE[1], "recall"), ("f1", PALETTE[2], "F1")]:
            ax.plot(taus, [r[key] for r in cs], color=col, marker="o", markeredgecolor="white", markeredgewidth=0.6, label=lab + " (strict)")
        cl = mv["curves"]["lenient"][d]
        ax.plot(taus, [r["f1"] for r in cl], color=INK2, linestyle="--", label="F1 (lenient rule)")
        ax.axvline(mv["chosen_tau"], color=GRID, linewidth=1)
        ax.set_xlabel(r"threshold $\tau$")
        ax.set_title(("dev: " if d == mv["dev_direction"] else "held-out: ") + DLAB[d], fontsize=8, color=INK)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("agreement with human labels")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.16), fontsize=7)
    save(fig, "fig_tau.pdf")

errors = load("errors.json")
if errors:
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.6), sharey=True)
    cats = [("correct", "#c3c2b7", "correct"), ("subject", PALETTE[0], "subject error"),
            ("verb", PALETTE[1], "verb error"), ("object", PALETTE[2], "object error")]
    for ax, d in zip(axes, DIRECTIONS):
        y = np.arange(len(MODELS))
        left = np.zeros(len(MODELS))
        for key, col, lab in cats:
            vals = np.array([errors[m][d][key]["mean"] for m in MODELS])
            ax.barh(y, vals, left=left, color=col, edgecolor="white", linewidth=1, height=0.62, label=lab)
            for yi, v, lft in zip(y, vals, left):
                if v >= 9:
                    ax.text(lft + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                            color="white" if key != "correct" else INK)
            left += vals
        ax.set_yticks(y)
        ax.set_yticklabels([DISPLAY[m] for m in MODELS])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("share of judged top-10 items (%)")
        ax.set_title(DLAB[d], fontsize=8, color=INK)
        ax.grid(axis="y", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.36), fontsize=7)
    save(fig, "fig_errors.pdf")

cross = load("cross_model.json")
if cross:
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.55), sharey=True, gridspec_kw={"wspace": 0.08})
    cols = MODELS + ["benchmark", "random"]
    for ax, d in zip(axes, DIRECTIONS):
        mat = np.array([[cross[d][sc][mi]["mean"] for mi in cols] for sc in MODELS])
        im = ax.imshow(mat, cmap=SEQ, vmin=40, vmax=100, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if mat[i, j] < 72 else INK)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([DISPLAY.get(c, c.capitalize()) for c in cols], rotation=45, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([DISPLAY[m] for m in MODELS], fontsize=6.5)
        ax.set_title(DLAB[d], fontsize=8, color=INK)
        ax.set_xlabel("negatives mined by")
        ax.grid(False)
    axes[1].tick_params(labelleft=False)
    axes[0].set_ylabel("scored by")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("pairwise accuracy (%)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    save(fig, "fig_cross.pdf")

matched = load("matched_negatives.json")
if matched:
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.1), sharey=True)
    sources = [("benchmark", "bench-\nmark"), ("random", "random"), ("pool_other", "pool\nother"),
               ("pool_self", "pool\nself")]
    for ax, d in zip(axes, DIRECTIONS):
        x = np.arange(len(sources))
        for i, m in enumerate(MODELS):
            means = [matched[m][d][s]["mean"] if matched[m][d][s] else np.nan for s, _ in sources]
            los = [matched[m][d][s]["lo"] if matched[m][d][s] else np.nan for s, _ in sources]
            his = [matched[m][d][s]["hi"] if matched[m][d][s] else np.nan for s, _ in sources]
            off = (i - (len(MODELS) - 1) / 2) * 0.12
            ax.errorbar(x + off, means, yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
                        fmt=MARK[m], color=COLOR[m], markeredgecolor="white", markeredgewidth=0.6, capsize=1.5,
                        elinewidth=0.8, label=DISPLAY[m])
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in sources], fontsize=7)
        ax.set_xlim(-0.5, len(sources) - 0.5)
        ax.set_title(DLAB[d], fontsize=8, color=INK)
        ax.set_ylim(25, 102)
    axes[0].set_ylabel("positive outranks negative (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.12), fontsize=6.5)
    save(fig, "fig_matched.pdf")
print("done")
