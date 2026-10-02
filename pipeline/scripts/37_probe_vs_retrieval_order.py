"""Does the probe order the embedding models as retrieval does? (paper, Sec. 5)

For the six embedding models (the fine-tuned Qwen2.5 embedder with its own similarity decision, not the prompted model):
Kendall's tau between pairwise accuracy and strict Success@1, and paired bootstrap intervals for the difference in
pairwise accuracy between selected pairs of models (SVO-Probes: resampling captions; SugarCrepe: resampling items).
Output: results/probe_vs_retrieval_order.json"""
import json
import numpy as np
from scipy.stats import kendalltau
from svo_eval.paths import RESULTS, MODELS, DIRECTIONS
from svo_eval.collection import load_collection
from svo_eval.embeddings import Scorer

full = json.load(open(RESULTS / "full_metrics.json"))
cfull = json.load(open(RESULTS / "coco_metrics.json"))
citems = json.load(open(RESULTS / "coco_pairwise_items.json"))
coll = load_collection()
df = coll.df
cap_id = df["sentence"].map(coll.cap_index).to_numpy()
out = {"svo": {}, "coco": {}}

# per-triplet pairwise correctness of every embedding model on SVO-Probes
corr = {}
for m in MODELS:
    e = Scorer(m).e
    ci = np.array([e.cap_index[c] for c in df["sentence"]])
    pi = np.array([e.img_index[int(i)] for i in df["pos_image_id"]])
    ni = np.array([e.img_index[int(i)] for i in df["neg_image_id"]])
    corr[m] = (np.einsum("ij,ij->i", e.cap[ci], e.img[pi]) > np.einsum("ij,ij->i", e.cap[ci], e.img[ni])).astype(float)
pw = {m: 100 * corr[m].mean() for m in MODELS}
out["svo"]["pairwise"] = pw
out["svo"]["S@1"] = {d: {m: full[m][d]["S@1"]["mean"] for m in MODELS} for d in DIRECTIONS}
cpw = {m: 100 * float(np.mean(citems[m])) for m in MODELS}
out["coco"]["pairwise"] = cpw
out["coco"]["S@1"] = {d: {m: cfull[m][d]["S@1"]["mean"] for m in MODELS} for d in DIRECTIONS}
for name, p in [("svo", pw), ("coco", cpw)]:
    out[name]["kendall"] = {}
    for d in DIRECTIONS:
        s1 = out[name]["S@1"][d]
        tau, pval = kendalltau([p[m] for m in MODELS], [s1[m] for m in MODELS])
        out[name]["kendall"][d] = {"tau": float(tau), "p": float(pval)}
        print(name, d, "pairwise", {m: round(p[m], 1) for m in MODELS}, "S@1", {m: round(s1[m], 1) for m in MODELS},
              f"Kendall tau {tau:.2f} (p {pval:.2f})")


def clustered_diff(a, b, clusters, n_boot=2000, seed=0):
    """paired bootstrap of mean(a) - mean(b), resampling clusters"""
    k = clusters.max() + 1
    da = np.bincount(clusters, weights=a - b, minlength=k)
    cnt = np.bincount(clusters, minlength=k).astype(float)
    keep = cnt > 0
    da, cnt = da[keep], cnt[keep]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(cnt), len(cnt))
        boots.append(da[idx].sum() / cnt[idx].sum())
    return {"diff": 100 * float((a - b).mean()), "lo": 100 * float(np.quantile(boots, 0.025)), "hi": 100 * float(np.quantile(boots, 0.975))}


out["svo"]["pairs"], out["coco"]["pairs"] = {}, {}
for a, b in [("FLAVA", "SigLIP2"), ("BLIP2", "SigLIP2"), ("FLAVA", "CLIP"), ("BLIP2", "Qwen25")]:
    r = clustered_diff(corr[a], corr[b], cap_id)
    out["svo"]["pairs"][f"{a}-{b}"] = r
    print(f"SVO  pairwise {a} - {b}: {r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]   S@1 t2i {full[a]['t2i']['S@1']['mean']:.1f} vs {full[b]['t2i']['S@1']['mean']:.1f}"
          f"   i2t {full[a]['i2t']['S@1']['mean']:.1f} vs {full[b]['i2t']['S@1']['mean']:.1f}")
for a, b in [("FLAVA", "SigLIP2"), ("FLAVA", "CLIP"), ("Qwen25", "CLIP")]:
    x, y = np.array(citems[a], float), np.array(citems[b], float)
    r = clustered_diff(x, y, np.arange(len(x)))
    out["coco"]["pairs"][f"{a}-{b}"] = r
    print(f"COCO pairwise {a} - {b}: {r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]   S@1 t2i {cfull[a]['t2i']['S@1']['mean']:.1f} vs {cfull[b]['t2i']['S@1']['mean']:.1f}"
          f"   i2t {cfull[a]['i2t']['S@1']['mean']:.1f} vs {cfull[b]['i2t']['S@1']['mean']:.1f}")
json.dump(out, open(RESULTS / "probe_vs_retrieval_order.json", "w"), indent=1)
print("saved")
