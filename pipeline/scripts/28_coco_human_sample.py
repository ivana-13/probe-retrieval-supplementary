"""Human validation sample of the COCO pool, its annotation sheet, the import of the finished sheet, and the
agreement between the LLM assessor and the human labels.

  make [--queries N] [--seed S]
      Draw N whole queries per direction (default 12, seed 0) from results/coco_pool_candidates.json. Candidates that
      are gold by construction (the caption's own image, the image's own caption) are not shown; they count as
      "correct". Writes results/coco_human_sample.json and annotation/coco_annotation.html, a self-contained sheet
      made for speed: the current pair is highlighted, keys 1-5 label it (1 correct, 2 incorrect, 3 subject, 4 verb,
      5 object incorrect) and move on, arrows move, progress is kept in the browser, "Download CSV" exports. Candidates
      are shown in a seeded random order, blind to the retrieving systems. Open the sheet from its folder so that the
      image paths resolve (annotation/ -> ../data/coco/val2017). Stopping early is fine: partial sheets import.
  import <csv>
      Read the downloaded CSV (direction, query, candidate, label) and write results/coco_pool_labels_human.json in
      the format of the LLM labels, gold-by-construction pairs of the sampled queries included as "correct".
  agreement [--judge o3]
      Cohen's kappa, F1 (class "correct"), accuracy of the LLM labels against the human labels on the labelled sampled
      pairs, per direction and overall, with bootstrap intervals, plus error-type agreement on the pairs both label
      with a specific error type. Gold-by-construction pairs are excluded. Writes results/coco_judge_validation.json.
"""
import csv
import html
import json
import random
import sys
import numpy as np
from svo_eval.paths import RESULTS, ECIR
from svo_eval.coco import load_coco

TYPES = ["subject incorrect", "verb incorrect", "object incorrect"]
LABELS = ["correct", "incorrect"] + TYPES          # "incorrect" = wrong without a stated category (human sheet only)
KEYS = {"1": "correct", "2": "incorrect", "3": "subject incorrect", "4": "verb incorrect", "5": "object incorrect"}
args = sys.argv[1:]
cmd = args[0] if args else "make"


def opt(name, default, cast):
    return cast(args[args.index(name) + 1]) if name in args else default


coll = load_coco()


def is_gold(d, q, c):
    return (int(c) in coll.gold("t2i", q)) if d == "t2i" else (c in coll.gold("i2t", int(q)))


SAMPLE = RESULTS / "coco_human_sample.json"
ANNOT_DIR = ECIR / "annotation"

if cmd == "make":
    n_q, seed = opt("--queries", 12, int), opt("--seed", 0, int)
    pool = json.load(open(RESULTS / "coco_pool_candidates.json", encoding="utf-8"))
    rng = random.Random(seed)
    sample = {"seed": seed, "queries_per_direction": n_q, "t2i": {}, "i2t": {}}
    n_pairs, n_gold = 0, 0
    for d in ("t2i", "i2t"):
        qs = sorted(pool[d])
        rng.shuffle(qs)
        for q in qs[:n_q]:
            cands = list(pool[d][q])
            gold = [c for c in cands if is_gold(d, q, c)]
            shown = [c for c in cands if c not in gold]
            rng.shuffle(shown)
            sample[d][q] = {"to_label": shown, "gold": gold}
            n_pairs += len(shown)
            n_gold += len(gold)
    json.dump(sample, open(SAMPLE, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"sample: {n_q} queries per direction, {n_pairs} pairs to label, {n_gold} gold pairs labelled automatically")

    ANNOT_DIR.mkdir(exist_ok=True)
    esc = html.escape

    def img_tag(image_id, cls=""):
        return f'<img class="{cls}" loading="lazy" src="../data/coco/val2017/{int(image_id):012d}.jpg" alt="image {int(image_id)}">'

    blocks, pair_no = [], 0
    for d in ("t2i", "i2t"):
        title = ("Part 1: caption queries. Is each image a correct retrieval for the caption?" if d == "t2i" else
                 "Part 2: image queries. Is each caption a correct description of the image?")
        blocks.append(f"<h2>{title}</h2>")
        for qi, (q, s) in enumerate(sample[d].items(), 1):
            head = (f'<div class="query"><span class="qn">Query {qi} of {n_q}</span><p class="qtext">{esc(q)}</p></div>' if d == "t2i"
                    else f'<div class="query"><span class="qn">Query {qi} of {n_q}</span>{img_tag(q, "qimg")}</div>')
            rows = []
            for c in s["to_label"]:
                pair_no += 1
                key = f"{d}|{q}|{c}"
                body = img_tag(c, "cimg") if d == "t2i" else f'<p class="ctext">{esc(str(c))}</p>'
                buttons = "".join(f'<button type="button" data-label="{lab}" title="key {k}">{k} {lab.replace(" incorrect", "")}</button>'
                                  for k, lab in KEYS.items())
                rows.append(f'<div class="pair" data-key="{esc(key)}" data-n="{pair_no}"><div class="num">{pair_no}</div>'
                            f'<div class="cand">{body}</div><div class="labels">{buttons}<span class="chosen"></span></div></div>')
            blocks.append(f'<section class="qblock">{head}{"".join(rows)}</section>')
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>COCO pool annotation</title>
<style>
:root {{ --bg:#f6f6f4; --fg:#1c1c1a; --muted:#6a6a65; --card:#fff; --line:#e2e2de; --acc:#2f6fed; --ok:#2e9e5b; --bad:#d64545; --cur:#fff7d6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141416; --fg:#ecece8; --muted:#a3a39d; --card:#1e1e21; --line:#2e2e33; --acc:#5b8dff; --ok:#4cc07a; --bad:#f06a6a; --cur:#3a3410; }} }}
* {{ box-sizing:border-box }} body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.45 system-ui,sans-serif; padding:0 16px 60vh }}
main {{ max-width:1100px; margin:0 auto }} h1 {{ font-size:22px; margin:18px 0 6px }} h2 {{ font-size:18px; margin:28px 0 10px }}
.bar {{ position:sticky; top:0; z-index:5; background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 14px; margin:12px 0;
       display:flex; gap:14px; align-items:center; flex-wrap:wrap; font-size:14px }}
.bar button {{ padding:7px 12px; border-radius:8px; border:1px solid var(--line); background:var(--acc); color:#fff; font-weight:600; cursor:pointer }}
.muted {{ color:var(--muted) }} kbd {{ border:1px solid var(--line); border-radius:4px; padding:0 5px; font-size:12px; background:var(--bg) }}
.qblock {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin:12px 0 }}
.query {{ display:flex; gap:14px; align-items:center; border-bottom:2px solid var(--acc); padding-bottom:10px; margin-bottom:6px }}
.qn {{ font-size:12px; font-weight:700; color:var(--muted); text-transform:uppercase; white-space:nowrap }} .qtext {{ font-size:18px; font-weight:600; margin:0 }}
img.qimg {{ max-height:300px; max-width:100%; border-radius:6px }} img.cimg {{ max-height:280px; max-width:100%; border-radius:6px; display:block }}
.pair {{ display:grid; grid-template-columns:36px 1fr 330px; gap:12px; align-items:center; padding:8px 6px; border-top:1px solid var(--line); border-radius:6px }}
.pair.cur {{ background:var(--cur); outline:2px solid var(--acc) }} .pair.done .num {{ color:var(--ok) }}
.num {{ color:var(--muted); font-size:13px; font-weight:600 }} .ctext {{ margin:0; font-size:17px }}
.labels {{ display:flex; flex-wrap:wrap; gap:4px; align-items:center }}
.labels button {{ padding:4px 8px; border-radius:6px; border:1px solid var(--line); background:var(--bg); color:var(--fg); cursor:pointer; font-size:13px }}
.labels button.on {{ background:var(--acc); color:#fff; border-color:var(--acc) }} .chosen {{ font-size:12px; color:var(--muted); width:100% }}
@media (max-width:760px) {{ .pair {{ grid-template-columns:28px 1fr }} .labels {{ grid-column:2 }} }}
</style></head><body><main>
<h1>COCO pool annotation</h1>
<p class="muted">Click a pair (or just start): the highlighted pair is the current one. Press <kbd>1</kbd> correct, <kbd>2</kbd> incorrect,
or <kbd>3</kbd> subject / <kbd>4</kbd> verb / <kbd>5</kbd> object incorrect if the category is obvious; the highlight moves to the next pair by itself.
<kbd>&uarr;</kbd> <kbd>&darr;</kbd> move without labelling, <kbd>Backspace</kbd> clears. Your labels are saved in this browser as you go; download the CSV when you stop, even partway.
Subject: the main subject of the caption is not in the image. Verb: the activity or relation does not match. Object: other details are wrong.</p>
<div class="bar"><button id="dl">Download CSV</button><button id="next" style="background:var(--muted)">Jump to next unlabelled</button><span id="prog"></span><span class="muted">Seed {seed}, {n_q} queries per direction.</span></div>
{"".join(blocks)}
</main>
<script>
(function () {{
  var KEYMAP = {json.dumps(KEYS)}, STORE = "coco_annotation_seed{seed}_q{n_q}", state = {{}};
  try {{ state = JSON.parse(localStorage.getItem(STORE) || "{{}}"); }} catch (e) {{ state = {{}}; }}
  var pairs = Array.prototype.slice.call(document.querySelectorAll(".pair")), cur = -1;
  function save() {{ try {{ localStorage.setItem(STORE, JSON.stringify(state)); }} catch (e) {{}} }}
  function render(p) {{
    var lab = state[p.dataset.key] || "";
    p.classList.toggle("done", !!lab);
    p.querySelectorAll("button").forEach(function (b) {{ b.classList.toggle("on", b.dataset.label === lab); }});
    p.querySelector(".chosen").textContent = lab ? "labelled: " + lab : "";
  }}
  function progress() {{
    var done = pairs.filter(function (p) {{ return !!state[p.dataset.key]; }}).length;
    document.getElementById("prog").textContent = done + " / " + pairs.length + " pairs labelled";
  }}
  function setCur(i, scroll) {{
    if (i < 0 || i >= pairs.length) return;
    if (cur >= 0) pairs[cur].classList.remove("cur");
    cur = i; pairs[cur].classList.add("cur");
    if (scroll) pairs[cur].scrollIntoView({{block: "center", behavior: "smooth"}});
  }}
  function label(i, lab) {{
    var p = pairs[i];
    if (lab) state[p.dataset.key] = lab; else delete state[p.dataset.key];
    save(); render(p); progress();
  }}
  function nextUnlabelled(from) {{
    for (var i = from + 1; i < pairs.length; i++) if (!state[pairs[i].dataset.key]) return i;
    for (var j = 0; j < pairs.length; j++) if (!state[pairs[j].dataset.key]) return j;
    return -1;
  }}
  pairs.forEach(function (p, i) {{
    render(p);
    p.addEventListener("click", function (e) {{ setCur(i, false); }});
    p.querySelectorAll("button").forEach(function (b) {{
      b.addEventListener("click", function (e) {{ e.stopPropagation(); label(i, b.dataset.label); setCur(Math.min(i + 1, pairs.length - 1), true); }});
    }});
  }});
  document.addEventListener("keydown", function (e) {{
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (cur < 0) setCur(0, true);
    if (KEYMAP[e.key]) {{ label(cur, KEYMAP[e.key]); setCur(Math.min(cur + 1, pairs.length - 1), true); e.preventDefault(); }}
    else if (e.key === "ArrowDown" || e.key === "j") {{ setCur(cur + 1, true); e.preventDefault(); }}
    else if (e.key === "ArrowUp" || e.key === "k") {{ setCur(cur - 1, true); e.preventDefault(); }}
    else if (e.key === "Backspace") {{ label(cur, ""); e.preventDefault(); }}
  }});
  document.getElementById("next").addEventListener("click", function () {{ var i = nextUnlabelled(cur); if (i >= 0) setCur(i, true); }});
  document.getElementById("dl").addEventListener("click", function () {{
    var lines = ["direction,query,candidate,label"];
    function q(s) {{ return '"' + String(s).replace(/"/g, '""') + '"'; }}
    pairs.forEach(function (p) {{
      var lab = state[p.dataset.key]; if (!lab) return;
      var k = p.dataset.key, i1 = k.indexOf("|"), i2 = k.lastIndexOf("|");
      lines.push([k.slice(0, i1), q(k.slice(i1 + 1, i2)), q(k.slice(i2 + 1)), lab].join(","));
    }});
    var blob = new Blob([lines.join("\\n")], {{type: "text/csv"}}), a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "coco_annotation_labels.csv"; a.click();
  }});
  progress();
  var first = nextUnlabelled(-1); setCur(first >= 0 ? first : 0, false);
}})();
</script></body></html>
"""
    (ANNOT_DIR / "coco_annotation.html").write_text(page, encoding="utf-8")
    print("sheet:", ANNOT_DIR / "coco_annotation.html")

elif cmd == "import":
    path = args[1]
    sample = json.load(open(SAMPLE, encoding="utf-8"))
    labels = {"t2i": {}, "i2t": {}}
    for d in ("t2i", "i2t"):
        for q, s in sample[d].items():
            labels[d][q] = {c: "correct" for c in s["gold"]}
            for c in s["to_label"]:
                labels[d][q][c] = ""
    n = 0
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            d, q, c, lab = row["direction"], row["query"], row["candidate"], row["label"].strip().lower()
            if lab not in LABELS:
                continue
            if q in labels[d] and c in labels[d][q]:
                labels[d][q][c] = lab
                n += 1
    out = RESULTS / "coco_pool_labels_human.json"
    json.dump(labels, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    missing = sum(1 for d in labels for v in labels[d].values() for x in v.values() if not x)
    print(f"imported {n} labels; {missing} sampled pairs left unlabelled (ignored in the agreement); saved {out.name}")

elif cmd == "agreement":
    from sklearn.metrics import cohen_kappa_score, f1_score
    from svo_eval.judges import binary_agreement, multiclass_agreement, bootstrap_stat
    judge = opt("--judge", "o3", str)
    human = json.load(open(RESULTS / "coco_pool_labels_human.json", encoding="utf-8"))
    llm = json.load(open(RESULTS / f"coco_pool_labels_{judge}.json", encoding="utf-8"))
    sample = json.load(open(SAMPLE, encoding="utf-8"))
    out = {"judge": judge}
    all_t, all_p, all_te, all_pe = [], [], [], []
    for d in ("t2i", "i2t"):
        t, p, te, pe = [], [], [], []
        for q, s in sample[d].items():
            for c in s["to_label"]:
                h, l = human.get(d, {}).get(q, {}).get(c, ""), llm.get(d, {}).get(q, {}).get(c, "")
                if h in LABELS and l in LABELS:
                    t.append(h == "correct"); p.append(l == "correct"); te.append(h); pe.append(l)
        t, p = np.array(t, bool), np.array(p, bool)
        res = {}
        if len(t):
            res = binary_agreement(t, p)
            res["f1_ci"] = bootstrap_stat(lambda a, b: f1_score(a, b), t, p)
            res["kappa_ci"] = bootstrap_stat(lambda a, b: cohen_kappa_score(a, b), t, p)
            typed = [(x, y) for x, y in zip(te, pe) if x in TYPES and y in TYPES]
            res["error_type_among_incorrect"] = multiclass_agreement([x for x, _ in typed], [y for _, y in typed], TYPES) if typed else None
            res["n_pairs"] = int(len(t)); res["n_typed"] = int(len(typed)); res["human_share_correct"] = float(t.mean())
        out[d] = res
        all_t.append(t); all_p.append(p); all_te += te; all_pe += pe
        print(d, {k: round(v, 3) for k, v in res.items() if isinstance(v, float)}, "n", len(t))
    t, p = np.concatenate(all_t), np.concatenate(all_p)
    if len(t):
        res = binary_agreement(t, p)
        res["f1_ci"] = bootstrap_stat(lambda a, b: f1_score(a, b), t, p)
        res["kappa_ci"] = bootstrap_stat(lambda a, b: cohen_kappa_score(a, b), t, p)
        res["n_pairs"] = int(len(t))
        out["overall"] = res
        print("overall", {k: round(v, 3) for k, v in res.items() if isinstance(v, float)}, "n", len(t))
    json.dump(out, open(RESULTS / "coco_judge_validation.json", "w"), indent=1)
    print("saved", RESULTS / "coco_judge_validation.json")
else:
    raise SystemExit("commands: make | import <csv> | agreement")
