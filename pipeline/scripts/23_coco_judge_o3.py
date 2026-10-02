"""Label the COCO pool candidates with an LLM assessor through OpenRouter, with the prompt structure of the
SVO-Probes o3 experiment (one call per query covering its whole candidate set, images attached in order).

Reads  results/coco_pool_candidates.json   (direction -> query -> candidate -> {"retrieved_by": ...})
Writes results/coco_pool_labels_<judge>.json {direction: {query: {candidate: "correct" | "<type> incorrect" | ""}}}
       results/coco_judge_usage_<judge>.jsonl one line per call: tokens, reasoning tokens, cost in USD as reported
                                              by OpenRouter (or estimated from list prices when not reported)
       results/coco_judge_raw_<judge>.jsonl   the raw answers, for auditing
The run resumes: queries whose candidates are all labelled are skipped. It stops when the cost of the current run
exceeds --budget USD (default 40).

Environment: OPENROUTER_API_KEY (required for real calls); optional OPENROUTER_MODEL (default openai/o3, the
o3-2025-04-16 checkpoint used on SVO-Probes) and OPENROUTER_BASE_URL (default https://openrouter.ai/api/v1).
Any OpenAI-compatible endpoint works through the same variables, for example
  OpenAI directly:  OPENROUTER_BASE_URL=https://api.openai.com/v1            OPENROUTER_MODEL=o3-2025-04-16
  Azure OpenAI v1:  OPENROUTER_BASE_URL=https://<resource>.openai.azure.com/openai/v1  OPENROUTER_MODEL=<deployment>
with the corresponding key in OPENROUTER_API_KEY. On OpenRouter the reasoning effort and the per-call cost use
OpenRouter's request fields; elsewhere the standard `reasoning_effort` field is sent and the cost is estimated from
the o3 list price ($2 / $8 per million input / output tokens).
Usage: python scripts/23_coco_judge_o3.py [t2i] [i2t] [--dry-run] [--pilot N] [--budget USD] [--effort low|medium|high]
                                          [--judge NAME] [--skip-gold]
  --dry-run   count calls, images and text tokens and print a cost range without any API call
  --pilot N   only the first N queries per direction; their labels are kept, so the full run later resumes after them
  --judge     name used in the output files (default o3)
  --skip-gold do not send candidates that are gold by construction (the caption's own image / the image's own
              caption); they are labelled "correct" directly. Default: send everything, as on SVO-Probes."""
import base64
import json
import os
import re
import sys
import time
from svo_eval.paths import RESULTS
from svo_eval.coco import load_coco

args = sys.argv[1:]


def opt(name, default=None, cast=str):
    if name in args:
        return cast(args[args.index(name) + 1])
    return default


directions = [a for a in args if a in ("t2i", "i2t")] or ["i2t", "t2i"]
DRY = "--dry-run" in args
PILOT = opt("--pilot", None, int)
BUDGET = opt("--budget", 40.0, float)
EFFORT = opt("--effort", "medium")
JUDGE = opt("--judge", "o3")
SKIP_GOLD = "--skip-gold" in args
MODEL = os.environ.get("OPENROUTER_MODEL", "openai/o3")
BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
IS_OPENROUTER = "openrouter.ai" in BASE_URL
PRICE_IN, PRICE_OUT = 2.0, 8.0                      # USD per million tokens, o3 list price (fallback estimate only)
# request fields: OpenRouter takes its unified `reasoning` object and can report the cost; OpenAI and Azure take
# the standard `reasoning_effort` field
REQUEST_EXTRA = ({"extra_body": {"reasoning": {"effort": EFFORT}, "usage": {"include": True}}} if IS_OPENROUTER
                 else {"reasoning_effort": EFFORT})

coll = load_coco()
pool = json.load(open(RESULTS / "coco_pool_candidates.json", encoding="utf-8"))
out_path = RESULTS / f"coco_pool_labels_{JUDGE}.json"
usage_path = RESULTS / f"coco_judge_usage_{JUDGE}.jsonl"
raw_path = RESULTS / f"coco_judge_raw_{JUDGE}.jsonl"
labels = json.load(open(out_path, encoding="utf-8")) if out_path.exists() else {"t2i": {}, "i2t": {}}

# The prompts are the SVO-Probes ones (repository root: o3.py and o3_text_retrieval.py) word for word, except that the
# typo `<imge_number>` is fixed. The I->T prompt receives the candidate captions as a Python list literal, as the
# original did; the model numbers them by position.
PROMPT_T2I = ("Assess if given images are correct retrievals for the text query provided {caption}.\n\n"
              "For each image, evaluate if it is correct. If it is incorrect, mention the specific category that best "
              "describes the error:\n- **Subject incorrect**: The subject in the caption does not match the image.\n"
              "- **Verb incorrect**: The activity described in the caption does not match the image.\n"
              "- **Object incorrect**: Other details (e.g., objects or contextual elements) in the caption do not align "
              "with the image.\n\n# Steps\n1. For each image, evaluate its correctness based on text query.\n"
              "2. If the image aligns well with the caption, classify it as `correct`.\n3. If it is incorrect, determine "
              "the category of the error:\n   - **Subject incorrect**\n   - **Verb incorrect**\n   - **Object incorrect** \n"
              "4. Output results using a structured, simple list.\n\n# Output Format\n\nThe results should be listed in "
              "this format:\n- `<image_number>: <classification>`\n\nFor example:\n`1: correct`\n`2: verb incorrect`\n`3: object incorrect`.")
PROMPT_I2T = ("Assess if given captions {captions} are correct retrievals for the image query provided.\n\n"
              "For each caption, evaluate if it is correct. If it is incorrect, mention the specific category that best "
              "describes the error:\n- **Subject incorrect**: The subject in the caption does not match the image.\n"
              "- **Verb incorrect**: The activity described in the caption does not match the image.\n"
              "- **Object incorrect**: Other details (e.g., objects or contextual elements) in the caption do not align "
              "with the image.\n\n# Steps\n1. For each caption, evaluate its correctness based on the image query "
              "(`{{image}}`).\n2. If the caption aligns well with the image, classify it as `correct`.\n3. If it is "
              "incorrect, determine the category of the error:\n   - **Subject incorrect**\n   - **Verb incorrect**\n"
              "   - **Object incorrect** \n4. Output results using a structured, simple list.\n\n# Output Format\n\n"
              "The results should be listed in this format:\n- `<caption_number>: <classification>`\n\nFor example:\n"
              "`1: correct`\n`2: verb incorrect`\n`3: object incorrect`.")
ANSWER = re.compile(r"(\d+)\s*[:.)-]\s*\**\s*(correct|subject incorrect|verb incorrect|object incorrect)", re.IGNORECASE)


def image_item(image_id):
    data = base64.b64encode(open(coll.image_path(image_id), "rb").read()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + data}}


def is_gold(d, q, c):
    return (int(c) in coll.gold("t2i", q)) if d == "t2i" else (c in coll.gold("i2t", int(q)))


def build(d, q, cand_list):
    """Message content for one query; returns (content, n_images, n_text_chars)."""
    if d == "t2i":
        text = PROMPT_T2I.format(caption=q)
        return [{"type": "text", "text": text}] + [image_item(int(c)) for c in cand_list], len(cand_list), len(text)
    text = PROMPT_I2T.format(captions=str([str(c) for c in cand_list]))
    return [{"type": "text", "text": text}, image_item(int(q))], 1, len(text)


def usage_of(r):
    u = r.usage.model_dump() if getattr(r, "usage", None) else {}
    details = u.get("completion_tokens_details") or {}
    rec = {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
           "reasoning_tokens": details.get("reasoning_tokens"), "cost_usd": u.get("cost"), "cost_source": "openrouter"}
    if rec["cost_usd"] is None and rec["prompt_tokens"] is not None:
        rec["cost_usd"] = (rec["prompt_tokens"] * PRICE_IN + (rec["completion_tokens"] or 0) * PRICE_OUT) / 1e6
        rec["cost_source"] = "estimated from list price"
    return rec


client = None
if not DRY:
    from openai import OpenAI
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set (use --dry-run to estimate without calls)")
    headers = {"HTTP-Referer": "https://github.com/svo-retrieval", "X-Title": "svo-retrieval pool judge"} if IS_OPENROUTER else None
    client = OpenAI(api_key=key, base_url=BASE_URL, default_headers=headers)

spent_before = 0.0
if usage_path.exists():
    for line in open(usage_path, encoding="utf-8"):
        try:
            spent_before += float(json.loads(line).get("cost_usd") or 0)
        except ValueError:
            pass
spent, calls, n_images, n_chars, skipped_done = 0.0, 0, 0, 0, 0
t0 = time.time()
stop = False
for d in directions:
    queries = list(pool[d].items())
    if PILOT:
        queries = queries[:PILOT]
    for q, cands in queries:
        if stop:
            break
        lab = labels[d].setdefault(q, {})
        cand_list = list(cands)
        if SKIP_GOLD:
            for c in cand_list:
                if is_gold(d, q, c) and not lab.get(c):
                    lab[c] = "correct"
            cand_list = [c for c in cand_list if not is_gold(d, q, c)]
        if all(lab.get(c) for c in cand_list):
            skipped_done += 1
            continue
        # only the candidates without a label are sent (each is judged on its own), so an earlier pilot or an
        # enlarged pool never repeats work
        cand_list = [c for c in cand_list if not lab.get(c)]
        content, ni, nc = build(d, q, cand_list)
        calls += 1
        n_images += ni
        n_chars += nc
        if DRY:
            continue
        answer, rec = "", None
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": content}], **REQUEST_EXTRA)
                answer = r.choices[0].message.content or ""
                rec = usage_of(r)
                break
            except Exception as e:
                status = getattr(e, "status_code", None)
                msg = f"{type(e).__name__}: {str(e)[:160]}"
                if status is not None and 400 <= status < 500 and status != 429:
                    # authentication, guardrail, bad request: retrying will not help, stop the run
                    raise SystemExit(f"{d} {str(q)[:40]!r}: {msg}\nStopping: fix the key, model or account settings and rerun.")
                wait = 15 * (attempt + 1)                # rate limits and transient errors
                print(f"  {d} {str(q)[:40]!r}: attempt {attempt + 1} failed ({msg}); retry in {wait}s", flush=True)
                time.sleep(wait)
        found = {k: v.lower() for k, v in ANSWER.findall(answer)}
        for i, c in enumerate(cand_list):
            lab[c] = found.get(str(i + 1), "")
        n_labelled = sum(1 for c in cand_list if lab[c])
        json.dump(labels, open(out_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        with open(raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"direction": d, "query": q, "n_candidates": len(cand_list), "answer": answer}, ensure_ascii=False) + "\n")
        if rec is not None:
            rec.update({"direction": d, "query": q, "n_candidates": len(cand_list), "n_labelled": n_labelled,
                        "model": MODEL, "effort": EFFORT, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
            with open(usage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            spent += float(rec.get("cost_usd") or 0)
        print(f"{d} call {calls}: {n_labelled}/{len(cand_list)} labelled, "
              f"{(rec or {}).get('prompt_tokens')} in / {(rec or {}).get('completion_tokens')} out tokens, "
              f"cost this run ${spent:.3f}", flush=True)
        if spent > BUDGET:
            print(f"budget of ${BUDGET:.2f} exceeded, stopping; rerun to continue", flush=True)
            stop = True

if DRY:
    est_text = n_chars / 4 + 40 * calls                              # about 4 characters per token
    img_low, img_high = n_images * 375, n_images * 765                 # o3: 75 + 150 per 512px tile; 2 or 4 tiles
    out_low, out_high = calls * 300, calls * 2500                     # answer plus reasoning at medium effort
    low = ((est_text + img_low) * PRICE_IN + out_low * PRICE_OUT) / 1e6
    high = ((est_text + img_high) * PRICE_IN + out_high * PRICE_OUT) / 1e6
    print(f"dry run: {calls} calls ({skipped_done} queries already complete), {n_images} images, "
          f"about {int(est_text):,} text tokens; estimated cost ${low:.2f} to ${high:.2f} at list price "
          f"(${PRICE_IN}/M in, ${PRICE_OUT}/M out); model {MODEL}, effort {EFFORT}")
else:
    total_pairs = sum(len(v) for d in directions for v in labels[d].values())
    labelled = sum(1 for d in directions for v in labels[d].values() for x in v.values() if x)
    print(f"done: {calls} calls in {(time.time() - t0) / 60:.1f} min, {skipped_done} queries were already complete; "
          f"labels {labelled}/{total_pairs}; cost this run ${spent:.2f}, all runs ${spent_before + spent:.2f}; saved {out_path.name}")
