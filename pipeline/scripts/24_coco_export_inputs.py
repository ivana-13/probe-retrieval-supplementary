"""Export the exact COCO/SugarCrepe inputs of the second collection so that every encoder, on this laptop or on a VM,
embeds the same strings in the same order (the layout that scripts/20_coco_embed.py writes).

Writes results/coco_inputs.json:
  captions   : the 24,789 unique cleaned val2017 captions (retrieval candidates and text queries), in collection order
  probe_texts: SugarCrepe captions and negative captions that are not collection captions (sorted), as in 20_coco_embed
  image_ids  : the 5,000 val2017 image ids (ascending)
  sugarcrepe : the probe pairs [{image_id, caption, negative_caption, subset}], in svo_eval.coco order
With --pool it also writes results/coco_pool_pairs.json from results/coco_pool_candidates.json:
  {direction: [[query, candidate], ...]} with query = caption (t2i) or image id (i2t), candidate = image id or caption.
Usage: python scripts/24_coco_export_inputs.py [--pool]"""
import json
import sys
from svo_eval.paths import RESULTS
from svo_eval.coco import load_coco

coll = load_coco()
captions = list(coll.captions)
probe_texts = sorted(({p["negative_caption"] for p in coll.probe} | {p["caption"] for p in coll.probe}) - set(captions))
out = {"captions": captions, "probe_texts": probe_texts, "image_ids": [int(i) for i in coll.images], "sugarcrepe": coll.probe}
json.dump(out, open(RESULTS / "coco_inputs.json", "w", encoding="utf-8"), ensure_ascii=False)
print("coco_inputs.json:", len(captions), "captions,", len(probe_texts), "probe texts,", len(out["image_ids"]), "images,",
      len(coll.probe), "SugarCrepe pairs")
if "--pool" in sys.argv:
    pool = json.load(open(RESULTS / "coco_pool_candidates.json", encoding="utf-8"))
    pairs = {d: [[q if d == "t2i" else int(q), int(c) if d == "t2i" else c] for q, cands in pool[d].items() for c in cands]
             for d in ("t2i", "i2t")}
    json.dump(pairs, open(RESULTS / "coco_pool_pairs.json", "w", encoding="utf-8"), ensure_ascii=False)
    print("coco_pool_pairs.json:", {d: len(v) for d, v in pairs.items()})
