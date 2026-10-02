"""Qwen3-VL-Embedding-2B embeddings of the COCO collection, the same way as the SVO-Probes run on the cluster
(data/qwen3_embed/retrieval_qwen3.py): images resized to 1024x1024, texts and images without instruction (the model's
default instruction), fp16, unit-norm (the scorer normalises again). Inputs come from results/coco_inputs.json
(scripts/24_coco_export_inputs.py) so that every encoder embeds the same strings in the same order; outputs follow the
layout of scripts/20_coco_embed.py under results/coco_emb/Qwen3_*. Each phase checkpoints to a .partial.npy file and
resumes from it after an interruption."""
import json
import sys
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from svo_eval.paths import RESULTS, DATA
from svo_eval.coco import load_coco

sys.path.insert(0, str(DATA / "qwen3_vl_embedding"))
from src.models.qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: E402

MODEL = "Qwen3"
OUT = RESULTS / "coco_emb"
OUT.mkdir(parents=True, exist_ok=True)
IMAGE_SIZE = 1024
BS_TEXT, BS_IMG = 16, 4

inp = json.load(open(RESULTS / "coco_inputs.json", encoding="utf-8"))
coll = load_coco()
embedder = Qwen3VLEmbedder(str(DATA / "models" / "qwen3-vl-embedding-2b"), dtype=torch.float16)


def enc(batch):
    with torch.no_grad():
        e = embedder.process(batch)
    return torch.as_tensor(e).float().cpu().numpy()


def run_phase(name, items, make_input, bs, chunk):
    """Encode items in order; the accumulated array is saved every `chunk` rows and the phase resumes from it."""
    part = OUT / f"{MODEL}_{name}.partial.npy"
    done = np.load(part) if part.exists() else np.zeros((0, 0), np.float32)
    start = len(done)
    rows = [done] if start else []
    pending, n_pending = [], 0
    todo = list(range(start, len(items), bs))
    for i in tqdm(todo, desc=f"{MODEL} {name}"):
        batch = items[i:i + bs]
        try:
            out = enc([make_input(x) for x in batch])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            out = np.concatenate([enc([make_input(x)]) for x in batch])
        pending.append(out)
        n_pending += len(out)
        if n_pending >= chunk:
            rows.append(np.concatenate(pending))
            pending, n_pending = [], 0
            np.save(part, np.concatenate(rows))
    if pending:
        rows.append(np.concatenate(pending))
    arr = np.concatenate(rows) if rows else np.zeros((0, 2048), np.float32)
    np.save(part, arr)
    return arr


def image_input(image_id):
    im = Image.open(coll.image_path(image_id)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    return {"text": "", "image": [im]}


cap = run_phase("caption_embeddings", inp["captions"], lambda t: {"text": t, "image": []}, BS_TEXT, 2000)
neg = run_phase("negative_embeddings", inp["probe_texts"], lambda t: {"text": t, "image": []}, BS_TEXT, 2000)
img = run_phase("image_embeddings", inp["image_ids"], image_input, BS_IMG, 200)
assert len(cap) == len(inp["captions"]) and len(neg) == len(inp["probe_texts"]) and len(img) == len(inp["image_ids"])
np.save(OUT / f"{MODEL}_caption_embeddings.npy", cap.astype(np.float32))
np.save(OUT / f"{MODEL}_negative_embeddings.npy", neg.astype(np.float32))
np.save(OUT / f"{MODEL}_image_embeddings.npy", img.astype(np.float32))
np.save(OUT / f"{MODEL}_captions.npy", np.array(inp["captions"], dtype=str))
np.save(OUT / f"{MODEL}_negatives.npy", np.array(inp["probe_texts"], dtype=str))
np.save(OUT / f"{MODEL}_image_id.npy", np.array(inp["image_ids"], dtype=int))
for name in ["caption_embeddings", "negative_embeddings", "image_embeddings"]:
    (OUT / f"{MODEL}_{name}.partial.npy").unlink(missing_ok=True)
print(MODEL, cap.shape, neg.shape, img.shape)
