"""Cosine scorer over the COCO embeddings written by scripts/20_coco_embed.py (collection captions, extra probe texts, images)."""
from functools import lru_cache
import numpy as np
from .paths import RESULTS

EMB = RESULTS / "coco_emb"
COCO_MODELS = ["CLIP", "BLIP2", "FLAVA", "SigLIP2", "Qwen25", "Qwen3"]   # same order as MODELS in paths.py


def _norm(a):
    a = np.asarray(a, dtype=np.float32)
    return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)


def available_models():
    return [m for m in COCO_MODELS if (EMB / f"{m}_image_embeddings.npy").exists()]


class CocoScorer:
    def __init__(self, model):
        self.model = model
        cap = _norm(np.load(EMB / f"{model}_caption_embeddings.npy"))
        extra = np.load(EMB / f"{model}_negative_embeddings.npy")
        extra = _norm(extra) if len(extra) else np.zeros((0, cap.shape[1]), np.float32)
        self.img = _norm(np.load(EMB / f"{model}_image_embeddings.npy"))
        caps = np.load(EMB / f"{model}_captions.npy", allow_pickle=True).astype(str)
        extras = np.load(EMB / f"{model}_negatives.npy", allow_pickle=True).astype(str)
        self.image_ids = np.load(EMB / f"{model}_image_id.npy", allow_pickle=True).astype(int)
        self.cap = cap                                   # collection captions only (retrieval candidates)
        self.captions = caps
        self.cap_index = {c: i for i, c in enumerate(caps)}
        self.text_index = dict(self.cap_index)           # any text: collection caption or probe text
        self.text = np.concatenate([cap, extra]) if len(extra) else cap
        for k, t in enumerate(extras):
            self.text_index.setdefault(str(t), len(cap) + k)
        self.img_index = {int(i): k for k, i in enumerate(self.image_ids)}

    def text_vec(self, text):
        return self.text[self.text_index[text]]

    def all_scores(self, direction, query):
        if direction == "t2i":
            return self.img @ self.text_vec(query)
        return self.cap @ self.img[self.img_index[int(query)]]

    def score(self, direction, query, cands):
        if direction == "t2i":
            return self.img[[self.img_index[int(c)] for c in cands]] @ self.text_vec(query)
        return np.stack([self.text_vec(c) for c in cands]) @ self.img[self.img_index[int(query)]]

    def topk(self, query, direction, k):
        s = self.all_scores(direction, query)
        idx = np.argpartition(-s, k)[:k]
        idx = idx[np.argsort(-s[idx])]
        return [int(self.image_ids[i]) for i in idx] if direction == "t2i" else [str(self.captions[i]) for i in idx]

    def candidates(self, direction):
        return self.image_ids if direction == "t2i" else self.captions
