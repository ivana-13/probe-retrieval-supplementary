import numpy as np
from .paths import RESULTS

E5 = "intfloat/e5-large-v2"


def _cos(W, a, b):
    if a == b:
        return 1.0
    va, vb = W.get(a), W.get(b)
    if va is None or vb is None:
        return 0.0
    return float(np.dot(va, vb))


def strict_match(q_trip, c_trips, tau, W):
    """Verb identical; subject-subject and object-object similarity at least tau for some candidate triplet."""
    qs, qv, qo = q_trip
    for cs, cv, co in c_trips:
        if cv == qv and _cos(W, qs, cs) >= tau and _cos(W, qo, co) >= tau:
            return True
    return False


def lenient_match(q_trip, c_trips, tau, W):
    """Previous notebook rule: each query role may match either the candidate subject or object."""
    qs, qv, qo = q_trip
    for cs, cv, co in c_trips:
        if cv == qv and max(_cos(W, qs, cs), _cos(W, qs, co)) >= tau and max(_cos(W, qo, cs), _cos(W, qo, co)) >= tau:
            return True
    return False


def semantic_hit(coll, d, q, cand, tau, W, rule=strict_match):
    if d == "t2i":
        return rule(coll.cap2trip[q], coll.img2trips.get(int(cand), set()), tau, W)
    return any(rule(qt, {coll.cap2trip[cand]}, tau, W) for qt in coll.img2trips.get(int(q), set()))


def word_embeddings(words):
    """E5-large-v2 mean-pooled, L2-normalised embeddings of bare words (as in the original notebooks), cached."""
    path = RESULTS / "e5_words.npz"
    cache = dict(np.load(path, allow_pickle=True)) if path.exists() else {}
    todo = [w for w in words if w not in cache]
    if todo:
        import torch
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(E5)
        mdl = AutoModel.from_pretrained(E5).eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        mdl.to(dev)
        with torch.no_grad():
            for i in range(0, len(todo), 64):
                batch = todo[i:i + 64]
                t = tok(batch, return_tensors="pt", padding=True).to(dev)
                h = mdl(**t).last_hidden_state
                m = t["attention_mask"].unsqueeze(-1).float()
                emb = (h * m).sum(1) / m.sum(1)
                emb = torch.nn.functional.normalize(emb, dim=1).cpu().numpy()
                for w, e in zip(batch, emb):
                    cache[w] = e
        np.savez(path, **cache)
    return {w: cache[w] for w in words}
