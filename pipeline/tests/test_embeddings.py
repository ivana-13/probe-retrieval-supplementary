import numpy as np
import pytest
from svo_eval.embeddings import Scorer, load_embeddings
from svo_eval.lists import load_lists
from svo_eval.paths import MODELS


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("direction", ["t2i", "i2t"])
def test_embeddings_reproduce_stored_top1(model, direction):
    """Every system's scorer must reproduce the annotated lists (the pool was mined under these rankings)."""
    s = Scorer(model)
    lists = load_lists(model, direction)
    qs = list(lists)[:500]
    agree = 0
    for q in qs:
        if direction == "i2t":
            q = int(q)
        try:
            agree += s.topk(q, direction, 1)[0] == lists[q if direction == "t2i" else str(q)][0]
        except KeyError:
            agree += s.topk(q, direction, 1)[0] == lists[q][0]
    assert agree >= 0.98 * len(qs), (model, direction, agree)


def test_scorer_shapes():
    e = load_embeddings("BLIP2")
    assert e.cap.shape[0] == 10978 and e.img.shape[0] == 13285
    np.testing.assert_allclose(np.linalg.norm(e.cap[:5], axis=1), 1.0, atol=1e-4)
    raw = load_embeddings("CLIP")
    assert abs(float(np.linalg.norm(raw.cap[0])) - 1.0) > 0.1     # CLIP is scored on unnormalised embeddings
