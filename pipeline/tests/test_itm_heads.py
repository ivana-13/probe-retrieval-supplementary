import pytest
import torch
from svo_eval.collection import load_collection
from svo_eval.itm_heads import MatchingHead

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU and the cached checkpoints")


def has_lavis():
    try:
        import lavis  # noqa: F401
        return True
    except ImportError:
        return False


# the BLIP-2 head is the LAVIS implementation, so its case runs only in lavis-env (scripts/README_lavis.md):
#   set PYTHONPATH=<pipeline> && lavis-env\Scripts\python.exe -m pytest tests/test_itm_heads.py -k BLIP2
needs_lavis = pytest.mark.skipif(not has_lavis(), reason="BLIP-2 head needs the LAVIS environment")


@needs_gpu
@pytest.mark.parametrize("model", [pytest.param("BLIP2", marks=needs_lavis), "FLAVA"])
def test_batched_probabilities_match_single_pair(model):
    df = load_collection().df.iloc[:24]
    pairs = [(int(r.pos_image_id), r.sentence) for r in df.itertuples()] + \
            [(int(r.neg_image_id), r.sentence) for r in df.itertuples()]
    head = MatchingHead(model, batch_size=16)
    batched = head.match_probs(pairs)
    head.batch_size = 1
    single = head.match_probs(pairs)
    assert len(batched) == len(pairs)
    for a, b in zip(batched, single):
        assert (a is None) == (b is None)
        if a is not None:
            assert abs(a - b) < 0.02, (model, a, b)
    probs = [p for p in batched if p is not None]
    assert any(p > 0.5 for p in probs) and any(p < 0.5 for p in probs)
