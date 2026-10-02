from svo_eval import pools
from svo_eval.paths import MODELS


def test_same_queries_everywhere():
    for d in ["t2i", "i2t"]:
        base = set(pools.load_pool("CLIP", d))
        assert len(base) == 100
        for m in MODELS:
            assert set(pools.load_pool(m, d)) == base, (m, d)


def test_fixed_pool_sizes_and_kappa():
    fp = pools.build_fixed_pool()
    n_i2t = sum(len(v) for v in fp["i2t"].values())
    n_t2i = sum(len(v) for v in fp["t2i"].values())
    # 3,719 judged T->I pairs minus 5 Qwen3 candidates outside the 13,285-image collection
    assert n_i2t == 3714 and n_t2i == 3714 and pools.OUT_OF_COLLECTION == {"t2i": 5, "i2t": 0}
    assert abs(pools.fleiss_kappa_i2t() - 0.642) < 0.002


def test_labels_present():
    p = pools.load_pool("CLIP", "i2t")
    j = next(iter(p.values()))[0]
    assert j["o3"] in ("correct", "subject incorrect", "verb incorrect", "object incorrect")
    assert len(j["votes"]) == 3
