import numpy as np
from svo_eval.similarity import strict_match, lenient_match


def test_strict_match_uses_roles():
    W = {"girl": np.array([1.0, 0]), "woman": np.array([0.95, 0.31]), "ball": np.array([0, 1.0])}
    assert strict_match(("girl", "sit", "ball"), {("woman", "sit", "ball")}, 0.9, W)
    assert not strict_match(("girl", "sit", "ball"), {("woman", "run", "ball")}, 0.9, W)
    assert not strict_match(("girl", "sit", "ball"), {("ball", "sit", "girl")}, 0.9, W)
    assert lenient_match(("girl", "sit", "ball"), {("ball", "sit", "girl")}, 0.9, W)
