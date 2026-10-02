import numpy as np
from svo_eval.judges import binary_agreement


def test_binary_agreement_perfect():
    y = np.array([1, 0, 1, 1, 0], bool)
    r = binary_agreement(y, y)
    assert r["accuracy"] == 1.0 and r["f1_correct"] == 1.0 and abs(r["kappa"] - 1.0) < 1e-9
