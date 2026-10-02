import numpy as np
from svo_eval import metrics as M


def test_first_gold_rank():
    assert M.first_gold_rank([5, 3, 9], {9}) == 3
    assert M.first_gold_rank([5, 3, 9], {7}) is None


def test_success_and_rr():
    ranks = [1, 3, None, 11]
    assert M.success_vector(ranks, 1).tolist() == [True, False, False, False]
    assert M.success_vector(ranks, 10).tolist() == [True, True, False, False]
    np.testing.assert_allclose(M.rr_vector(ranks, 10), [1.0, 1 / 3, 0.0, 0.0])


def test_bootstrap_ci_contains_mean():
    v = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 1], dtype=float)
    mean, lo, hi = M.bootstrap_ci(v, n_boot=500, seed=1)
    assert lo <= mean <= hi and abs(mean - 0.7) < 1e-9


def test_mcnemar_symmetric_and_bounded():
    a = np.array([1, 1, 1, 0, 0, 0, 1, 1], bool)
    b = np.array([1, 0, 0, 0, 0, 1, 1, 1], bool)
    p = M.mcnemar_p(a, b)
    assert 0 <= p <= 1 and abs(p - M.mcnemar_p(b, a)) < 1e-12


def test_fleiss_kappa_perfect_agreement():
    counts = np.array([[3, 0], [0, 3], [3, 0]])
    assert abs(M.fleiss_kappa(counts) - 1.0) < 1e-9
