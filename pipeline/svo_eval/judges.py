import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score, confusion_matrix


def binary_agreement(y_true, y_pred):
    t = np.asarray(y_true, bool)
    p = np.asarray(y_pred, bool)
    return {"accuracy": float((t == p).mean()),
            "f1_correct": float(f1_score(t, p)),
            "precision_correct": float(precision_score(t, p, zero_division=0)),
            "recall_correct": float(recall_score(t, p, zero_division=0)),
            "precision_incorrect": float(precision_score(~t, ~p, zero_division=0)),
            "recall_incorrect": float(recall_score(~t, ~p, zero_division=0)),
            "kappa": float(cohen_kappa_score(t, p)), "n": int(len(t))}


def multiclass_agreement(t, p, labels):
    return {"weighted_f1": float(f1_score(t, p, labels=labels, average="weighted", zero_division=0)),
            "macro_f1": float(f1_score(t, p, labels=labels, average="macro", zero_division=0)),
            "kappa": float(cohen_kappa_score(t, p, labels=labels)),
            "confusion": confusion_matrix(t, p, labels=labels).tolist(), "labels": labels, "n": int(len(t))}


def bootstrap_stat(fn, t, p, n_boot=1000, seed=0):
    t = np.asarray(t)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(t), len(t))
        vals.append(fn(t[idx], p[idx]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))
