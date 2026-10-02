import pandas as pd
from svo_eval.qwen_csv import load_qwen25_pairwise, load_qwen25_itm, load_qwen3_itm


def _vm_csv(tmp_path):
    """A tiny unfiltered file whose rows 0-3 map onto three collection triplets (row 2 is outside the collection)."""
    p = tmp_path / "vm.csv"
    pd.DataFrame({"sentence": ["A", "B", "Z", "C"], "pos_triplet": ["a,b,c"] * 4, "neg_triplet": ["a,b,d"] * 4,
                  "pos_url": [""] * 4, "neg_url": [""] * 4, "pos_image_id": [1, 2, 99, 3], "neg_image_id": [11, 12, 98, 13],
                  "subj_neg": [True, False, False, True], "verb_neg": [False, True, False, False],
                  "obj_neg": [False, False, True, False]}).to_csv(p, index=False)
    return p


def test_pairwise_alignment_and_unanswered(tmp_path, monkeypatch):
    import svo_eval.qwen_csv as Q
    local = tmp_path / "local.csv"
    pd.DataFrame({"sentence": ["A", "B", "C"], "pos_triplet": ["a,b,c"] * 3, "neg_triplet": ["a,b,d"] * 3, "pos_url": [""] * 3,
                  "neg_url": [""] * 3, "pos_image_id": [1, 2, 3], "neg_image_id": [11, 12, 13], "subj_neg": [True, False, True],
                  "verb_neg": [False, True, False], "obj_neg": [False, False, False]}).to_csv(local, index=False)
    monkeypatch.setattr(Q, "COLLECTION_CSV", local)
    p = tmp_path / "pw.csv"
    pd.DataFrame({"index": [0, 1, 2, 3], "correct": [True, None, True, False],
                  "subj_neg": [True, False, False, True], "verb_neg": [False, True, False, False],
                  "obj_neg": [False, False, True, False]}).to_csv(p, index=False)
    r = load_qwen25_pairwise(p, vm_csv=_vm_csv(tmp_path))
    assert r["n_rows"] == 4 and r["n_in_collection"] == 3 and r["n_unanswered"] == 1     # row 2 dropped, row 1 unanswered
    assert r["correct"].tolist() == [True, False]


def test_itm_alignment(tmp_path, monkeypatch):
    import svo_eval.qwen_csv as Q
    local = tmp_path / "local.csv"
    pd.DataFrame({"sentence": ["A", "B", "C"], "pos_triplet": ["a,b,c"] * 3, "neg_triplet": ["a,b,d"] * 3, "pos_url": [""] * 3,
                  "neg_url": [""] * 3, "pos_image_id": [1, 2, 3], "neg_image_id": [11, 12, 13], "subj_neg": [True, False, True],
                  "verb_neg": [False, True, False], "obj_neg": [False, False, False]}).to_csv(local, index=False)
    monkeypatch.setattr(Q, "COLLECTION_CSV", local)
    p = tmp_path / "itm.csv"
    pd.DataFrame({"index": [0, 1, 2, 3], "pos_pred": ["yes", None, "yes", "no"], "neg_pred": ["no", "yes", "no", "no"],
                  "pos_correct": [True, False, True, False], "neg_correct": [True, False, True, True],
                  "subj_neg": [True, False, False, True], "verb_neg": [False, True, False, False],
                  "obj_neg": [False, False, True, False]}).to_csv(p, index=False)
    r = load_qwen25_itm(p, vm_csv=_vm_csv(tmp_path))
    assert r["n_in_collection"] == 3 and r["n_unanswered"] == 1 and len(r["correct"]) == 5
    assert r["correct"].tolist() == [True, False, True, False, True]      # positives of rows 0,3 then negatives of rows 0,1,3
    assert r["is_positive"].tolist() == [True, True, False, False, False]


def test_real_files_align_to_the_collection():
    pw = load_qwen25_pairwise()
    assert pw["n_rows"] == 36841 and pw["n_in_collection"] == 33685 and pw["n_unanswered"] == 8
    itm = load_qwen25_itm()
    assert itm["n_in_collection"] == 33685 and itm["n_unanswered"] == 0 and len(itm["correct"]) == 2 * 33685
    q3 = load_qwen3_itm()
    assert q3["n_rows"] == 33685 and q3["n_unanswered"] == 0 and len(q3["correct"]) == 2 * 33685
