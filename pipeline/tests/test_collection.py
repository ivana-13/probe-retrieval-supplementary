from svo_eval.collection import load_collection


def test_counts_match_spec():
    c = load_collection()
    assert len(c.df) == 33685
    assert len(c.captions) == 10978
    assert len(c.images) == 13285
    assert len(c.evaluable_queries("i2t")) == 11455


def test_gold_and_negatives_consistent():
    c = load_collection()
    cap = "Girl is standing in the grass."
    assert 0 in c.cap2imgs[cap]
    assert c.cap2trip[cap] == ("girl", "stand", "grass")
    assert 1 in c.cap2negimgs[cap]
    assert c.cap2negtype[cap][1] == "subject"
    # every benchmark-negative caption of image 0 is a gold caption of some negative image of a row where 0 is positive
    neg_images_of_0 = set(c.df[c.df.pos_image_id == 0].neg_image_id)
    for negcap in c.img2negcaps.get(0, set()):
        assert c.cap2imgs[negcap] & neg_images_of_0
        assert 0 not in c.cap2imgs[negcap]
