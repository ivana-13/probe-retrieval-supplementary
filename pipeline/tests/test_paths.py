from svo_eval import paths


def test_registry_files_exist():
    for m in paths.MODELS:
        for d in paths.DIRECTIONS:
            assert paths.list_file(m, d).exists(), (m, d)
            assert paths.pool_file(m, d).exists(), (m, d)
    for m in paths.DUAL_ENCODERS:
        for d in paths.DIRECTIONS:
            assert paths.pool_file(m, d, judged_by_llm=True).exists()


def test_pool_key_naming_bug_is_handled():
    assert paths.pool_key("CLIP", "i2t") == "FLAVA retrieved captions"
    assert paths.pool_key("CLIP", "t2i") == "CLIP retrieved images"
    assert paths.pool_key("Qwen25", "i2t") == "Qwen2.5 retrieved captions"
