from aic2026.core.retrieval import rrf_fuse, borda_fuse, weighted_sum_fuse


def test_rrf_is_reciprocal_rank_not_mean():
    lists = {"visual": ["a", "b"], "ocr": ["b"]}
    fused = rrf_fuse(lists, k=60)
    # a: 1/(60+1); b: 1/(60+1) + 1/(60+2)
    assert abs(fused["a"] - 1 / 61) < 1e-12
    assert abs(fused["b"] - (1 / 61 + 1 / 62)) < 1e-12
    # b appears in two channels -> must beat a (a mean would have tied ~equal)
    assert fused["b"] > fused["a"]


def test_rrf_rewards_agreement_across_channels():
    # 'x' is rank-1 in one channel; 'y' is rank-2 in all three -> y should win.
    lists = {"visual": ["x", "y"], "ocr": ["z", "y"], "asr": ["w", "y"]}
    fused = rrf_fuse(lists, k=60)
    top = max(fused, key=fused.get)
    assert top == "y"


def test_borda_orders_by_rank_points():
    lists = {"a": ["p", "q", "r"]}
    fused = borda_fuse(lists)
    assert fused["p"] == 3 and fused["q"] == 2 and fused["r"] == 1


def test_weighted_sum_normalises_per_channel():
    scored = {"visual": [("a", 10.0), ("b", 0.0)], "ocr": [("a", 1.0), ("b", 0.0)]}
    fused = weighted_sum_fuse(scored, {"visual": 1.0, "ocr": 1.0})
    # both channels put a at top after min-max -> a == 2.0, b == 0.0
    assert abs(fused["a"] - 2.0) < 1e-9
    assert abs(fused["b"] - 0.0) < 1e-9
