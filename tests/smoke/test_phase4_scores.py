from billwise.categorization.inference import top_k_scores


def test_phase4_top_k_scores():
    score_map = {
        "Dairy": 0.62,
        "Beverages": 0.11,
        "Produce": 0.05,
        "Fruits": 0.21,
        "Frozen / Processed": 0.01,
    }

    result = top_k_scores(score_map, 3)
    keys = list(result.keys())

    assert len(result) == 3
    assert keys[0] == "Dairy"
    assert keys[1] == "Fruits"
    assert keys[2] == "Beverages"