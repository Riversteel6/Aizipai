from tools.fit_response_proxy_margins import fit_binary_margin


def test_fit_binary_margin_separates_claims_without_seed_search() -> None:
    rows = [
        {"score": -2.0, "target_positive": False, "positive_key_correct": False},
        {"score": -1.0, "target_positive": False, "positive_key_correct": False},
        {"score": 2.0, "target_positive": True, "positive_key_correct": True},
        {"score": 3.0, "target_positive": True, "positive_key_correct": True},
    ]

    fitted = fit_binary_margin(rows)

    assert fitted["balanced_accuracy"] == 1.0
    assert -1.0 < fitted["threshold"] < 2.0
