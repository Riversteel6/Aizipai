from tools.evaluate_anchored_structural_discard_proxies import PROXY_TYPES


def test_reproduction_has_one_proxy_for_every_frozen_teacher() -> None:
    assert set(PROXY_TYPES) == {
        "aggressive_meld",
        "defensive_search",
        "independent_balanced",
        "independent_denial",
        "independent_pressure",
        "information_set_search",
        "red_black_search",
    }
