from tools.evaluate_hybrid_response_proxies import (
    HYBRID_RESPONSE_PROXY_TYPES,
    percentile_95,
)


def test_hybrid_response_mapping_covers_every_teacher() -> None:
    assert len(HYBRID_RESPONSE_PROXY_TYPES) == 7
    assert "independent_pressure" in HYBRID_RESPONSE_PROXY_TYPES


def test_percentile_95_uses_upper_observation() -> None:
    assert percentile_95([1.0] * 18 + [5.0] * 2) == 5.0
