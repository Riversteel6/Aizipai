import hashlib
import json

import pytest

from tools.evaluate_hybrid_response_proxies import HYBRID_RESPONSE_PROXY_TYPES
from tools.evaluate_proxy_public_belief_validator import verify_capability_reports


def test_full_action_proxy_mapping_is_complete() -> None:
    assert set(HYBRID_RESPONSE_PROXY_TYPES) == {
        "aggressive_meld",
        "defensive_search",
        "independent_balanced",
        "independent_denial",
        "independent_pressure",
        "information_set_search",
        "red_black_search",
    }


def test_optional_implementation_reports_are_hash_and_gate_checked(tmp_path) -> None:
    inputs = {}
    for prefix in (
        "belief",
        "discard_proxy",
        "response_proxy",
        "implementation_equivalence",
        "heavy_state_gate",
    ):
        path = tmp_path / f"{prefix}.json"
        raw = json.dumps({"focused_gate_pass": True}).encode("utf-8")
        path.write_bytes(raw)
        inputs[f"{prefix}_report"] = str(path)
        inputs[f"{prefix}_report_sha256"] = hashlib.sha256(raw).hexdigest()

    verify_capability_reports(inputs)

    inputs["heavy_state_gate_report_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="heavy_state_gate_hash_mismatch"):
        verify_capability_reports(inputs)
