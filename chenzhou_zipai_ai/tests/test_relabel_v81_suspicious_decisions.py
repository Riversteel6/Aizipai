from __future__ import annotations

from tools import relabel_v81_suspicious_decisions as relabel


def test_response_relabel_uses_root_search_key_not_selected_label(monkeypatch) -> None:
    monkeypatch.setattr(
        relabel,
        "replay_case",
        lambda *_args, **_kwargs: {
            "current_rc3": {
                "product_path": {
                    "selected_action": "CHI",
                    "selected_label": "八",
                    "policy_version": "test",
                    "strategy_context": {"elapsed_ms": 12.0},
                    "event": {
                        "search_selected_key": "CHI:runtime_001",
                        "reason": "root_response_completed",
                    },
                }
            }
        },
    )
    row = {
        "_source_index": 7,
        "category": "response_internal_better_alternative_ignored",
        "seed": 1,
        "opponent": "x",
        "event_index": 0,
        "production_key": "PASS",
        "actual_selected_key": "PASS",
        "public_view": {"stock_count": 40},
    }

    result = relabel.relabel_row(row, run_product=True)

    assert result["source_index"] == 7
    assert result["after_selected"] == "CHI:runtime_001"
    assert result["product_final_authorization"] == "root_response_completed"
    assert result["semantic_action_changed"] is True


def test_hu_without_new_search_event_uses_product_action(monkeypatch) -> None:
    monkeypatch.setattr(relabel, "evaluate_hu_oracle", lambda *_args, **_kwargs: type(
        "Oracle", (), {"can_hu": True, "reasons": ("exact",)}
    )())
    monkeypatch.setattr(relabel, "rules_for_room", lambda **_kwargs: {})
    monkeypatch.setattr(
        relabel,
        "replay_case",
        lambda *_args, **_kwargs: {
            "current_rc3": {
                "product_path": {
                    "selected_action": "HU",
                    "selected_label": "一",
                    "policy_version": "test",
                    "strategy_context": {"route": "hard_priority", "elapsed_ms": 0.0},
                    "event": None,
                }
            }
        },
    )
    row = {
        "_source_index": 8,
        "category": "legal_hu_passed",
        "seed": 1,
        "opponent": "x",
        "event_index": 0,
        "pending_card": "一",
        "production_key": "HU",
        "actual_selected_key": "PASS",
        "public_view": {"seat": 0, "hand": [], "all_melds": [[], []]},
    }

    result = relabel.relabel_row(row, run_product=True)

    assert result["after_selected"] == "HU"
    assert result["oracle_status"] == "EXACT_ACTION"
    assert result["acceptance"] == "PASS"
    assert result["semantic_action_changed"] is True
    assert result["delta_pwin"] is None


def test_chi_runtime_key_is_compared_by_plan_semantics(monkeypatch) -> None:
    monkeypatch.setattr(
        relabel,
        "replay_case",
        lambda *_args, **_kwargs: {
            "current_rc3": {
                "product_path": {
                    "selected_action": "CHI",
                    "policy_version": "test",
                    "strategy_context": {"elapsed_ms": 1.0},
                    "event": {
                        "search_selected_key": "CHI:runtime_009",
                        "candidates": [
                            {
                                "key": "CHI:runtime_009",
                                "action_type": "CHI",
                                "consumed_from_hand": ["二", "一"],
                                "meld_groups": [["三", "二", "一"]],
                            }
                        ],
                    },
                }
            }
        },
    )
    row = {
        "category": "response_internal_better_alternative_ignored",
        "production_key": "CHI:old_hash",
        "actual_selected_key": "CHI:old_hash",
        "actual_candidate_stats": {
            "consumed_from_hand": ["一", "二"],
            "meld_groups": [["一", "二", "三"]],
        },
        "public_view": {"stock_count": 40},
    }

    result = relabel.relabel_row(row, run_product=True)

    assert result["semantic_action_changed"] is False
