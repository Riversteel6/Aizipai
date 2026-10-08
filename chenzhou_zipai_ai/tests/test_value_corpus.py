"""Action-value corpus coverage and leakage tests."""

from copy import deepcopy

from ai.value_corpus import build_value_corpus


def _audit(seed: int, sequence: int) -> dict:
    return {
        "complete": True,
        "game": {
            "seed": seed,
            "players": 2,
            "wildcard_enabled": False,
            "candidate_seat": sequence % 2,
            "dealer": 0,
        },
        "public_state": {
            "seat": sequence % 2,
            "hand": ["一", "二", "三"],
            "own_melds": [],
            "all_melds": [[], []],
            "discards": [[], []],
            "remaining_counts": [["一", 3], ["二", 3]],
            "stock_count": 39,
            "hand_sizes": [3, 3],
        },
        "state_before_hash": f"state-{sequence}",
        "trace_sequence": sequence,
        "phase": "discard",
        "turn": sequence,
        "seat": sequence % 2,
        "selected_key": "DISCARD:一",
        "best_key": "DISCARD:二",
        "completed_paired_worlds": 8,
        "confidently_suboptimal": False,
        "candidate_stats": [
            {
                "key": "DISCARD:一",
                "visits": 8,
                "average_reward": 0.1,
                "win_rate": 0.25,
                "loss_rate": 0.125,
                "draw_rate": 0.625,
                "mean_outcome_score": 0.5,
                "mean_signed_xi": 1.0,
                "heuristic_value": 10.0,
            },
            {
                "key": "DISCARD:二",
                "visits": 8,
                "average_reward": 0.2,
                "win_rate": 0.375,
                "loss_rate": 0.125,
                "draw_rate": 0.5,
                "mean_outcome_score": 1.0,
                "mean_signed_xi": 2.0,
                "heuristic_value": 8.0,
            },
        ],
    }


def test_value_corpus_keeps_same_deal_in_one_split_and_all_actions_in_group():
    report = build_value_corpus((_audit(10, 1), _audit(10, 2), _audit(11, 3)))

    assert report["ok"]
    assert not report["training_ready"]
    assert report["samples"] == 6
    assert report["decision_groups"] == 3
    assert report["minimum_actions_per_group"] == 2
    assert report["maximum_actions_per_group"] == 2
    assert report["split_leakage"] == 0
    assert not report["training_ready"]
    assert any(
        failure.endswith("_deals_below_10")
        for failure in report["readiness_failures"]
    )
    split_by_deal = {}
    for row in report["rows"]:
        split_by_deal.setdefault(row["deal_key"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in split_by_deal.values())
    assert all(row["targets"]["visits"] == 8 for row in report["rows"])
    assert "decision_groups_below_2000" in report["readiness_failures"]


def test_value_corpus_rejects_missing_public_state():
    audit = _audit(10, 1)
    audit["public_state"] = {}

    report = build_value_corpus((audit,))

    assert not report["ok"]
    assert report["samples"] == 0
    assert report["failures"][0]["reason"] == "missing_game_or_public_state"


def test_value_corpus_preserves_direct_chi_candidate_structure():
    audit = _audit(12, 4)
    audit["candidate_stats"] = [
        {
            "key": "PASS",
            "action_type": "PASS",
            "visits": 8,
        },
        {
            "key": "CHI:option-hash",
            "action_type": "CHI",
            "option_id": "option-hash",
            "consumed_from_hand": ["四", "六"],
            "meld_groups": [["四", "五", "六"]],
            "followup_discard": "九",
            "visits": 8,
        },
    ]

    report = build_value_corpus((audit,))

    chi = next(row for row in report["rows"] if row["action"]["type"] == "CHI")
    assert chi["action"]["label"] is None
    assert chi["action"]["option_id"] == "option-hash"
    assert chi["action"]["consumed_from_hand"] == ["四", "六"]
    assert chi["action"]["meld_groups"] == [["四", "五", "六"]]
    assert chi["action"]["followup_discard"] == "九"


def test_value_corpus_preserves_visible_meld_and_discard_labels():
    audit = _audit(13, 5)
    audit["public_state"]["own_melds"] = [
        {"type": "peng", "labels": ["五", "五", "五"]}
    ]
    audit["public_state"]["all_melds"] = [
        [{"type": "peng", "labels": ["五", "五", "五"]}],
        [{"type": "wei", "labels": ["柒", "柒", "柒"]}],
    ]
    audit["public_state"]["discards"] = [["九"], ["贰", "拾"]]

    report = build_value_corpus((audit,))

    state = report["rows"][0]["state_features"]
    assert state["own_melds"][0]["labels"] == ["五", "五", "五"]
    assert state["all_melds"][1][0]["type"] == "wei"
    assert state["discards"] == [["九"], ["贰", "拾"]]


def test_value_corpus_prefers_deeper_audit_for_same_decision():
    shallow = _audit(14, 6)
    deep = deepcopy(shallow)
    deep["completed_paired_worlds"] = 64
    deep["candidate_stats"][0]["average_reward"] = 0.75

    report = build_value_corpus((shallow, deep))

    assert report["ok"]
    assert report["superseded_audits"] == 1
    assert report["samples"] == 2
    first = next(
        row
        for row in report["rows"]
        if row["action"]["key"] == "DISCARD:一"
    )
    assert first["audit_health"]["paired_worlds"] == 64
    assert first["targets"]["mean_reward"] == 0.75
