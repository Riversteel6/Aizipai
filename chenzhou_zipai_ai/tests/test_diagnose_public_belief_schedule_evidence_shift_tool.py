from tools.diagnose_public_belief_schedule_evidence_shift import changed_state_ids


def test_changed_state_ids_compares_only_shared_source_rows() -> None:
    source_21 = [
        {"state_before_hash": "a", "candidate_selected_label": "一"},
        {"state_before_hash": "b", "candidate_selected_label": "二"},
    ]
    source_112 = [
        {"state_before_hash": "b", "candidate_selected_label": "三"},
    ]

    assert changed_state_ids(source_21, source_112) == ("b",)
