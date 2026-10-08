from tools.confirmed_hand_ledger import load_confirmed_ledger, record_confirmed_action


def context(*, hand=None, target=None, pending=None):
    return {
        "hand_signature": hand or [["一", 2], ["九", 1]],
        "planned_hand_target": target,
        "opponent_pending_card": pending,
    }


def test_confirmed_discard_commits_one_semantic_card(tmp_path):
    path = tmp_path / "ledger.json"

    result = record_confirmed_action(
        path,
        mode="1v1-wang",
        action="discard",
        context=context(target={"label": "一"}),
    )

    assert result["state_version"] == 1
    assert result["trusted"] is True
    assert result["hand_counts"] == {"一": 1, "九": 1}
    assert load_confirmed_ledger(path) == result


def test_failed_or_unconfirmed_actions_never_call_ledger(tmp_path):
    path = tmp_path / "ledger.json"
    assert load_confirmed_ledger(path) == {}


def test_unknown_combination_is_recorded_but_not_promoted(tmp_path):
    path = tmp_path / "ledger.json"

    result = record_confirmed_action(
        path,
        mode="1v1-no-wang",
        action="chi_option",
        context=context(),
    )

    assert result["state_version"] == 1
    assert result["shadow_only"] is True
    assert result["trusted"] is False
    assert result["pending_reconciliation_reason"] == "confirmed_combination_requires_reconciliation"
