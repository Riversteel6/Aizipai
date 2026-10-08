import json

from tools.select_counterfactual_states import select_stratified_states
from tools.select_counterfactual_states import load_excluded_state_hashes


def test_stratified_selection_is_balanced_and_repeatable() -> None:
    candidates = [
        {
            "state_before_hash": f"{stratum}-{index}",
            "opponent_stratum": stratum,
            "game_seed": index,
            "trace_sequence": index,
        }
        for stratum in ("A", "B")
        for index in range(6)
    ]

    first = select_stratified_states(
        candidates,
        per_stratum=3,
        seed=17,
    )
    second = select_stratified_states(
        list(reversed(candidates)),
        per_stratum=3,
        seed=17,
    )

    assert first == second
    assert len(first) == 6
    assert sum(row["opponent_stratum"] == "A" for row in first) == 3
    assert sum(row["opponent_stratum"] == "B" for row in first) == 3


def test_load_excluded_state_hashes_reads_report_rows(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(
            {
                "rows": [
                    {"state_before_hash": "a"},
                    {"state_before_hash": "b"},
                ]
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "rows": [
                    {"state_before_hash": "b"},
                    {"state_before_hash": "c"},
                    {"other": "ignored"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_excluded_state_hashes([first, second]) == {
        "a",
        "b",
        "c",
    }
