from engine.cards import SMALL_LABELS
from tools.build_sequential_public_evidence_dataset import (
    count_forbidden_keys,
    extract_game_events,
    public_projection,
)


def _state(
    *,
    seat: int,
    discards: list[list[str]],
    hand_sizes: list[int],
    stock_count: int = 39,
) -> dict:
    return {
        "seat": seat,
        "hand": [SMALL_LABELS[2]],
        "remaining_counts": {SMALL_LABELS[3]: 3},
        "all_melds": [[], []],
        "own_melds": [],
        "discards": discards,
        "hand_sizes": hand_sizes,
        "stock_count": stock_count,
    }


def test_public_projection_drops_private_fields() -> None:
    projected = public_projection(
        _state(seat=1, discards=[[], []], hand_sizes=[20, 20]),
        observer_seat=0,
    )

    assert projected["seat"] == 0
    assert "hand" not in projected
    assert "remaining_counts" not in projected


def test_extract_game_events_maps_private_pass_to_no_claim() -> None:
    first, second = SMALL_LABELS[:2]
    game = {
        "players": 2,
        "candidate_seat": 0,
        "wildcard_enabled": False,
        "decision_trace": [
            {
                "sequence": 1,
                "seat": 0,
                "phase": "discard",
                "selected_key": f"DISCARD:{first}",
                "public_state": _state(
                    seat=0,
                    discards=[[], []],
                    hand_sizes=[21, 20],
                ),
                "legal_actions": [
                    {"key": f"DISCARD:{first}", "type": "DISCARD"}
                ],
            },
            {
                "sequence": 2,
                "seat": 1,
                "phase": "response_root",
                "selected_key": "PASS",
                "public_state": _state(
                    seat=1,
                    discards=[[first], []],
                    hand_sizes=[20, 20],
                ),
                "legal_actions": [
                    {"key": "PASS", "type": "PASS"},
                    {"key": "CHI:test", "type": "CHI"},
                ],
            },
            {
                "sequence": 3,
                "seat": 1,
                "phase": "discard",
                "selected_key": f"DISCARD:{second}",
                "public_state": _state(
                    seat=1,
                    discards=[[first], []],
                    hand_sizes=[20, 21],
                    stock_count=38,
                ),
                "legal_actions": [
                    {"key": f"DISCARD:{second}", "type": "DISCARD"}
                ],
            },
        ],
    }

    events, diagnostics = extract_game_events(
        game,
        game_id="g",
        profile="p",
        fold=0,
    )

    assert [event["action_kind"] for event in events] == [
        "NO_CLAIM",
        "DISCARD",
    ]
    assert diagnostics["explicit_pass"] == 1
    assert diagnostics["reconstructed_context_mismatches"] == 0
    assert all(count_forbidden_keys(event) == 0 for event in events)
