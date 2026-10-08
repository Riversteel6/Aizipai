from ai.full_game_simulator import PublicView, SimMeld
import pytest

from ai.opponent_context import (
    opponent_context_features,
    opponent_context_interactions,
)


def test_opponent_context_matches_mapping_and_public_view() -> None:
    public_view = PublicView(
        seat=0,
        hand=("一", "二"),
        own_melds=(),
        all_melds=(
            (),
            (
                SimMeld("peng", ("三", "三", "三")),
                SimMeld("normal_sequence", ("四", "五", "六")),
            ),
        ),
        discards=(("七",), ("八", "九", "十", "壹")),
        remaining_counts=(),
        stock_count=20,
        hand_sizes=(2, 8),
    )
    mapping = {
        "seat": 0,
        "all_melds": [
            [],
            [
                {"type": "peng", "labels": ["三", "三", "三"]},
                {
                    "type": "normal_sequence",
                    "labels": ["四", "五", "六"],
                },
            ],
        ],
        "discards": [["七"], ["八", "九", "十", "壹"]],
        "stock_count": 20,
        "hand_sizes": [2, 8],
    }

    features = opponent_context_features(public_view)

    assert features == opponent_context_features(mapping)
    assert features == (
        20 / 30,
        4 / 20,
        2 / 6,
        0.5,
        0.5,
        0.0,
        8 / 20,
        6 / 20,
    )


def test_opponent_context_interactions_validate_feature_count() -> None:
    context = (0.5,) * 8

    assert opponent_context_interactions(0.2, context) == pytest.approx(
        (0.1,) * 8
    )
    with pytest.raises(
        ValueError,
        match="opponent_context_interaction_feature_mismatch",
    ):
        opponent_context_interactions(0.2, context[:-1])
