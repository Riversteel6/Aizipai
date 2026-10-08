"""Tests for hu checking."""

from collections import Counter
import random

from engine.hu_checker import (
    _candidate_groups_with,
    _count_state,
    _label_sort_key,
    _solve_grouping,
    _solve_grouping_counts,
    _state_key,
    best_grouping,
    best_grouping_normalized,
    can_form_three_card_meld,
    concealed_group_xi,
    explain_hu,
)
from engine.melds import complete_three_card_meld
from engine.rules import rules_for_room
from engine.cards import WILD_LABEL, normal_labels


COMPLETE_HAND = [
    "贰", "柒", "拾",
    "壹", "贰", "叁",
    "九", "九", "九",
    "一", "二", "三",
    "四", "五", "六",
    "七", "八", "九",
    "肆", "伍", "陆",
]


def test_normalized_grouping_fast_path_is_exact_on_random_engine_hands():
    rng = random.Random(20260824)
    labels = [*normal_labels(), WILD_LABEL]

    for required_pair_count in (0, 1):
        for group_count in range(1, 8):
            size = group_count * 3 + required_pair_count * 2
            for _ in range(80):
                hand = [rng.choice(labels) for _ in range(size)]
                expected = tuple(
                    tuple(group)
                    for group in best_grouping(
                        hand,
                        required_pair_count=required_pair_count,
                    )
                )
                actual = best_grouping_normalized(
                    hand,
                    required_pair_count=required_pair_count,
                )
                assert actual == expected


def test_wildcard_can_complete_key_melds():
    assert can_form_three_card_meld(["二", "七", "王"])
    assert can_form_three_card_meld(["一", "二", "王"])
    assert can_form_three_card_meld(["八", "八", "王"])
    assert can_form_three_card_meld(["贰", "柒", "王"])


def test_all_wang_group_uses_highest_legal_concealed_xi():
    rules = rules_for_room(wildcard_enabled=True)

    assert concealed_group_xi(["王", "王", "王"], rules=rules) == 6
    rules["wildcard"]["wildcard_groups_count_xi"] = False
    assert concealed_group_xi(["王", "王", "王"], rules=rules) == 0


def test_explain_hu_returns_groups_and_xi():
    breakdown = explain_hu(COMPLETE_HAND)

    assert breakdown.can_hu
    assert breakdown.total_xi >= breakdown.min_xi
    assert ["贰", "柒", "拾"] in breakdown.hand_groups
    assert breakdown.best_partition == breakdown.hand_groups
    assert breakdown.to_dict()["best_partition"] == breakdown.best_partition
    assert breakdown.reasons
    assert breakdown.partition_complete
    assert breakdown.group_count == 7


def test_explain_hu_best_partition_includes_existing_melds_first():
    hand = [
        "贰", "柒", "拾",
        "壹", "贰", "叁",
        "一", "二", "三",
        "四", "五", "六",
        "七", "八", "九",
        "肆", "伍", "陆",
    ]
    breakdown = explain_hu(hand, existing_groups=[["玖", "玖", "玖"]])

    assert breakdown.best_partition[0] == ["玖", "玖", "玖"]
    assert ["贰", "柒", "拾"] in breakdown.best_partition
    assert breakdown.to_dict()["can_hu"] is True


def test_explain_hu_rejects_scoring_subset_with_leftover_card():
    breakdown = explain_hu([*COMPLETE_HAND, "十"])

    assert not breakdown.can_hu
    assert not breakdown.partition_complete


def test_explain_hu_accepts_one_pair_after_a_four_card_meld():
    existing = [
        ["三", "三", "三", "三"],
        ["一", "二", "三"],
        ["四", "五", "六"],
        ["七", "八", "九"],
        ["壹", "贰", "叁"],
        ["肆", "伍", "陆"],
    ]

    breakdown = explain_hu(["王", "王"], existing_groups=existing)

    assert breakdown.can_hu
    assert breakdown.partition_complete
    assert breakdown.group_count == 7
    assert ["王", "王"] in breakdown.hand_groups


def test_explain_hu_accepts_wildcard_as_half_of_quad_compensation_pair():
    existing = [
        ["三", "三", "三", "三"],
        ["一", "二", "三"],
        ["四", "五", "六"],
        ["七", "八", "九"],
        ["壹", "贰", "叁"],
        ["肆", "伍", "陆"],
    ]

    breakdown = explain_hu(["拾", "王"], existing_groups=existing)

    assert breakdown.can_hu
    assert ["拾", "王"] in breakdown.hand_groups


def test_explain_hu_does_not_allow_a_pair_without_a_four_card_meld():
    existing = [
        ["一", "二", "三"],
        ["四", "五", "六"],
        ["七", "八", "九"],
        ["壹", "贰", "叁"],
        ["肆", "伍", "陆"],
        ["柒", "捌", "玖"],
    ]

    breakdown = explain_hu(["王", "王"], existing_groups=existing)

    assert not breakdown.can_hu
    assert not breakdown.partition_complete


def test_explain_hu_accepts_pair_after_multiple_four_card_melds():
    existing = [
        ["三", "三", "三", "三"],
        ["四", "四", "四", "四"],
        ["一", "二", "三"],
        ["七", "八", "九"],
        ["壹", "贰", "叁"],
    ]

    breakdown = explain_hu(
        ["伍", "伍", "伍", "王", "王"],
        existing_groups=existing,
    )

    assert breakdown.can_hu
    assert breakdown.partition_complete
    assert breakdown.total_xi == 33
    assert breakdown.group_count == 7
    assert len(breakdown.hand_groups) == 2
    assert any(len(group) == 2 for group in breakdown.hand_groups)


def test_precomputed_candidate_groups_match_exhaustive_enumeration():
    states = (
        Counter("一一二三四七十"),
        Counter("一一壹壹二贰"),
        Counter("二七王八八"),
        Counter("壹贰王王拾"),
        Counter("王王王"),
    )

    for counts in states:
        first = min(counts, key=_label_sort_key)
        labels = sorted(counts, key=_label_sort_key)
        exhaustive = []
        seen = set()
        for second_index, second in enumerate(labels):
            for third in labels[second_index:]:
                group = [first, second, third]
                needed = Counter(group)
                if any(
                    counts[label] < amount
                    for label, amount in needed.items()
                ):
                    continue
                if complete_three_card_meld(group) is None:
                    continue
                key = tuple(sorted(group, key=_label_sort_key))
                if key in seen:
                    continue
                seen.add(key)
                exhaustive.append(list(key))

        assert _candidate_groups_with(first, counts) == exhaustive


def test_compact_grouping_solver_matches_legacy_solver_on_random_hands():
    rng = random.Random(20260808)
    deck = [
        label
        for label in (*normal_labels(), WILD_LABEL)
        for _ in range(4)
    ]
    for _ in range(2_000):
        rng.shuffle(deck)
        required_pairs = rng.randrange(2)
        valid_lengths = (
            (2, 5, 8, 11, 14, 17, 20)
            if required_pairs
            else (3, 6, 9, 12, 15, 18, 21)
        )
        hand = list(deck[: rng.choice(valid_lengths)])
        legacy = _solve_grouping(
            _state_key(Counter(hand)),
            "config/rules.yaml",
            required_pairs,
        )
        compact = _solve_grouping_counts(
            _count_state(hand),
            "config/rules.yaml",
            required_pairs,
        )

        assert compact == legacy
