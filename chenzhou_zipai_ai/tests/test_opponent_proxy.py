from collections import Counter

from ai.features import (
    QUICK_POTENTIAL_LABELS,
    quick_potential,
    quick_potential_after_each_removal_from_counts,
    quick_potential_from_counts,
)
from ai.full_game_simulator import PublicView
from ai.opponent_proxy import (
    FastAggressiveMeldProxyPolicy,
    FastDefensiveSearchProxyPolicy,
    FastIndependentDenialStructuralDiscardProxyPolicy,
    FastIndependentPressureStructuralDiscardProxyPolicy,
    FastIndependentStructuralDiscardProxyPolicy,
    FastInformationSetProxyPolicy,
    FastRedBlackSearchProxyPolicy,
    HybridIndependentResponseProxyPolicy,
    HybridInformationSetResponseProxyPolicy,
)
from engine.rules import rules_for_room


def test_fast_information_set_proxy_returns_legal_discard() -> None:
    view = PublicView(
        seat=0,
        hand=("一", "二", "四", "六", "九"),
        own_melds=(),
        all_melds=((), ()),
        discards=((), ()),
        remaining_counts=(
            ("一", 3),
            ("二", 3),
            ("三", 4),
            ("四", 3),
            ("五", 4),
            ("六", 3),
            ("七", 4),
            ("八", 4),
            ("九", 3),
            ("十", 4),
        ),
        stock_count=35,
        hand_sizes=(5, 5),
    )

    selected = FastInformationSetProxyPolicy().choose_discard(
        view,
        rules_for_room(wildcard_enabled=False, players=2),
    )

    assert selected in view.hand


def test_count_based_quick_potential_is_exact() -> None:
    hand = ("一", "一", "二", "三", "七", "十", "壹", "贰", "叁", "王")
    held = Counter(hand)
    counts = tuple(held[label] for label in QUICK_POTENTIAL_LABELS)

    assert quick_potential_from_counts(counts) == quick_potential(hand)


def test_batched_removal_quick_potential_is_exact_for_every_card() -> None:
    hand = (
        "一",
        "一",
        "二",
        "三",
        "七",
        "十",
        "壹",
        "贰",
        "叁",
        "王",
    )
    held = Counter(hand)
    counts = tuple(held[label] for label in QUICK_POTENTIAL_LABELS)

    values = quick_potential_after_each_removal_from_counts(counts)

    for index, amount in enumerate(counts):
        if amount <= 0:
            assert values[index] is None
            continue
        after = list(counts)
        after[index] -= 1
        assert values[index] == quick_potential_from_counts(tuple(after))


def test_batched_removal_quick_potential_reuses_immutable_count_state() -> None:
    held = Counter(("一", "一", "二", "三", "七", "十", "壹", "贰"))
    counts = tuple(held[label] for label in QUICK_POTENTIAL_LABELS)
    quick_potential_after_each_removal_from_counts.cache_clear()

    first = quick_potential_after_each_removal_from_counts(counts)
    second = quick_potential_after_each_removal_from_counts(counts)
    cache = quick_potential_after_each_removal_from_counts.cache_info()

    assert second is first
    assert cache.hits == 1
    assert cache.misses == 1
    assert cache.maxsize == 65_536
    quick_potential_after_each_removal_from_counts.cache_clear()


def test_structural_fast_proxies_return_legal_discards() -> None:
    view = PublicView(
        seat=0,
        hand=("一", "二", "四", "六", "九"),
        own_melds=(),
        all_melds=((), ()),
        discards=((), ()),
        remaining_counts=(("一", 3), ("二", 3), ("三", 4), ("四", 3)),
        stock_count=35,
        hand_sizes=(5, 5),
    )
    rules = rules_for_room(wildcard_enabled=False, players=2)

    for policy_type in (
        FastAggressiveMeldProxyPolicy,
        FastDefensiveSearchProxyPolicy,
        FastIndependentStructuralDiscardProxyPolicy,
        FastIndependentDenialStructuralDiscardProxyPolicy,
        FastIndependentPressureStructuralDiscardProxyPolicy,
        FastRedBlackSearchProxyPolicy,
    ):
        selected = policy_type().choose_discard(view, rules)
        assert selected in view.hand


def test_search_proxy_response_gain_uses_frozen_margins() -> None:
    view = PublicView(
        seat=0,
        hand=("一", "一", "二", "四", "六", "九"),
        own_melds=(),
        all_melds=((), ()),
        discards=((), ()),
        remaining_counts=(("一", 2), ("二", 3), ("三", 4), ("四", 3)),
        stock_count=35,
        hand_sizes=(6, 6),
        pending_card="一",
        pending_source_seat=1,
    )
    rules = rules_for_room(wildcard_enabled=False, players=2)
    policy = FastInformationSetProxyPolicy()

    gain = policy._peng_response_gain(view, "一", rules)

    assert gain is not None
    assert policy.choose_peng(view, "一", rules) == (gain > policy.peng_margin)


def test_hybrid_response_proxies_keep_fast_legal_discard() -> None:
    view = PublicView(
        seat=0,
        hand=("一", "二", "四", "六", "九"),
        own_melds=(),
        all_melds=((), ()),
        discards=((), ()),
        remaining_counts=(("一", 3), ("二", 3), ("三", 4), ("四", 3)),
        stock_count=35,
        hand_sizes=(5, 5),
    )
    rules = rules_for_room(wildcard_enabled=False, players=2)

    for policy in (
        HybridInformationSetResponseProxyPolicy(),
        HybridIndependentResponseProxyPolicy(),
    ):
        assert policy.choose_discard(view, rules) in view.hand
