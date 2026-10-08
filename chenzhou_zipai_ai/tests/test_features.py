import random

from ai.features import (
    QUICK_POTENTIAL_LABELS,
    _quick_potential_cached,
    best_quick_potential_after_discard_from_counts,
    quick_potential,
    quick_potential_after_each_removal_from_counts,
)


def test_quick_potential_is_order_independent_and_reuses_cache():
    _quick_potential_cached.cache_clear()

    first = quick_potential(["二", "七", "十", "二", "王"])
    second = quick_potential(("王", "二", "十", "七", "二"))

    assert second == first
    info = _quick_potential_cached.cache_info()
    assert info.misses == 1
    assert info.hits == 1


def test_best_discard_feature_matches_full_removal_vector_on_random_counts():
    rng = random.Random(20260824)
    for _ in range(2_000):
        counts = tuple(rng.randrange(5) for _ in QUICK_POTENTIAL_LABELS)
        vector = quick_potential_after_each_removal_from_counts(counts)
        expected = max(
            (
                int(vector[index])
                for index, amount in enumerate(counts[:-1])
                if 0 < amount < 3
            ),
            default=None,
        )
        assert best_quick_potential_after_discard_from_counts(counts) == expected
