import pytest

from tools.diagnose_proxy_public_belief_validator_range import select_state_ids


def test_select_state_ids_uses_one_based_inclusive_bounds() -> None:
    assert select_state_ids(
        ("a", "b", "c", "d"),
        start_index=2,
        end_index=3,
    ) == ("b", "c")


@pytest.mark.parametrize(
    ("start_index", "end_index"),
    ((0, 1), (2, 1), (1, 5)),
)
def test_select_state_ids_rejects_invalid_bounds(
    start_index: int,
    end_index: int,
) -> None:
    with pytest.raises(ValueError, match="validator_range_bounds"):
        select_state_ids(
            ("a", "b", "c", "d"),
            start_index=start_index,
            end_index=end_index,
        )
