from __future__ import annotations

from ai.dual_discard_validator import _balanced_world_shards


def test_balanced_world_shards_cover_each_world_once() -> None:
    shards = _balanced_world_shards(112, 5)

    assert shards == ((0, 23), (23, 23), (46, 22), (68, 22), (90, 22))
    assert [
        world
        for offset, amount in shards
        for world in range(offset, offset + amount)
    ] == list(range(112))
