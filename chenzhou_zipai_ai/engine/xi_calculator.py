"""Xi calculation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from engine.melds import MeldPattern, classify_meld
from engine.rules import load_rules


@dataclass(frozen=True)
class XiBreakdown:
    total_xi: int
    details: list[dict[str, Any]]
    is_enough_to_hu: bool
    min_xi: int
    xi_gap: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_xi": self.total_xi,
            "details": list(self.details),
            "is_enough_to_hu": self.is_enough_to_hu,
            "min_xi": self.min_xi,
            "xi_gap": self.xi_gap,
        }


def meld_xi(
    labels: list[str],
    *,
    kind: str | None = None,
    config_path: str = "config/rules.yaml",
    rules: dict[str, Any] | None = None,
) -> int:
    config = rules or load_rules(config_path)
    return _meld_xi_from_config(labels, kind=kind, config=config)


def _meld_xi_from_config(labels: list[str], *, kind: str | None, config: dict[str, Any]) -> int:
    pattern = classify_meld(labels) if kind is None else MeldPattern(tuple(labels), kind)
    suit = pattern.suit or _infer_suit(labels)
    if pattern.kind in {"ti", "hidden_quad"}:
        return int(config["xi"]["ti"].get(suit, 0))
    if pattern.kind in {"pao", "quad"}:
        return int(config["xi"]["pao"].get(suit, 0))
    if pattern.kind in {"wei", "hidden_triplet"}:
        return int(config["xi"]["wei"].get(suit, 0))
    if pattern.kind == "peng":
        return int(config["xi"]["peng"].get(suit, 0))
    if pattern.kind == "special_123":
        return int(config["xi"]["special_123"].get(suit, 0))
    if pattern.kind == "special_2710":
        return int(config["xi"]["special_2710"].get(suit, 0))
    return 0


def total_xi(
    groups: list[list[str]],
    *,
    config_path: str = "config/rules.yaml",
    rules: dict[str, Any] | None = None,
) -> int:
    return sum(meld_xi(group, config_path=config_path, rules=rules) for group in groups)


def calculate_xi(
    melds,
    rules: dict[str, Any] | None = None,
    *,
    config_path: str = "config/rules.yaml",
):
    if _looks_like_protected_melds(melds):
        from ai.pro_brain import calculate_xi as _calculate_xi

        return _calculate_xi(melds, rules)

    config = rules or load_rules(config_path)
    minimum = int(config.get("rules", {}).get("min_xi", 9))
    details = [_meld_detail(item, config) for item in (melds or [])]
    total = sum(item["xi"] for item in details)
    return XiBreakdown(
        total_xi=total,
        details=details,
        is_enough_to_hu=total >= minimum,
        min_xi=minimum,
        xi_gap=max(0, minimum - total),
    )


def estimate_potential_xi(allocation, rules: dict | None = None):
    from ai.pro_brain import estimate_potential_xi as _estimate_potential_xi

    return _estimate_potential_xi(allocation, rules)


def _looks_like_protected_melds(melds) -> bool:
    return bool(melds) and all(hasattr(item, "xi_value") and hasattr(item, "labels") for item in melds)


def _meld_detail(meld, config: dict[str, Any]) -> dict[str, Any]:
    if isinstance(meld, Mapping):
        labels = list(meld.get("labels") or meld.get("cards") or [])
        kind = meld.get("kind") or meld.get("type")
    elif isinstance(meld, Sequence) and not isinstance(meld, (str, bytes)):
        labels = [str(label) for label in meld]
        kind = None
    else:
        labels = []
        kind = None
    xi = _meld_xi_from_config(labels, kind=str(kind) if kind else None, config=config)
    pattern = classify_meld(labels) if kind is None else MeldPattern(tuple(labels), str(kind))
    return {
        "type": pattern.kind,
        "cards": labels,
        "labels": labels,
        "xi": xi,
        "suit": pattern.suit or _infer_suit(labels),
    }


def _infer_suit(labels: list[str]) -> str:
    for label in labels:
        if label in {"一", "二", "三", "四", "五", "六", "七", "八", "九", "十"}:
            return "small"
        if label in {"壹", "贰", "叁", "肆", "伍", "陆", "柒", "捌", "玖", "拾"}:
            return "big"
    return "small"
