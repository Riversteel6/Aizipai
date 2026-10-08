"""Settlement scoring and unified EV scoring compatibility helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine.cards import RED_LABELS, WILD_LABEL
from engine.red_black_rules import classify_red_black
from engine.rules import load_rules


_LEGACY_AI_EXPORTS = {"ActionEval", "EVScorer", "evaluate_action_ev", "simulate_action"}


@dataclass(frozen=True)
class ScoreResult:
    total_xi: int
    min_xi: int
    can_hu: bool
    xi_to_tun: str
    base_tun: int
    extra_tun: int
    tun: int
    red_count: int
    black_count: int
    red_black_mode: str
    red_black_points: float
    red_black_multiplier: float
    piao_points: float
    piao_mode: str
    zimo_double_enabled: bool
    is_self_draw: bool
    zimo_multiplier: float
    final_score: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_xi": self.total_xi,
            "min_xi": self.min_xi,
            "can_hu": self.can_hu,
            "xi_to_tun": self.xi_to_tun,
            "base_tun": self.base_tun,
            "extra_tun": self.extra_tun,
            "tun": self.tun,
            "red_count": self.red_count,
            "black_count": self.black_count,
            "red_black_mode": self.red_black_mode,
            "red_black_points": self.red_black_points,
            "red_black_multiplier": self.red_black_multiplier,
            "piao_points": self.piao_points,
            "piao_mode": self.piao_mode,
            "zimo_double_enabled": self.zimo_double_enabled,
            "is_self_draw": self.is_self_draw,
            "zimo_multiplier": self.zimo_multiplier,
            "final_score": self.final_score,
            "details": dict(self.details),
        }


def score_hu(
    hu_result: Any,
    rule_config: Any | None = None,
    *,
    is_self_draw: bool | None = None,
    piao_points: int | float | None = None,
    red_count: int | None = None,
    black_count: int | None = None,
) -> ScoreResult:
    """Convert a HuResult-like object into a configurable settlement result."""

    config = _rules_dict(rule_config)
    rules = config.get("rules", {}) if isinstance(config.get("rules", {}), Mapping) else {}
    scoring = config.get("scoring", {}) if isinstance(config.get("scoring", {}), Mapping) else {}
    total_xi = int(_value(hu_result, "total_xi", 0) or 0)
    min_xi = int(_value(hu_result, "min_xi", rules.get("min_xi", 9)) or 9)
    can_hu = bool(_value(hu_result, "can_hu", total_xi >= min_xi)) and total_xi >= min_xi

    xi_to_tun = str(_rule_value(config, "xi_to_tun", "3_to_1"))
    base_tun = int(_rule_value(config, "base_tun_at_9_xi", 1))
    extra_tun = _extra_tun(total_xi, min_xi, xi_to_tun) if can_hu else 0
    tun = base_tun + extra_tun if can_hu else 0

    partition_labels = _partition_labels(_value(hu_result, "partition", []))
    resolved_red_count = int(red_count if red_count is not None else _value(hu_result, "red_count", -1))
    resolved_black_count = int(black_count if black_count is not None else _value(hu_result, "black_count", -1))
    if resolved_red_count < 0:
        resolved_red_count = sum(1 for label in partition_labels if label in RED_LABELS)
    if resolved_black_count < 0:
        resolved_black_count = sum(1 for label in partition_labels if label and label not in RED_LABELS and label != WILD_LABEL)

    red_black_mode = str(_rule_value(config, "red_black_mode", "red_black_mingtang"))
    rb_raw_points, rb_multiplier, rb_details = _red_black_points(
        red_count=resolved_red_count,
        black_count=resolved_black_count,
        mode=red_black_mode,
        scoring=scoring,
    )
    red_black_points = rb_raw_points * rb_multiplier if can_hu else 0.0

    resolved_piao_points = _resolve_piao_points(hu_result, piao_points)
    piao_mode = str(_rule_value(config, "piao_mode", "none"))
    zimo_double_enabled = _as_bool(_rule_value(config, "zimo_double", False))
    resolved_self_draw = _resolve_self_draw(hu_result, is_self_draw)
    zimo_multiplier = 2.0 if zimo_double_enabled and resolved_self_draw and can_hu else 1.0

    pre_multiplier_score = float(tun) + red_black_points + resolved_piao_points
    final_score = pre_multiplier_score * zimo_multiplier if can_hu else 0.0
    details = {
        "base_tun_at_min_xi": base_tun if can_hu else 0,
        "xi_over_min": max(total_xi - min_xi, 0),
        "pre_multiplier_score": round(pre_multiplier_score, 3) if can_hu else 0.0,
        "red_black": rb_details,
    }
    if not can_hu:
        details["reject_reason"] = _value(hu_result, "reject_reason", "cannot_hu_or_xi_not_enough")
    return ScoreResult(
        total_xi=total_xi,
        min_xi=min_xi,
        can_hu=can_hu,
        xi_to_tun=xi_to_tun,
        base_tun=base_tun if can_hu else 0,
        extra_tun=extra_tun,
        tun=tun,
        red_count=resolved_red_count,
        black_count=resolved_black_count,
        red_black_mode=red_black_mode,
        red_black_points=round(red_black_points, 3),
        red_black_multiplier=rb_multiplier if can_hu else 1.0,
        piao_points=resolved_piao_points if can_hu else 0.0,
        piao_mode=piao_mode,
        zimo_double_enabled=zimo_double_enabled,
        is_self_draw=resolved_self_draw,
        zimo_multiplier=zimo_multiplier,
        final_score=round(final_score, 3),
        details=details,
    )


calculate_score = score_hu
settle_hu = score_hu


def _rules_dict(rule_config: Any | None) -> dict[str, Any]:
    if rule_config is None:
        return load_rules()
    if isinstance(rule_config, (str, Path)):
        return load_rules(rule_config)
    if isinstance(rule_config, Mapping):
        return dict(rule_config)
    raw = getattr(rule_config, "raw", None)
    if isinstance(raw, Mapping):
        return dict(raw)
    raise TypeError("rule_config must be a RuleConfig, dict, path, or None")


def _rule_value(config: Mapping[str, Any], key: str, default: Any) -> Any:
    scoring = config.get("scoring", {})
    if isinstance(scoring, Mapping) and key in scoring:
        return scoring[key]
    rules = config.get("rules", {})
    if isinstance(rules, Mapping) and key in rules:
        return rules[key]
    return default


def _value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _extra_tun(total_xi: int, min_xi: int, xi_to_tun: str) -> int:
    over = max(total_xi - min_xi, 0)
    normalized = xi_to_tun.strip().lower().replace(" ", "")
    if normalized in {"1_to_1", "1:1", "1xi1tun", "1息1囤"}:
        return over
    if normalized in {"3_to_1", "3:1", "3xi1tun", "3息1囤"}:
        return over // 3
    raise ValueError(f"Unsupported xi_to_tun mode: {xi_to_tun}")


def _partition_labels(partition: Any) -> list[str]:
    if not isinstance(partition, Sequence) or isinstance(partition, (str, bytes)):
        return []
    labels: list[str] = []
    for meld in partition:
        if isinstance(meld, Mapping):
            raw_labels = meld.get("labels", meld.get("cards", []))
        elif isinstance(meld, Sequence) and not isinstance(meld, (str, bytes)):
            raw_labels = meld
        else:
            raw_labels = getattr(meld, "labels", getattr(meld, "cards", []))
        if isinstance(raw_labels, Sequence) and not isinstance(raw_labels, (str, bytes)):
            labels.extend(str(label) for label in raw_labels)
    return labels


def _red_black_points(
    *,
    red_count: int,
    black_count: int,
    mode: str,
    scoring: Mapping[str, Any],
) -> tuple[float, float, dict[str, Any]]:
    outcome = classify_red_black(
        red_count,
        {"rules": {"red_black_mode": mode}, "scoring": dict(scoring)},
        card_count=red_count + black_count,
    )
    details = {
        "special_hand": outcome.kind,
        "qualifies": outcome.qualifies,
        "raw_points": round(outcome.raw_points, 3),
        "multiplier": outcome.multiplier,
        "red_count": red_count,
        "black_count": black_count,
    }
    return outcome.raw_points, outcome.multiplier, details


def _resolve_piao_points(hu_result: Any, explicit: int | float | None) -> float:
    if explicit is not None:
        return float(explicit)
    for key in ("piao_points", "piao_score", "piao"):
        value = _value(hu_result, key, None)
        if value is not None:
            return float(value)
    return 0.0


def _resolve_self_draw(hu_result: Any, explicit: bool | None) -> bool:
    if explicit is not None:
        return bool(explicit)
    for key in ("is_self_draw", "self_draw", "zimo"):
        value = _value(hu_result, key, None)
        if value is not None:
            return _as_bool(value)
    win_type = str(_value(hu_result, "win_type", "") or "").strip().lower()
    return win_type in {"zimo", "self_draw", "self-draw", "selfdraw"}


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "zimo", "self_draw"}
    return bool(value)


def __getattr__(name: str) -> Any:
    if name not in _LEGACY_AI_EXPORTS:
        raise AttributeError(name)
    from ai import pro_brain

    return getattr(pro_brain, name)

__all__ = [
    "ActionEval",
    "EVScorer",
    "ScoreResult",
    "calculate_score",
    "evaluate_action_ev",
    "score_hu",
    "settle_hu",
    "simulate_action",
]
