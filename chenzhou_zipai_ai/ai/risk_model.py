"""Opponent risk estimation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from engine.cards import RED_LABELS, WILD_LABEL, normalize_card_label
from engine.rules import load_rules


def danger_score(
    label: str,
    *,
    visible_labels: list[str] | None = None,
    remaining_deck_count: int | None = None,
    rules: dict[str, Any] | None = None,
    config_path: str = "config/rules.yaml",
) -> tuple[float, list[str]]:
    config = rules or load_rules(config_path)
    ai_weights = config.get("ai_weights", {})
    weights = config.get("weights", {})
    visible = Counter(visible_labels or [])
    reasons: list[str] = []
    score = 0.0

    if label == WILD_LABEL:
        score += abs(
            float(
                weights.get(
                    "wildcard_discard_penalty",
                    ai_weights.get("discard_wildcard_penalty", -500),
                )
            )
        )
        reasons.append("王不能轻易打出")
    if label in RED_LABELS:
        score += 35
        reasons.append("红牌可能服务二七十/红黑路线")
        if remaining_deck_count is not None and remaining_deck_count <= 10:
            score += abs(
                float(
                    weights.get(
                        "red_card_late_danger",
                        ai_weights.get("late_red_danger_penalty", -80),
                    )
                )
            )
            reasons.append("后盘红牌危险加重")
    if visible[label] >= 3:
        score -= 45
        reasons.append("已出现三张，安全度提高")
    elif visible[label] >= 2:
        score -= 25
        reasons.append("已出现两张，风险下降")
    elif visible[label] == 0 and remaining_deck_count is not None and remaining_deck_count <= 15:
        score += 30
        reasons.append("后盘未见牌，可能危险")

    return max(0.0, score), reasons


def visible_labels_from_memory(memory: dict | None) -> list[str]:
    if not memory:
        return []
    result: list[str] = []
    result.extend(_normalize_visible_labels(memory.get("my_discards", [])))
    opponents = _iter_opponent_entries(memory)
    if not opponents:
        result.extend(_normalize_visible_labels(memory.get("opponent_discards", [])))
    for key in ("my_meld_groups",):
        for group in memory.get(key, []):
            result.extend(_normalize_visible_labels(group))
    if not opponents:
        for group in memory.get("opponent_meld_groups", []):
            result.extend(_normalize_visible_labels(group))
    for opponent in opponents:
        for key in ("discards", "opponent_discards"):
            result.extend(_normalize_visible_labels(opponent.get(key, [])))
        for key in ("meld_groups", "opponent_meld_groups", "exposed_melds"):
            for group in opponent.get(key, []):
                result.extend(_normalize_visible_labels(group))
    return result


def _normalize_visible_labels(raw_labels) -> list[str]:
    if not isinstance(raw_labels, list):
        return []
    result: list[str] = []
    for raw in raw_labels:
        label = normalize_card_label(str(raw))
        if label and label != "暗":
            result.append(label)
    return result


def _iter_opponent_entries(memory: dict | None) -> list[dict]:
    if not memory:
        return []
    entries: list[dict] = []
    for key in ("opponents", "opponent_states", "other_players"):
        raw = memory.get(key, [])
        if isinstance(raw, dict):
            raw = raw.values()
        if not isinstance(raw, list) and not hasattr(raw, "__iter__"):
            continue
        for item in raw:
            if isinstance(item, dict):
                entries.append(item)
    return entries


def _iter_visible_meld_groups(memory: dict | None) -> list[list[str]]:
    if not memory:
        return []
    groups: list[list[str]] = []
    opponents = _iter_opponent_entries(memory)
    raw_groups = [] if opponents else list(memory.get("opponent_meld_groups") or [])
    for opponent in opponents:
        for key in ("meld_groups", "opponent_meld_groups", "exposed_melds"):
            raw_groups.extend(opponent.get(key, []))
    for raw_group in raw_groups:
        if isinstance(raw_group, dict):
            raw_cards = raw_group.get("cards") or raw_group.get("labels") or []
        else:
            raw_cards = raw_group
        if not isinstance(raw_cards, list):
            continue
        labels = [
            normalize_card_label(str(label))
            for label in raw_cards
            if str(label) != "暗" and normalize_card_label(str(label))
        ]
        if labels:
            groups.append(labels)
    return groups


def opponent_meld_risk(
    label: str,
    memory: dict | None,
    *,
    remaining_deck_count: int | None = None,
    rules: dict[str, Any] | None = None,
    config_path: str = "config/rules.yaml",
) -> tuple[float, list[str]]:
    """Estimate discard risk from opponent's exposed same-label melds."""
    target = normalize_card_label(label)
    if not target:
        return 0.0, []

    config = rules or load_rules(config_path)
    ai_weights = config.get("ai_weights", {})
    weights = config.get("weights", {})
    possible_pao = abs(float(ai_weights.get("possible_pao_danger_penalty", -200)))
    pao_risk = float(weights.get("opponent_pao_risk_penalty", 500))

    reasons: list[str] = []
    score = 0.0
    max_same_count = 0
    meld_groups = _iter_visible_meld_groups(memory)
    for group in meld_groups:
        same_count = sum(1 for group_label in group if group_label == target)
        max_same_count = max(max_same_count, same_count)
        if same_count >= 3:
            score += max(possible_pao, pao_risk)
            reasons.append(f"对手落地已有{target}{target}{target}，打出同牌可能送跑/明龙")
        elif same_count == 2:
            score += max(80.0, possible_pao * 0.5)
            reasons.append(f"对手落地已有一对{target}，打出同牌可能补对子/刻子")

    if score > 0 and target in RED_LABELS:
        score += 35
        reasons.append("同牌风险叠加红牌二七十价值")
    if score > 0 and remaining_deck_count is not None and remaining_deck_count <= 15:
        late_bonus = 50.0 if max_same_count >= 3 else 25.0
        score += late_bonus
        reasons.append("后盘落地同牌风险加重")

    return score, reasons


def opponent_meld_risks_by_seat(
    label: str,
    memory: dict | None,
    *,
    remaining_deck_count: int | None = None,
    rules: dict[str, Any] | None = None,
    config_path: str = "config/rules.yaml",
) -> list[dict[str, Any]]:
    """Return independent meld risks so two opponents are never merged."""
    risks: list[dict[str, Any]] = []
    for index, opponent in enumerate(_iter_opponent_entries(memory)):
        score, reasons = opponent_meld_risk(
            label,
            {"opponents": [opponent]},
            remaining_deck_count=remaining_deck_count,
            rules=rules,
            config_path=config_path,
        )
        risks.append(
            {
                "seat": opponent.get("seat", index + 1),
                "relative_offset": opponent.get("relative_offset"),
                "is_next_seat": bool(opponent.get("is_next_seat", False)),
                "danger_score": score,
                "reasons": reasons,
            }
        )
    return risks


@dataclass(frozen=True)
class DangerScore:
    danger_score: float
    risk_type: str
    reasons: list[str]

    def to_dict(self) -> dict:
        return {
            "danger_score": self.danger_score,
            "risk_type": self.risk_type,
            "reasons": list(self.reasons),
        }


def evaluate_discard_danger(state: dict | None, card_instance) -> DangerScore:
    label = getattr(card_instance, "label", str(card_instance))
    remaining = None
    visible: list[str] = []
    if isinstance(state, dict):
        remaining = state.get("remaining_deck_count") or state.get("remaining_cards_estimate")
        visible_raw = state.get("visible_cards") or state.get("discards") or []
        if isinstance(visible_raw, list):
            visible = visible_raw
    score, reasons = danger_score(label, visible_labels=visible, remaining_deck_count=remaining)
    if label in RED_LABELS:
        risk_type = "red_or_key"
    elif reasons:
        risk_type = "late_unknown"
    else:
        risk_type = "low"
    return DangerScore(score, risk_type, reasons or ["early_stage_low_danger"])
