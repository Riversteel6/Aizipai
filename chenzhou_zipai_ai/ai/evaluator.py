"""Position and action evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ai.hand_protection import ProtectedMeld
from ai.pro_brain import (
    EVScorer,
    allocate_hand_structures as allocate_professional_structures,
    analyze_hand as analyze_professional_hand,
    build_decision_context as build_professional_context,
    generate_legal_actions as generate_professional_legal_actions,
)
from engine.cards import normalize_cards


@dataclass(frozen=True)
class DiscardEvaluation:
    label: str
    score: float
    base_score: float
    orphan_score: float
    structure_loss: float
    breaks_melds: list[list[str]]
    danger_score: float
    after_xi_potential: int
    danger: float
    total_score: float
    reasons: list[str] = field(default_factory=list)
    penalties: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "score": self.score,
            "base_score": self.base_score,
            "orphan_score": self.orphan_score,
            "structure_loss": self.structure_loss,
            "breaks_melds": self.breaks_melds,
            "danger_score": self.danger_score,
            "after_xi_potential": self.after_xi_potential,
            "danger": self.danger,
            "total_score": self.total_score,
            "reasons": self.reasons,
            "penalties": self.penalties,
        }


def evaluate_discard(
    hand: list[str],
    label: str,
    *,
    memory: dict | None = None,
    remaining_deck_count: int | None = None,
    config_path: str = "config/rules.yaml",
    protected_melds: list[ProtectedMeld] | None = None,
    rules: dict | None = None,
) -> DiscardEvaluation:
    normalized_candidates = normalize_cards([label])
    if not normalized_candidates:
        raise ValueError(f"Cannot normalize discard label: {label!r}")
    label = normalized_candidates[0]
    normalized_hand = normalize_cards(hand)
    counter = Counter(normalized_hand)
    if counter[label] <= 0:
        raise ValueError(f"Cannot discard {label!r}; not in hand.")
    for item in evaluate_discards(
        normalized_hand,
        memory=memory,
        remaining_deck_count=remaining_deck_count,
        config_path=config_path,
        protected_melds=protected_melds,
        rules=rules,
    ):
        if item.label == label:
            return item
    raise ValueError(f"No professional discard evaluation for {label!r}.")


def evaluate_discards(
    hand: list[str],
    *,
    memory: dict | None = None,
    remaining_deck_count: int | None = None,
    config_path: str = "config/rules.yaml",
    protected_melds: list[ProtectedMeld] | None = None,
    rules: dict | None = None,
) -> list[DiscardEvaluation]:
    normalized_hand = normalize_cards(hand)
    context = build_professional_context(
        {
            "hand": normalized_hand,
            "legal_actions": [{"type": "DISCARD"}],
            "memory": memory or {},
            "remaining_deck_count": remaining_deck_count,
        },
        rules=rules,
        config_path=config_path,
    )
    allocation = allocate_professional_structures(context)
    analysis = analyze_professional_hand(context, allocation)
    actions, rejected = generate_professional_legal_actions(context, allocation, analysis)
    scorer = EVScorer()
    rows: list[DiscardEvaluation] = []
    for action in [*actions, *rejected]:
        if action.type != "DISCARD" or not action.label:
            continue
        action_eval = scorer.evaluate(context, action, allocation, analysis)
        rows.append(_discard_evaluation_from_action_eval(action_eval))
    best_by_label: dict[str, DiscardEvaluation] = {}
    for row in rows:
        if row.label not in best_by_label or row.score > best_by_label[row.label].score:
            best_by_label[row.label] = row
    return sorted(best_by_label.values(), key=lambda item: item.score, reverse=True)


def calculate_structure_loss_if_discard(
    label: str,
    hand: list[str],
    *,
    protected_melds: list[ProtectedMeld] | None = None,
    config_path: str = "config/rules.yaml",
    rules: dict | None = None,
) -> tuple[float, list[str], list[tuple[str, ...]]]:
    normalized_hand = normalize_cards(hand)
    normalized_label = normalize_cards([label])[0] if normalize_cards([label]) else ""
    if not normalized_label:
        return 0.0, [], []
    if normalized_hand.count(normalized_label) <= 0:
        return 0.0, [], []
    context = build_professional_context(
        {"hand": normalized_hand, "legal_actions": [{"type": "DISCARD"}]},
        rules=rules,
        config_path=config_path,
    )
    allocation = allocate_professional_structures(context)
    matching = [card.id for card in allocation.card_instances if card.label == normalized_label]
    if not matching:
        return 0.0, [], []
    reasons: list[str] = []
    breaks: list[tuple[str, ...]] = []
    loss = 0.0
    for meld in allocation.protected_melds:
        if not set(matching) & set(meld.card_ids):
            continue
        if meld.protect_level == "hard":
            loss += float(context.rules.get("weights", {}).get("hard_protected_break_penalty", 10000))
        else:
            loss += float(meld.structure_value)
        reasons.append(f"拆{_protect_type_desc(meld.type)}{''.join(meld.labels)}（{meld.protect_level}）")
        breaks.append(tuple(meld.labels))
    return round(loss, 3), reasons, breaks


def _discard_evaluation_from_action_eval(action_eval) -> DiscardEvaluation:
    payload = action_eval.to_dict()
    label = action_eval.action.label or ""
    return DiscardEvaluation(
        label=label,
        score=action_eval.ev,
        base_score=action_eval.hand_value_after,
        orphan_score=action_eval.score_gain,
        structure_loss=action_eval.structure_loss,
        breaks_melds=[list(item) for item in payload.get("breaks_melds", [])],
        danger_score=action_eval.danger_loss,
        after_xi_potential=int(action_eval.xi_gain),
        danger=action_eval.danger_loss,
        total_score=action_eval.ev,
        reasons=[action_eval.reason] if action_eval.reason else [],
        penalties=[action_eval.reject_reason] if action_eval.reject_reason else [],
    )


def _protect_type_desc(meld_type: str) -> str:
    if meld_type == "exact_triplet":
        return "刻/刻型结构"
    if meld_type == "exact_quad":
        return "刻型结构（4张）"
    if meld_type == "mixed_same_rank_triplet":
        return "同点混搭"
    if meld_type == "special_123":
        return "一二三/壹贰叁"
    if meld_type == "special_2710":
        return "二七十/贰柒拾"
    if meld_type == "normal_sequence":
        return "顺子"
    if meld_type == "wildcard_meld":
        return "王结构"
    return "结构"


def structure_loss(
    label: str,
    hand: list[str],
    *,
    config_path: str = "config/rules.yaml",
) -> tuple[float, list[str]]:
    loss, reasons, _ = calculate_structure_loss_if_discard(label, hand, config_path=config_path)
    return loss, reasons
