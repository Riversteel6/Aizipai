"""Choose chi/compare option groups after the button response opens candidates."""

from __future__ import annotations

from dataclasses import dataclass

from engine.cards import BIG_LABELS, RED_LABELS, SMALL_LABELS
from engine.melds import classify_meld
from engine.xi_calculator import meld_xi


@dataclass(frozen=True)
class OptionEvaluation:
    labels: list[str]
    action: str
    score: float
    type: str
    xi: int
    reason: str

    def to_dict(self) -> dict:
        return {
            "labels": self.labels,
            "action": self.action,
            "score": self.score,
            "type": self.type,
            "xi": self.xi,
            "reason": self.reason,
        }


def evaluate_option(labels: list[str], *, action: str = "chi", config_path: str = "config/rules.yaml") -> OptionEvaluation:
    if action == "chi" and len(labels) == 2:
        return _evaluate_two_card_chi_option(labels)
    pattern = classify_meld(labels)
    if pattern.kind == "mixed_same_rank" and labels[0] == labels[2] != labels[1]:
        return OptionEvaluation(labels, action, -100.0, "unknown", 0, "候选牌显示顺序异常，拒绝按同点混搭执行")
    xi = meld_xi(labels, kind=pattern.kind, config_path=config_path)
    score = float(xi * 100)
    reasons: list[str] = []
    if pattern.kind == "special_2710":
        score += 320
        reasons.append("二七十/贰柒拾高价值")
    elif pattern.kind == "special_123":
        score += 260
        reasons.append("一二三/壹贰叁有胡息")
    elif pattern.kind == "peng":
        score += 180
        reasons.append("三同牌结构稳定")
    elif pattern.kind == "mixed_same_rank":
        score += 140
        reasons.append("同点混搭候选")
    elif pattern.kind == "sequence":
        score += 80
        reasons.append("普通顺子")
    elif pattern.kind.startswith("hidden"):
        score += 200
        reasons.append("暗牌组有胡息")
    else:
        score -= 100
        reasons.append("组合价值不明确")
    hidden_or_wild = sum(1 for label in labels if label in {"王", "暗"})
    if hidden_or_wild:
        score -= hidden_or_wild * 25
        reasons.append("使用王/暗牌信息，保守扣分")
    return OptionEvaluation(labels, action, round(score, 3), pattern.kind, xi, "；".join(reasons))


def _evaluate_two_card_chi_option(labels: list[str]) -> OptionEvaluation:
    ranks: list[int] = []
    suits: list[str] = []
    for label in labels:
        if label in SMALL_LABELS:
            ranks.append(SMALL_LABELS.index(label) + 1)
            suits.append("small")
        elif label in BIG_LABELS:
            ranks.append(BIG_LABELS.index(label) + 1)
            suits.append("big")
        else:
            return OptionEvaluation(labels, "chi", -100.0, "unknown", 0, "两张候选含未知牌")
    if len(set(suits)) != 1:
        return OptionEvaluation(labels, "chi", -80.0, "unknown", 0, "两张候选大小字不一致")
    gap = abs(ranks[0] - ranks[1])
    if gap not in {1, 2}:
        return OptionEvaluation(labels, "chi", -60.0, "unknown", 0, "两张候选不能与一张待吃牌组成顺子")
    score = 120.0 if gap == 2 else 90.0
    red_count = sum(1 for label in labels if label in RED_LABELS)
    score += red_count * 20
    reason = "两张候选可与待吃牌组成顺子"
    if red_count:
        reason += "；含红牌"
    return OptionEvaluation(labels, "chi", round(score, 3), "two_card_sequence_candidate", 0, reason)


def choose_option(
    options: list[list[str]],
    *,
    action: str = "chi",
    config_path: str = "config/rules.yaml",
) -> OptionEvaluation | None:
    if not options:
        return None
    evaluations = [evaluate_option(option, action=action, config_path=config_path) for option in options]
    return sorted(evaluations, key=lambda item: item.score, reverse=True)[0]
