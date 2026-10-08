"""Offline self-play calibration for long-horizon discard EV."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ai.monte_carlo import simulate_discards
from ai.pro_brain import allocate_hand_structures, build_decision_context, choose_action
from engine.cards import BIG_LABELS, SMALL_LABELS, WILD_LABEL


DECK = [*SMALL_LABELS, *BIG_LABELS] * 4 + [WILD_LABEL] * 4


@dataclass(frozen=True)
class SelfPlayDecision:
    index: int
    hand: list[str]
    policy_action: str
    policy_label: str | None
    candidate_stage: str
    policy_ev: float
    monte_carlo_best_label: str | None
    monte_carlo_best_score: float | None
    policy_monte_carlo_score: float | None
    long_term_gap: float | None
    hard_protected_labels: list[str]
    soft_protected_labels: list[str]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "hand": list(self.hand),
            "policy_action": self.policy_action,
            "policy_label": self.policy_label,
            "candidate_stage": self.candidate_stage,
            "policy_ev": self.policy_ev,
            "monte_carlo_best_label": self.monte_carlo_best_label,
            "monte_carlo_best_score": self.monte_carlo_best_score,
            "policy_monte_carlo_score": self.policy_monte_carlo_score,
            "long_term_gap": self.long_term_gap,
            "hard_protected_labels": list(self.hard_protected_labels),
            "soft_protected_labels": list(self.soft_protected_labels),
            "warnings": list(self.warnings),
        }


def _sample_hand(rng: random.Random, hand_size: int) -> list[str]:
    size = min(max(1, hand_size), len(DECK))
    return rng.sample(DECK, size)


def _mc_score_for_label(rows, label: str | None) -> float | None:
    if label is None:
        return None
    for row in rows:
        if row.label == label:
            return float(row.avg_score)
    return None


def _decision_warnings(decision, allocation, mc_best_label: str | None, long_term_gap: float | None) -> list[str]:
    warnings: list[str] = []
    hard_ids = set(allocation.hard_protected_instances)
    if (
        decision.selected_action == "DISCARD"
        and decision.selected_card_id in hard_ids
        and decision.candidate_stage != "forced_break_hard_protection"
    ):
        warnings.append("policy_discarded_hard_protected")
    if long_term_gap is not None and long_term_gap < -150:
        warnings.append("monte_carlo_prefers_different_discard")
    if decision.selected_action == "DISCARD" and mc_best_label and decision.selected_label != mc_best_label:
        warnings.append("policy_mc_label_mismatch")
    return warnings


def _as_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _raise_weight(current: float, *, pct: float, minimum_delta: float, cap: float) -> float:
    return round(min(cap, max(current + minimum_delta, current * (1.0 + pct))), 3)


def derive_weight_adjustments(
    report: dict[str, Any],
    *,
    weights: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert self-play evidence into conservative weight updates."""
    weights = weights or {}
    count = max(1, int(report.get("count") or 0))
    label_mismatches = int(report.get("label_mismatches") or 0)
    mismatch_rate = label_mismatches / count
    hard_violations = len(report.get("hard_violations") or [])
    avg_gap = _as_float(report.get("avg_long_term_gap"), 0.0)
    worst_gap = _as_float(report.get("worst_long_term_gap"), 0.0)
    updates: list[dict[str, Any]] = []

    def add_update(key: str, *, fallback: float, pct: float, minimum_delta: float, cap: float, reason: str) -> None:
        old_value = _as_float(weights.get(key), fallback)
        new_value = _raise_weight(old_value, pct=pct, minimum_delta=minimum_delta, cap=cap)
        if new_value <= old_value:
            return
        updates.append(
            {
                "key": key,
                "old_value": round(old_value, 3),
                "new_value": new_value,
                "delta": round(new_value - old_value, 3),
                "reason": reason,
            }
        )

    if hard_violations:
        add_update(
            "hard_protected_break_penalty",
            fallback=10000,
            pct=0.20,
            minimum_delta=1000,
            cap=30000,
            reason=f"self_play found {hard_violations} hard protected discard violations",
        )
        add_update(
            "exact_triplet_break_penalty",
            fallback=3000,
            pct=0.10,
            minimum_delta=300,
            cap=12000,
            reason="raise exact triplet protection after hard protected violation",
        )

    if mismatch_rate >= 0.25 or worst_gap <= -150:
        add_update(
            "ting_improvement",
            fallback=120,
            pct=0.15,
            minimum_delta=20,
            cap=600,
            reason=(
                f"monte_carlo mismatch_rate={mismatch_rate:.2f} "
                f"worst_gap={round(worst_gap, 3)}"
            ),
        )
        add_update(
            "weak_potential_discard_bonus",
            fallback=120,
            pct=0.10,
            minimum_delta=10,
            cap=500,
            reason="favor discarding weak potentials when long-horizon simulation disagrees",
        )

    if avg_gap <= -75:
        add_update(
            "orphan_discard_bonus",
            fallback=300,
            pct=0.10,
            minimum_delta=20,
            cap=900,
            reason=f"avg_long_term_gap={round(avg_gap, 3)} suggests policy needs more discard mobility",
        )

    return {
        "metrics": {
            "count": count,
            "label_mismatches": label_mismatches,
            "mismatch_rate": round(mismatch_rate, 3),
            "hard_violations": hard_violations,
            "avg_long_term_gap": round(avg_gap, 3),
            "worst_long_term_gap": round(worst_gap, 3),
        },
        "suggested_weight_updates": updates,
    }


def apply_weight_adjustments(
    *,
    config_path: Path,
    output_path: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Write a tuned rules file from self-play calibration evidence."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    weights = config.setdefault("weights", {})
    adjustments = derive_weight_adjustments(report, weights=weights)
    applied: list[dict[str, Any]] = []
    for update in adjustments["suggested_weight_updates"]:
        key = str(update["key"])
        weights[key] = update["new_value"]
        if key in config.get("ai_weights", {}):
            config["ai_weights"][key] = update["new_value"]
        applied.append(update)
    config.setdefault("calibration", {})["last_self_play_adjustment"] = {
        "generated_at": datetime.now().isoformat(),
        "source": "run_self_play_calibration",
        "metrics": adjustments["metrics"],
        "applied_updates": applied,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return {
        "ok": True,
        "config_path": str(config_path),
        "output_path": str(output_path),
        "applied_updates": applied,
        "metrics": adjustments["metrics"],
    }


def run_self_play_calibration(
    *,
    count: int = 30,
    hand_size: int = 14,
    simulations: int = 40,
    seed: int = 20260531,
    output_dir: Path | None = None,
    config_path: str = "config/rules.yaml",
) -> dict[str, Any]:
    rng = random.Random(seed)
    rows: list[SelfPlayDecision] = []
    for index in range(1, count + 1):
        hand = _sample_hand(rng, hand_size)
        state = {
            "context_id": f"self_play_{index:04d}",
            "hand": hand,
            "legal_actions": [{"type": "DISCARD"}],
            "remaining_deck_count": max(0, len(DECK) - hand_size - index),
            "phase": "self_play_calibration",
        }
        context = build_decision_context(state, config_path=config_path)
        allocation = allocate_hand_structures(context)
        decision = choose_action(state, config_path=config_path)
        mc_rows = simulate_discards(hand, simulations=simulations, seed=seed + index, config_path=config_path)
        mc_best = mc_rows[0] if mc_rows else None
        policy_mc_score = _mc_score_for_label(mc_rows, decision.selected_label)
        best_score = float(mc_best.avg_score) if mc_best is not None else None
        long_term_gap = (
            None
            if policy_mc_score is None or best_score is None
            else round(policy_mc_score - best_score, 3)
        )
        warnings = _decision_warnings(
            decision,
            allocation,
            mc_best.label if mc_best is not None else None,
            long_term_gap,
        )
        rows.append(
            SelfPlayDecision(
                index=index,
                hand=hand,
                policy_action=decision.action,
                policy_label=decision.selected_label,
                candidate_stage=decision.candidate_stage,
                policy_ev=round(decision.ev, 3),
                monte_carlo_best_label=mc_best.label if mc_best is not None else None,
                monte_carlo_best_score=round(best_score, 3) if best_score is not None else None,
                policy_monte_carlo_score=round(policy_mc_score, 3) if policy_mc_score is not None else None,
                long_term_gap=long_term_gap,
                hard_protected_labels=allocation.hard_protected_labels,
                soft_protected_labels=allocation.soft_protected_labels,
                warnings=warnings,
            )
        )
    hard_violations = [
        row.to_dict()
        for row in rows
        if "policy_discarded_hard_protected" in row.warnings
    ]
    label_mismatches = sum(1 for row in rows if "policy_mc_label_mismatch" in row.warnings)
    gaps = [row.long_term_gap for row in rows if row.long_term_gap is not None]
    report = {
        "ok": not hard_violations,
        "generated_at": datetime.now().isoformat(),
        "count": count,
        "hand_size": hand_size,
        "simulations": simulations,
        "seed": seed,
        "hard_violations": hard_violations,
        "label_mismatches": label_mismatches,
        "avg_long_term_gap": round(sum(gaps) / len(gaps), 3) if gaps else None,
        "worst_long_term_gap": min(gaps) if gaps else None,
        "rows": [row.to_dict() for row in rows],
    }
    report["calibration_adjustments"] = derive_weight_adjustments(
        report,
        weights=(context.rules.get("weights") if rows else None),
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "self_play_calibration.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "self_play_weight_adjustments.json").write_text(
            json.dumps(report["calibration_adjustments"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_markdown_report(report, output_dir / "self_play_calibration.md")
    return report


def _write_markdown_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Self-Play Calibration",
        "",
        f"- ok: {report['ok']}",
        f"- decisions: {report['count']}",
        f"- simulations: {report['simulations']}",
        f"- seed: {report['seed']}",
        f"- label_mismatches: {report['label_mismatches']}",
        f"- avg_long_term_gap: {report['avg_long_term_gap']}",
        f"- worst_long_term_gap: {report['worst_long_term_gap']}",
        "",
        "| # | hand | policy | stage | mc_best | gap | warnings |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        warnings = ",".join(row["warnings"])
        lines.append(
            f"| {row['index']} | {' '.join(row['hand'])} | "
            f"{row['policy_action']} {row.get('policy_label') or ''} | "
            f"{row['candidate_stage']} | {row.get('monte_carlo_best_label') or ''} | "
            f"{row.get('long_term_gap')} | {warnings} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_training_example(path: str | Path, example: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(example, ensure_ascii=False) + "\n")
