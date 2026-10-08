"""Audit an offline policy variant against captured production discard roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.ismcts import public_view_from_dict
from ai.opponent_league import (
    POLICY_FACTORIES,
    _summarize_discard_shadow,
    _wildcard_route_bonus,
    create_policy,
)
from engine.cards import WILD_LABEL
from engine.rules import rules_for_room


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--candidate", choices=sorted(POLICY_FACTORIES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-limit", type=int, default=40)
    args = parser.parse_args()
    report = audit_variant(
        args.input,
        candidate=args.candidate,
        sample_limit=max(0, args.sample_limit),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "ok",
                    "candidate",
                    "roots",
                    "overrides",
                    "override_rate",
                    "errors",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def audit_variant(
    paths: list[Path],
    *,
    candidate: str,
    sample_limit: int,
) -> dict[str, Any]:
    policy = create_policy(candidate)
    roots: dict[str, dict[str, Any]] = {}
    for path in paths:
        source = json.loads(path.read_text(encoding="utf-8"))
        for row in source.get("rows") or []:
            for event in row.get("candidate_discard_events") or []:
                if not event.get("eligible") or not event.get("public_view"):
                    continue
                state_id = _state_id(event)
                roots.setdefault(
                    state_id,
                    {
                        "state_id": state_id,
                        "source_report": str(path),
                        "players": int(event["players"]),
                        "wildcard_enabled": bool(event["wildcard_enabled"]),
                        "production_label": str(event["production_label"]),
                        "heuristic_gap": float(event["heuristic_gap"]),
                        "public_view": event["public_view"],
                    },
                )
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for root in roots.values():
        view = public_view_from_dict(root["public_view"])
        rules = rules_for_room(
            wildcard_enabled=bool(root["wildcard_enabled"]),
            players=int(root["players"]),
        )
        production = str(root["production_label"])
        bypass_variant = _should_bypass_variant(policy, view)
        if bypass_variant:
            label = production
        else:
            try:
                label = policy.choose_discard(view, rules)
            except ValueError as exc:
                errors.append({"state_id": root["state_id"], "error": str(exc)})
                continue
        route_bonus = (
            _wildcard_route_bonus(
                view,
                label,
                rules,
                per_step=float(getattr(policy, "route_bonus_per_step", 0.0)),
            )
            if hasattr(policy, "route_bonus_per_step")
            else 0.0
        )
        rows.append(
            {
                **{key: value for key, value in root.items() if key != "public_view"},
                "candidate_label": label,
                "override": label != production,
                "candidate_route_bonus": route_bonus,
                "hand": list(view.hand),
                "stock_count": view.stock_count,
            }
        )
    overrides = [row for row in rows if row["override"]]
    by_mode = Counter(
        f"{row['players']}p_{'wang' if row['wildcard_enabled'] else 'no_wang'}"
        for row in overrides
    )
    diagnostics = (
        policy.diagnostics()
        if callable(getattr(policy, "diagnostics", None))
        else {}
    )
    discard_events = (
        list(policy.discard_events())
        if callable(getattr(policy, "discard_events", None))
        else []
    )
    return {
        "ok": not errors,
        "candidate": candidate,
        "source_reports": [str(path) for path in paths],
        "roots": len(rows),
        "overrides": len(overrides),
        "override_rate": round(len(overrides) / len(rows), 6) if rows else 0.0,
        "by_mode": dict(sorted(by_mode.items())),
        "candidate_diagnostics": diagnostics,
        "discard_shadow": _summarize_discard_shadow(discard_events),
        "errors": len(errors),
        "error_samples": errors[:sample_limit],
        "override_samples": sorted(
            overrides,
            key=lambda row: (row["heuristic_gap"], row["state_id"]),
        )[:sample_limit],
    }


def _state_id(event: dict[str, Any]) -> str:
    payload = {
        "public_view": event["public_view"],
        "production_label": event["production_label"],
        "candidates": event.get("candidates") or [],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _should_bypass_variant(policy: Any, view: Any) -> bool:
    return (
        bool(getattr(policy, "require_wildcard_in_hand", False))
        and WILD_LABEL not in view.hand
    )


if __name__ == "__main__":
    raise SystemExit(main())
