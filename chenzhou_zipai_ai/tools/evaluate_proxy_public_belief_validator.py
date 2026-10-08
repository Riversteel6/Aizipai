"""Run the frozen public-belief validator with validated full-action proxies."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.evaluate_anchored_structural_discard_proxies import sha256_file
from tools.evaluate_hybrid_response_proxies import HYBRID_RESPONSE_PROXY_TYPES
from tools.evaluate_public_belief_validator import run_preregistered_validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    raw = args.preregistration.read_bytes()
    preregistration = json.loads(raw.decode("utf-8"))
    verify_capability_reports(preregistration["inputs"])
    report = run_preregistered_validation(
        preregistration,
        preregistration_path=args.preregistration,
        preregistration_sha256=hashlib.sha256(raw).hexdigest(),
        progress_every=max(0, int(args.progress_every)),
        profile_factories=HYBRID_RESPONSE_PROXY_TYPES,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "focused_gate_pass": report["focused_gate_pass"],
                "states": report["states"],
                "health_errors": report["health_errors"],
                "candidate_harmful_overrides": report["candidate"][
                    "harmful_overrides"
                ],
                "anchor_harmful_overrides": report["anchor"]["harmful_overrides"],
                "candidate_missed_overrides": report["candidate"]["missed_overrides"],
                "candidate_confidently_positive_overrides": report["candidate"][
                    "confidently_positive_overrides"
                ],
                "candidate_mean_regret": report["candidate"][
                    "mean_regret_to_heldout_best"
                ],
                "p95_elapsed_ms": report["p95_elapsed_ms"],
                "failures": report["focused_gate_failures"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def verify_capability_reports(inputs: Mapping[str, Any]) -> None:
    prefixes = ["belief", "discard_proxy", "response_proxy"]
    prefixes.extend(
        prefix
        for prefix in ("implementation_equivalence", "heavy_state_gate")
        if f"{prefix}_report" in inputs
    )
    for prefix in prefixes:
        path = Path(str(inputs[f"{prefix}_report"]))
        if sha256_file(path) != str(inputs[f"{prefix}_report_sha256"]):
            raise ValueError(f"proxy_public_belief_{prefix}_hash_mismatch")
        report = json.loads(path.read_text(encoding="utf-8"))
        if not bool(report.get("focused_gate_pass")):
            raise ValueError(f"proxy_public_belief_{prefix}_did_not_pass")


if __name__ == "__main__":
    raise SystemExit(main())
