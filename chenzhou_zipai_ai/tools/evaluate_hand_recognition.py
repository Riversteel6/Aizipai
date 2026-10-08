"""Measure screenshot-only hand recognition agreement with protocol-bound action frames."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.hand_recognizer import clear_classification_cache, read_image, recognize_hand


def collect_protocol_bound_frames(logs_root: str | Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for events_path in Path(logs_root).rglob("session_events.jsonl"):
        session_root = events_path.parent
        event_by_frame: dict[str, dict[str, Any]] = {}
        try:
            with events_path.open("r", encoding="utf-8") as file:
                for line in file:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("event_type") != "FRAME_RECOGNIZED":
                        continue
                    data = event.get("data") or {}
                    details = data.get("hand_details") or []
                    if not details or not all(item.get("source") == "protocol_overlay" for item in details):
                        continue
                    frame_id = str(event.get("frame_id") or "")
                    if frame_id:
                        event_by_frame.setdefault(frame_id, data)
        except OSError:
            continue

        for frame_id, data in event_by_frame.items():
            screenshots = list(
                (session_root / "rounds").glob(f"*/screenshots/{frame_id}_before_action.png")
            )
            if not screenshots:
                continue
            details = data.get("hand_details") or []
            frames.append(
                {
                    "screenshot": screenshots[0],
                    "truth": list(data.get("normalized_hand") or data.get("raw_hand") or []),
                    "fully_coordinate_bound": all(
                        item.get("x") is not None and item.get("y") is not None for item in details
                    ),
                    "fully_exact_label_bound": all(
                        item.get("x") is not None
                        and item.get("y") is not None
                        and item.get("protocol_binding") == "exact_label"
                        for item in details
                    ),
                }
            )
    return sorted(frames, key=lambda item: str(item["screenshot"]))


def evaluate_frames(
    frames: list[dict[str, Any]],
    *,
    config_path: str | Path,
) -> dict[str, Any]:
    clear_classification_cache()
    results: list[dict[str, Any]] = []
    for item in frames:
        image = read_image(item["screenshot"])
        started = perf_counter()
        cards = recognize_hand(image, config_path=config_path)
        elapsed_ms = (perf_counter() - started) * 1000
        truth = list(item["truth"])
        prediction = [card.name for card in cards]
        missing = list((Counter(truth) - Counter(prediction)).elements())
        extra = list((Counter(prediction) - Counter(truth)).elements())
        results.append(
            {
                "fully_coordinate_bound": bool(item["fully_coordinate_bound"]),
                "fully_exact_label_bound": bool(item["fully_exact_label_bound"]),
                "exact": not missing and not extra,
                "truth_count": len(truth),
                "prediction_count": len(prediction),
                "missing_count": len(missing),
                "extra_count": len(extra),
                "elapsed_ms": elapsed_ms,
            }
        )

    visible = [item for item in results if item["fully_coordinate_bound"]]
    exact_bound = [item for item in results if item["fully_exact_label_bound"]]
    elapsed = [float(item["elapsed_ms"]) for item in results]
    return {
        "reference": "protocol_overlay_agreement_not_manual_ground_truth",
        "frames": len(results),
        "strict_exact_frames": sum(bool(item["exact"]) for item in results),
        "strict_exact_rate": _rate(sum(bool(item["exact"]) for item in results), len(results)),
        "fully_coordinate_bound_frames": len(visible),
        "fully_coordinate_bound_exact_frames": sum(bool(item["exact"]) for item in visible),
        "fully_coordinate_bound_exact_rate": _rate(
            sum(bool(item["exact"]) for item in visible),
            len(visible),
        ),
        "fully_exact_label_bound_frames": len(exact_bound),
        "fully_exact_label_bound_exact_frames": sum(bool(item["exact"]) for item in exact_bound),
        "fully_exact_label_bound_exact_rate": _rate(
            sum(bool(item["exact"]) for item in exact_bound),
            len(exact_bound),
        ),
        "count_exact_frames": sum(
            item["truth_count"] == item["prediction_count"] for item in results
        ),
        "under_count_frames": sum(
            item["prediction_count"] < item["truth_count"] for item in results
        ),
        "over_count_frames": sum(
            item["prediction_count"] > item["truth_count"] for item in results
        ),
        "latency_ms": {
            "mean": round(statistics.fmean(elapsed), 3) if elapsed else 0.0,
            "median": round(statistics.median(elapsed), 3) if elapsed else 0.0,
            "p95": round(_percentile(elapsed, 0.95), 3) if elapsed else 0.0,
            "max": round(max(elapsed), 3) if elapsed else 0.0,
        },
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", default=str(ROOT / "logs"))
    parser.add_argument("--config", default=str(ROOT / "config" / "screen_1080x2400.yaml"))
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--output")
    args = parser.parse_args()

    frames = collect_protocol_bound_frames(args.logs_root)
    if args.limit > 0 and len(frames) > args.limit:
        if args.limit == 1:
            frames = frames[:1]
        else:
            frames = [
                frames[round(index * (len(frames) - 1) / (args.limit - 1))]
                for index in range(args.limit)
            ]
    report = evaluate_frames(frames, config_path=args.config)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
