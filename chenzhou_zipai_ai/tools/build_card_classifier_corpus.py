"""Extract conservative pseudo-labelled card crops from bounded device frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.cards import BIG_LABELS, SMALL_LABELS, WILD_LABEL
from vision.hand_recognizer import _candidate_scores, detect_hand_slots, load_templates, read_image


VALID_LABELS = {*SMALL_LABELS, *BIG_LABELS, WILD_LABEL}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data/card_classifier_corpus")
    parser.add_argument("--config", type=Path, default=ROOT / "config/screen_1080x2400.yaml")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--exclude-frame", action="append", default=[])
    parser.add_argument("--minimum-score", type=float, default=0.82)
    parser.add_argument("--minimum-margin", type=float, default=0.15)
    parser.add_argument("--max-per-label", type=int, default=240)
    args = parser.parse_args()

    frames = sorted(args.session_root.rglob("*_before_action.png"))
    excluded = {str(value) for value in args.exclude_frame}
    frames = [frame for frame in frames if frame.stem not in excluded]
    if args.limit > 0:
        frames = frames[: args.limit]
    templates = load_templates(ROOT / "data/templates/hand_auto", grayscale=False)
    args.output.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    manifest: list[dict[str, object]] = []
    rejected_low_score = 0
    rejected_low_margin = 0
    for frame in frames:
        image = read_image(frame)
        slots = detect_hand_slots(image, config_path=args.config)
        for slot in slots:
            crop = image[slot.y : slot.y + slot.h, slot.x : slot.x + slot.w]
            scores = _candidate_scores(crop, templates)
            if not scores:
                continue
            score, label, template = scores[0]
            runner_up = next(
                (
                    value
                    for value, other_label, _other_template in scores[1:]
                    if other_label != label
                ),
                0.0,
            )
            if label not in VALID_LABELS or score < args.minimum_score:
                rejected_low_score += 1
                continue
            if score - runner_up < args.minimum_margin:
                rejected_low_margin += 1
                continue
            if counts[label] >= max(1, args.max_per_label):
                continue
            digest = hashlib.blake2b(
                np.ascontiguousarray(cv2.resize(crop, (48, 48))).tobytes(),
                digest_size=12,
            ).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            label_dir = args.output / label
            label_dir.mkdir(parents=True, exist_ok=True)
            target = label_dir / f"{frame.stem}_{digest}.png"
            ok, encoded = cv2.imencode(".png", crop)
            if not ok:
                raise RuntimeError(f"card_crop_encode_failed:{frame}")
            encoded.tofile(str(target))
            counts[label] += 1
            manifest.append(
                {
                    "file": str(target),
                    "source_frame": str(frame),
                    "label": label,
                    "template": template,
                    "template_score": round(float(score), 6),
                    "template_margin": round(float(score - runner_up), 6),
                    "slot": {
                        "x": slot.x,
                        "y": slot.y,
                        "w": slot.w,
                        "h": slot.h,
                        "source": slot.source,
                    },
                }
            )
    report = {
        "schema_version": "aizipai-card-corpus-v1",
        "source_session": str(args.session_root),
        "frames_scanned": len(frames),
        "excluded_frames": sorted(excluded),
        "minimum_score": args.minimum_score,
        "minimum_margin": args.minimum_margin,
        "accepted_crops": len(manifest),
        "accepted_by_label": dict(sorted(counts.items())),
        "labels_covered": len(counts),
        "rejected_low_score": rejected_low_score,
        "rejected_low_margin": rejected_low_margin,
        "rows": manifest,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: report[key] for key in ("frames_scanned", "accepted_crops", "labels_covered", "accepted_by_label")},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
