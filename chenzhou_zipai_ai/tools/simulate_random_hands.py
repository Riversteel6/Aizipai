"""Generate random hands and look for obvious professional-brain violations."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.pro_brain import allocate_hand_structures, build_decision_context, choose_action
from engine.cards import BIG_LABELS, SMALL_LABELS, WILD_LABEL


DECK = [*SMALL_LABELS, *BIG_LABELS] * 4 + [WILD_LABEL] * 4


def simulate_random_hands(count: int = 100, hand_size: int = 14, seed: int = 20260531) -> dict:
    rng = random.Random(seed)
    violations: list[dict] = []
    for index in range(count):
        hand = rng.sample(DECK, hand_size)
        context = build_decision_context(hand)
        allocation = allocate_hand_structures(context)
        decision = choose_action(hand)
        if decision.selected_card_id in set(allocation.hard_protected_instances):
            violations.append(
                {
                    "index": index,
                    "hand": hand,
                    "reason": "discarded_hard_protected",
                    "decision": decision.to_dict(),
                    "allocation": allocation.to_dict(),
                }
            )
    return {
        "ok": not violations,
        "hands": count,
        "violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-hand sanity runner.")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--hand-size", type=int, default=14)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = simulate_random_hands(args.count, args.hand_size, args.seed)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"simulate_random_hands ok={result['ok']} hands={result['hands']} violations={len(result['violations'])}")
        for item in result["violations"][:10]:
            print(f"  {item['index']}: {item['reason']} hand={' '.join(item['hand'])}")
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
