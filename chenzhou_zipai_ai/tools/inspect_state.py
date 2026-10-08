"""Inspect current game state from a screenshot or read-only ADB capture."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
SRC = WORKSPACE / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aizipai.rules.engine import RuleEngine
from aizipai.state.models import ActionType, ButtonState, Card, Color, GameState, Phase, Suit
from capture.adb_capture import save_screen
from vision.button_detector import ButtonDetection, detect_buttons
from vision.count_detector import detect_remaining_deck_count
from vision.discard_button_detector import detect_discard_button
from vision.discard_recognizer import recognize_discards
from vision.hand_recognizer import HandCard, read_image, recognize_hand
from vision.neural_card_classifier import DEFAULT_MODEL_PATH
from vision.history_memory import (
    VisionMemory,
    build_sanity_checks,
    enrich_meld_groups,
    load_memory,
    save_memory,
)
from vision.meld_recognizer import recognize_meld_groups
from vision.option_detector import OptionCandidate, detect_option_candidates
from vision.option_recognizer import recognize_options
from vision.pending_card_recognizer import PendingCard, recognize_pending_cards
from vision.regions import load_regions_for_size
from vision.seat_role_detector import detect_seat_role
from engine.cards import normalize_card_label

SMALL_LABELS = "一二三四五六七八九十"
BIG_LABELS = "壹贰叁肆伍陆柒捌玖拾"
RED_RANKS = {2, 7, 10}
RESPONSE_BUTTONS = {"chi", "peng", "pao", "hu", "pass"}
_VISION_EXECUTOR = ThreadPoolExecutor(max_workers=3, thread_name_prefix="aizipai-vision")


class VisionPipelineTimeout(RuntimeError):
    """A bounded vision branch failed to return before the frame deadline."""


def _vision_timeout_seconds() -> float:
    configured = os.environ.get("AIZIPAI_VISION_TIMEOUT_SECONDS", "15")
    try:
        return max(0.05, float(configured))
    except ValueError:
        return 15.0


def _await_parallel_vision(
    futures: tuple[tuple[str, Future], ...],
    *,
    timeout_seconds: float | None = None,
) -> dict[str, object]:
    timeout = _vision_timeout_seconds() if timeout_seconds is None else max(0.05, timeout_seconds)
    deadline = time.monotonic() + timeout
    results: dict[str, object] = {}
    try:
        for label, future in futures:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise FutureTimeoutError()
            try:
                results[label] = future.result(timeout=remaining)
            except FutureTimeoutError as error:
                raise VisionPipelineTimeout(
                    f"vision_pipeline_timeout branch={label} timeout_seconds={timeout:.2f}"
                ) from error
        return results
    finally:
        for _, future in futures:
            if not future.done():
                future.cancel()


def _runtime_hand_hybrid_mode() -> str:
    configured = os.environ.get("AIZIPAI_CARD_HYBRID_MODE")
    if configured:
        return configured
    return "shadow" if DEFAULT_MODEL_PATH.exists() else "off"


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    try:
        if candidate.exists():
            return candidate
    except OSError:
        pass
    for base in (ROOT, WORKSPACE):
        resolved = base / candidate
        if resolved.exists():
            return resolved
    return candidate


def _select_pending_action_card(
    pending_action_card: PendingCard | None,
    opponent_pending_card: PendingCard | None,
    buttons: list[ButtonDetection],
) -> PendingCard | None:
    button_names = {button.name for button in buttons}
    if button_names & RESPONSE_BUTTONS and opponent_pending_card is not None:
        if pending_action_card is None:
            return opponent_pending_card
        if opponent_pending_card.confidence >= 0.75:
            return opponent_pending_card
        if opponent_pending_card.confidence >= pending_action_card.confidence + 0.12:
            return opponent_pending_card
    return pending_action_card


def _is_own_draw_pending_card(
    pending_action_card: PendingCard | None,
    buttons: list[ButtonDetection],
    discard_button: ButtonDetection | None,
) -> bool:
    if pending_action_card is None:
        return False
    if pending_action_card.region_name != "pending_action_card":
        return False
    if pending_action_card.confidence < 0.75:
        return False
    button_names = {button.name for button in buttons}
    return not bool(button_names & RESPONSE_BUTTONS) and discard_button is None


def _pending_card_as_hand_card(card: PendingCard, index: int) -> dict:
    return {
        "name": normalize_card_label(card.name),
        "card_id": f"h{index:03d}",
        "confidence": card.confidence,
        "x": card.x,
        "y": card.y,
        "w": card.w,
        "h": card.h,
        "center": card.center,
        "template": card.template,
        "clickable": True,
        "source": "pending_action_card",
    }
BUTTON_ACTIONS = {
    "chi": ActionType.CHI,
    "peng": ActionType.PENG,
    "pass": ActionType.PASS,
    "hu": ActionType.HU,
    "pao": ActionType.PAO,
    "ti": ActionType.TI,
}


def card_from_label(label: str) -> Card:
    normalized = normalize_card_label(label)
    if normalized == "王":
        return Card(rank=1, suit=Suit.SMALL, color=Color.RED, is_wildcard=True)
    if normalized in SMALL_LABELS:
        rank = SMALL_LABELS.index(normalized) + 1
        return Card(rank=rank, suit=Suit.SMALL, color=_color_for_rank(rank))
    if normalized in BIG_LABELS:
        rank = BIG_LABELS.index(normalized) + 1
        return Card(rank=rank, suit=Suit.BIG, color=_color_for_rank(rank))
    raise ValueError(f"Unknown card label: {label} normalized={normalized}")


def _safe_card_from_label(label: str) -> Card | None:
    try:
        return card_from_label(label)
    except ValueError:
        return None


def _norm(label: str) -> str:
    return normalize_card_label(label) or label.strip()


def _color_for_rank(rank: int) -> Color:
    return Color.RED if rank in RED_RANKS else Color.BLACK


def _button_states(buttons: list[ButtonDetection]) -> list[ButtonState]:
    states: list[ButtonState] = []
    for button in buttons:
        action = BUTTON_ACTIONS.get(button.name)
        if action is None:
            continue
        states.append(
            ButtonState(
                action=action,
                visible=True,
                confidence=button.confidence,
                bbox=(button.x, button.y, button.w, button.h),
            )
        )
    return states


def _metadata(
    *,
    screenshot: Path,
    hand: list[HandCard],
    buttons: list[ButtonDetection],
    options: list[OptionCandidate],
    discard_button_visible: bool,
    opponent_priority_pending: bool,
) -> dict:
    raw_labels = [card.name for card in hand]
    labels = [normalize_card_label(card.name) for card in hand]
    counts = Counter(labels)
    return {
        "screenshot_path": str(screenshot),
        "vision_confidence": min((card.confidence for card in hand), default=0.0),
        "hand_count": len(hand),
        "hand_labels": labels,
        "button_names": [button.name for button in buttons],
        "discard_button_visible": discard_button_visible,
        "option_count": len(options),
        "opponent_priority_pending": opponent_priority_pending,
        "auto_quad_pending": any(count >= 4 for count in counts.values()),
    }


def inspect_screenshot(
    screenshot: str | Path,
    *,
    config_path: str | Path = "config/screen_1080x2400.yaml",
    assume_my_turn: bool = False,
    opponent_priority_pending: bool = False,
    expected_total: int | None = None,
    infer_expected_total_from_phase: bool = False,
    timings: dict[str, float] | None = None,
    precomputed_buttons: list[ButtonDetection] | None = None,
    precomputed_hand_labels: list[str] | None = None,
) -> dict:
    def timed(label: str, operation):
        started = time.perf_counter()
        value = operation()
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        if timings is not None:
            timings[label] = elapsed_ms
        return value

    screenshot_path = Path(screenshot)
    config_path = _resolve_project_path(config_path)
    image = timed("read_image_ms", lambda: read_image(screenshot_path))
    hand_recognition: dict[str, object] = {}
    parallel_vision = os.environ.get("AIZIPAI_PARALLEL_VISION") == "1"
    def hand_operation(expected_hand_count: int | None = None):
        if precomputed_hand_labels is not None:
            hand_recognition.update(
                {
                    "recognition_source": "confirmed_hand_ledger",
                    "ledger_reuse": True,
                    "expected_hand_count": expected_hand_count,
                }
            )
            return [
                HandCard(
                    name=str(label),
                    confidence=1.0,
                    x=index,
                    y=0,
                    w=1,
                    h=1,
                    template="confirmed_hand_ledger",
                    # A response action never taps the hand, but the policy's
                    # follow-up evaluator must still treat confirmed cards as
                    # available. False here changes CHI/PASS semantics.
                    clickable=True,
                    card_id=f"ledger_{index + 1:03d}",
                    recognition_source="confirmed_hand_ledger",
                )
                for index, label in enumerate(precomputed_hand_labels)
            ]
        return recognize_hand(
            image,
            config_path=config_path,
            expected_hand_count=expected_hand_count,
            diagnostics=hand_recognition,
            hybrid_mode=_runtime_hand_hybrid_mode(),
        )
    option_operation = lambda: (
        timed(
            "option_candidates_ms",
            lambda: detect_option_candidates(image, config_path=config_path),
        ),
        timed(
            "option_recognition_ms",
            lambda: recognize_options(image, config_path=config_path),
        ),
    )
    pending_operation = lambda: recognize_pending_cards(image, config_path=config_path)
    surface_first = infer_expected_total_from_phase
    if parallel_vision and not surface_first:
        hand_future = _VISION_EXECUTOR.submit(timed, "hand_ms", hand_operation)
        options_future = _VISION_EXECUTOR.submit(option_operation)
        pending_future = _VISION_EXECUTOR.submit(timed, "pending_cards_ms", pending_operation)
    if precomputed_buttons is None:
        buttons = timed("buttons_ms", lambda: detect_buttons(image, config_path=config_path))
    else:
        buttons = list(precomputed_buttons)
        if timings is not None:
            timings["buttons_ms"] = 0.0
    discard_button = timed(
        "discard_button_ms",
        lambda: detect_discard_button(image, config_path=config_path),
    )
    own_discard_surface = bool(discard_button is not None and not buttons)
    if parallel_vision and surface_first and not own_discard_surface:
        hand_future = _VISION_EXECUTOR.submit(timed, "hand_ms", hand_operation)
        options_future = _VISION_EXECUTOR.submit(option_operation)
        pending_future = _VISION_EXECUTOR.submit(
            timed,
            "pending_cards_ms",
            pending_operation,
        )
    if parallel_vision and (not surface_first or not own_discard_surface):
        vision_results = _await_parallel_vision(
            (
                ("hand", hand_future),
                ("options", options_future),
                ("pending_cards", pending_future),
            )
        )
        hand = vision_results["hand"]
        options, option_details = vision_results["options"]
        pending_cards = vision_results["pending_cards"]
    else:
        if surface_first and own_discard_surface:
            options, option_details = [], []
            pending_cards = {
                "pending_action_card": None,
                "opponent_pending_card": None,
            }
        else:
            options, option_details = option_operation()
            pending_cards = timed("pending_cards_ms", pending_operation)
    detected_pending_action_card = pending_cards.get("pending_action_card")
    opponent_pending_card = pending_cards.get("opponent_pending_card")
    pending_action_card = _select_pending_action_card(detected_pending_action_card, opponent_pending_card, buttons)
    remaining_deck_count = timed(
        "remaining_count_ms",
        lambda: detect_remaining_deck_count(image, config_path=config_path),
    )
    seat_role = timed("seat_role_ms", lambda: detect_seat_role(image, config_path=config_path))
    discards = timed("discards_ms", lambda: recognize_discards(image, config_path=config_path))
    meld_groups = timed(
        "meld_groups_ms",
        lambda: recognize_meld_groups(image, config_path=config_path),
    )
    melds = {
        region_name: [cell for group in groups for cell in group]
        for region_name, groups in meld_groups.items()
    }
    meld_cell_count = sum(
        len(group)
        for group in meld_groups.get("my_melds", [])
    )
    effective_expected_total = expected_total
    if effective_expected_total is None and infer_expected_total_from_phase:
        effective_expected_total = _expected_total_from_surface(
            buttons=buttons,
            discard_button=discard_button,
            options=option_details or options,
        )
    if surface_first and own_discard_surface:
        expected_hand_count = (
            max(0, effective_expected_total - meld_cell_count)
            if effective_expected_total is not None
            else None
        )
        hand = timed("hand_ms", lambda: hand_operation(expected_hand_count))
    elif not parallel_vision:
        hand = timed("hand_ms", hand_operation)
    controlled_count = len(hand) + meld_cell_count
    if (
        effective_expected_total is not None
        and 1 <= effective_expected_total - controlled_count <= 3
        and discard_button is not None
        and not buttons
    ):
        expected_visible_hand = max(0, effective_expected_total - meld_cell_count)
        recovered_diagnostics: dict[str, object] = {}
        recovered_hand = timed(
            "hand_recovery_ms",
            lambda: recognize_hand(
                image,
                config_path=config_path,
                recover_missing_top_row=True,
                expected_hand_count=expected_visible_hand,
                diagnostics=recovered_diagnostics,
                hybrid_mode="enforce",
            ),
        )
        recovered_controlled_count = len(recovered_hand) + meld_cell_count
        if controlled_count < recovered_controlled_count <= effective_expected_total:
            hand = recovered_hand
            hand_recognition = recovered_diagnostics
            hand_recognition["recovery_pass"] = "partial_occlusion_dual_agreement"
    phase = Phase.AWAIT_ACTION if assume_my_turn or (discard_button is not None and not buttons) else Phase.WAITING
    raw_hand_labels = [card.name for card in hand]
    normalized_hand_labels = [normalize_card_label(name) for name in raw_hand_labels]
    state = GameState(
        round_id=screenshot_path.stem,
        seat=0,
        turn=0,
        phase=phase,
        hand=[card for card in ( _safe_card_from_label(name) for name in normalized_hand_labels ) if card is not None],
        pending_action_card=_safe_card_from_label(pending_action_card.name) if pending_action_card else None,
        opponent_pending_card=_safe_card_from_label(opponent_pending_card.name) if opponent_pending_card else None,
        buttons=_button_states(buttons),
        metadata=_metadata(
            screenshot=screenshot_path,
            hand=[
                HandCard(
                    name=label,
                    confidence=card.confidence,
                    x=card.x,
                    y=card.y,
                    w=card.w,
                    h=card.h,
                    template=card.template,
                    clickable=card.clickable,
                    raw_confidence=card.raw_confidence,
                    runner_up_name=card.runner_up_name,
                    runner_up_confidence=card.runner_up_confidence,
                    recognition_source=card.recognition_source,
                )
                for card, label in zip(hand, normalized_hand_labels)
            ],
            buttons=buttons,
            options=options,
            discard_button_visible=discard_button is not None,
            opponent_priority_pending=opponent_priority_pending,
        ),
    )
    state.metadata["remaining_deck_count"] = remaining_deck_count
    state.metadata["seat_role"] = seat_role.to_dict()
    state.metadata["hand_recognition"] = hand_recognition
    state.metadata["pending_action_card"] = pending_action_card.to_dict() if pending_action_card else None
    state.metadata["detected_pending_action_card"] = (
        detected_pending_action_card.to_dict() if detected_pending_action_card else None
    )
    state.metadata["opponent_pending_card"] = opponent_pending_card.to_dict() if opponent_pending_card else None
    legal_actions = RuleEngine().legal_actions(state)
    meld_group_dict = {
        name: [[cell.to_dict() for cell in group] for group in groups]
        for name, groups in meld_groups.items()
    }
    hand_details = _ensure_card_id_prefix(hand)
    hand_details = [
        {**item, "name": normalize_card_label(item["name"])} for item in hand_details
    ]
    low_confidence_threshold = 0.70
    recognition_warnings = [
        warning
        for item in hand_details
        if (warning := _low_confidence_warning(item, threshold=low_confidence_threshold)) is not None
    ]
    result = {
        "screenshot": str(screenshot_path),
        "phase": state.phase,
        "raw_hand": raw_hand_labels,
        "hand": normalized_hand_labels,
        "hand_details": hand_details,
        "hand_count": len(hand),
        "hand_recognition": hand_recognition,
        "remaining_deck_count": remaining_deck_count,
        "seat_role": seat_role.to_dict(),
        "pending_action_card": pending_action_card.to_dict() if pending_action_card else None,
        "detected_pending_action_card": detected_pending_action_card.to_dict() if detected_pending_action_card else None,
        "opponent_pending_card": opponent_pending_card.to_dict() if opponent_pending_card else None,
        "discards": {name: [card.to_dict() for card in cards] for name, cards in discards.items()},
        "melds": {name: [cell.to_dict() for cell in cells] for name, cells in melds.items()},
        "meld_groups": meld_group_dict,
        "meld_group_summaries": enrich_meld_groups(meld_group_dict),
        "buttons": [button.to_dict() for button in buttons],
        "discard_button": discard_button.to_dict() if discard_button else None,
        "options": [option.to_dict() for option in options],
        "option_details": _normalize_option_details(option_details),
        "metadata": state.metadata,
        "legal_actions": [action.to_dict() for action in legal_actions],
    }
    result["flow_state"] = "play"
    result["recognition_warnings"] = recognition_warnings
    result["recognition_errors"] = []
    result["sanity_checks"] = build_sanity_checks(result, expected_total=effective_expected_total)
    return result


def _low_confidence_warning(item: dict, *, threshold: float) -> str | None:
    confidence = float(item.get("confidence", 0.0))
    if confidence >= threshold:
        return None
    if "secondary_grid_" in str(item.get("template") or ""):
        return None
    if (
        item.get("recognition_source") == "hybrid"
        and item.get("hybrid_reason") == "independent_agreement"
    ):
        return None
    return f"low_confidence:{item['name']}:{confidence:.2f}"


def _expected_total_from_surface(
    *,
    buttons: list[ButtonDetection],
    discard_button: object | None,
    options: list[object],
) -> int:
    button_names = {str(button.name).lower() for button in buttons}
    response_visible = bool(button_names & RESPONSE_BUTTONS)
    own_discard_turn = discard_button is not None and not response_visible and not options
    return 21 if own_discard_turn else 20


def _normalize_option_details(options: list[object]) -> list[dict]:
    normalized: list[dict] = []
    for option in options:
        if isinstance(option, dict):
            details = option
        else:
            details = option.to_dict()
        labels = details.get("labels", [])
        if not isinstance(labels, (list, tuple)):
            labels = [labels]
        normalized.append(
            {
                **details,
                "labels": [normalize_card_label(item) or item for item in labels],
            }
        )
    return normalized


def _ensure_card_id_prefix(hand: list[HandCard]) -> list[dict]:
    result: list[dict] = []
    for index, card in enumerate(hand, start=1):
        payload = card.to_dict()
        if not payload.get("card_id"):
            payload["card_id"] = f"h{index:03d}"
        result.append(payload)
    return result


def write_preview(
    screenshot: str | Path,
    result: dict,
    *,
    config_path: str | Path = "config/screen_1080x2400.yaml",
    output: str | Path | None = None,
    max_width: int = 1000,
) -> Path:
    image = read_image(screenshot)
    height, width = image.shape[:2]
    for region in load_regions_for_size(config_path, width, height):
        if region.name not in {
            "my_melds",
            "opponent_melds",
            "remaining_deck_count",
            "opponent_discards",
            "my_discards",
        }:
            continue
        color = (255, 0, 255) if region.name == "remaining_deck_count" else (255, 180, 0)
        if region.name in {"opponent_discards", "my_discards"}:
            color = (60, 220, 255)
        cv2.rectangle(image, (region.x, region.y), (region.x2, region.y2), color, 3)
        cv2.putText(
            image,
            region.name,
            (region.x, max(24, region.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    for index, card in enumerate(result["hand_details"], 1):
        x, y, w, h = int(card["x"]), int(card["y"]), int(card["w"]), int(card["h"])
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 4)
        label_origin = (x, max(26, y - 10))
        cv2.rectangle(
            image,
            (label_origin[0] - 3, label_origin[1] - 22),
            (label_origin[0] + 34, label_origin[1] + 6),
            (255, 0, 255),
            -1,
        )
        cv2.putText(
            image,
            str(index),
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(index),
            (20, 40 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    for button in result["buttons"]:
        x, y, w, h = int(button["x"]), int(button["y"]), int(button["w"]), int(button["h"])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(
            image,
            button["name"],
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    for option in result["options"]:
        x, y, w, h = int(option["x"]), int(option["y"]), int(option["w"]), int(option["h"])
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 3)
        label = f"{option['region_name']}#{option['index']}"
        cv2.putText(
            image,
            label,
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    for option in result.get("option_details", []):
        label = "".join(option["labels"])
        cv2.putText(
            image,
            label,
            (int(option["x"]), int(option["y"] + option["h"] + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    for cards in result["discards"].values():
        for card in cards:
            x, y, w, h = int(card["x"]), int(card["y"]), int(card["w"]), int(card["h"])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 180, 255), 2)
            cv2.putText(
                image,
                card["name"],
                (x, max(22, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )

    for cells in result["melds"].values():
        for cell in cells:
            x, y, w, h = int(cell["x"]), int(cell["y"]), int(cell["w"]), int(cell["h"])
            color = (0, 0, 255) if cell["hidden"] else (0, 180, 255)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                image,
                cell["name"],
                (x, max(22, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

    if image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        image = cv2.resize(image, (max_width, round(image.shape[0] * scale)))

    output_path = Path(output) if output is not None else _default_preview_path(screenshot)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError(f"Failed to write preview: {output_path}")
    return output_path


def _default_preview_path(screenshot: str | Path) -> Path:
    return Path("data/crops/previews") / f"{Path(screenshot).stem}_state_preview.jpg"


def _print_summary(result: dict) -> None:
    print(f"screenshot={result['screenshot']}")
    print(f"phase={result['phase']}")
    print(f"hand_count={result['hand_count']}")
    print(f"remaining_deck_count={result['remaining_deck_count']}")
    print("hand=" + " ".join(result["hand"]))
    for region_name, cards in result["discards"].items():
        print(f"{region_name}=" + " ".join(card["name"] for card in cards))
    for region_name, cells in result["melds"].items():
        print(f"{region_name}=" + " ".join(cell["name"] for cell in cells))
    for region_name, groups in result["meld_groups"].items():
        print(
            f"{region_name}_groups="
            + " | ".join(" ".join(cell["name"] for cell in group) for group in groups)
        )
    for region_name, groups in result["meld_group_summaries"].items():
        print(
            f"{region_name}_types="
            + " | ".join(f"{' '.join(group['labels'])}:{group['type']}" for group in groups)
        )
    checks = result["sanity_checks"]
    expected = checks["expected_total"] if checks["expected_total"] is not None else "unset"
    print(
        "sanity="
        + f"controlled={checks['controlled_card_count']} expected={expected} ok={checks['ok']}"
    )
    for warning in checks["warnings"]:
        print(f"warning={warning}")
    print("buttons=" + " ".join(item["name"] for item in result["buttons"]))
    print(
        "options="
        + " ".join(
            f"{item['region_name']}#{item['index']}:{item['card_count']}"
            for item in result["options"]
        )
    )
    print(
        "option_details="
        + " ".join(
            f"{item['region_name']}#{item['index']}:{''.join(item['labels'])}"
            for item in result.get("option_details", [])
        )
    )
    print(
        "metadata="
        + " ".join(
            f"{key}={value}"
            for key, value in result["metadata"].items()
            if key in {"opponent_priority_pending", "auto_quad_pending", "option_count"}
        )
    )
    print("legal_actions=" + " ".join(item["type"] for item in result["legal_actions"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect game state from screenshot vision.")
    parser.add_argument("screenshot", nargs="?", help="Screenshot path. Omit with --adb.")
    parser.add_argument("--adb", action="store_true", help="Capture a fresh screenshot via ADB first.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="data/screenshots")
    parser.add_argument("--config", default="config/screen_1080x2400.yaml")
    parser.add_argument("--assume-my-turn", action="store_true")
    parser.add_argument("--opponent-priority-pending", action="store_true")
    parser.add_argument("--expected-total", type=int, default=None)
    parser.add_argument("--memory-file", default=None)
    parser.add_argument("--reset-memory", action="store_true")
    parser.add_argument("--preview", action="store_true", help="Write a small annotated JPG preview.")
    parser.add_argument("--preview-out", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.adb:
        screenshot = save_screen(args.out, device_id=args.device)
    elif args.screenshot:
        screenshot = Path(args.screenshot)
    else:
        raise SystemExit("Provide a screenshot path or use --adb.")

    result = inspect_screenshot(
        screenshot,
        config_path=args.config,
        assume_my_turn=args.assume_my_turn,
        opponent_priority_pending=args.opponent_priority_pending,
        expected_total=args.expected_total,
    )
    if args.memory_file:
        memory = load_memory(args.memory_file) if not args.reset_memory else VisionMemory()
        memory.update_from_snapshot(result)
        save_memory(memory, args.memory_file)
        result["memory"] = memory.to_dict()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_summary(result)
        if args.memory_file:
            memory = result["memory"]
            print(f"memory_file={args.memory_file}")
            print(f"memory_frames_seen={memory['frames_seen']}")
            print("memory_my_discards=" + " ".join(memory["my_discards"]))
            print("memory_opponent_discards=" + " ".join(memory["opponent_discards"]))
            print(
                "memory_my_meld_groups="
                + " | ".join(" ".join(group) for group in memory["my_meld_groups"])
            )
            print(
                "memory_opponent_meld_groups="
                + " | ".join(" ".join(group) for group in memory["opponent_meld_groups"])
            )
    if args.preview:
        preview = write_preview(screenshot, result, config_path=args.config, output=args.preview_out)
        print(f"preview={preview}")


if __name__ == "__main__":
    main()
