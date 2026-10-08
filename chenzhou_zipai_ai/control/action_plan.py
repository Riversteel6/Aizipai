"""Build non-executing click plans from vision decisions."""

from __future__ import annotations

from collections import Counter

from engine.cards import label_for, normalize_card_label, normalize_cards, parse_label
from engine.melds import classify_meld


BUTTON_ACTIONS = {"hu", "peng", "pass"}
AUTO_MELD_ACTIONS = {"pao", "ti", "ming_long", "long", "wei", "auto_quad", "wait_auto_meld"}
LEGAL_CHI_OPTION_KINDS = {"sequence", "special_123", "special_2710", "mixed_same_rank"}
PLAY_ACTIONS = {
    "hu",
    "peng",
    "chi",
    "expand_chi_options",
    "pass",
    "discard",
    "chi_option",
    "compare_option",
}
AUXILIARY_ACTIONS = {"settlement_ready", "confirm_discard_warning", "compact_hand"}
EXECUTABLE_ACTIONS = PLAY_ACTIONS | AUXILIARY_ACTIONS
EXECUTION_MODES = {"tap_sequence", "discard_drag", "select_then_discard_button", "drag_sequence"}


def action_plan_contract_error(plan: dict) -> str | None:
    """Return the first semantic error that makes a play plan unsafe to execute."""

    if not isinstance(plan, dict):
        return "action_plan_not_mapping"
    action = str(plan.get("action") or "").lower()
    if not action:
        return "action_plan_missing_action"
    ready = plan.get("ready") is True
    validation = plan.get("validation")
    validation_passed = isinstance(validation, dict) and validation.get("passed") is True
    clicks = plan.get("clicks")

    if not ready:
        if action in EXECUTABLE_ACTIONS and validation_passed:
            return "unready_play_action_marked_valid"
        return None
    if action not in EXECUTABLE_ACTIONS:
        return "ready_non_play_action"
    if not isinstance(clicks, list) or not clicks:
        return "ready_plan_missing_clicks"
    if not validation_passed:
        return "ready_plan_validation_failed"
    if validation.get("target_matches_policy") is False:
        return "action_plan_policy_mismatch"
    if action in PLAY_ACTIONS:
        policy = plan.get("policy_selected_action")
        policy_action = str(policy.get("type") or "").lower() if isinstance(policy, dict) else ""
        if policy_action != action or validation.get("target_matches_policy") is not True:
            return "action_plan_policy_mismatch"
    execution_mode = str(plan.get("execution_mode") or "tap_sequence")
    if execution_mode not in EXECUTION_MODES:
        return "action_plan_execution_mode_invalid"
    for click in clicks:
        click_error = _click_contract_error(click, execution_mode=execution_mode)
        if click_error is not None:
            return click_error
    return None


def _click_contract_error(click: object, *, execution_mode: str) -> str | None:
    if not isinstance(click, dict):
        return "action_plan_click_not_mapping"
    if not str(click.get("target") or "").strip():
        return "action_plan_click_target_missing"
    required_coordinates = ["x", "y"]
    if execution_mode == "drag_sequence":
        required_coordinates.extend(["from_x", "from_y"])
    for key in required_coordinates:
        try:
            coordinate = int(click.get(key))
        except (TypeError, ValueError):
            return f"action_plan_click_{key}_invalid"
        if not 0 <= coordinate <= 10_000:
            return f"action_plan_click_{key}_out_of_bounds"
    try:
        delay_ms = int(click.get("delay_ms", 80))
    except (TypeError, ValueError):
        return "action_plan_click_delay_invalid"
    if not 0 <= delay_ms <= 5_000:
        return "action_plan_click_delay_out_of_bounds"
    if execution_mode in {"drag_sequence", "discard_drag"}:
        try:
            duration_ms = int(click.get("duration_ms", 520))
        except (TypeError, ValueError):
            return "action_plan_drag_duration_invalid"
        if not 1 <= duration_ms <= 5_000:
            return "action_plan_drag_duration_out_of_bounds"
    return None


def build_action_plan(state: dict, decision: dict) -> dict:
    action = decision["action"]
    candidate_stage = decision.get("candidate_stage")
    break_hard_protection = bool(decision.get("break_hard_protection"))
    hard_protected = set(decision.get("hard_protected", []))
    decision_label = normalize_card_label(decision.get("label")) if decision.get("label") else None
    decision_card_id = decision.get("selected_card_id") or decision.get("card_id")

    if action in AUTO_MELD_ACTIONS:
        return _wait_auto_meld_plan(action, decision)

    if action == "expand_chi_options":
        return _expand_chi_options_plan(state, decision)

    if action == "chi":
        return _chi_plan(state, decision)

    if action == "compare_option":
        return _compare_option_plan(state, decision)

    if action in BUTTON_ACTIONS:
        button = _find_button(state, action)
        if not button:
            return _empty_plan(
                action=action,
                reason=f"未找到 {action} 按钮",
                reason_code="action_plan_policy_mismatch",
            )
        return _click_plan(
            action=action,
            target_type="button",
            target_label=action,
            target_card_id=None,
            clicks=[_center_click(button, target=f"button:{action}")],
            policy_selected_action={"type": action, "label": None},
        )

    if action == "discard" and decision_label:
        if not _is_discard_actionable(state):
            return _empty_plan(
                action=action,
                reason="discard_not_actionable_without_turn_signal",
                reason_code="discard_not_actionable_without_turn_signal",
                target_label=decision_label,
            )
        if decision_label in hard_protected and candidate_stage != "forced_break_hard_protection":
            return _empty_plan(
                action=action,
                reason="blocked_discard_hard_protected",
                reason_code="blocked_discard_hard_protection",
                target_label=decision_label,
            )
        card = _find_hand_card(state, decision_label, card_id=decision_card_id)
        if not card:
            return _empty_plan(
                action=action,
                reason="selected_card_not_clickable",
                reason_code="selected_card_not_clickable",
                target_label=decision_label,
            )
        if candidate_stage == "forced_break_hard_protection":
            clicks, execution_mode = _discard_execution_clicks(state, card, decision_label)
            reason = (
                f"hard_protected={sorted(hard_protected)}; "
                f"break_hard_protection={break_hard_protection}; "
                f"break_reason={decision.get('break_reason')}"
            )
            return _click_plan(
                action=action,
                target_type="hand_card",
                target_label=decision_label,
                target_card_id=card.get("card_id"),
                clicks=clicks,
                policy_selected_action={"type": "discard", "label": decision_label},
                extra_reason=reason,
                execution_mode=execution_mode,
            )
        clicks, execution_mode = _discard_execution_clicks(state, card, decision_label)
        return _click_plan(
            action=action,
            target_type="hand_card",
            target_label=decision_label,
            target_card_id=card.get("card_id"),
            clicks=clicks,
            policy_selected_action={"type": "discard", "label": decision_label},
            extra_reason=f"点击手牌 {decision_label}",
            execution_mode=execution_mode,
        )

    return _empty_plan(
        action=action,
        reason="无需点击或暂不支持",
        reason_code="action_plan_policy_mismatch",
    )


def _empty_plan(
    *,
    action: str,
    reason: str,
    reason_code: str = "action_plan_policy_mismatch",
    target_label: str | None = None,
) -> dict:
    return {
        "action": action,
        "ready": False,
        "reason": reason,
        "target_type": None,
        "target_label": target_label,
        "target_card_id": None,
        "clicks": [],
        "policy_selected_action": {"type": action, "label": target_label if action == "discard" else None},
        "validation": {
            "passed": False,
            "reason_code": reason_code,
            "checks": ["action_plan_policy_mismatch"],
        },
    }


def _chi_plan(state: dict, decision: dict) -> dict:
    option_candidates = _option_candidates(state, "chi_options")
    if not option_candidates:
        return _empty_plan(
            action="chi",
            reason="chi_options_not_visible",
            reason_code="chi_options_not_visible",
        )

    semantic_option = _find_semantic_option_intent(option_candidates, decision)
    if semantic_option is not None:
        button = _find_button(state, "chi")
        if not button:
            return _empty_plan(
                action="chi",
                reason="未找到 chi 按钮",
                reason_code="button_not_found",
            )
        option_labels = repair_chi_option_labels(_option_labels(semantic_option), state)
        guard_reason = chi_option_guard_reason(option_labels, state)
        if guard_reason is not None:
            return _empty_plan(
                action="chi",
                reason=describe_chi_option_guard(guard_reason, option_labels, state),
                reason_code=guard_reason,
                target_label="".join(option_labels),
            )
        option_id = _option_id(
            semantic_option,
            "chi_options",
            option_candidates.index(semantic_option) + 1,
        )
        return _click_plan(
            action="chi",
            target_type="button",
            target_label="chi",
            target_card_id=None,
            clicks=[_center_click(button, target="button:chi")],
            policy_selected_action={"type": "chi", "label": None},
            extra_reason=(
                f"策略已按规则选定吃牌 {''.join(option_labels)}，"
                "展开候选仅用于映射屏幕位置"
            ),
            target_option_id=option_id,
            target_option_cards=option_labels,
        )

    selected_option = _find_selected_option(option_candidates, decision, region_name="chi_options")
    if selected_option is None:
        return _empty_plan(
            action="chi",
            reason="selected_option_not_clickable",
            reason_code="selected_option_not_clickable",
        )

    option_id = _option_id(selected_option, "chi_options", option_candidates.index(selected_option) + 1)
    option_labels = repair_chi_option_labels(_option_labels(selected_option) or _selected_option_cards(decision), state)
    guard_reason = chi_option_guard_reason(option_labels, state)
    if guard_reason is not None:
        return _empty_plan(
            action="chi",
            reason=describe_chi_option_guard(guard_reason, option_labels, state),
            reason_code=guard_reason,
            target_label="".join(option_labels) if option_labels else option_id,
        )
    option_target = f"chi:{''.join(option_labels)}" if option_labels else f"chi:{option_id}"
    return _click_plan(
        action="chi_option",
        target_type="option_column",
        target_label="".join(option_labels) if option_labels else None,
        target_card_id=None,
        clicks=[_center_click(selected_option, target=option_target)],
        policy_selected_action={
            "type": "chi_option",
            "label": None,
            "option_id": option_id,
            "option_cards": option_labels,
        },
        extra_reason=f"吃牌候选已展开，选择候选 {''.join(option_labels)}",
        target_option_id=option_id,
        target_option_cards=option_labels,
    )


def _expand_chi_options_plan(state: dict, decision: dict) -> dict:
    button = _find_button(state, "chi")
    if not button:
        return _empty_plan(
            action="expand_chi_options",
            reason="未找到 chi 按钮",
            reason_code="button_not_found",
        )
    if _option_candidates(state, "chi_options"):
        return _empty_plan(
            action="expand_chi_options",
            reason="chi_options_already_visible",
            reason_code="chi_options_already_visible",
        )
    return _click_plan(
        action="expand_chi_options",
        target_type="button",
        target_label="chi",
        target_card_id=None,
        clicks=[_center_click(button, target="button:chi")],
        policy_selected_action={"type": "expand_chi_options", "label": None},
        extra_reason="展开吃牌候选，仅用于评估，不代表选择吃牌",
    )


def _compare_option_plan(state: dict, decision: dict) -> dict:
    option_candidates = _option_candidates(state, "compare_options")
    if not option_candidates:
        return _empty_plan(
            action="compare_option",
            reason="selected_option_not_clickable",
            reason_code="selected_option_not_clickable",
        )

    selected_option = _find_selected_option(option_candidates, decision, region_name="compare_options")
    if selected_option is None:
        return _empty_plan(
            action="compare_option",
            reason="selected_option_not_clickable",
            reason_code="selected_option_not_clickable",
        )

    option_id = _option_id(selected_option, "compare_options", option_candidates.index(selected_option) + 1)
    option_labels = _option_labels(selected_option)
    option_target = f"compare:{''.join(option_labels)}" if option_labels else f"compare:{option_id}"
    return _click_plan(
        action="compare_option",
        target_type="option_column",
        target_label="".join(option_labels) if option_labels else None,
        target_card_id=None,
        clicks=[_center_click(selected_option, target=option_target)],
        policy_selected_action={
            "type": "compare_option",
            "label": None,
            "option_id": option_id,
            "option_cards": option_labels,
        },
        extra_reason=f"选择比牌/下火候选 {''.join(option_labels)}",
        target_option_id=option_id,
        target_option_cards=option_labels,
    )


def _wait_auto_meld_plan(action: str, decision: dict) -> dict:
    auto_label = normalize_card_label(decision.get("label")) if decision.get("label") else None
    return {
        "action": "wait_auto_meld",
        "ready": False,
        "reason": decision.get("reason", "等待自动动作完成"),
        "target_type": None,
        "target_label": auto_label,
        "target_card_id": None,
        "clicks": [],
        "policy_selected_action": {"type": action, "label": auto_label},
        "validation": {
            "passed": False,
            "reason_code": "wait_auto_meld",
            "checks": ["wait_auto_meld"],
        },
    }


def _click_plan(
    *,
    action: str,
    target_type: str,
    target_label: str | None,
    target_card_id: str | None,
    clicks: list[dict],
    policy_selected_action: dict[str, str | None],
    extra_reason: str | None = None,
    target_option_id: str | None = None,
    target_option_cards: list[str] | None = None,
    execution_mode: str = "tap_sequence",
) -> dict:
    target_matches_policy = policy_selected_action.get("type") == action and (
        policy_selected_action.get("label") is None or policy_selected_action.get("label") == target_label
    )
    target_clickable = bool(clicks) and all(
        isinstance(click, dict) and click.get("x") is not None and click.get("y") is not None
        for click in clicks
    )
    return {
        "action": action,
        "ready": True,
        "reason": extra_reason or f"点击 {target_type} {target_label or ''}".strip(),
        "execution_mode": execution_mode,
        "target_type": target_type,
        "target_label": target_label,
        "target_card_id": target_card_id,
        "target_option_id": target_option_id,
        "target_option_cards": list(target_option_cards or []),
        "clicks": clicks,
        "policy_selected_action": policy_selected_action,
        "validation": {
            "passed": target_clickable
            and target_matches_policy
            and (bool(target_card_id) if target_type == "hand_card" else True),
            "checks": [
                "target_type",
                "target_matches_policy",
                "target_clickable",
            ],
            "target_matches_policy": target_matches_policy,
        },
    }


def _find_button(state: dict, name: str) -> dict | None:
    button = next(
        (
            button
            for button in state.get("buttons", [])
            if (button.get("name") or button.get("type")) == name
        ),
        None,
    )
    if button is not None:
        return button
    if name == "hu" and "hu" in _legal_action_types(state) and _protocol_fallback_buttons_trusted(state):
        return _fallback_protocol_hu_button(state)
    return None


def _protocol_fallback_buttons_trusted(state: dict) -> bool:
    metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
    return bool(metadata.get("allow_protocol_fallback_buttons"))


def _legal_action_types(state: dict) -> set[str]:
    result: set[str] = set()
    for item in state.get("legal_actions") or []:
        value = item.get("type") if isinstance(item, dict) else item
        if value:
            result.add(str(value).lower())
    return result


def _fallback_protocol_hu_button(state: dict) -> dict:
    legal = _legal_action_types(state)
    center_x = 1745 if legal & {"chi", "peng"} else 1985
    center_y = 510
    width = 150
    height = 170
    return {
        "name": "hu",
        "source": "protocol_fallback_hu_button",
        "confidence": 0.75,
        "x": int(center_x - width / 2),
        "y": int(center_y - height / 2),
        "w": width,
        "h": height,
        "center": [center_x, center_y],
    }


def _option_candidates(state: dict, region_name: str) -> list[dict]:
    explicit_key = "chi_options" if region_name == "chi_options" else "compare_options"
    explicit = state.get(explicit_key) or []
    if explicit:
        return [item for item in explicit if isinstance(item, dict)]
    return [
        item
        for item in state.get("option_details", [])
        if isinstance(item, dict) and item.get("region_name") == region_name
    ]


def _find_selected_option(options: list[dict], decision: dict, *, region_name: str) -> dict | None:
    selected_option_id = (
        decision.get("selected_option_id")
        or decision.get("option_id")
        or (decision.get("selected_action") or {}).get("option_id")
    )
    selected_cards = _selected_option_cards(decision)
    for index, option in enumerate(options, start=1):
        option_id = _option_id(option, region_name, index)
        if selected_option_id and option_id == str(selected_option_id):
            return option if _has_click_box(option) else None
    if selected_cards:
        for option in options:
            if _option_labels(option) == selected_cards:
                return option if _has_click_box(option) else None
    return None


def _find_semantic_option_intent(options: list[dict], decision: dict) -> dict | None:
    selected_option_id = (
        decision.get("selected_option_id")
        or decision.get("option_id")
        or (decision.get("selected_action") or {}).get("option_id")
    )
    selected_cards = _selected_option_cards(decision)
    for index, option in enumerate(options, start=1):
        if not option.get("semantic_only"):
            continue
        option_id = _option_id(option, "chi_options", index)
        if selected_option_id and option_id == str(selected_option_id):
            return option
        if selected_cards and _option_labels(option) == selected_cards:
            return option
    return None


def _selected_option_cards(decision: dict) -> list[str]:
    for value in (
        decision.get("selected_option_cards"),
        decision.get("option_cards"),
        (decision.get("selected_action") or {}).get("option_cards"),
    ):
        if isinstance(value, list) and value:
            return normalize_cards([str(item) for item in value])
    selected_eval = decision.get("selected_action_eval") or {}
    for value in (
        selected_eval.get("option_cards"),
        (selected_eval.get("action") or {}).get("option_cards"),
    ):
        if isinstance(value, list) and value:
            return normalize_cards([str(item) for item in value])
    return []


def _option_labels(option: dict) -> list[str]:
    labels = option.get("labels") or option.get("cards") or option.get("option_cards") or []
    return normalize_cards([str(item) for item in labels]) if isinstance(labels, list) else []


def repair_chi_option_labels(labels: list[str], state: dict) -> list[str]:
    option_labels = normalize_cards([str(item) for item in labels])
    if chi_option_guard_reason(option_labels, state) is None:
        return option_labels
    repaired = _repair_uniform_chi_size_mismatch(option_labels, state)
    if repaired and chi_option_guard_reason(repaired, state) is None:
        return repaired
    repaired = _repair_same_rank_chi_size_mismatch(option_labels, state)
    if repaired and chi_option_guard_reason(repaired, state) is None:
        return repaired
    repaired = _repair_pending_same_rank_chi_misread(option_labels, state)
    if repaired and chi_option_guard_reason(repaired, state) is None:
        return repaired
    repaired = _repair_pending_sequence_chi_misread(option_labels, state)
    if repaired and chi_option_guard_reason(repaired, state) is None:
        return repaired
    return option_labels


def chi_option_guard_reason(labels: list[str], state: dict) -> str | None:
    option_labels = normalize_cards([str(item) for item in labels])
    pending = _pending_chi_card(state)
    if len(option_labels) == 3:
        pattern = classify_meld(option_labels)
        if pattern.kind not in LEGAL_CHI_OPTION_KINDS:
            return "invalid_chi_option"
        if pending and pending not in option_labels:
            return "chi_option_missing_pending_card"
        return None
    if len(option_labels) == 2:
        if not pending:
            return "chi_option_pending_card_missing"
        for index in range(3):
            candidate = option_labels[:index] + [pending] + option_labels[index:]
            if classify_meld(candidate).kind in LEGAL_CHI_OPTION_KINDS:
                return None
        return "invalid_chi_option"
    return "invalid_chi_option"


def describe_chi_option_guard(reason_code: str, labels: list[str], state: dict) -> str:
    label_text = "".join(normalize_cards([str(item) for item in labels])) or "空候选"
    pending = _pending_chi_card(state)
    if reason_code == "chi_option_pending_card_missing":
        return f"吃牌候选 {label_text} 缺少待吃牌上下文，禁止点击"
    if reason_code == "chi_option_missing_pending_card":
        return f"吃牌候选 {label_text} 不包含待吃牌 {pending}，禁止点击"
    return f"吃牌候选 {label_text} 不是合法吃牌牌型，禁止点击"


def _repair_uniform_chi_size_mismatch(labels: list[str], state: dict) -> list[str] | None:
    if len(labels) != 3:
        return None
    parsed = [parse_label(label) for label in labels]
    if any(item is None for item in parsed):
        return None
    ranks = [item[1] for item in parsed if item is not None]
    sorted_ranks = sorted(ranks)
    is_sequence_like = (
        (len(set(sorted_ranks)) == 3 and sorted_ranks == list(range(sorted_ranks[0], sorted_ranks[0] + 3)))
        or sorted_ranks in ([1, 2, 3], [2, 7, 10])
    )
    if not is_sequence_like:
        return None

    hand_counts = Counter(_state_hand_labels(state))
    pending = _pending_chi_card(state)
    candidates: list[tuple[int, list[str]]] = []
    for suit in ("small", "big"):
        candidate = [label_for(suit, rank) for rank in ranks]
        candidate_counts = Counter(candidate)
        if pending in candidate_counts:
            candidate_counts[pending] -= 1
            if candidate_counts[pending] <= 0:
                candidate_counts.pop(pending, None)
        hand_match = sum(min(count, hand_counts.get(label, 0)) for label, count in candidate_counts.items())
        required_matches = max(1, len(candidate_counts)) if pending else min(2, len(candidate_counts))
        if hand_match >= required_matches:
            candidates.append((hand_match, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return None
    return candidates[0][1]


def _repair_same_rank_chi_size_mismatch(labels: list[str], state: dict) -> list[str] | None:
    if len(labels) != 3:
        return None
    parsed = [parse_label(label) for label in labels]
    if any(item is None for item in parsed):
        return None
    ranks = {item[1] for item in parsed if item is not None}
    suits = {item[0] for item in parsed if item is not None}
    if len(ranks) != 1 or len(suits) != 1:
        return None
    rank = next(iter(ranks))
    observed_suit = next(iter(suits))
    opposite_suit = "small" if observed_suit == "big" else "big"
    observed_label = label_for(observed_suit, rank)
    opposite_label = label_for(opposite_suit, rank)
    hand_counts = Counter(_state_hand_labels(state))
    pending = _pending_chi_card(state)
    candidate_orders = (
        [
            [opposite_label, opposite_label, observed_label],
            [opposite_label, observed_label, observed_label],
            [observed_label, observed_label, opposite_label],
            [observed_label, opposite_label, opposite_label],
        ]
        if observed_suit == "big"
        else [
            [observed_label, opposite_label, opposite_label],
            [observed_label, observed_label, opposite_label],
            [opposite_label, opposite_label, observed_label],
            [opposite_label, observed_label, observed_label],
        ]
    )
    candidates: list[tuple[int, int, list[str]]] = []
    seen: set[tuple[str, ...]] = set()
    for candidate in candidate_orders:
        key = tuple(candidate)
        if key in seen:
            continue
        seen.add(key)
        if classify_meld(candidate).kind not in LEGAL_CHI_OPTION_KINDS:
            continue
        candidate_counts = Counter(candidate)
        if pending and pending not in candidate_counts:
            continue
        if pending in candidate_counts:
            candidate_counts[pending] -= 1
            if candidate_counts[pending] <= 0:
                candidate_counts.pop(pending, None)
        hand_match = sum(min(count, hand_counts.get(label, 0)) for label, count in candidate_counts.items())
        required_matches = max(1, len(candidate_counts)) if pending else min(2, len(candidate_counts))
        if hand_match >= required_matches:
            pending_bonus = 1 if pending else 0
            candidates.append((hand_match, pending_bonus, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _repair_pending_same_rank_chi_misread(labels: list[str], state: dict) -> list[str] | None:
    if len(labels) != 3:
        return None
    pending = _pending_chi_card(state)
    pending_parsed = parse_label(pending) if pending else None
    if pending_parsed is None:
        return None
    parsed = [parse_label(label) for label in labels]
    if any(item is None for item in parsed):
        return None
    pending_rank = pending_parsed[1]
    observed_same_rank = sum(1 for item in parsed if item is not None and item[1] == pending_rank)
    if observed_same_rank < 2:
        return None

    small_label = label_for("small", pending_rank)
    big_label = label_for("big", pending_rank)
    hand_counts = Counter(_state_hand_labels(state))
    candidate_orders = [
        [small_label, small_label, big_label],
        [small_label, big_label, big_label],
        [big_label, big_label, small_label],
        [big_label, small_label, small_label],
    ]
    candidates: list[tuple[int, int, list[str]]] = []
    for candidate in candidate_orders:
        if classify_meld(candidate).kind not in LEGAL_CHI_OPTION_KINDS:
            continue
        candidate_counts = Counter(candidate)
        if pending not in candidate_counts:
            continue
        candidate_counts[pending] -= 1
        if candidate_counts[pending] <= 0:
            candidate_counts.pop(pending, None)
        hand_match = sum(min(count, hand_counts.get(label, 0)) for label, count in candidate_counts.items())
        required_matches = sum(candidate_counts.values())
        if hand_match < required_matches:
            continue
        exact_position_matches = sum(1 for left, right in zip(labels, candidate) if left == right)
        candidates.append((exact_position_matches, hand_match, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _repair_pending_sequence_chi_misread(labels: list[str], state: dict) -> list[str] | None:
    if len(labels) != 3:
        return None
    pending = _pending_chi_card(state)
    pending_parsed = parse_label(pending) if pending else None
    if pending_parsed is None:
        return None
    parsed = [parse_label(label) for label in labels]
    if any(item is None for item in parsed):
        return None
    pending_suit, pending_rank = pending_parsed
    hand_counts = Counter(_state_hand_labels(state))
    candidate_orders: list[list[str]] = []
    for start_rank in range(pending_rank - 2, pending_rank + 1):
        ranks = [start_rank, start_rank + 1, start_rank + 2]
        if ranks[0] < 1 or ranks[-1] > 10:
            continue
        candidate = [label_for(pending_suit, rank) for rank in ranks]
        if pending in candidate:
            candidate_orders.append(candidate)
    for ranks in ([1, 2, 3], [2, 7, 10]):
        if pending_rank in ranks:
            candidate = [label_for(pending_suit, rank) for rank in ranks]
            if pending in candidate and candidate not in candidate_orders:
                candidate_orders.append(candidate)

    observed_ranks = Counter(item[1] for item in parsed if item is not None)
    candidates: list[tuple[int, int, int, list[str]]] = []
    for candidate in candidate_orders:
        if classify_meld(candidate).kind not in LEGAL_CHI_OPTION_KINDS:
            continue
        candidate_counts = Counter(candidate)
        candidate_counts[pending] -= 1
        if candidate_counts[pending] <= 0:
            candidate_counts.pop(pending, None)
        hand_match = sum(min(count, hand_counts.get(label, 0)) for label, count in candidate_counts.items())
        required_matches = sum(candidate_counts.values())
        if hand_match < required_matches:
            continue
        candidate_ranks = Counter(parse_label(label)[1] for label in candidate if parse_label(label) is not None)
        rank_overlap = sum(min(count, observed_ranks.get(rank, 0)) for rank, count in candidate_ranks.items())
        if rank_overlap < 1:
            continue
        exact_position_matches = sum(1 for left, right in zip(labels, candidate) if left == right)
        candidates.append((exact_position_matches, rank_overlap, hand_match, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates[0][3]


def _state_hand_labels(state: dict) -> list[str]:
    labels: list[str] = []
    for key in ("hand", "normalized_hand", "raw_hand"):
        value = state.get(key)
        if isinstance(value, list):
            labels.extend(normalize_cards([str(item) for item in value]))
    for item in state.get("hand_details") or []:
        if isinstance(item, dict):
            label = normalize_card_label(str(item.get("name") or item.get("label") or ""))
            if label:
                labels.append(label)
    return labels


def _pending_chi_card(state: dict) -> str:
    candidates = [
        state.get("pending_card"),
        state.get("pending_action_card"),
        state.get("opponent_pending_card"),
        state.get("source_card"),
        state.get("external_card"),
        state.get("drawn_card"),
        (state.get("metadata") or {}).get("pending_card"),
        (state.get("metadata") or {}).get("source_card"),
    ]
    for value in candidates:
        label = _normalize_state_card(value)
        if label:
            return label
    return ""


def _normalize_state_card(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("name", "label", "card", "cardval", "value"):
            label = _normalize_state_card(value.get(key))
            if label:
                return label
        return ""
    if isinstance(value, (list, tuple)):
        for item in value:
            label = _normalize_state_card(item)
            if label:
                return label
        return ""
    return normalize_card_label(str(value))


def _option_id(option: dict, region_name: str, index: int) -> str:
    prefix = "chi" if region_name == "chi_options" else "compare"
    return str(option.get("option_id") or f"{prefix}_{index:03d}")


def _has_click_box(item: dict) -> bool:
    if item.get("center") is not None:
        return True
    return all(item.get(key) is not None for key in ("x", "y", "w", "h"))


def _find_hand_card(state: dict, label: str, *, card_id: str | None = None) -> dict | None:
    if card_id:
        card = next(
            (
                item
                for item in state.get("hand_details", [])
                if str(item.get("card_id") or item.get("id") or "") == str(card_id)
                and normalize_card_label(item.get("name") or item.get("label") or "") == label
                and item.get("clickable", True)
            ),
            None,
        )
        if card is not None:
            return card
    return next(
        (
            card
            for card in state.get("hand_details", [])
            if normalize_card_label(card.get("name") or card.get("label") or "") == label
            and card.get("clickable", True)
        ),
        None,
    )


def _discard_execution_clicks(state: dict, card: dict, label: str) -> tuple[list[dict], str]:
    hand_click = _center_click(card, target=f"hand:{label}")
    discard_button = state.get("discard_button")
    if isinstance(discard_button, dict) and _has_click_box(discard_button):
        return [
            hand_click,
            _center_click(discard_button, target="button:discard"),
        ], "select_then_discard_button"
    return [hand_click], "discard_drag"


def _is_discard_actionable(state: dict) -> bool:
    discard_button = state.get("discard_button")
    if isinstance(discard_button, dict) and _has_click_box(discard_button):
        return True
    if "legal_actions" not in state:
        return True
    for action in state.get("legal_actions") or []:
        if not isinstance(action, dict):
            continue
        if action.get("type") != "discard":
            continue
        if action.get("source") in {"qs_own_draw", "manual", "vision_discard_button"}:
            return True
    metadata = state.get("metadata") or {}
    protocol_metadata = metadata.get("protocol_state") or {}
    return metadata.get("discard_turn_source") == "qs_own_draw" or protocol_metadata.get("discard_turn_source") == "qs_own_draw"


def _center_click(item: dict, *, target: str) -> dict:
    if item.get("center") is not None:
        x, y = item["center"]
        return {"target": target, "x": int(x), "y": int(y), "delay_ms": 80}
    x = int(item["x"] + item["w"] / 2)
    y = int(item["y"] + item["h"] / 2)
    return {"target": target, "x": x, "y": y, "delay_ms": 80}
