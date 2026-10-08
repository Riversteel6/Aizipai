"""Validation helpers for completed game logs."""

from __future__ import annotations

from pathlib import Path

from .replay_loader import load_round_bundle_from_path
from .schemas import SAFE_HALT_REASONS


def _events_by_decision(bundle) -> dict[str, list[dict]]:
    by_decision: dict[str, list[dict]] = {}
    for row in bundle.events:
        decision_id = row.get("decision_id")
        if decision_id is not None:
            by_decision.setdefault(str(decision_id), []).append(row)
    return by_decision


def _event_field(row: dict, key: str):
    data = row.get("data", {})
    if isinstance(data, dict) and data.get(key) is not None:
        return data.get(key)
    return row.get(key)


def _has_matching_screen_after_action(events: list[dict], tap_row: dict) -> bool:
    after_screenshot = _event_field(tap_row, "after_screenshot")
    frame_id = _event_field(tap_row, "frame_id")
    decision_id = _event_field(tap_row, "decision_id")
    for row in events:
        if row.get("event_type") != "SCREEN_AFTER_ACTION":
            continue
        if _event_field(row, "after_screenshot") != after_screenshot:
            continue
        if frame_id is not None and _event_field(row, "frame_id") != frame_id:
            continue
        if decision_id is not None and _event_field(row, "decision_id") != decision_id:
            continue
        return True
    return False


def _tap_planned_clicks(data: dict) -> list[dict]:
    clicks = data.get("planned_clicks")
    if isinstance(clicks, list):
        return [click for click in clicks if isinstance(click, dict)]
    plan = data.get("action_plan")
    if isinstance(plan, dict) and isinstance(plan.get("clicks"), list):
        return [click for click in plan["clicks"] if isinstance(click, dict)]
    return []


def _click_count(data: dict, key: str) -> int | None:
    value = data.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _has_click_xy(click: dict) -> bool:
    return click.get("x") is not None and click.get("y") is not None


def _selected_action_name(selected: dict) -> str:
    action = selected.get("action") or selected.get("policy_action")
    payload = selected.get("selected_action")
    if isinstance(payload, dict):
        action = payload.get("type") or payload.get("action") or action
    elif payload:
        action = payload
    return str(action or "").upper()


def _plan_policy_action(plan_data: dict) -> str:
    action_plan = plan_data.get("action_plan") if isinstance(plan_data.get("action_plan"), dict) else {}
    policy = action_plan.get("policy_selected_action") or plan_data.get("policy_selected_action") or {}
    if isinstance(policy, dict):
        return str(policy.get("type") or policy.get("action") or "").upper()
    return str(policy or "").upper()


def validate_round_bundle(round_path: str | Path) -> dict:
    """Validate one round bundle and return a flat summary dict."""
    source = Path(round_path)
    bundle = load_round_bundle_from_path(source)
    by_decision = _events_by_decision(bundle)
    errors: list[str] = []

    state_index: dict[str, dict] = {}
    for row in bundle.states:
        frame_id = row.get("frame_id")
        if frame_id:
            state_index[str(frame_id)] = row

    for decision in bundle.decisions:
        decision_id = str(decision.get("decision_id", ""))
        if not decision_id:
            errors.append("决策记录缺少 decision_id")
            continue

        frame_id = str(decision.get("frame_id", ""))
        if not frame_id:
            errors.append(f"决策 {decision_id} 缺少 frame_id")
        elif frame_id not in state_index:
            errors.append(f"决策 {decision_id} 的 frame_id={frame_id} 未找到对应状态")
        else:
            frame_state = state_index[frame_id].get("state", {})
            frame_screenshot = frame_state.get("screenshot_path") or frame_state.get("screenshot")
            if not frame_screenshot:
                errors.append(f"决策 {decision_id} 的 frame_id={frame_id} 对应状态缺少截图路径")
            elif source.is_dir():
                screenshot_path = (source / frame_screenshot).resolve()
                if not screenshot_path.exists():
                    errors.append(f"决策 {decision_id} 的截图文件不存在: {frame_screenshot}")

        selected = decision.get("decision", {})
        action = selected.get("action")
        if not action:
            errors.append(f"决策 {decision_id} 决策缺少 action")
        selected_action = _selected_action_name(selected)

        decision_events = by_decision.get(decision_id, [])
        if not any(row.get("event_type") == "STRUCTURE_ALLOCATED" for row in decision_events):
            errors.append(f"决策 {decision_id} 没有 STRUCTURE_ALLOCATED 记录")
        if not any(row.get("event_type") == "HAND_ANALYZED" for row in decision_events):
            errors.append(f"决策 {decision_id} 没有 HAND_ANALYZED 记录")
        eval_events = [row for row in decision_events if row.get("event_type") == "ACTION_EVALUATED"]
        if not eval_events:
            errors.append(f"决策 {decision_id} 没有 ACTION_EVALUATED 记录")
        else:
            if not any(row.get("event_type") == "ACTION_EVALUATION_STARTED" for row in decision_events):
                errors.append(f"决策 {decision_id} 没有 ACTION_EVALUATION_STARTED 记录")
            selected_label = selected.get("label") or selected.get("selected_label")
            if isinstance(selected.get("selected_action"), dict):
                selected_payload = selected["selected_action"]
                selected_label = selected_payload.get("label") or selected_label
            if selected_action and selected_action not in {"PASS", "SAFE_HALT"}:
                matched_eval = False
                for row in eval_events:
                    data = row.get("data", {})
                    eval_type = str(data.get("type") or data.get("action", {}).get("type", "")).upper()
                    eval_label = data.get("label") or data.get("action", {}).get("label")
                    if eval_type == selected_action and (not selected_label or eval_label == selected_label):
                        matched_eval = True
                        break
                if not matched_eval:
                    errors.append(f"决策 {decision_id} selected_action 找不到对应 ACTION_EVALUATED")

        plan_events = [row for row in decision_events if row.get("event_type") == "ACTION_PLAN_CREATED"]
        if not plan_events:
            errors.append(f"决策 {decision_id} 没有 ACTION_PLAN_CREATED 记录")
        else:
            plan_data = plan_events[-1].get("data", {})
            if not isinstance(plan_data, dict):
                errors.append(f"决策 {decision_id} ACTION_PLAN_CREATED 数据格式错误")
            else:
                reason = plan_data.get("reason")
                if not reason:
                    errors.append(f"决策 {decision_id} action_plan missing reason")

                plan = plan_data.get("action_plan", {}) if isinstance(plan_data.get("action_plan"), dict) else {}
                plan_clicks = plan.get("clicks") if isinstance(plan.get("clicks"), list) else []
                if selected_action == "EXPAND_CHI_OPTIONS":
                    if _plan_policy_action(plan_data) != "EXPAND_CHI_OPTIONS":
                        errors.append(f"决策 {decision_id} expand_chi_options action_plan 未标记为展开候选")
                    if plan_data.get("ready") is not True:
                        errors.append(f"决策 {decision_id} expand_chi_options action_plan 未 ready")
                    if [str(click.get("target") or "") for click in plan_clicks] != ["button:chi"]:
                        errors.append(f"决策 {decision_id} expand_chi_options 必须只点击 button:chi")
                    if plan.get("target_label") not in {None, "chi"}:
                        errors.append(f"决策 {decision_id} expand_chi_options target_label 非 chi")

                if selected_action == "SAFE_HALT":
                    if plan_data.get("ready"):
                        errors.append(f"决策 {decision_id} SAFE_HALT action_plan 不应 ready")
                    if plan_clicks:
                        errors.append(f"决策 {decision_id} SAFE_HALT action_plan 不应包含点击")
                    if not any(row.get("event_type") == "SAFE_HALT" for row in decision_events):
                        errors.append(f"决策 {decision_id} SAFE_HALT 缺少 SAFE_HALT 事件")

                if plan_data.get("ready", False) and selected:
                    selected_label = selected.get("label")
                    plan_label = plan.get("target_label")
                    if selected_label and plan_label and selected_label != plan_label and action == "discard":
                        errors.append(
                            f"决策 {decision_id} action_plan 与策略目标不一致: "
                            f"{selected_label} != {plan_label}"
                        )

                if action == "discard":
                    selected_label = selected.get("label") or selected.get("selected_label")
                    if not selected_label:
                        errors.append(f"决策 {decision_id} 出牌动作缺少 selected label")
        validation_events = [row for row in decision_events if row.get("event_type") == "ACTION_PLAN_VALIDATED"]
        if not validation_events:
            errors.append(f"决策 {decision_id} 没有 ACTION_PLAN_VALIDATED 记录")
        else:
            validation_data = validation_events[-1].get("data", {})
            if "passed" not in validation_data:
                errors.append(f"决策 {decision_id} ACTION_PLAN_VALIDATED 缺少 passed")
            if not validation_data.get("checks"):
                errors.append(f"决策 {decision_id} ACTION_PLAN_VALIDATED 缺少 checks")

    for row in bundle.events:
        if row.get("event_type") == "SAFE_HALT":
            data = row.get("data", {})
            reason = data.get("halt_reason") or data.get("reason")
            if reason not in SAFE_HALT_REASONS:
                errors.append(f"未知 SAFE_HALT reason: {reason}")
            if not (data.get("screenshot_path") or data.get("screenshots")):
                errors.append(f"SAFE_HALT 缺少 screenshot_path: {row.get('decision_id')}")

    for row in bundle.events:
        if row.get("event_type") != "TAP_EXECUTED":
            continue
        data = row.get("data", {})
        decision_id = _event_field(row, "decision_id") or "unknown"
        execute_enabled = bool(data.get("execute_enabled"))
        dry_run = bool(data.get("dry_run"))
        tap_executed = bool(data.get("tap_executed"))
        before_screenshot = data.get("before_screenshot")
        after_screenshot = data.get("after_screenshot")
        planned_clicks = _tap_planned_clicks(data)
        executed_clicks = data.get("executed_clicks")
        would_clicks = data.get("would_clicks")

        if planned_clicks:
            click_count = _click_count(data, "click_count")
            planned_click_count = _click_count(data, "planned_click_count")
            if click_count is not None and click_count != len(planned_clicks):
                errors.append(f"TAP_EXECUTED click_count 与 planned_clicks 不一致: {decision_id}")
            if planned_click_count is not None and planned_click_count != len(planned_clicks):
                errors.append(f"TAP_EXECUTED planned_click_count 与 planned_clicks 不一致: {decision_id}")
            if any(not _has_click_xy(click) for click in planned_clicks):
                errors.append(f"TAP_EXECUTED planned_clicks 缺少点击坐标: {decision_id}")
            if dry_run and not tap_executed and len(planned_clicks) > 1:
                if not isinstance(would_clicks, list) or len(would_clicks) != len(planned_clicks):
                    errors.append(f"TAP_EXECUTED dry_run 多步动作缺少 would_clicks: {decision_id}")

        if execute_enabled and not dry_run and not before_screenshot:
            errors.append(f"TAP_EXECUTED execute 模式缺少 before_screenshot: {decision_id}")
        if tap_executed:
            if dry_run:
                errors.append(f"TAP_EXECUTED tap_executed 不能同时 dry_run: {decision_id}")
            if not before_screenshot:
                errors.append(f"TAP_EXECUTED 实际点击缺少 before_screenshot: {decision_id}")
            if not after_screenshot:
                errors.append(f"TAP_EXECUTED 实际点击缺少 after_screenshot: {decision_id}")
            if not data.get("executed_at"):
                errors.append(f"TAP_EXECUTED 实际点击缺少 executed_at: {decision_id}")
            if after_screenshot and not _has_matching_screen_after_action(bundle.events, row):
                errors.append(f"TAP_EXECUTED 实际点击缺少匹配 SCREEN_AFTER_ACTION: {decision_id}")
            if planned_clicks:
                if not isinstance(executed_clicks, list):
                    errors.append(f"TAP_EXECUTED 实际点击缺少 executed_clicks: {decision_id}")
                elif len(executed_clicks) != len(planned_clicks):
                    errors.append(f"TAP_EXECUTED executed_clicks 与 planned_clicks 数量不一致: {decision_id}")
                elif any(not isinstance(click, dict) or not _has_click_xy(click) for click in executed_clicks):
                    errors.append(f"TAP_EXECUTED executed_clicks 缺少点击坐标: {decision_id}")
                executed_click_count = _click_count(data, "executed_click_count")
                if executed_click_count is not None and executed_click_count != len(planned_clicks):
                    errors.append(f"TAP_EXECUTED executed_click_count 与 planned_clicks 不一致: {decision_id}")
            elif data.get("tap_x") is None or data.get("tap_y") is None:
                errors.append(f"TAP_EXECUTED 实际点击缺少点击坐标: {decision_id}")

    for row in bundle.events:
        if row.get("event_type") == "ACTION_EVALUATED":
            eval_data = row.get("data", {})
            if "score_breakdown" not in eval_data or not isinstance(eval_data["score_breakdown"], dict):
                errors.append(f"决策评估缺少 score_breakdown: {row.get('decision_id')}")
            if eval_data.get("allowed") is False and not (eval_data.get("reject_reason") or eval_data.get("reason")):
                errors.append(f"被拒动作缺少 reject reason: {row.get('decision_id')}")
            eval_type = str(eval_data.get("type") or eval_data.get("action", {}).get("type", "")).upper()
            if eval_type in {"CHI", "EXPAND_CHI_OPTIONS", "PENG"}:
                response_key = "peng_ev" if eval_type == "PENG" else "chi_ev"
                for key in ("pass_ev", response_key, "ev_delta_vs_pass"):
                    if key not in eval_data:
                        errors.append(f"{eval_type} 评估缺少 {key}: {row.get('decision_id')}")

    for row in bundle.events:
        if row.get("event_type") == "STRUCTURE_ALLOCATED":
            data = row.get("data", {})
            for key in ("locked_counts", "free_counts_after_locked", "hard_protected_card_ids"):
                if key not in data:
                    errors.append(f"STRUCTURE_ALLOCATED 缺少 {key}: {row.get('decision_id')}")

    for decision_id, rows in by_decision.items():
        if not rows:
            continue
        has_plan = any(row.get("event_type") == "ACTION_PLAN_CREATED" for row in rows)
        if not has_plan:
            errors.append(f"决策 {decision_id} 缺少 ACTION_PLAN_CREATED")
        has_plan_validation = any(row.get("event_type") == "ACTION_PLAN_VALIDATED" for row in rows)
        if not has_plan_validation:
            errors.append(f"决策 {decision_id} 缺少 ACTION_PLAN_VALIDATED")
        has_eval = any(row.get("event_type") == "ACTION_EVALUATED" for row in rows)
        if not has_eval:
            errors.append(f"决策 {decision_id} 缺少 ACTION_EVALUATED")
        has_eval_started = any(row.get("event_type") == "ACTION_EVALUATION_STARTED" for row in rows)
        if has_eval and not has_eval_started:
            errors.append(f"决策 {decision_id} 缺少 ACTION_EVALUATION_STARTED")

    for row in bundle.events:
        if row.get("event_type") == "LEGAL_ACTIONS_GENERATED":
            data = row.get("data", {})
            for item in data.get("rejected_actions", []) or []:
                if not (item.get("reject_reason") or item.get("reason")):
                    errors.append(f"LEGAL_ACTIONS_GENERATED rejected action 缺少 reason: {row.get('decision_id')}")

    for decision in bundle.decisions:
        reason = decision.get("reason")
        selected = decision.get("decision", {}) if isinstance(decision.get("decision"), dict) else {}
        if _selected_action_name(selected) == "SAFE_HALT" and not reason:
            errors.append(f"decision {decision.get('decision_id')} SAFE_HALT 需记录原因")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "summary": {
            "decisions": len(bundle.decisions),
            "events": len(bundle.events),
            "states": len(bundle.states),
            "actions": len(bundle.actions),
            "safe_halts": sum(1 for row in bundle.events if row.get("event_type") == "SAFE_HALT"),
            "session_id": bundle.session_id,
            "round_id": bundle.round_id,
        },
    }


def validate_round_path(round_path: str | Path) -> None:
    result = validate_round_bundle(round_path)
    if not result["ok"]:
        for item in result["errors"]:
            print(f"ERROR: {item}")
        raise SystemExit(1)
    print(f"validate_logs ok: {round_path}")
