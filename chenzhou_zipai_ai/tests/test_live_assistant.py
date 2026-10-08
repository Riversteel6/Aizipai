"""Tests for live-assistant flow click plans."""

import copy
import json
from collections import Counter
from types import SimpleNamespace

from tools.live_assistant import (
    _confirm_discard_warning_plan,
    _defer_opening_compact_for_recognition_retry,
    _expire_stale_guard_state,
    _handle_opening_compact,
    _unchanged_executed_frame_wait,
    _flow_click_plan,
    _can_continue_response_window_without_seat,
    _continue_nonfatal_halts,
    _is_play_action_plan,
    _loop_interval_after_result,
    _matches_required_plan,
    _prepare_protocol_payload_for_loop,
    _reuse_pending_chi_option_plan,
    _pending_chi_option_plan_from_surface,
    _pending_compare_option_plan_from_surface,
    _result_context_signature,
    _plan_signature,
    _can_reuse_auto_seat_result,
    commit_executed_action_guard,
    abort_external_action_guard,
    prepare_external_action_guard,
    reset_transient_external_action_guard,
    infer_seat_role,
    print_summary,
    run_loop,
    run_once,
    _mark_midgame_count_relaxed,
    _should_confirm_discard_warning,
    _should_auto_compact_dealer_hand,
    _should_block_repeated_execution,
    _should_halt_for_failed_chi_expand,
    _should_hold_for_response_options,
    _bounded_untrusted_pending_card_recheck,
    _stabilize_pending_chi_candidate,
    _should_skip_recent_flow_ready_click,
    _should_relax_midgame_count,
    _should_capture_after_action,
    _should_retry_dealer_after_first_discard,
    _wait_for_stable_opening_layout,
    _expected_total_for_current_phase,
    _recover_current_phase_count_shortage,
    execute_action_plan,
    should_stop_for_user,
)


def test_manual_runtime_restart_clears_only_transient_action_guard(tmp_path):
    guard_file = tmp_path / "guard.json"
    guard_file.write_text(
        json.dumps(
            {
                "inflight_action": {"signature": ["pass", []]},
                "pending_response": {"action": "chi", "attempts": 2},
                "last_executed_signature": ["discard", [["hand:九", 900, 900]]],
                "last_context": {"flow_state": "play"},
            }
        ),
        encoding="utf-8",
    )

    assert reset_transient_external_action_guard(guard_file)
    saved = json.loads(guard_file.read_text(encoding="utf-8"))
    assert "inflight_action" not in saved
    assert "pending_response" not in saved
    assert saved["last_executed_signature"][0] == "discard"
    assert saved["last_context"] == {"flow_state": "play"}


def test_manual_runtime_restart_allows_unfinished_chi_expansion_retry(tmp_path):
    guard_file = tmp_path / "guard.json"
    guard_file.write_text(
        json.dumps(
            {
                "last_executed_signature": [
                    "expand_chi_options",
                    [["button:chi", 1925, 520]],
                ],
                "last_context": {"flow_state": "play"},
                "last_executed_at": 100.0,
            }
        ),
        encoding="utf-8",
    )

    assert reset_transient_external_action_guard(guard_file)
    saved = json.loads(guard_file.read_text(encoding="utf-8"))
    assert "last_executed_signature" not in saved
    assert "last_context" not in saved
    assert "last_executed_at" not in saved


def test_settlement_epoch_blocks_same_page_but_allows_next_round(tmp_path):
    guard_file = tmp_path / "guard.json"
    signature = ("settlement_ready", (("settlement_ready", 0, 0),))
    first_round = {"flow_state": "settlement_ready", "settlement_epoch": "session-a:1"}
    next_round = {"flow_state": "settlement_ready", "settlement_epoch": "session-a:2"}
    action_plan = {"action": "settlement_ready"}

    assert prepare_external_action_guard(
        guard_file,
        signature=signature,
        context_signature=first_round,
    ) == "armed"
    assert commit_executed_action_guard(
        guard_file,
        signature=signature,
        context_signature=first_round,
        action_plan=action_plan,
    )

    assert prepare_external_action_guard(
        guard_file,
        signature=signature,
        context_signature=first_round,
    ) == "duplicate_execution_same_context"
    assert prepare_external_action_guard(
        guard_file,
        signature=signature,
        context_signature=next_round,
    ) == "armed"


def test_shared_execution_commit_records_chi_expand_pending_response(monkeypatch, tmp_path):
    monkeypatch.setattr("tools.live_assistant.time.time", lambda: 100.0)
    guard_file = tmp_path / "guard.json"
    guard_file.write_text(
        json.dumps(
            {
                "inflight_action": {"signature": ["old", []]},
                "unrelated_state": "keep",
            }
        ),
        encoding="utf-8",
    )
    plan = {
        "action": "expand_chi_options",
        "ready": True,
        "clicks": [{"target": "button:chi", "x": 1960, "y": 507}],
    }
    signature = ("expand_chi_options", (("button:chi", 1960, 507),))
    context = {"flow_state": "play", "hand_signature": ["三", "三", "叁"]}

    assert commit_executed_action_guard(
        guard_file,
        signature=signature,
        context_signature=context,
        action_plan=plan,
    )

    saved = json.loads(guard_file.read_text(encoding="utf-8"))
    assert saved["unrelated_state"] == "keep"
    assert "inflight_action" not in saved
    assert saved["last_executed_signature"] == [
        "expand_chi_options",
        [["button:chi", 1960, 507]],
    ]
    assert saved["last_context"] == context
    assert saved["last_executed_at"] == 100.0
    assert saved["pending_response"] == {
        "action": "chi",
        "created_at": 100.0,
        "expires_at": 104.0,
        "attempts": 1,
        "context": context,
        "last_signature": ["expand_chi_options", [["button:chi", 1960, 507]]],
    }


def test_shared_execution_commit_advances_pending_transaction_to_compare(monkeypatch, tmp_path):
    monkeypatch.setattr("tools.live_assistant.time.time", lambda: 105.0)
    guard_file = tmp_path / "guard.json"
    guard_file.write_text(
        json.dumps({"pending_response": {"action": "chi", "expires_at": 108.0}}),
        encoding="utf-8",
    )
    plan = {
        "action": "chi_option",
        "ready": True,
        "target_option_id": "chi_001",
        "target_option_cards": ["叁", "叁", "三"],
        "clicks": [{"target": "chi:叁叁三", "x": 1560, "y": 220}],
    }
    signature = ("chi_option", (("chi:叁叁三", 1560, 220),))

    assert commit_executed_action_guard(
        guard_file,
        signature=signature,
        context_signature={"option_stage": "chi"},
        action_plan=plan,
    )

    saved = json.loads(guard_file.read_text(encoding="utf-8"))
    assert saved["pending_response"] == {
        "action": "compare",
        "created_at": 105.0,
        "expires_at": 109.0,
        "attempts": 1,
        "context": {"option_stage": "chi"},
        "last_signature": ["chi_option", [["chi:叁叁三", 1560, 220]]],
        "selected_chi_option_cards": ["叁", "叁", "三"],
        "selected_chi_option_id": "chi_001",
    }
    assert saved["last_executed_signature"][0] == "chi_option"


def test_pending_semantic_chi_maps_exact_visible_option_without_rethinking():
    result = {
        "hand": ["七", "九", "一"],
        "hand_count": 3,
        "sanity_checks": {"controlled_card_count": 3},
        "pending_action_card": {"name": "八"},
        "buttons": [],
        "option_details": [
            {
                "region_name": "chi_options",
                "labels": ["九", "八", "七"],
                "center": [360, 120],
            }
        ],
    }
    previous_context = _result_context_signature(result)
    guard = {
        "pending_response": {
            "action": "chi",
            "intended_option_cards": ["七", "八", "九"],
            "context": previous_context,
        }
    }

    reused = _reuse_pending_chi_option_plan(result, guard)

    assert reused is not None
    assert reused["pending_chi_plan_cache_hit"] is True
    assert reused["action_plan"]["action"] == "chi_option"
    assert Counter(reused["action_plan"]["target_option_cards"]) == Counter(["七", "八", "九"])
    assert reused["action_plan"]["clicks"][0]["x"] == 360


def test_pending_semantic_chi_is_rejected_when_hand_changed():
    original = {
        "hand": ["七", "九", "一"],
        "sanity_checks": {"controlled_card_count": 3},
        "pending_action_card": {"name": "八"},
        "buttons": [],
        "option_details": [],
    }
    changed = {
        **original,
        "hand": ["七", "九", "二"],
        "option_details": [
            {
                "region_name": "chi_options",
                "labels": ["七", "八", "九"],
                "center": [360, 120],
            }
        ],
    }
    guard = {
        "pending_response": {
            "action": "chi",
            "intended_option_cards": ["七", "八", "九"],
            "context": _result_context_signature(original),
        }
    }

    assert _reuse_pending_chi_option_plan(changed, guard) is None


def test_pending_chi_surface_path_maps_exact_option_without_full_state_scan(monkeypatch):
    original = {
        "hand": ["七", "九", "一"],
        "sanity_checks": {"controlled_card_count": 3},
        "opponent_pending_card": {"name": "八"},
        "buttons": [{"name": "chi"}, {"name": "pass"}],
        "flow": {"state": "play"},
    }
    guard = {
        "pending_response": {
            "action": "chi",
            "intended_option_cards": ["七", "八", "九"],
            "context": _result_context_signature(original),
            "expires_at": 104.0,
        }
    }
    option = SimpleNamespace(
        to_dict=lambda: {
            "region_name": "chi_options",
            "labels": ["九", "八", "七"],
            "confidence": 0.94,
            "center": [360, 120],
        }
    )
    buttons = [SimpleNamespace(name="pass", to_dict=lambda: {"name": "pass"})]
    monkeypatch.setattr("tools.live_assistant.time.time", lambda: 100.0)
    monkeypatch.setattr("tools.live_assistant.read_image", lambda _path: object())
    monkeypatch.setattr(
        "tools.live_assistant.detect_hand_slots",
        lambda *_args, **_kwargs: [SimpleNamespace(source="contour")] * 3,
    )
    monkeypatch.setattr(
        "tools.live_assistant.recognize_pending_cards",
        lambda *_args, **_kwargs: {
            # The center action-card region may contain an unrelated low-confidence
            # match while the source-specific opponent region remains correct.
            "pending_action_card": SimpleNamespace(name="五", to_dict=lambda: {"name": "五"}),
            "opponent_pending_card": SimpleNamespace(name="八", to_dict=lambda: {"name": "八"}),
        },
    )
    monkeypatch.setattr("tools.live_assistant.recognize_options", lambda *_args, **_kwargs: [option])

    result = _pending_chi_option_plan_from_surface(
        "same-frame",
        guard,
        flow=SimpleNamespace(state="play", to_dict=lambda: {"state": "play"}),
        buttons=buttons,
    )

    assert result is not None
    assert result["pending_chi_surface_fast_path"] is True
    assert result["action_plan"]["action"] == "chi_option"
    assert result["action_plan"]["clicks"][0]["x"] == 360
    assert Counter(result["action_plan"]["target_option_cards"]) == Counter(["七", "八", "九"])


def test_pending_chi_surface_path_falls_back_on_hu_or_nonmatching_option(monkeypatch):
    original = {
        "hand": ["七", "九", "一"],
        "sanity_checks": {"controlled_card_count": 3},
        "pending_action_card": {"name": "八"},
        "buttons": [{"name": "chi"}],
        "flow": {"state": "play"},
    }
    guard = {
        "pending_response": {
            "action": "chi",
            "intended_option_cards": ["七", "八", "九"],
            "context": _result_context_signature(original),
            "expires_at": 104.0,
        }
    }
    option = SimpleNamespace(
        to_dict=lambda: {
            "region_name": "chi_options",
            "labels": ["一", "二", "三"],
            "confidence": 0.95,
            "center": [360, 120],
        }
    )
    monkeypatch.setattr("tools.live_assistant.time.time", lambda: 100.0)
    monkeypatch.setattr("tools.live_assistant.read_image", lambda _path: object())
    monkeypatch.setattr(
        "tools.live_assistant.detect_hand_slots",
        lambda *_args, **_kwargs: [SimpleNamespace(source="contour")] * 3,
    )
    monkeypatch.setattr(
        "tools.live_assistant.recognize_pending_cards",
        lambda *_args, **_kwargs: {
            "pending_action_card": SimpleNamespace(name="八", to_dict=lambda: {"name": "八"}),
            "opponent_pending_card": None,
        },
    )
    monkeypatch.setattr("tools.live_assistant.recognize_options", lambda *_args, **_kwargs: [option])
    flow = SimpleNamespace(state="play", to_dict=lambda: {"state": "play"})

    assert _pending_chi_option_plan_from_surface(
        "same-frame",
        guard,
        flow=flow,
        buttons=[SimpleNamespace(name="hu", to_dict=lambda: {"name": "hu"})],
    ) is None
    assert _pending_chi_option_plan_from_surface(
        "same-frame",
        guard,
        flow=flow,
        buttons=[SimpleNamespace(name="pass", to_dict=lambda: {"name": "pass"})],
    ) is None


def test_pending_chi_surface_path_falls_back_when_hand_slots_or_incoming_source_changed(monkeypatch):
    original = {
        "hand": ["七", "九", "一"],
        "sanity_checks": {"controlled_card_count": 3},
        "opponent_pending_card": {"name": "八"},
        "buttons": [{"name": "chi"}],
        "flow": {"state": "play"},
    }
    guard = {
        "pending_response": {
            "action": "chi",
            "intended_option_cards": ["七", "八", "九"],
            "context": _result_context_signature(original),
            "expires_at": 104.0,
        }
    }
    option = SimpleNamespace(
        to_dict=lambda: {
            "region_name": "chi_options",
            "labels": ["七", "八", "九"],
            "confidence": 0.95,
            "center": [360, 120],
        }
    )
    flow = SimpleNamespace(state="play", to_dict=lambda: {"state": "play"})
    buttons = [SimpleNamespace(name="pass", to_dict=lambda: {"name": "pass"})]
    monkeypatch.setattr("tools.live_assistant.time.time", lambda: 100.0)
    monkeypatch.setattr("tools.live_assistant.read_image", lambda _path: object())
    monkeypatch.setattr("tools.live_assistant.recognize_options", lambda *_args, **_kwargs: [option])
    monkeypatch.setattr(
        "tools.live_assistant.detect_hand_slots",
        lambda *_args, **_kwargs: [SimpleNamespace(source="contour")] * 2,
    )
    monkeypatch.setattr(
        "tools.live_assistant.recognize_pending_cards",
        lambda *_args, **_kwargs: {
            "pending_action_card": None,
            "opponent_pending_card": SimpleNamespace(name="八"),
        },
    )

    assert _pending_chi_option_plan_from_surface(
        "same-frame", guard, flow=flow, buttons=buttons
    ) is None

    monkeypatch.setattr(
        "tools.live_assistant.detect_hand_slots",
        lambda *_args, **_kwargs: [SimpleNamespace(source="contour")] * 3,
    )
    monkeypatch.setattr(
        "tools.live_assistant.recognize_pending_cards",
        lambda *_args, **_kwargs: {
            "pending_action_card": SimpleNamespace(name="八"),
            "opponent_pending_card": None,
        },
    )
    assert _pending_chi_option_plan_from_surface(
        "same-frame", guard, flow=flow, buttons=buttons
    ) is None


def test_mobile_pending_chi_surface_path_bypasses_full_inspection(monkeypatch, tmp_path):
    original = {
        "hand": ["七", "九", "一"],
        "sanity_checks": {"controlled_card_count": 3},
        "pending_action_card": {"name": "八"},
        "buttons": [{"name": "chi"}, {"name": "pass"}],
        "flow": {"state": "play"},
    }
    guard_file = tmp_path / "guard.json"
    guard_file.write_text(
        json.dumps(
            {
                "pending_response": {
                    "action": "chi",
                    "intended_option_cards": ["七", "八", "九"],
                    "context": _result_context_signature(original),
                    "expires_at": 9_999_999_999.0,
                }
            }
        ),
        encoding="utf-8",
    )
    option = SimpleNamespace(
        to_dict=lambda: {
            "region_name": "chi_options",
            "labels": ["七", "八", "九"],
            "confidence": 0.94,
            "center": [360, 120],
        }
    )
    flow = SimpleNamespace(
        state="play",
        confidence=1.0,
        center=None,
        to_dict=lambda: {"state": "play", "confidence": 1.0},
    )
    monkeypatch.setattr("tools.live_assistant._capture_runtime_screen", lambda *_args, **_kwargs: tmp_path / "frame")
    monkeypatch.setattr("tools.live_assistant.detect_flow_state_from_path", lambda _path: flow)
    monkeypatch.setattr("tools.live_assistant.read_image", lambda _path: object())
    monkeypatch.setattr(
        "tools.live_assistant.detect_hand_slots",
        lambda *_args, **_kwargs: [SimpleNamespace(source="contour")] * 3,
    )
    monkeypatch.setattr(
        "tools.live_assistant.recognize_pending_cards",
        lambda *_args, **_kwargs: {
            "pending_action_card": SimpleNamespace(name="八", to_dict=lambda: {"name": "八"}),
            "opponent_pending_card": None,
        },
    )
    monkeypatch.setattr("tools.live_assistant.recognize_options", lambda *_args, **_kwargs: [option])

    def unexpected_full_chain(*_args, **_kwargs):
        raise AssertionError("pending CHI candidate frame entered full recognition or policy")

    monkeypatch.setattr("tools.live_assistant.inspect_screenshot", unexpected_full_chain)
    monkeypatch.setattr("tools.live_assistant.recommend_from_screenshot", unexpected_full_chain)

    result = run_once(
        runtime_screenshot_path=tmp_path / "frame",
        reuse_runtime_screenshot=True,
        guard_file=guard_file,
        memory_file=tmp_path / "memory.json",
        seat_role="auto",
        allow_opening_hand_compact=False,
        precomputed_buttons=[SimpleNamespace(name="pass", to_dict=lambda: {"name": "pass"})],
    )

    assert result["pending_chi_surface_fast_path"] is True
    assert result["action_plan"]["action"] == "chi_option"
    assert result["action_plan"]["ready"] is True


def test_mobile_pending_compare_surface_path_bypasses_full_policy(monkeypatch, tmp_path):
    guard_file = tmp_path / "guard.json"
    guard_file.write_text(
        json.dumps(
            {
                "pending_response": {
                    "action": "compare",
                    "context": {
                        "flow_state": "play",
                        "controlled_card_count": 20,
                        "hand_signature": [["一", 2], ["二", 1]],
                        "remaining_deck_count": 35,
                    },
                    "expires_at": 9_999_999_999.0,
                }
            }
        ),
        encoding="utf-8",
    )
    options = [
        SimpleNamespace(
            to_dict=lambda: {
                "region_name": "compare_options",
                "option_id": "compare_001",
                "labels": ["一", "二", "三"],
                "confidence": 0.96,
                "center": [420, 160],
                "index": 1,
                "x": 380,
                "y": 80,
            }
        )
    ]
    flow = SimpleNamespace(
        state="play",
        confidence=1.0,
        center=None,
        to_dict=lambda: {"state": "play", "confidence": 1.0},
    )
    monkeypatch.setattr("tools.live_assistant._capture_runtime_screen", lambda *_a, **_k: tmp_path / "frame")
    monkeypatch.setattr("tools.live_assistant.detect_flow_state_from_path", lambda _path: flow)
    monkeypatch.setattr("tools.live_assistant.read_image", lambda _path: object())
    monkeypatch.setattr("tools.live_assistant.recognize_options", lambda *_a, **_k: options)

    def unexpected_full_chain(*_args, **_kwargs):
        raise AssertionError("pending compare frame entered full recognition or V9 policy")

    monkeypatch.setattr("tools.live_assistant.inspect_screenshot", unexpected_full_chain)
    monkeypatch.setattr("tools.live_assistant.recommend_from_screenshot", unexpected_full_chain)

    result = run_once(
        runtime_screenshot_path=tmp_path / "frame",
        reuse_runtime_screenshot=True,
        guard_file=guard_file,
        memory_file=tmp_path / "memory.json",
        seat_role="auto",
        allow_opening_hand_compact=False,
        precomputed_buttons=[SimpleNamespace(name="pass", to_dict=lambda: {"name": "pass"})],
    )

    assert result["pending_compare_surface_fast_path"] is True
    assert result["action_plan"]["action"] == "compare_option"
    assert result["action_plan"]["ready"] is True
    assert result["action_plan"]["clicks"] == [
        {"target": "compare:一二三", "x": 420, "y": 160, "delay_ms": 80}
    ]


def test_partial_hu_surface_buttons_only_feed_pending_transaction(monkeypatch, tmp_path):
    guard_file = tmp_path / "guard.json"
    guard_file.write_text("{}", encoding="utf-8")
    flow = SimpleNamespace(
        state="play",
        confidence=1.0,
        center=None,
        to_dict=lambda: {"state": "play", "confidence": 1.0},
    )
    partial_buttons = []
    seen: dict[str, object] = {}
    monkeypatch.setattr("tools.live_assistant._capture_runtime_screen", lambda *_a, **_k: tmp_path / "frame")
    monkeypatch.setattr("tools.live_assistant.detect_flow_state_from_path", lambda _path: flow)
    def fake_pending_chi(*_args, **kwargs):
        seen["surface_buttons"] = kwargs["buttons"]
        return None

    monkeypatch.setattr(
        "tools.live_assistant._pending_chi_option_plan_from_surface",
        fake_pending_chi,
    )
    monkeypatch.setattr(
        "tools.live_assistant._pending_compare_option_plan_from_surface",
        lambda *_a, **_k: None,
    )

    def fake_recommend(*_args, **kwargs):
        seen["precomputed_state"] = kwargs.get("precomputed_state")
        return {
            "action_plan": {"action": "wait", "ready": False},
            "flow": flow.to_dict(),
            "hand": [],
            "sanity_checks": {"ok": True, "controlled_card_count": 20},
        }

    monkeypatch.setattr(
        "tools.live_assistant.recommend_from_screenshot",
        fake_recommend,
    )

    run_once(
        runtime_screenshot_path=tmp_path / "frame",
        reuse_runtime_screenshot=True,
        guard_file=guard_file,
        memory_file=tmp_path / "memory.json",
        seat_role="manual",
        auto_compact_dealer_hand=False,
        pending_surface_buttons=partial_buttons,
    )

    assert seen["surface_buttons"] is partial_buttons
    assert seen["precomputed_state"] is None


def test_external_execution_guard_arms_once_and_rejects_same_context(tmp_path):
    guard_file = tmp_path / "guard.json"
    signature = ("pass", (("button:pass", 2100, 510),))
    context = {"flow_state": "play", "button_names": ["pass"]}

    assert prepare_external_action_guard(
        guard_file,
        signature=signature,
        context_signature=context,
    ) == "armed"
    assert prepare_external_action_guard(
        guard_file,
        signature=signature,
        context_signature=context,
    ) == "duplicate_execution_same_context"

    saved = json.loads(guard_file.read_text(encoding="utf-8"))
    assert saved["inflight_action"]["signature"] == [
        "pass",
        [["button:pass", 2100, 510]],
    ]
    assert saved["inflight_action"]["context"] == context


def test_external_execution_guard_abort_only_clears_matching_inflight(tmp_path):
    guard_file = tmp_path / "guard.json"
    signature = ("peng", (("button:peng", 1950, 510),))
    other = ("pass", (("button:pass", 2100, 510),))
    context = {"flow_state": "play", "button_names": ["peng", "pass"]}
    assert prepare_external_action_guard(
        guard_file,
        signature=signature,
        context_signature=context,
    ) == "armed"

    assert abort_external_action_guard(
        guard_file,
        signature=other,
        context_signature=context,
    )
    assert "inflight_action" in json.loads(guard_file.read_text(encoding="utf-8"))

    assert abort_external_action_guard(
        guard_file,
        signature=signature,
        context_signature=context,
    )
    assert "inflight_action" not in json.loads(guard_file.read_text(encoding="utf-8"))
    assert prepare_external_action_guard(
        guard_file,
        signature=signature,
        context_signature=context,
    ) == "armed"


def test_chi_candidate_expand_defers_redundant_after_action_capture():
    expand_plan = {
        "action": "expand_chi_options",
        "ready": True,
        "target_type": "button",
        "clicks": [{"target": "button:chi", "x": 1900, "y": 520}],
    }
    selected_chi_plan = {
        "action": "chi",
        "ready": True,
        "target_type": "option_column",
        "clicks": [{"target": "chi:二七十", "x": 1600, "y": 220}],
    }

    assert not _should_capture_after_action(
        expand_plan,
        capture_after_action=True,
    )
    assert _should_capture_after_action(
        selected_chi_plan,
        capture_after_action=True,
    )
    assert not _should_capture_after_action(
        selected_chi_plan,
        capture_after_action=False,
    )


def test_controlled_total_depends_on_turn_phase_not_seat_role():
    player_discard_turn = {
        "seat_role": {"role": "player", "expected_total": 20},
        "discard_button": {"name": "discard"},
        "buttons": [],
    }
    dealer_waiting = {
        "seat_role": {"role": "dealer", "expected_total": 21},
        "discard_button": None,
        "buttons": [],
    }
    dealer_response = {
        "seat_role": {"role": "dealer", "expected_total": 21},
        "discard_button": None,
        "buttons": [{"name": "peng"}, {"name": "pass"}],
    }

    assert _expected_total_for_current_phase(player_discard_turn) == 21
    assert _expected_total_for_current_phase(dealer_waiting) == 20
    assert _expected_total_for_current_phase(dealer_response) == 20


def test_phase_count_shortage_rechecks_with_phase_total(monkeypatch, tmp_path):
    initial = {"sanity_checks": {"controlled_card_count": 20}}
    recovered = {
        "hand": ["八"] * 18,
        "sanity_checks": {"controlled_card_count": 21, "expected_total": 21, "ok": True},
    }
    calls = []

    def fake_inspect(path, **kwargs):
        calls.append((path, kwargs))
        return recovered

    monkeypatch.setattr("tools.live_assistant.inspect_screenshot", fake_inspect)
    output = _recover_current_phase_count_shortage(
        initial,
        screenshot=tmp_path / "screen.png",
        expected_total=21,
        opponent_priority_pending=False,
    )

    assert output is recovered
    assert calls[0][1]["expected_total"] == 21


def test_phase_count_shortage_keeps_initial_when_recheck_adds_no_card(monkeypatch, tmp_path):
    initial = {"sanity_checks": {"controlled_card_count": 20}}
    monkeypatch.setattr(
        "tools.live_assistant.inspect_screenshot",
        lambda *args, **kwargs: {"sanity_checks": {"controlled_card_count": 20}},
    )

    output = _recover_current_phase_count_shortage(
        initial,
        screenshot=tmp_path / "screen.png",
        expected_total=21,
        opponent_priority_pending=False,
    )

    assert output is initial
from vision.flow_detector import FlowDetection
from vision.history_memory import VisionMemory, load_memory, save_memory


def test_flow_click_plan_uses_detected_center():
    plan = _flow_click_plan(
        FlowDetection("settlement_ready", 0.9, x=1900, y=900, w=220, h=90),
        "settlement_ready",
        "结算界面准备按钮",
    )

    assert plan["ready"] is True
    assert plan["clicks"] == [
        {"target": "settlement_ready", "x": 2010, "y": 945, "delay_ms": 80}
    ]


def test_flow_click_plan_is_not_ready_without_location():
    plan = _flow_click_plan(
        FlowDetection("play", 0.0),
        "settlement_ready",
        "结算界面准备按钮",
    )

    assert plan["ready"] is False
    assert plan["clicks"] == []


def test_flow_click_plan_is_not_ready_at_low_confidence():
    plan = _flow_click_plan(
        FlowDetection("settlement_ready", 0.5, x=1900, y=900, w=220, h=90),
        "settlement_ready",
        "结算界面准备按钮",
    )

    assert plan["ready"] is False
    assert plan["validation"]["reason_code"] == "recognition_uncertain"


def test_recent_flow_ready_click_waits_instead_of_clicking_again(monkeypatch):
    monkeypatch.setattr("tools.live_assistant.time.time", lambda: 100.0)
    guard = {
        "last_flow_ready_click": {
            "action": "settlement_ready",
            "center": [1171, 615],
            "expires_at": 110.0,
        }
    }

    assert _should_skip_recent_flow_ready_click(
        guard,
        FlowDetection("settlement_ready", 1.0, x=820, y=453, w=703, h=324),
    )


def test_same_flow_ready_page_does_not_retry_even_after_time_passes(monkeypatch):
    monkeypatch.setattr("tools.live_assistant.time.time", lambda: 120.0)
    guard = {
        "last_flow_ready_click": {
            "action": "settlement_ready",
            "center": [1171, 615],
            "created_at": 100.0,
        }
    }

    assert _should_skip_recent_flow_ready_click(
        guard,
        FlowDetection("settlement_ready", 1.0, x=820, y=453, w=703, h=324),
    )


def test_flow_ready_does_not_retry_an_inflight_click_after_restart():
    guard = {
        "inflight_action": {
            "signature": ["settlement_ready", [["settlement_ready", 1171, 615]]],
            "context": {"flow_state": "settlement_ready"},
        }
    }

    assert _should_skip_recent_flow_ready_click(
        guard,
        FlowDetection("settlement_ready", 1.0, x=820, y=453, w=703, h=324),
    )


def test_settlement_ready_resets_previous_round_memory_before_click(monkeypatch, tmp_path):
    screenshot = tmp_path / "settlement.png"
    screenshot.write_text("fake", encoding="utf-8")
    memory_file = tmp_path / "memory.json"
    assert save_memory(
        VisionMemory(
            frames_seen=20,
            my_discards=["一"],
            my_meld_groups=[["壹", "贰", "叁"]],
            last_hand=["九", "九", "九"],
            last_hand_expected_total=20,
        ),
        memory_file,
    )
    execute_calls = []
    monkeypatch.setattr("tools.live_assistant.save_screen", lambda **kwargs: screenshot)
    monkeypatch.setattr(
        "tools.live_assistant.detect_flow_state_from_path",
        lambda path: FlowDetection("settlement_ready", 1.0, x=820, y=453, w=703, h=324),
    )
    monkeypatch.setattr(
        "tools.live_assistant.execute_action_plan",
        lambda *args, **kwargs: execute_calls.append(True) or True,
    )

    output = run_once(
        device_id="dev",
        execute_settlement_ready=True,
        memory_file=memory_file,
        guard_file=tmp_path / "guard.json",
        logs_root=tmp_path / "logs",
    )

    assert execute_calls == [True]
    assert output["executed"] is True
    assert load_memory(memory_file) == VisionMemory()


def test_execution_gate_rejects_ready_plan_missing_explicit_action_field():
    plan = {
        "ready": True,
        "target_type": "button",
        "target_label": "hu",
        "policy_selected_action": {"type": "hu", "label": None},
        "clicks": [{"target": "button:hu", "x": 1985, "y": 510, "delay_ms": 80}],
    }

    assert not _is_play_action_plan(plan)
    assert _matches_required_plan(plan, None, None)


def test_does_not_confirm_discard_warning_without_detected_dialog_anchor(monkeypatch):
    monkeypatch.setattr("tools.live_assistant.time.time", lambda: 100.0)
    result = {
        "flow": {"state": "play"},
        "hand": [],
        "buttons": [],
        "discard_button": None,
        "option_details": [],
        "action_plan": {"ready": False, "reason": "未找到 pass 按钮"},
    }
    guard = {
        "last_executed_signature": ["discard", [["hand:四", 1172, 1004]]],
        "last_executed_at": 97.0,
    }

    assert not _should_confirm_discard_warning(result, guard)


def test_confirms_discard_warning_only_with_high_confidence_dialog_anchor(monkeypatch):
    monkeypatch.setattr("tools.live_assistant.time.time", lambda: 100.0)
    result = {
        "flow": {"state": "play"},
        "hand": [],
        "buttons": [],
        "discard_button": None,
        "option_details": [],
        "metadata": {
            "discard_warning": {
                "detected": True,
                "confidence": 0.96,
                "confirm_center": [1402, 716],
            }
        },
        "action_plan": {"ready": False, "reason": "未找到 pass 按钮"},
    }
    guard = {
        "last_executed_signature": ["discard", [["hand:四", 1172, 1004]]],
        "last_executed_at": 80.0,
    }

    assert _should_confirm_discard_warning(result, guard)
    plan = _confirm_discard_warning_plan(result)
    assert plan["clicks"] == [
        {"target": "button:confirm_discard_warning", "x": 1402, "y": 716, "delay_ms": 80}
    ]


def test_does_not_confirm_very_stale_discard_warning(monkeypatch):
    monkeypatch.setattr("tools.live_assistant.time.time", lambda: 100.0)
    result = {
        "flow": {"state": "play"},
        "hand": [],
        "buttons": [],
        "discard_button": None,
        "option_details": [],
        "action_plan": {"ready": False, "reason": "未找到 pass 按钮"},
    }
    guard = {
        "last_executed_signature": ["discard", [["hand:四", 1172, 1004]]],
        "last_executed_at": 30.0,
    }

    assert not _should_confirm_discard_warning(result, guard)


def test_holds_repeated_chi_expand_while_waiting_for_options():
    result = {
        "action_plan": {
            "action": "expand_chi_options",
            "target_type": "button",
            "ready": True,
            "clicks": [{"target": "button:chi", "x": 1960, "y": 507}],
        },
        "option_details": [],
        "options": [],
    }
    pending = {"action": "chi", "expires_at": 104.0, "attempts": 1}

    assert _should_hold_for_response_options(result, pending)


def test_pending_semantic_chi_rejects_pass_or_different_visible_candidate():
    pending = {
        "action": "chi",
        "intended_option_cards": ["七", "八", "九"],
        "expires_at": 104.0,
    }
    visible = {
        "option_details": [{"region_name": "chi_options", "labels": ["七", "八", "九"]}],
        "options": [["七", "八", "九"]],
        "option_stage": "chi",
    }
    assert _should_hold_for_response_options(
        {**visible, "action_plan": {"action": "pass", "ready": True}},
        pending,
    )
    assert _should_hold_for_response_options(
        {
            **visible,
            "action_plan": {
                "action": "chi_option",
                "target_type": "option_column",
                "target_option_cards": ["一", "二", "三"],
                "ready": True,
            },
        },
        pending,
    )
    assert not _should_hold_for_response_options(
        {
            **visible,
            "action_plan": {
                "action": "chi_option",
                "target_type": "option_column",
                "target_option_cards": ["九", "八", "七"],
                "ready": True,
            },
        },
        pending,
    )


def test_pending_chi_retargets_only_after_two_stable_visible_frames():
    guard = {
        "pending_response": {
            "action": "chi",
            "intended_option_cards": ["七", "八", "九"],
        }
    }
    result = {
        "option_details": [
            {
                "region_name": "chi_options",
                "labels": ["一", "二", "三"],
                "center": [1200, 220],
            }
        ],
        "action_plan": {
            "action": "chi_option",
            "target_type": "option_column",
            "target_option_cards": ["一", "二", "三"],
            "ready": True,
        },
    }

    first_status, first_guard = _stabilize_pending_chi_candidate(result, guard)
    assert first_status == "recheck"
    assert first_guard["pending_response"]["intended_option_cards"] == ["七", "八", "九"]

    second_status, second_guard = _stabilize_pending_chi_candidate(result, first_guard)
    assert second_status == "release"
    assert second_guard["pending_response"]["intended_option_cards"] == ["一", "二", "三"]
    assert second_guard["pending_response"]["candidate_recovery"] == {
        "released": True,
        "stable_frames": 2,
        "selected_cards": ["一", "二", "三"],
    }


def test_untrusted_pending_card_recheck_is_bounded_then_passes_safely():
    result = {
        "flow": {"state": "play"},
        "hand": ["七", "九", "一"],
        "sanity_checks": {"controlled_card_count": 3},
        "buttons": [
            {"name": "chi", "x": 1800, "y": 420, "w": 180, "h": 180},
            {"name": "pass", "x": 2100, "y": 420, "w": 200, "h": 200},
        ],
        "decision": {"action": "expand_chi_options"},
        "action_plan": {
            "action": "wait_chi_candidate_recheck",
            "ready": False,
            "clicks": [],
        },
    }

    first, first_guard, first_changed = _bounded_untrusted_pending_card_recheck(result, {})
    second, second_guard, second_changed = _bounded_untrusted_pending_card_recheck(result, first_guard)
    third, third_guard, third_changed = _bounded_untrusted_pending_card_recheck(result, second_guard)

    assert first_changed and second_changed and third_changed
    assert first["action_plan"]["action"] == "wait_chi_candidate_recheck"
    assert second["action_plan"]["action"] == "wait_chi_candidate_recheck"
    assert third["action_plan"]["action"] == "pass"
    assert third["action_plan"]["ready"] is True
    assert third["action_plan"]["clicks"][0]["target"] == "button:pass"
    assert third["action_plan"]["validation"]["reason_code"] == "pending_card_untrusted_bounded_pass"
    assert third["pending_card_untrusted_bounded_pass"] is True
    assert "pending_card_recheck" not in third_guard


def test_untrusted_pending_card_recheck_restarts_when_the_surface_context_changes():
    result = {
        "flow": {"state": "play"},
        "hand": ["七", "九", "一"],
        "sanity_checks": {"controlled_card_count": 3},
        "buttons": [{"name": "chi"}, {"name": "pass"}],
        "decision": {"action": "expand_chi_options"},
        "action_plan": {"action": "wait_chi_candidate_recheck", "ready": False, "clicks": []},
    }
    _first, first_guard, _changed = _bounded_untrusted_pending_card_recheck(result, {})
    changed_result = copy.deepcopy(result)
    changed_result["hand"] = ["七", "九", "二"]

    second, second_guard, _changed = _bounded_untrusted_pending_card_recheck(changed_result, first_guard)

    assert second["action_plan"]["action"] == "wait_chi_candidate_recheck"
    assert second_guard["pending_card_recheck"]["stable_frames"] == 1


def test_pending_chi_exact_candidate_keeps_zero_delay_path():
    guard = {
        "pending_response": {
            "action": "chi",
            "intended_option_cards": ["七", "八", "九"],
        }
    }
    result = {
        "option_details": [
            {
                "region_name": "chi_options",
                "labels": ["九", "八", "七"],
                "center": [1200, 220],
            }
        ],
        "action_plan": {
            "action": "chi_option",
            "target_type": "option_column",
            "target_option_cards": ["九", "八", "七"],
            "ready": True,
        },
    }

    status, unchanged = _stabilize_pending_chi_candidate(result, guard)
    assert status is None
    assert unchanged == guard


def test_pending_chi_never_retargets_to_a_nonvisible_candidate():
    guard = {
        "pending_response": {
            "action": "chi",
            "intended_option_cards": ["七", "八", "九"],
        }
    }
    result = {
        "option_details": [
            {
                "region_name": "chi_options",
                "labels": ["一", "二", "三"],
                "center": [1200, 220],
            }
        ],
        "action_plan": {
            "action": "chi_option",
            "target_type": "option_column",
            "target_option_cards": ["四", "五", "六"],
            "ready": True,
        },
    }

    status, unchanged = _stabilize_pending_chi_candidate(result, guard)
    assert status is None
    assert unchanged == guard


def test_pending_chi_changed_candidate_restarts_stability_count():
    guard = {
        "pending_response": {
            "action": "chi",
            "intended_option_cards": ["七", "八", "九"],
        }
    }
    first_result = {
        "option_details": [
            {
                "region_name": "chi_options",
                "labels": ["一", "二", "三"],
                "center": [1200, 220],
            }
        ],
        "action_plan": {
            "action": "chi_option",
            "target_type": "option_column",
            "target_option_cards": ["一", "二", "三"],
            "ready": True,
        },
    }
    changed_result = {
        "option_details": [
            {
                "region_name": "chi_options",
                "labels": ["四", "五", "六"],
                "center": [1200, 220],
            }
        ],
        "action_plan": {
            "action": "chi_option",
            "target_type": "option_column",
            "target_option_cards": ["四", "五", "六"],
            "ready": True,
        },
    }

    first_status, first_guard = _stabilize_pending_chi_candidate(first_result, guard)
    second_status, second_guard = _stabilize_pending_chi_candidate(changed_result, first_guard)

    assert first_status == "recheck"
    assert second_status == "recheck"
    assert second_guard["pending_response"]["intended_option_cards"] == ["七", "八", "九"]
    assert second_guard["pending_response"]["candidate_recovery"]["stable_frames"] == 1


def test_pending_pass_never_enters_chi_candidate_recovery():
    guard = {
        "pending_response": {
            "action": "pass",
            "intended_option_cards": ["七", "八", "九"],
        }
    }
    result = {
        "option_details": [
            {
                "region_name": "chi_options",
                "labels": ["一", "二", "三"],
                "center": [1200, 220],
            }
        ],
        "action_plan": {
            "action": "chi_option",
            "target_type": "option_column",
            "target_option_cards": ["一", "二", "三"],
            "ready": True,
        },
    }

    status, unchanged = _stabilize_pending_chi_candidate(result, guard)
    assert status is None
    assert unchanged == guard


def test_run_once_rechecks_then_releases_a_stable_visible_chi_candidate(
    monkeypatch,
    tmp_path,
):
    screenshot = tmp_path / "frame.png"
    screenshot.write_text("fake", encoding="utf-8")
    guard_file = tmp_path / "guard.json"
    guard_file.write_text(
        json.dumps(
            {
                "pending_response": {
                    "action": "chi",
                    "intended_option_cards": ["七", "八", "九"],
                    "expires_at": 9_999_999_999.0,
                }
            }
        ),
        encoding="utf-8",
    )
    result = {
        "screenshot": str(screenshot),
        "flow": {"state": "play"},
        "hand": ["一"] * 20,
        "sanity_checks": {"ok": True, "controlled_card_count": 20},
        "option_details": [
            {
                "region_name": "chi_options",
                "labels": ["一", "二", "三"],
                "center": [1200, 220],
            }
        ],
        "options": [["一", "二", "三"]],
        "option_stage": "chi",
        "decision": {"action": "chi"},
        "action_plan": {
            "action": "chi_option",
            "target_type": "option_column",
            "target_option_cards": ["一", "二", "三"],
            "ready": True,
            "clicks": [{"target": "chi:一二三", "x": 1200, "y": 220}],
        },
    }
    flow = SimpleNamespace(
        state="play",
        confidence=1.0,
        center=None,
        to_dict=lambda: {"state": "play", "confidence": 1.0},
    )
    monkeypatch.setattr("tools.live_assistant.detect_flow_state_from_path", lambda _path: flow)
    monkeypatch.setattr(
        "tools.live_assistant.recommend_from_screenshot",
        lambda *_args, **_kwargs: copy.deepcopy(result),
    )

    first = run_once(
        runtime_screenshot_path=screenshot,
        reuse_runtime_screenshot=True,
        guard_file=guard_file,
        memory_file=tmp_path / "memory.json",
        seat_role="player",
        expected_total=20,
    )
    assert first["action_plan"]["action"] == "wait_chi_candidate_recheck"
    assert first["action_plan"]["ready"] is False

    second = run_once(
        runtime_screenshot_path=screenshot,
        reuse_runtime_screenshot=True,
        guard_file=guard_file,
        memory_file=tmp_path / "memory.json",
        seat_role="player",
        expected_total=20,
    )
    assert second["action_plan"]["action"] == "chi_option"
    assert second["action_plan"]["ready"] is True
    assert second["pending_chi_candidate_recovered"] is True


def test_duplicate_guard_allows_one_expired_chi_expand_retry(monkeypatch):
    monkeypatch.setattr("tools.live_assistant.time.time", lambda: 105.0)
    context = {
        "flow_state": "play",
        "remaining_deck_count": 20,
        "controlled_card_count": 20,
        "hand_signature": ["四", "五", "六"],
        "option_stage": None,
        "decision_action": "expand_chi_options",
    }
    signature = ("expand_chi_options", (("button:chi", 1960, 507),))
    guard = {
        "last_executed_signature": ["expand_chi_options", [["button:chi", 1960, 507]]],
        "last_context": context,
        "pending_response": {
            "action": "chi",
            "expires_at": 104.0,
            "attempts": 1,
            "context": context,
        },
    }

    assert not _should_block_repeated_execution(guard, signature, context)


def test_failed_chi_expand_halts_after_retry_budget(monkeypatch):
    monkeypatch.setattr("tools.live_assistant.time.time", lambda: 109.0)
    result = {
        "action_plan": {
            "action": "expand_chi_options",
            "target_type": "button",
            "ready": True,
            "clicks": [{"target": "button:chi", "x": 1960, "y": 507}],
        },
        "option_details": [],
        "options": [],
        "option_stage": None,
    }
    guard = {
        "pending_response": {
            "action": "chi",
            "expires_at": 108.0,
            "attempts": 2,
        }
    }

    assert _should_halt_for_failed_chi_expand(result, guard)


def test_infer_seat_role_prefers_dealer_marker_metadata_when_count_matches():
    result = {
        "seat_role": {
            "role": "dealer",
            "expected_total": 21,
            "confidence": 0.8,
            "reason": "dealer_marker_gold_pixels:200",
        },
        "sanity_checks": {"controlled_card_count": 21},
        "discard_button": {"name": "discard"},
    }

    inferred = infer_seat_role(result)

    assert inferred["role"] == "dealer"
    assert inferred["expected_total"] == 21


def test_infer_seat_role_trusts_dealer_marker_when_count_is_twenty():
    result = {
        "seat_role": {
            "role": "dealer",
            "expected_total": 21,
            "confidence": 1.0,
            "reason": "dealer_marker_gold_pixels:5514",
        },
        "sanity_checks": {"controlled_card_count": 20},
        "buttons": [{"name": "chi"}, {"name": "pass"}],
    }

    inferred = infer_seat_role(result)

    assert inferred["role"] == "dealer"
    assert inferred["expected_total"] == 21
    assert inferred["reason"] == "dealer_marker_gold_pixels:5514"


def test_infer_seat_role_keeps_dealer_marker_identity_in_midgame_count_shortage():
    result = {
        "seat_role": {
            "role": "dealer",
            "expected_total": 21,
            "confidence": 1.0,
            "reason": "dealer_marker_gold_pixels:5488",
        },
        "sanity_checks": {"controlled_card_count": 19, "my_meld_cell_count": 3},
        "remaining_deck_count": 39,
        "discard_button": {"name": "discard"},
    }

    inferred = infer_seat_role(result)

    assert inferred["role"] == "dealer"
    assert inferred["expected_total"] == 21
    assert inferred["reason"] == "dealer_marker_gold_pixels:5488"


def test_infer_seat_role_does_not_let_midgame_count_override_dealer_marker():
    result = {
        "seat_role": {
            "role": "dealer",
            "expected_total": 21,
            "confidence": 1.0,
            "reason": "dealer_marker_gold_pixels:6153",
        },
        "sanity_checks": {"controlled_card_count": 22, "my_meld_cell_count": 11, "ok": True},
        "remaining_deck_count": 33,
        "action_plan": {
            "action": "discard",
            "ready": True,
            "target_type": "hand_card",
        },
    }

    inferred = infer_seat_role(result)

    assert inferred["role"] == "dealer"
    assert inferred["expected_total"] == 21
    assert inferred["reason"] == "dealer_marker_gold_pixels:6153"
    assert _can_continue_response_window_without_seat(result)


def test_unknown_seat_midgame_discard_still_blocks_response_window():
    result = {
        "sanity_checks": {"controlled_card_count": 22, "my_meld_cell_count": 11, "ok": True},
        "remaining_deck_count": 33,
        "buttons": [{"name": "chi"}, {"name": "pass"}],
        "action_plan": {
            "action": "discard",
            "ready": True,
            "target_type": "hand_card",
        },
    }

    assert not _can_continue_response_window_without_seat(result)


def test_infer_seat_role_trusts_dealer_marker_for_opening_occlusion_shortage():
    result = {
        "seat_role": {
            "role": "dealer",
            "expected_total": 21,
            "confidence": 1.0,
            "reason": "dealer_marker_gold_pixels:5488",
        },
        "sanity_checks": {"controlled_card_count": 19, "my_meld_cell_count": 0},
        "remaining_deck_count": 45,
        "discard_button": {"name": "discard"},
    }

    inferred = infer_seat_role(result)

    assert inferred["role"] == "dealer"
    assert inferred["expected_total"] == 21
    assert inferred["reason"].startswith("dealer_marker_opening_shortage_19")


def test_infer_seat_role_trusts_dealer_marker_at_eighteen_with_no_round_history():
    result = {
        "seat_role": {
            "role": "dealer",
            "expected_total": 21,
            "confidence": 1.0,
            "reason": "dealer_marker_gold_pixels:6131",
        },
        "sanity_checks": {"controlled_card_count": 18, "my_meld_cell_count": 0},
        "remaining_deck_count": 43,
        "discard_button": {"name": "discard"},
        "buttons": [],
        "option_details": [],
        "discards": {"my_discards": [], "opponent_discards": []},
    }

    inferred = infer_seat_role(result)

    assert inferred["role"] == "dealer"
    assert inferred["expected_total"] == 21
    assert inferred["reason"].startswith("dealer_marker_opening_shortage_18")


def test_auto_seat_result_with_missing_expected_total_is_not_reused():
    result = {
        "sanity_checks": {"ok": True, "controlled_card_count": 20, "expected_total": None},
        "hand": ["一"] * 20,
        "action_plan": {"action": "pass", "reason": "ok"},
    }

    assert not _can_reuse_auto_seat_result(result, {"role": "player", "expected_total": 20})


def test_infer_seat_role_falls_back_to_unknown_when_own_turn_twenty_cards():
    result = {
        "sanity_checks": {"controlled_card_count": 20},
        "discard_button": {"name": "discard"},
    }

    inferred = infer_seat_role(result)

    assert inferred["role"] == "unknown"
    assert inferred["expected_total"] is None


def test_loop_stops_for_play_actions():
    assert should_stop_for_user({"action_plan": {"action": "discard"}, "flow": {"state": "play"}})
    assert should_stop_for_user({"action_plan": {"action": "hu"}, "flow": {"state": "play"}})
    assert should_stop_for_user({"action_plan": {"action": "pass"}, "flow": {"state": "play"}})


def test_loop_continues_for_ready_play_actions_when_execution_enabled():
    result = {"action_plan": {"action": "discard", "ready": True}, "flow": {"state": "play"}}

    assert not should_stop_for_user(result, execute_play_actions=True)
    assert not should_stop_for_user(
        {"action_plan": {"action": "pass", "ready": True}, "flow": {"state": "play"}},
        execute_play_actions=True,
    )


def test_loop_stops_for_unready_play_actions_even_when_execution_enabled():
    result = {"action_plan": {"action": "discard", "ready": False}, "flow": {"state": "play"}}

    assert should_stop_for_user(result, execute_play_actions=True)


def test_loop_stops_for_unready_pass_when_button_is_missing():
    result = {
        "action_plan": {
            "action": "pass",
            "ready": False,
            "reason": "未找到 pass 按钮",
            "validation": {"passed": False},
        },
        "flow": {"state": "play"},
    }

    assert should_stop_for_user(result, execute_play_actions=True)


def test_loop_continues_for_wait_and_settlement_ready():
    assert not should_stop_for_user({"action_plan": {"action": "wait_auto_meld"}, "flow": {"state": "play"}})
    assert not should_stop_for_user(
        {"action_plan": {"action": "settlement_ready"}, "flow": {"state": "settlement_ready"}}
    )


def test_automatic_loop_waits_through_nonfatal_halts_by_default():
    assert _continue_nonfatal_halts(
        loop=True,
        execute_play_actions=True,
        explicitly_continue=False,
        explicitly_stop=False,
    )
    assert not _continue_nonfatal_halts(
        loop=True,
        execute_play_actions=True,
        explicitly_continue=False,
        explicitly_stop=True,
    )


def test_loop_stops_before_capture_when_external_runtime_is_unhealthy(monkeypatch):
    monkeypatch.setattr(
        "tools.live_assistant.run_once",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("run_once should not be called")),
    )

    result = run_loop(
        max_steps=1,
        external_health_check=lambda: "packet_capture_stopped",
    )

    assert result["executed"] is False
    assert result["action_plan"]["action"] == "safe_halt"
    assert result["action_plan"]["reason"] == "packet_capture_stopped"


def test_execute_action_plan_taps_all_clicks(monkeypatch):
    calls = []

    def fake_tap(x, y, device_id=None):
        calls.append((x, y, device_id))

    monkeypatch.setattr("tools.live_assistant.tap", fake_tap)
    monkeypatch.setattr("tools.live_assistant.time.sleep", lambda seconds: None)

    executed = execute_action_plan(
        {
                "action": "chi_option",
                "ready": True,
                "policy_selected_action": {"type": "chi_option", "label": None},
                "validation": {"passed": True, "target_matches_policy": True},
                "clicks": [
                    {"target": "chi:option_1", "x": 10, "y": 20, "delay_ms": 80},
                    {"target": "compare:option_1", "x": 30, "y": 40, "delay_ms": 80},
            ],
        },
        device_id="dev",
    )

    assert executed is True
    assert calls == [(10, 20, "dev"), (30, 40, "dev")]


def test_execute_action_plan_blocks_plan_without_action_type(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "tools.live_assistant.tap",
        lambda x, y, device_id=None: calls.append((x, y, device_id)),
    )

    executed = execute_action_plan(
        {
            "ready": True,
            "validation": {"passed": True},
            "clicks": [{"target": "button:pass", "x": 10, "y": 20}],
        },
        device_id="dev",
    )

    assert executed is False
    assert calls == []


def test_run_once_fatally_halts_a_ready_plan_that_breaks_the_action_contract(monkeypatch, tmp_path):
    screenshot = tmp_path / "screenshot.png"
    screenshot.write_text("fake", encoding="utf-8")
    execute_calls = []
    malformed = {
        "screenshot": str(screenshot),
        "flow": {"state": "play"},
        "action_plan": {
            "ready": True,
            "validation": {"passed": True},
            "clicks": [{"target": "button:pass", "x": 1900, "y": 510}],
        },
        "sanity_checks": {"controlled_card_count": 20},
        "decision": {"action": "pass"},
        "hand": ["一", "二", "三"],
    }

    monkeypatch.setattr("tools.live_assistant.save_screen", lambda device_id="dev": screenshot)
    monkeypatch.setattr("tools.live_assistant.detect_flow_state_from_path", lambda path: FlowDetection("play", 0.0))
    monkeypatch.setattr("tools.live_assistant.recommend_from_screenshot", lambda *args, **kwargs: copy.deepcopy(malformed))
    monkeypatch.setattr("tools.live_assistant.execute_action_plan", lambda *args, **kwargs: execute_calls.append(True))

    output = run_once(device_id="dev", execute_play_actions=True, guard_file=tmp_path / "guard.json")

    assert execute_calls == []
    assert output["fatal"] is True
    assert output["action_plan"]["reason"] == "action_plan_contract_error:action_plan_missing_action"


def test_execute_action_plan_runs_compact_drag_sequence(monkeypatch):
    drags = []
    monkeypatch.setattr(
        "tools.live_assistant.drag",
        lambda x1, y1, x2, y2, **kwargs: drags.append((x1, y1, x2, y2, kwargs)),
    )

    executed = execute_action_plan(
        {
            "action": "compact_hand",
            "ready": True,
            "execution_mode": "drag_sequence",
            "validation": {"passed": True},
            "clicks": [
                {
                    "target": "hand:compact",
                    "from_x": 1700,
                    "from_y": 1000,
                    "x": 700,
                    "y": 1000,
                    "duration_ms": 520,
                }
            ],
        },
        device_id="dev",
    )

    assert executed is True
    assert drags == [(1700, 1000, 700, 1000, {"duration_ms": 520, "device_id": "dev"})]


def test_execute_action_plan_drags_discard_when_plan_uses_drag_mode(monkeypatch):
    taps = []
    drags = []

    def fake_tap(x, y, device_id=None):
        taps.append((x, y, device_id))

    def fake_discard_drag(x, y, *, device_id=None, config_path=None):
        drags.append((x, y, device_id, str(config_path)))

    monkeypatch.setattr("tools.live_assistant.tap", fake_tap)
    monkeypatch.setattr("tools.live_assistant.execute_discard_drag", fake_discard_drag)

    executed = execute_action_plan(
        {
            "action": "discard",
                "ready": True,
                "execution_mode": "discard_drag",
                "policy_selected_action": {"type": "discard", "label": "Ҽ"},
                "validation": {"passed": True, "target_matches_policy": True},
            "clicks": [
                {"target": "hand:壹", "x": 509, "y": 1004, "delay_ms": 80},
            ],
        },
        device_id="dev",
    )

    assert executed is True
    assert taps == []
    assert drags == [(509, 1004, "dev", "D:\\Codex Work\\aizipai\\chenzhou_zipai_ai\\config\\screen_1080x2400.yaml")]


def test_execute_action_plan_selects_card_then_taps_discard_button(monkeypatch):
    taps = []
    drags = []

    def fake_tap(x, y, device_id=None):
        taps.append((x, y, device_id))

    def fake_discard_drag(x, y, *, device_id=None, config_path=None):
        drags.append((x, y, device_id, str(config_path)))

    monkeypatch.setattr("tools.live_assistant.tap", fake_tap)
    monkeypatch.setattr("tools.live_assistant.execute_discard_drag", fake_discard_drag)
    monkeypatch.setattr("tools.live_assistant.time.sleep", lambda seconds: None)

    executed = execute_action_plan(
        {
            "action": "discard",
                "ready": True,
                "execution_mode": "select_then_discard_button",
                "policy_selected_action": {"type": "discard", "label": "Ҽ"},
                "validation": {"passed": True, "target_matches_policy": True},
            "clicks": [
                {"target": "hand:壹", "x": 509, "y": 1004, "delay_ms": 80},
                {"target": "button:discard", "x": 1172, "y": 511, "delay_ms": 80},
            ],
        },
        device_id="dev",
    )

    assert executed is True
    assert taps == [(509, 1004, "dev"), (1172, 511, "dev")]
    assert drags == []


def test_execute_action_plan_blocks_missing_validation(monkeypatch):
    calls = []

    def fake_tap(x, y, device_id=None):
        calls.append((x, y, device_id))

    monkeypatch.setattr("tools.live_assistant.tap", fake_tap)

    executed = execute_action_plan(
        {
            "ready": True,
            "clicks": [{"x": 10, "y": 20, "delay_ms": 80}],
        },
        device_id="dev",
    )

    assert executed is False
    assert calls == []


def test_execute_action_plan_blocks_failed_validation(monkeypatch):
    calls = []

    def fake_tap(x, y, device_id=None):
        calls.append((x, y, device_id))

    monkeypatch.setattr("tools.live_assistant.tap", fake_tap)

    executed = execute_action_plan(
        {
            "ready": True,
            "validation": {"passed": False, "reason_code": "action_plan_policy_mismatch"},
            "clicks": [{"x": 10, "y": 20, "delay_ms": 80}],
        },
        device_id="dev",
    )

    assert executed is False
    assert calls == []


def test_plan_signature_tracks_action_and_clicks():
    signature = _plan_signature(
        {
            "action": "chi_option",
            "ready": True,
            "clicks": [{"target": "chi:叁肆伍", "x": 1710, "y": 208}],
        }
    )

    assert signature == ("chi_option", (("chi:叁肆伍", 1710, 208),))


def test_result_context_signature_changes_with_state():
    base = {
        "flow": {"state": "play"},
        "remaining_deck_count": 34,
        "sanity_checks": {"controlled_card_count": 20},
        "hand": ["王", "贰", "三", "四", "五"],
        "option_stage": "chi",
        "decision": {"action": "chi"},
    }
    another = {
        "flow": {"state": "play"},
        "remaining_deck_count": 35,
        "sanity_checks": {"controlled_card_count": 20},
        "hand": ["王", "贰", "三", "四", "五"],
        "option_stage": "chi",
        "decision": {"action": "chi"},
    }

    assert _result_context_signature(base) != _result_context_signature(another)


def test_result_context_signature_uses_the_entire_hand_not_only_first_eight_cards():
    base = {
        "flow": {"state": "play"},
        "hand": ["一", "二", "三", "四", "五", "六", "七", "八", "九"],
        "decision": {"action": "discard"},
    }
    changed_tail = {**base, "hand": ["一", "二", "三", "四", "五", "六", "七", "八", "十"]}

    assert _result_context_signature(base) != _result_context_signature(changed_tail)


def test_run_once_blocks_repeated_execution_if_last_guard_matches(monkeypatch, tmp_path):
    screenshot = tmp_path / "screenshot.png"
    screenshot.write_text("fake", encoding="utf-8")
    execute_calls: list[bool] = []

    action_plan = {
        "action": "chi_option",
        "ready": True,
        "reason": "test",
        "policy_selected_action": {"type": "chi_option", "label": None},
        "validation": {"passed": True, "target_matches_policy": True},
        "clicks": [{"target": "chi:AB", "x": 1922, "y": 194, "delay_ms": 80}],
    }
    result = {
        "screenshot": str(screenshot),
        "flow": {"state": "play"},
        "action_plan": action_plan,
        "sanity_checks": {"controlled_card_count": 20},
        "remaining_deck_count": 34,
        "option_stage": "chi",
        "decision": {"action": "chi"},
        "hand": ["A", "B", "C", "D", "E"],
    }

    def fake_save_screen(device_id="3B1F5WEA9BBUX9ZQ"):
        return screenshot

    def fake_execute(*args, **kwargs):
        execute_calls.append(True)
        return True

    monkeypatch.setattr("tools.live_assistant.save_screen", fake_save_screen)
    monkeypatch.setattr("tools.live_assistant.detect_flow_state_from_path", lambda path: FlowDetection("play", 0.0))
    monkeypatch.setattr("tools.live_assistant.recommend_from_screenshot", lambda *args, **kwargs: copy.deepcopy(result))
    monkeypatch.setattr("tools.live_assistant._load_guard", lambda _: {
        "last_executed_signature": ["chi_option", [["chi:AB", 1922, 194]]],
        "last_context": _result_context_signature(result),
    })
    monkeypatch.setattr("tools.live_assistant.execute_action_plan", fake_execute)

    output = run_once(
        device_id="3B1F5WEA9BBUX9ZQ",
        execute_play_actions=True,
        expected_total=20,
        guard_file=tmp_path / "guard.json",
    )

    assert execute_calls == []
    assert output["action_plan"]["action"] == "wait_duplicate_execution"
    assert output["action_plan"]["reason"] == "上一次执行后画面未变化，禁止重复点击同一目标，等待下一帧"


def test_run_once_executes_when_last_guard_context_differs(monkeypatch, tmp_path):
    screenshot = tmp_path / "screenshot.png"
    screenshot.write_text("fake", encoding="utf-8")
    execute_calls: list[bool] = []

    result = {
        "screenshot": str(screenshot),
        "flow": {"state": "play"},
        "action_plan": {
            "action": "discard",
            "ready": True,
            "reason": "test",
            "policy_selected_action": {"type": "discard", "label": "八"},
            "validation": {"passed": True, "target_matches_policy": True},
            "clicks": [{"target": "hand:八", "x": 1540, "y": 1006, "delay_ms": 80}],
        },
        "sanity_checks": {"controlled_card_count": 19},
        "remaining_deck_count": 39,
        "option_stage": "discard",
        "decision": {"action": "discard"},
        "hand": ["A", "B", "C", "D", "E"],
    }

    monkeypatch.setattr("tools.live_assistant.save_screen", lambda device_id="dev": screenshot)
    monkeypatch.setattr("tools.live_assistant.detect_flow_state_from_path", lambda path: FlowDetection("play", 0.0))
    monkeypatch.setattr("tools.live_assistant.recommend_from_screenshot", lambda *args, **kwargs: copy.deepcopy(result))
    old_context = _result_context_signature(result)
    old_context["remaining_deck_count"] = 38
    monkeypatch.setattr("tools.live_assistant._load_guard", lambda _: {
        "last_executed_signature": ["discard", [["hand:八", 1540, 1006]]],
        "last_context": old_context,
    })
    monkeypatch.setattr("tools.live_assistant.execute_action_plan", lambda *args, **kwargs: execute_calls.append(True) or True)

    output = run_once(
        device_id="dev",
        execute_play_actions=True,
        expected_total=20,
        guard_file=tmp_path / "guard.json",
    )

    assert execute_calls == [True]
    assert output["executed"] is True


def test_run_once_allows_response_action_when_auto_seat_is_uncertain(monkeypatch, tmp_path):
    screenshot = tmp_path / "screenshot.png"
    screenshot.write_text("fake", encoding="utf-8")
    execute_calls: list[dict] = []

    result = {
        "screenshot": str(screenshot),
        "flow": {"state": "play"},
        "action_plan": {
            "action": "expand_chi_options",
            "ready": True,
            "reason": "展开吃牌候选，仅用于评估，不代表选择吃牌",
            "policy_selected_action": {"type": "expand_chi_options", "label": None},
            "validation": {"passed": True, "target_matches_policy": True},
            "clicks": [{"target": "button:chi", "x": 1925, "y": 520, "delay_ms": 80}],
        },
        "buttons": [{"name": "chi"}, {"name": "pass"}],
        "sanity_checks": {"controlled_card_count": 17, "expected_total": None},
        "remaining_deck_count": None,
        "option_stage": None,
        "decision": {"action": "expand_chi_options"},
        "decision_id": "decision_000001",
        "hand": ["一", "一", "壹", "壹", "二", "二", "贰", "贰", "三", "三", "七", "七", "九", "九", "玖", "十", "拾"],
    }

    monkeypatch.setattr("tools.live_assistant.save_screen", lambda device_id="dev": screenshot)
    monkeypatch.setattr("tools.live_assistant.detect_flow_state_from_path", lambda path: FlowDetection("play", 0.0))
    monkeypatch.setattr(
        "tools.live_assistant.inspect_screenshot",
        lambda *args, **kwargs: {
            **copy.deepcopy(result),
            "seat_role": {
                "role": "unknown",
                "expected_total": None,
                "confidence": 0.0,
                "reason": "response_window",
            },
        },
    )
    monkeypatch.setattr("tools.live_assistant.recommend_from_screenshot", lambda *args, **kwargs: copy.deepcopy(result))
    monkeypatch.setattr("tools.live_assistant.execute_action_plan", lambda plan, **kwargs: execute_calls.append(copy.deepcopy(plan)) or True)

    output = run_once(
        device_id="dev",
        execute_play_actions=True,
        seat_role="auto",
        guard_file=tmp_path / "guard.json",
    )

    assert execute_calls
    assert execute_calls[0]["action"] == "expand_chi_options"
    assert output["executed"] is True
    assert output["seat_role"]["expected_total"] is None


def test_run_once_blocks_execution_after_logger_write_failure(monkeypatch, tmp_path):
    screenshot = tmp_path / "screenshot.png"
    screenshot.write_text("fake", encoding="utf-8")
    execute_calls: list[bool] = []

    class FailingLogger:
        session_active = True
        round_active = True

        def __init__(self) -> None:
            self.log_write_failures = 0
            self.safe_halts: list[str] = []
            self.errors: list[str] = []

        def next_frame_id(self):
            return "frame_000001"

        def log_event(self, *args, **kwargs):
            self.log_write_failures += 1

        def log_error(self, *args, **kwargs):
            self.errors.append(kwargs.get("message") or (args[1] if len(args) > 1 else ""))

        def log_safe_halt(self, *args, **kwargs):
            self.safe_halts.append(kwargs.get("reason"))

    result = {
        "screenshot": str(screenshot),
        "flow": {"state": "play"},
        "action_plan": {
            "action": "discard",
            "ready": True,
            "reason": "test",
            "policy_selected_action": {"type": "discard", "label": "九"},
            "validation": {"passed": True, "target_matches_policy": True},
            "clicks": [{"target": "hand:九", "x": 100, "y": 200, "delay_ms": 80}],
        },
        "sanity_checks": {"controlled_card_count": 20},
        "remaining_deck_count": 34,
        "option_stage": "discard",
        "decision": {"action": "discard"},
        "decision_id": "decision_000001",
        "hand": ["一", "二", "三", "九"],
    }

    monkeypatch.setattr("tools.live_assistant.save_screen", lambda device_id="dev": screenshot)
    monkeypatch.setattr("tools.live_assistant.detect_flow_state_from_path", lambda path: FlowDetection("play", 0.0))
    monkeypatch.setattr("tools.live_assistant.recommend_from_screenshot", lambda *args, **kwargs: copy.deepcopy(result))

    def fake_execute(*args, **kwargs):
        execute_calls.append(True)
        return True

    monkeypatch.setattr("tools.live_assistant.execute_action_plan", fake_execute)

    logger = FailingLogger()
    output = run_once(
        device_id="dev",
        execute_play_actions=True,
        expected_total=20,
        logger=logger,
        guard_file=tmp_path / "guard.json",
    )

    assert execute_calls == []
    assert output["action_plan"]["action"] == "safe_halt"
    assert output["action_plan"]["reason"] == "logger_write_failed"
    assert output["critical_error"]["type"] == "LOGGER_WRITE_FAILED"
    assert "CRITICAL_LOGGER_WRITE_FAILED" in logger.errors
    assert "logger_write_failed" in logger.safe_halts


def test_run_once_allows_execution_when_logger_failure_is_historical(monkeypatch, tmp_path):
    screenshot = tmp_path / "screenshot.png"
    screenshot.write_text("fake", encoding="utf-8")
    execute_calls: list[dict] = []

    class HistoricalFailureLogger:
        session_active = True
        round_active = True

        def __init__(self) -> None:
            self.log_write_failures = 1
            self.taps: list[dict] = []

        def next_frame_id(self):
            return "frame_000001"

        def log_event(self, *args, **kwargs):
            return None

        def log_safe_halt(self, *args, **kwargs):
            raise AssertionError("historical logger failures must not safe-halt current clicks")

        def log_tap(self, *args, **kwargs):
            self.taps.append(kwargs)

    result = {
        "screenshot": str(screenshot),
        "flow": {"state": "play"},
        "action_plan": {
            "action": "hu",
            "ready": True,
            "reason": "点击 button hu",
            "policy_selected_action": {"type": "hu", "label": None},
            "validation": {"passed": True, "target_matches_policy": True},
            "clicks": [{"target": "button:hu", "x": 1985, "y": 510, "delay_ms": 80}],
        },
        "sanity_checks": {"controlled_card_count": 8},
        "remaining_deck_count": 73,
        "option_stage": None,
        "decision": {"action": "hu"},
        "decision_id": "decision_000001",
        "hand": ["一", "一", "一", "肆"],
    }

    monkeypatch.setattr("tools.live_assistant.save_screen", lambda device_id="dev": screenshot)
    monkeypatch.setattr("tools.live_assistant.detect_flow_state_from_path", lambda path: FlowDetection("play", 0.0))
    monkeypatch.setattr("tools.live_assistant.recommend_from_screenshot", lambda *args, **kwargs: copy.deepcopy(result))

    def fake_execute(plan, **kwargs):
        execute_calls.append(copy.deepcopy(plan))
        return True

    monkeypatch.setattr("tools.live_assistant.execute_action_plan", fake_execute)

    logger = HistoricalFailureLogger()
    output = run_once(
        device_id="dev",
        execute_play_actions=True,
        expected_total=20,
        logger=logger,
        guard_file=tmp_path / "guard.json",
    )

    assert output["executed"] is True
    assert execute_calls[0]["action"] == "hu"
    assert logger.taps


def test_run_once_does_not_execute_when_write_ahead_guard_cannot_be_saved(monkeypatch, tmp_path):
    screenshot = tmp_path / "screenshot.png"
    screenshot.write_text("fake", encoding="utf-8")
    execute_calls = []
    result = {
        "screenshot": str(screenshot),
        "flow": {"state": "play"},
        "action_plan": {
            "action": "pass",
            "ready": True,
            "policy_selected_action": {"type": "pass", "label": None},
            "validation": {"passed": True, "target_matches_policy": True},
            "clicks": [{"target": "button:pass", "x": 1900, "y": 510}],
        },
        "sanity_checks": {"controlled_card_count": 20},
        "decision": {"action": "pass"},
        "hand": ["一", "二", "三"],
    }

    monkeypatch.setattr("tools.live_assistant.save_screen", lambda device_id="dev": screenshot)
    monkeypatch.setattr("tools.live_assistant.detect_flow_state_from_path", lambda path: FlowDetection("play", 0.0))
    monkeypatch.setattr("tools.live_assistant.recommend_from_screenshot", lambda *args, **kwargs: copy.deepcopy(result))
    monkeypatch.setattr("tools.live_assistant._save_guard", lambda *args, **kwargs: False)
    monkeypatch.setattr("tools.live_assistant.execute_action_plan", lambda *args, **kwargs: execute_calls.append(True))

    output = run_once(
        device_id="dev",
        execute_play_actions=True,
        guard_file=tmp_path / "guard.json",
    )

    assert execute_calls == []
    assert output["fatal"] is True
    assert output["action_plan"]["reason"] == "action_guard_write_failed:before_execution"


def test_run_loop_blocks_repeated_play_action_if_state_stays_same(monkeypatch):
    base = {
        "action_plan": {
            "action": "chi_option",
            "ready": True,
            "reason": "test",
            "clicks": [{"target": "chi:AB", "x": 1922, "y": 194, "delay_ms": 80}],
        },
        "executed": True,
        "flow": {"state": "play"},
        "sanity_checks": {"controlled_card_count": 20},
        "remaining_deck_count": 34,
        "option_stage": "chi",
        "decision": {"action": "chi"},
        "hand": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"],
    }

    monkeypatch.setattr("tools.live_assistant.run_once", lambda **_: copy.deepcopy(base))
    monkeypatch.setattr("tools.live_assistant.time.sleep", lambda _: None)
    result = run_loop(
        execute_play_actions=True,
        max_steps=3,
    )

    assert result["action_plan"]["action"] == "wait_opponent_priority"


def test_run_loop_waits_for_repeated_settlement_ready_without_halting(monkeypatch):
    base = {
        "action_plan": {
            "action": "settlement_ready",
            "ready": True,
            "clicks": [{"target": "settlement_ready", "x": 1171, "y": 615, "delay_ms": 80}],
        },
        "executed": True,
        "flow": {"state": "settlement_ready"},
        "sanity_checks": {},
        "remaining_deck_count": None,
        "hand": [],
        "decision": {},
    }

    monkeypatch.setattr("tools.live_assistant.run_once", lambda **_: copy.deepcopy(base))
    monkeypatch.setattr("tools.live_assistant.time.sleep", lambda _: None)

    result = run_loop(
        execute_play_actions=True,
        max_steps=3,
    )

    assert result["action_plan"]["action"] == "wait_next_round"
    assert result["action_plan"]["reason"] == "已点击准备，等待牌局开始"


def test_run_loop_retries_transient_runtime_errors(monkeypatch):
    calls = []
    sleeps = []

    def fake_run_once(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise OSError("temporary screenshot failure")
        return {
            "action_plan": {"action": "wait", "ready": False, "clicks": []},
            "executed": False,
            "flow": {"state": "play"},
        }

    monkeypatch.setattr("tools.live_assistant.run_once", fake_run_once)
    monkeypatch.setattr("tools.live_assistant.time.sleep", lambda seconds: sleeps.append(seconds))

    result = run_loop(max_steps=2, interval_seconds=0.5)

    assert len(calls) == 2
    assert sleeps
    assert result["action_plan"]["action"] == "wait"


def test_loop_shortens_only_chi_chain_recheck_interval():
    assert _loop_interval_after_result(
        {"executed": True, "action_plan": {"action": "expand_chi_options"}},
        0.5,
    ) == 0.15
    assert _loop_interval_after_result(
        {"executed": True, "action_plan": {"action": "chi"}},
        0.5,
    ) == 0.15
    assert _loop_interval_after_result(
        {"executed": True, "action_plan": {"action": "compare_option"}},
        0.5,
    ) == 0.15
    assert _loop_interval_after_result(
        {"executed": True, "action_plan": {"action": "discard"}},
        0.5,
    ) == 0.5
    assert _loop_interval_after_result(
        {"executed": False, "action_plan": {"action": "chi"}},
        0.5,
    ) == 0.5


def test_run_loop_stops_on_unexpected_programming_error(monkeypatch):
    monkeypatch.setattr("tools.live_assistant.run_once", lambda **kwargs: (_ for _ in ()).throw(ValueError("bug")))

    result = run_loop(max_steps=3)

    assert result["fatal"] is True
    assert result["runtime_error"]["transient"] is False
    assert result["action_plan"]["action"] == "safe_halt"


def test_run_loop_passes_bounded_runtime_screenshot_path(monkeypatch, tmp_path):
    seen = []
    target = tmp_path / "live_current.png"

    def fake_run_once(**kwargs):
        seen.append(kwargs["runtime_screenshot_path"])
        return {
            "action_plan": {"action": "wait", "ready": False, "clicks": []},
            "executed": False,
            "flow": {"state": "play"},
        }

    monkeypatch.setattr("tools.live_assistant.run_once", fake_run_once)
    monkeypatch.setattr("tools.live_assistant.time.sleep", lambda _: None)

    run_loop(max_steps=1, runtime_screenshot_path=target)

    assert seen == [target]


def test_run_loop_wraps_existing_protocol_log_in_incremental_tail(monkeypatch, tmp_path):
    protocol_log = tmp_path / "qs_packets.jsonl"
    protocol_log.write_text("", encoding="utf-8")
    seen_protocol_payloads = []

    def fake_run_once(**kwargs):
        seen_protocol_payloads.append(kwargs.get("protocol_payload"))
        return {
            "action_plan": {"action": "wait", "ready": False, "clicks": []},
            "executed": False,
            "flow": {"state": "play"},
            "sanity_checks": {"controlled_card_count": 20},
            "hand": ["一", "二"],
            "decision": {"action": "wait"},
        }

    monkeypatch.setattr("tools.live_assistant.run_once", fake_run_once)
    monkeypatch.setattr("tools.live_assistant.time.sleep", lambda _: None)

    run_loop(protocol_payload=protocol_log, max_steps=1)

    assert seen_protocol_payloads
    assert seen_protocol_payloads[0] is _prepare_protocol_payload_for_loop(protocol_log) or hasattr(
        seen_protocol_payloads[0],
        "read_latest",
    )


def test_required_plan_blocks_drifted_execution_target():
    plan = {
        "action": "discard",
        "ready": True,
        "clicks": [{"target": "hand:玖", "x": 1685, "y": 1004}],
    }

    assert _matches_required_plan(plan, "discard", "hand:玖")
    assert not _matches_required_plan(plan, "discard", "hand:伍")


def test_dealer_after_first_discard_retries_as_twenty_cards():
    result = {
        "sanity_checks": {"ok": False, "controlled_card_count": 20},
        "discard_button": None,
    }

    assert _should_retry_dealer_after_first_discard(result, 21)


def test_dealer_response_buttons_retry_as_twenty_cards():
    result = {
        "sanity_checks": {"ok": False, "controlled_card_count": 20},
        "discard_button": None,
        "buttons": [{"name": "chi"}, {"name": "pass"}],
        "decision": {"action": "pass"},
    }

    assert _should_retry_dealer_after_first_discard(result, 21)


def test_dealer_own_turn_does_not_retry_as_twenty_cards():
    result = {
        "sanity_checks": {"ok": False, "controlled_card_count": 20},
        "decision": {"action": "discard"},
        "memory": {"frames_seen": 1},
    }

    assert not _should_retry_dealer_after_first_discard(result, 21)


def test_dealer_later_own_turn_does_not_retry_as_twenty_cards():
    result = {
        "sanity_checks": {"ok": False, "controlled_card_count": 20},
        "decision": {"action": "discard"},
        "memory": {"frames_seen": 3},
    }

    assert not _should_retry_dealer_after_first_discard(result, 21)


def test_dealer_visible_discard_button_does_not_retry_as_twenty_cards():
    result = {
        "sanity_checks": {"ok": False, "controlled_card_count": 20},
        "discard_button": {"name": "discard"},
        "memory": {"frames_seen": 3},
    }

    assert not _should_retry_dealer_after_first_discard(result, 21)


def test_protocol_binding_shortage_still_auto_compacts_dealer_opening():
    result = {
        "seat_role": {
            "role": "dealer",
            "confidence": 1.0,
            "reason": "dealer_marker_gold_pixels:42",
        },
        "discard_button": {"name": "discard"},
        "buttons": [],
        "options": [],
        "option_details": [],
        "metadata": {
            "hand_count": 21,
            "protocol_hand_binding": {
                "protocol_count": 21,
                "visible_slot_count": 20,
                "missing_coordinate_count": 1,
            },
        },
        "sanity_checks": {
            "ok": True,
            "controlled_card_count": 21,
            "my_meld_cell_count": 0,
        },
        "discards": {"my_discards": []},
        "hand": [{"name": "壹"}] * 21,
        "action_plan": {"action": "discard", "ready": True},
    }

    assert _should_auto_compact_dealer_hand(result)


def test_dealer_opening_shortage_compacts_without_an_extra_retry_frame(tmp_path):
    guard_file = tmp_path / "guard.json"
    result = {
        "sanity_checks": {"controlled_card_count": 18},
        "action_plan": {"action": "discard", "ready": False},
    }

    result = _defer_opening_compact_for_recognition_retry(
        result,
        guard_file=guard_file,
        guard_state={},
    )

    assert result is None
    assert not guard_file.exists()


def test_single_tap_opening_shortage_waits_without_moving_cards():
    result = {
        "sanity_checks": {
            "controlled_card_count": 18,
            "my_meld_cell_count": 0,
        }
    }

    output = _wait_for_stable_opening_layout(result)

    assert output["executed"] is False
    assert output["action_plan"]["action"] == "wait_initial_layout"
    assert output["action_plan"]["clicks"] == []
    assert output["action_plan"]["validation"]["reason_code"] == "opening_layout_not_stable"


def test_mobile_single_tap_shortage_never_calls_compact(monkeypatch, tmp_path):
    screenshot = tmp_path / "screenshot.png"
    screenshot.write_text("fake", encoding="utf-8")
    recognized = {
        "seat_role": {
            "role": "dealer",
            "expected_total": 21,
            "confidence": 1.0,
            "reason": "dealer_marker_gold_pixels:42",
        },
        "discard_button": {"name": "discard"},
        "buttons": [],
        "options": [],
        "option_details": [],
        "metadata": {"hand_count": 18},
        "sanity_checks": {
            "ok": False,
            "controlled_card_count": 18,
            "my_meld_cell_count": 0,
        },
        "discards": {"my_discards": []},
        "hand": ["壹"] * 18,
        "hand_details": [],
    }
    compact_calls = []

    monkeypatch.setattr("tools.live_assistant.save_screen", lambda **kwargs: screenshot)
    monkeypatch.setattr(
        "tools.live_assistant.detect_flow_state_from_path",
        lambda path: FlowDetection("play", 1.0),
    )
    monkeypatch.setattr(
        "tools.live_assistant.inspect_screenshot",
        lambda *args, **kwargs: copy.deepcopy(recognized),
    )
    monkeypatch.setattr(
        "tools.live_assistant.run_compact",
        lambda **kwargs: compact_calls.append(kwargs),
    )

    output = run_once(
        device_id="dev",
        execute_play_actions=False,
        seat_role="auto",
        guard_file=tmp_path / "guard.json",
        allow_opening_hand_compact=False,
    )

    assert compact_calls == []
    assert output["action_plan"]["action"] == "wait_initial_layout"


def test_dealer_opening_shortage_does_not_compact_without_discard_button():
    result = {
        "seat_role": {
            "role": "dealer",
            "confidence": 1.0,
            "reason": "dealer_marker_gold_pixels:42",
        },
        "discard_button": None,
        "buttons": [],
        "options": [],
        "option_details": [],
        "metadata": {"hand_count": 20},
        "sanity_checks": {
            "ok": False,
            "controlled_card_count": 20,
            "my_meld_cell_count": 0,
        },
        "discards": {"my_discards": []},
        "hand": [{"name": "壹"}] * 20,
        "action_plan": {"action": "pass", "ready": False, "reason": "未找到 pass 按钮"},
    }

    assert not _should_auto_compact_dealer_hand(result)


def test_protocol_binding_shortage_compacts_before_using_a_safe_discard_target():
    result = {
        "seat_role": {
            "role": "dealer",
            "confidence": 1.0,
            "reason": "dealer_marker_gold_pixels:42",
        },
        "discard_button": {"name": "discard"},
        "buttons": [],
        "options": [],
        "option_details": [],
        "metadata": {
            "hand_count": 21,
            "protocol_hand_binding": {
                "protocol_count": 21,
                "visible_slot_count": 19,
                "missing_coordinate_count": 2,
            },
        },
        "sanity_checks": {
            "ok": True,
            "controlled_card_count": 21,
            "my_meld_cell_count": 0,
        },
        "discards": {"my_discards": []},
        "hand_details": [
            {
                "card_id": "h004",
                "name": "玖",
                "x": 585,
                "y": 928,
                "w": 144,
                "h": 152,
                "center": [657, 1004],
                "clickable": True,
            }
        ],
        "hand": [{"name": "壹"}] * 21,
        "action_plan": {
            "action": "discard",
            "ready": True,
            "execution_mode": "discard_drag",
            "target_card_id": "h004",
            "clicks": [{"target": "hand:玖", "x": 657, "y": 1004}],
        },
    }

    assert _should_auto_compact_dealer_hand(result)


def test_stale_compact_guard_is_expired_without_expiring_other_actions():
    stale_compact = {
        "inflight_action": {
            "signature": ["compact_hand", [["hand:compact", 700, 900]]],
            "started_at": 80.0,
        },
        "pending_opening_recognition_retry": {
            "attempts": 1,
            "created_at": 80.0,
        },
    }
    fresh_discard = {
        "inflight_action": {
            "signature": ["discard", [["hand:玖", 700, 900]]],
            "started_at": 1.0,
        }
    }

    assert "inflight_action" not in _expire_stale_guard_state(stale_compact, now=100.0)
    assert "pending_opening_recognition_retry" not in _expire_stale_guard_state(
        stale_compact,
        now=100.0,
    )
    assert "inflight_action" in _expire_stale_guard_state(fresh_discard, now=100.0)


def test_unchanged_frame_waits_before_repeating_strategy_after_execution():
    result = {
        "flow": {"state": "play"},
        "remaining_deck_count": 42,
        "sanity_checks": {"controlled_card_count": 20},
        "hand": ["一", "二", "三"],
        "buttons": [],
        "discard_button": {"name": "discard"},
        "pending_action_card": None,
        "opponent_pending_card": None,
        "option_stage": None,
    }
    previous_context = _result_context_signature(
        {**result, "decision": {"action": "discard"}}
    )
    guard = {
        "last_executed_signature": ["discard", [["hand:三", 900, 900]]],
        "last_context": previous_context,
    }

    waiting = _unchanged_executed_frame_wait(result, guard)

    assert waiting is not None
    assert waiting["action_plan"]["action"] == "wait_unchanged_after_execution"
    assert waiting["executed"] is False
    assert _unchanged_executed_frame_wait({**result, "hand": ["一", "二"]}, guard) is None


def test_chi_expand_uses_response_wait_instead_of_generic_unchanged_guard():
    result = {
        "flow": {"state": "play"},
        "remaining_deck_count": 42,
        "sanity_checks": {"controlled_card_count": 20},
        "hand": ["一", "二", "三"],
        "buttons": [{"name": "chi"}, {"name": "pass"}],
        "discard_button": None,
        "pending_action_card": None,
        "opponent_pending_card": {"label": "三"},
        "option_stage": None,
        "decision": {"action": "expand_chi_options"},
    }
    guard = {
        "last_executed_signature": [
            "expand_chi_options",
            [["button:chi", 1925, 520]],
        ],
        "last_context": _result_context_signature(result),
        "pending_response": {
            "action": "chi",
            "expires_at": 104.0,
            "attempts": 1,
        },
    }

    assert _unchanged_executed_frame_wait(result, guard) is None


def test_print_summary_handles_runtime_error_without_screenshot_or_flow(capsys):
    print_summary(
        {
            "executed": False,
            "runtime_error": {"message": "boom"},
            "action_plan": {
                "action": "safe_halt",
                "ready": False,
                "reason": "runtime_exception",
                "clicks": [],
            },
        }
    )

    output = capsys.readouterr().out
    assert "screenshot=unavailable" in output
    assert "flow=unknown" in output
    assert "runtime_error=boom" in output


def test_dealer_shortage_compacts_before_calling_v81_and_plans_two_drags(monkeypatch, tmp_path):
    screenshot = tmp_path / "screenshot.png"
    screenshot.write_text("fake", encoding="utf-8")
    guard_file = tmp_path / "guard.json"
    guard_file.write_text(
        json.dumps(
            {
                "pending_opening_recognition_retry": {
                    "attempts": 1,
                    "created_at": 100.0,
                }
            }
        ),
        encoding="utf-8",
    )
    hand_details = [
        {
            "card_id": f"h{index:03d}",
            "name": "壹",
            "center": [600 + index * 50, 900],
            "clickable": True,
        }
        for index in range(20)
    ]
    recognized = {
        "screenshot": str(screenshot),
        "seat_role": {
            "role": "dealer",
            "expected_total": 21,
            "confidence": 1.0,
            "reason": "dealer_marker_gold_pixels:42",
        },
        "discard_button": {"name": "discard"},
        "buttons": [],
        "options": [],
        "option_details": [],
        "metadata": {"hand_count": 20},
        "sanity_checks": {
            "ok": False,
            "controlled_card_count": 20,
            "my_meld_cell_count": 0,
        },
        "discards": {"my_discards": []},
        "hand": ["壹"] * 20,
        "hand_details": hand_details,
    }
    recommend_calls = []
    inspect_calls = []

    monkeypatch.setattr("tools.live_assistant.time.time", lambda: 100.0)
    monkeypatch.setattr("tools.live_assistant.save_screen", lambda **kwargs: screenshot)
    monkeypatch.setattr(
        "tools.live_assistant.detect_flow_state_from_path",
        lambda path: FlowDetection("play", 1.0),
    )
    def fake_inspect(*args, **kwargs):
        inspect_calls.append(kwargs)
        return copy.deepcopy(recognized)

    monkeypatch.setattr("tools.live_assistant.inspect_screenshot", fake_inspect)
    monkeypatch.setattr(
        "tools.live_assistant.run_compact",
        lambda **kwargs: {
            "reason": "planned",
            "planned_drags": [
                {
                    "label": "壹",
                    "start_x": 1700,
                    "start_y": 900,
                    "end_x": 700,
                    "end_y": 900,
                },
                {
                    "label": "贰",
                    "start_x": 1600,
                    "start_y": 900,
                    "end_x": 850,
                    "end_y": 900,
                },
                {
                    "label": "叁",
                    "start_x": 1500,
                    "start_y": 900,
                    "end_x": 1000,
                    "end_y": 900,
                },
            ],
        },
    )
    monkeypatch.setattr(
        "tools.live_assistant.recommend_from_screenshot",
        lambda *args, **kwargs: recommend_calls.append(kwargs),
    )

    output = run_once(
        device_id="dev",
        execute_play_actions=False,
        seat_role="auto",
        guard_file=guard_file,
    )

    assert recommend_calls == []
    assert len(inspect_calls) == 1
    assert inspect_calls[0]["infer_expected_total_from_phase"] is True
    assert output["action_plan"]["action"] == "compact_hand"
    assert len(output["action_plan"]["clicks"]) == 2


def test_dealer_complete_count_with_untrusted_card_does_not_recompact(monkeypatch, tmp_path):
    result = {
        "seat_role": {
            "role": "dealer",
            "expected_total": 21,
            "confidence": 1.0,
            "reason": "dealer_marker_gold_pixels:42",
        },
        "discard_button": {"name": "discard"},
        "buttons": [],
        "options": [],
        "option_details": [],
        "metadata": {"hand_count": 21},
        "sanity_checks": {
            "ok": True,
            "controlled_card_count": 21,
            "my_meld_cell_count": 0,
        },
        "discards": {"my_discards": []},
        "hand": ["壹"] * 21,
        "hand_details": [
            {
                "card_id": f"h{index:03d}",
                "name": "壹",
                "center": [500 + index * 50, 900],
                "confidence": 0.72 if index == 20 else 0.99,
                "hybrid_reason": (
                    "insufficient_independent_evidence" if index == 20 else "independent_agreement"
                ),
                "clickable": True,
            }
            for index in range(21)
        ],
    }

    assert _should_auto_compact_dealer_hand(result) is False


def test_dealer_shortage_batches_exactly_the_missing_rightmost_cards(monkeypatch, tmp_path):
    result = {
        "flow": {"state": "play"},
        "hand": ["壹"] * 18,
        "hand_details": [
            {"name": "壹", "center": [500 + index * 60, 900]}
            for index in range(18)
        ],
        "sanity_checks": {"controlled_card_count": 18},
        "discard_button": {"name": "discard"},
    }
    monkeypatch.setattr(
        "tools.live_assistant.run_compact",
        lambda **kwargs: {
            "planned_drags": [
                {
                    "label": label,
                    "start_x": start_x,
                    "start_y": 900,
                    "end_x": end_x,
                    "end_y": 900,
                }
                for label, start_x, end_x in (
                    ("玖", 1900, 650),
                    ("捌", 1750, 800),
                    ("柒", 1600, 950),
                    ("陆", 1450, 1100),
                )
            ]
        },
    )

    output = _handle_opening_compact(
        result,
        screenshot=tmp_path / "screen.png",
        device_id="dev",
        guard_file=tmp_path / "guard.json",
        guard_state={},
        execute_play_actions=False,
        logger=None,
        frame_id="frame-1",
        external_health_check=None,
    )

    assert [click["label"] for click in output["action_plan"]["clicks"]] == ["玖", "捌"]


def test_dealer_shortage_never_batches_two_drags_into_one_target_column(monkeypatch, tmp_path):
    result = {
        "flow": {"state": "play"},
        "hand": ["壹"] * 19,
        "hand_details": [
            {"name": "壹", "center": [500 + index * 60, 900]}
            for index in range(19)
        ],
        "sanity_checks": {"controlled_card_count": 19},
        "discard_button": {"name": "discard"},
    }
    monkeypatch.setattr(
        "tools.live_assistant.run_compact",
        lambda **kwargs: {
            "planned_drags": [
                {
                    "label": "玖",
                    "start_x": 1900,
                    "start_y": 900,
                    "end_x": 650,
                    "end_y": 900,
                },
                {
                    "label": "捌",
                    "start_x": 1750,
                    "start_y": 900,
                    "end_x": 650,
                    "end_y": 900,
                },
                {
                    "label": "柒",
                    "start_x": 1600,
                    "start_y": 900,
                    "end_x": 800,
                    "end_y": 900,
                },
            ]
        },
    )

    output = _handle_opening_compact(
        result,
        screenshot=tmp_path / "screen.png",
        device_id="dev",
        guard_file=tmp_path / "guard.json",
        guard_state={},
        execute_play_actions=False,
        logger=None,
        frame_id="frame-1",
        external_health_check=None,
    )

    assert [click["label"] for click in output["action_plan"]["clicks"]] == ["玖", "柒"]
    assert [click["x"] for click in output["action_plan"]["clicks"]] == [650, 800]


def test_opening_no_movable_cards_gets_one_recheck_then_stops_repeating(
    monkeypatch,
    tmp_path,
):
    result = {
        "flow": {"state": "play"},
        "hand": ["壹"] * 20,
        "hand_details": [
            {
                "card_id": f"h{index:03d}",
                "name": "壹",
                "center": [600 + index * 45, 900],
                "clickable": index < 17,
            }
            for index in range(20)
        ],
        "hand_recognition": {"rejected_candidates": []},
        "sanity_checks": {"controlled_card_count": 20},
        "discard_button": {"name": "discard"},
    }
    guard_file = tmp_path / "guard.json"
    monkeypatch.setattr(
        "tools.live_assistant.run_compact",
        lambda **kwargs: {
            "reason": "no_right_side_cards_to_compact",
            "planned_drags": [],
        },
    )

    first = _handle_opening_compact(
        copy.deepcopy(result),
        screenshot=tmp_path / "screen.png",
        device_id="dev",
        guard_file=guard_file,
        guard_state={},
        execute_play_actions=False,
        logger=None,
        frame_id="frame-1",
        external_health_check=None,
    )

    assert first["action_plan"]["action"] == "wait_opening_locked_recheck"
    assert first["action_plan"]["validation"]["reason_code"] == "opening_locked_recheck"
    saved = json.loads(guard_file.read_text(encoding="utf-8"))
    assert saved["opening_compact_recovery"]["no_movable_rechecks"] == 1

    second = _handle_opening_compact(
        copy.deepcopy(result),
        screenshot=tmp_path / "screen.png",
        device_id="dev",
        guard_file=guard_file,
        guard_state=saved,
        execute_play_actions=False,
        logger=None,
        frame_id="frame-2",
        external_health_check=None,
    )

    assert second["action_plan"]["action"] == "safe_halt"
    assert second["action_plan"]["validation"]["reason_code"] == "opening_compact_exhausted"
    assert "不会继续重复整理" in second["action_plan"]["reason"]


def test_opening_no_movable_recheck_allows_new_drag_after_layout_changes(
    monkeypatch,
    tmp_path,
):
    previous = {
        "opening_compact_recovery": {
            "attempts": 2,
            "no_progress": 0,
            "no_movable_rechecks": 1,
            "no_movable_signature": [[600, 900, True]],
        }
    }
    result = {
        "flow": {"state": "play"},
        "hand": ["贰"] * 20,
        "hand_details": [
            {"name": "贰", "center": [650 + index * 45, 900], "clickable": True}
            for index in range(20)
        ],
        "sanity_checks": {"controlled_card_count": 20},
        "discard_button": {"name": "discard"},
    }
    guard_file = tmp_path / "guard.json"
    guard_file.write_text(json.dumps(previous), encoding="utf-8")
    monkeypatch.setattr(
        "tools.live_assistant.run_compact",
        lambda **kwargs: {
            "planned_drags": [
                {
                    "label": "贰",
                    "start_x": 1700,
                    "start_y": 900,
                    "end_x": 650,
                    "end_y": 900,
                }
            ],
            "reason": "planned",
        },
    )

    output = _handle_opening_compact(
        result,
        screenshot=tmp_path / "screen.png",
        device_id="dev",
        guard_file=guard_file,
        guard_state=previous,
        execute_play_actions=False,
        logger=None,
        frame_id="frame-2",
        external_health_check=None,
    )

    assert output["action_plan"]["action"] == "compact_hand"
    assert output["action_plan"]["clicks"][0]["label"] == "贰"


def test_successful_compact_logs_drag_coordinates_without_crashing(monkeypatch, tmp_path):
    result = {
        "flow": {"state": "play"},
        "hand": ["壹"] * 20,
        "hand_details": [
            {"name": "壹", "center": [500 + index * 60, 900]}
            for index in range(20)
        ],
        "sanity_checks": {"controlled_card_count": 20},
        "discard_button": {"name": "discard"},
    }
    monkeypatch.setattr(
        "tools.live_assistant.run_compact",
        lambda **kwargs: {
            "planned_drags": [
                {
                    "label": "玖",
                    "start_x": 1900,
                    "start_y": 900,
                    "end_x": 650,
                    "end_y": 900,
                }
            ]
        },
    )
    monkeypatch.setattr("tools.live_assistant.execute_action_plan", lambda *args, **kwargs: True)
    after_screenshot = tmp_path / "after.png"
    monkeypatch.setattr("tools.live_assistant.save_screen", lambda **kwargs: after_screenshot)
    monkeypatch.setattr(
        "tools.live_assistant.inspect_screenshot",
        lambda *args, **kwargs: {
            "hand_details": [{"name": "壹", "center": [500 + index * 60, 900]} for index in range(21)]
        },
    )

    class StrictLogger:
        round_active = True

        def __init__(self):
            self.logged = None

        def log_tap(
            self,
            frame_id,
            decision_id,
            plan,
            *,
            execute_enabled,
            dry_run,
            tap_executed,
            tap_x,
            tap_y,
            **kwargs,
        ):
            self.logged = {"tap_x": tap_x, "tap_y": tap_y, **kwargs}

    logger = StrictLogger()
    output = _handle_opening_compact(
        result,
        screenshot=tmp_path / "before.png",
        device_id="dev",
        guard_file=tmp_path / "guard.json",
        guard_state={},
        execute_play_actions=True,
        logger=logger,
        frame_id="frame-1",
        external_health_check=None,
    )

    assert output["executed"] is True
    assert logger.logged["tap_x"] == 650
    assert logger.logged["tap_y"] == 900
    assert logger.logged["target_label"] == "玖"


def test_verified_opening_compact_no_progress_halts_instead_of_waiting_forever(
    monkeypatch,
    tmp_path,
):
    result = {
        "flow": {"state": "play"},
        "hand": ["壹"] * 20,
        "hand_details": [
            {"name": "壹", "center": [600 + index * 50, 900]}
            for index in range(20)
        ],
        "sanity_checks": {"controlled_card_count": 20},
        "discard_button": {"name": "discard"},
        "decision": {"action": "discard"},
    }
    compact_plan = {
        "action": "compact_hand",
        "ready": True,
        "clicks": [
            {
                "target": "hand:compact",
                "x": 700,
                "y": 900,
                "from_x": 1700,
                "from_y": 900,
                "label": "壹",
            }
        ],
    }
    signature = _plan_signature(compact_plan)
    context = _result_context_signature(result)
    guard_state = {
        "last_executed_signature": [
            signature[0],
            [list(click) for click in signature[1]],
        ],
        "last_context": context,
        "opening_compact_recovery": {
            "attempts": 1,
            "no_progress": 1,
            "updated_at": 100.0,
        },
    }
    monkeypatch.setattr(
        "tools.live_assistant.run_compact",
        lambda **kwargs: {
            "planned_drags": [
                {
                    "label": "壹",
                    "start_x": 1700,
                    "start_y": 900,
                    "end_x": 700,
                    "end_y": 900,
                }
            ]
        },
    )

    output = _handle_opening_compact(
        result,
        screenshot=tmp_path / "screen.png",
        device_id="dev",
        guard_file=tmp_path / "guard.json",
        guard_state=guard_state,
        execute_play_actions=True,
        logger=None,
        frame_id="frame-1",
        external_health_check=None,
    )

    assert output["action_plan"]["action"] == "safe_halt"
    assert output["action_plan"]["validation"]["reason_code"] == "opening_compact_no_progress"
    assert "停止重复同一坐标" in output["action_plan"]["reason"]


def test_complete_auto_dealer_frame_calls_v81_once_with_precomputed_state(monkeypatch, tmp_path):
    screenshot = tmp_path / "screenshot.png"
    screenshot.write_text("fake", encoding="utf-8")
    recognized = {
        "screenshot": str(screenshot),
        "seat_role": {
            "role": "dealer",
            "expected_total": 21,
            "confidence": 1.0,
            "reason": "dealer_marker_gold_pixels:42",
        },
        "discard_button": {"name": "discard"},
        "buttons": [],
        "options": [],
        "option_details": [],
        "metadata": {"hand_count": 21},
        "sanity_checks": {
            "ok": True,
            "controlled_card_count": 21,
            "expected_total": None,
            "my_meld_cell_count": 0,
        },
        "discards": {"my_discards": []},
        "hand": ["壹"] * 21,
        "hand_details": [
            {
                "card_id": f"h{index:03d}",
                "name": "壹",
                "center": [500 + index * 30, 900],
                "clickable": True,
            }
            for index in range(21)
        ],
    }
    recommend_calls = []

    monkeypatch.setattr("tools.live_assistant.save_screen", lambda **kwargs: screenshot)
    monkeypatch.setattr(
        "tools.live_assistant.detect_flow_state_from_path",
        lambda path: FlowDetection("play", 1.0),
    )
    monkeypatch.setattr(
        "tools.live_assistant.inspect_screenshot",
        lambda *args, **kwargs: copy.deepcopy(recognized),
    )

    def fake_recommend(*args, **kwargs):
        recommend_calls.append(kwargs)
        return {
            **copy.deepcopy(recognized),
            "sanity_checks": {
                **recognized["sanity_checks"],
                "expected_total": 21,
            },
            "decision": {"action": "wait", "label": None},
            "action_plan": {
                "action": "wait",
                "ready": False,
                "reason": "test",
                "clicks": [],
            },
        }

    monkeypatch.setattr("tools.live_assistant.recommend_from_screenshot", fake_recommend)

    run_once(
        device_id="dev",
        execute_play_actions=False,
        seat_role="auto",
        guard_file=tmp_path / "guard.json",
    )

    assert len(recommend_calls) == 1
    assert recommend_calls[0]["expected_total"] == 21
    assert recommend_calls[0]["precomputed_state"]["hand"] == ["壹"] * 21


def test_relaxes_midgame_count_when_deck_has_advanced_and_discard_is_visible():
    result = {
        "sanity_checks": {"ok": False, "controlled_card_count": 16, "expected_total": 21},
        "discard_button": {"name": "discard"},
        "remaining_deck_count": 43,
        "action_plan": {"action": "discard"},
    }

    assert _should_relax_midgame_count(result)


def test_does_not_relax_opening_like_count_mismatch():
    result = {
        "sanity_checks": {"ok": False, "controlled_card_count": 16, "expected_total": 21},
        "discard_button": {"name": "discard"},
        "remaining_deck_count": 44,
        "action_plan": {"action": "discard"},
    }

    assert not _should_relax_midgame_count(result)


def test_mark_midgame_count_relaxed_preserves_seat_role():
    result = {
        "metadata": {},
        "sanity_checks": {"warnings": []},
    }

    patched = _mark_midgame_count_relaxed(result, {"role": "dealer", "expected_total": 21})

    assert patched["metadata"]["count_validation_mode"] == "midgame_relaxed"
    assert patched["sanity_checks"]["mode"] == "midgame_relaxed"
    assert patched["seat_role"]["role"] == "dealer"
