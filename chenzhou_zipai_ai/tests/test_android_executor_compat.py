from __future__ import annotations

import base64
from concurrent.futures import TimeoutError as FutureTimeoutError
import importlib
import json
import pickle
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def _without_volatile_timing(value):
    if isinstance(value, dict):
        return {
            key: _without_volatile_timing(item)
            for key, item in value.items()
            if key not in {"elapsed_ms", "cache_hit", "created_at", "timestamp"}
        }
    if isinstance(value, list):
        return [_without_volatile_timing(item) for item in value]
    return value


class _FakeStrategyProcessPool:
    next_request_id = 0
    requests: dict[int, tuple[str, str]] = {}
    submitted: list[tuple[str, int]] = []
    cancelled: list[int] = []
    restart_calls = 0
    fail_await = False

    @classmethod
    def reset(cls) -> None:
        cls.next_request_id = 0
        cls.requests = {}
        cls.submitted = []
        cls.cancelled = []
        cls.restart_calls = 0
        cls.fail_await = False

    @classmethod
    def workerCount(cls) -> int:
        return 2

    @classmethod
    def submitPrioritized(cls, task_name: str, payload: str, priority: int) -> int:
        cls.next_request_id += 1
        request_id = cls.next_request_id
        cls.requests[request_id] = (task_name, payload)
        cls.submitted.append((task_name, int(priority)))
        return request_id

    @classmethod
    def awaitResult(cls, request_id: int, _timeout_ms: int) -> str:
        if cls.fail_await:
            raise RuntimeError(f"strategy_request_timeout:{request_id}")
        task_name, payload = cls.requests.pop(int(request_id))
        args, kwargs = pickle.loads(base64.b64decode(payload))
        if task_name == "runtime.batch":
            batched_task_name, calls = args
            module_name, function_name = batched_task_name.rsplit(".", 1)
            function = getattr(importlib.import_module(module_name), function_name)
            result = [
                function(*call_args, **call_kwargs)
                for call_args, call_kwargs in calls
            ]
        else:
            module_name, function_name = task_name.rsplit(".", 1)
            function = getattr(importlib.import_module(module_name), function_name)
            result = function(*args, **kwargs)
        return base64.b64encode(pickle.dumps(result, protocol=4)).decode("ascii")

    @classmethod
    def cancel(cls, request_id: int) -> bool:
        cls.cancelled.append(int(request_id))
        return cls.requests.pop(int(request_id), None) is not None

    @classmethod
    def cancelAll(cls) -> None:
        cls.requests.clear()

    @classmethod
    def restartWorkers(cls) -> None:
        cls.restart_calls += 1
        cls.requests.clear()


@pytest.fixture
def android_bridge(monkeypatch):
    import ai.dual_discard_validator as validator
    import ai.dual_validated_candidate as candidate

    assert not validator._SHARED_EXECUTORS
    original_executor = validator.ProcessPoolExecutor
    original_validation_executor = candidate.ThreadPoolExecutor
    _FakeStrategyProcessPool.reset()
    monkeypatch.setenv("ANDROID_ARGUMENT", "1")
    monkeypatch.setenv("AIZIPAI_MOBILE_RUNTIME", "1")
    monkeypatch.setenv("AIZIPAI_PARALLEL_VISION", "1")
    monkeypatch.setenv("AIZIPAI_VISION_TIMEOUT_SECONDS", "6")
    java_module = types.ModuleType("java")
    java_module.jclass = lambda _name: _FakeStrategyProcessPool
    monkeypatch.setitem(sys.modules, "java", java_module)
    module_name = "android_app.mobile_runtime.mobile_bridge"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    try:
        yield module
    finally:
        validator.ProcessPoolExecutor = original_executor
        candidate.ThreadPoolExecutor = original_validation_executor
        validator._SHARED_EXECUTORS.clear()
        sys.modules.pop(module_name, None)


def test_android_executor_map_preserves_input_order(android_bridge) -> None:
    executor = android_bridge._AndroidProcessExecutor(max_workers=20)

    assert list(executor.map(abs, [-3, -1, -2], chunksize=1)) == [3, 1, 2]
    assert _FakeStrategyProcessPool.submitted == [
        ("builtins.abs", 0),
        ("builtins.abs", 0),
        ("builtins.abs", 0),
    ]


def test_android_executor_map_cancels_unconsumed_requests(android_bridge) -> None:
    executor = android_bridge._AndroidProcessExecutor(max_workers=20)
    results = executor.map(abs, [-3, -1, -2], chunksize=1)

    assert next(results) == 3
    results.close()

    assert _FakeStrategyProcessPool.cancelled == [2, 3]
    assert _FakeStrategyProcessPool.requests == {}


def test_android_executor_map_rejects_invalid_chunksize(android_bridge) -> None:
    executor = android_bridge._AndroidProcessExecutor(max_workers=20)

    with pytest.raises(ValueError, match="chunksize"):
        executor.map(abs, [-1], chunksize=0)


def test_android_remote_timeout_restarts_workers_and_clears_generation(
    android_bridge,
) -> None:
    executor = android_bridge._AndroidProcessExecutor(max_workers=20)
    future = executor.submit(abs, -1)
    _FakeStrategyProcessPool.fail_await = True

    with pytest.raises(FutureTimeoutError, match="strategy_request_timeout"):
        future.result(timeout=0.01)

    assert _FakeStrategyProcessPool.restart_calls == 1
    assert _FakeStrategyProcessPool.requests == {}


@pytest.mark.parametrize("wildcard_enabled", [False, True])
def test_android_executor_map_roundtrip_matches_reference_action_evals(
    android_bridge,
    wildcard_enabled: bool,
) -> None:
    from ai.pro_brain import (
        _evaluate_action_batch_task,
        _evaluate_action_task,
        allocate_hand_structures,
        analyze_hand,
        build_decision_context,
        generate_legal_actions,
    )
    from engine.rules import rules_for_room

    rules = rules_for_room(wildcard_enabled=wildcard_enabled, players=2)
    hand = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "壹", "贰"]
    if wildcard_enabled:
        hand[-1] = "王"
    context = build_decision_context(
        hand,
        rules=rules,
    )
    allocation = allocate_hand_structures(context)
    analysis = analyze_hand(context, allocation)
    legal_actions, _ = generate_legal_actions(context, allocation, analysis)
    payloads = [
        (context, action, allocation, analysis)
        for action in legal_actions
    ]
    assert len(payloads) >= 4

    expected = [_evaluate_action_task(payload) for payload in payloads]
    assert _evaluate_action_batch_task(payloads) == expected
    actual = list(
        android_bridge._AndroidProcessExecutor(max_workers=20).map(
            _evaluate_action_task,
            payloads,
            chunksize=1,
        )
    )

    assert actual == expected
    assert _FakeStrategyProcessPool.submitted == [
        ("runtime.batch", 0),
    ] * min(len(payloads), 4)


def test_mobile_policy_uses_bounded_medium_grain_world_shards(
    android_bridge,
) -> None:
    from ai.opponent_league import create_policy

    policy = create_policy(
        "professional_v81_two_player_exact_discard_sharded_research"
    )

    assert policy.discard_validator.config.coverage_parallel_shards == 4
    assert policy.discard_validator.config.confirmation_parallel_shards == 2
    assert policy.response_search.parallel_shards == 4
    assert policy.search.refinement.parallel_shards == 4
    assert policy.search.discard_selection.parallel_shards == 4
    assert policy.search.discard_confirmation.parallel_shards == 4
    assert policy.search.direct_discard_confirmation_from_refinement
    assert policy.search.reuse_candidate_priors_as_coverage


@pytest.mark.parametrize("wildcard_enabled", [False, True])
def test_android_parallel_baseline_preserves_frozen_v81_decision(
    android_bridge,
    wildcard_enabled: bool,
) -> None:
    import ai.frozen_two_player_strategy as frozen
    from ai.pro_brain import choose_action as choose_production_action
    from engine.rules import rules_for_room

    rules = rules_for_room(wildcard_enabled=wildcard_enabled, players=2)
    hand = [
        "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
        "壹", "贰", "叁", "肆", "伍", "陆", "柒", "捌", "玖", "拾", "九",
    ]
    if wildcard_enabled:
        hand[-1] = "王"
    state = {
        "hand": hand,
        "legal_actions": [{"type": "DISCARD"}],
        "remaining_deck_count": 39,
    }

    sequential = choose_production_action(
        state,
        rules=rules,
        parallel_evaluation=False,
    )
    parallel = choose_production_action(
        state,
        rules=rules,
        parallel_evaluation=True,
    )
    assert parallel.to_dict() == sequential.to_dict()

    frozen._SELECTION_CACHE.clear()
    expected = frozen.choose_action(state, rules=rules, baseline_decision=sequential)
    actual = frozen.choose_action(state, rules=rules, baseline_decision=parallel)

    assert _without_volatile_timing(actual.to_dict()) == _without_volatile_timing(
        expected.to_dict()
    )


def test_mobile_worker_and_android_pool_allow_action_eval_task(
    monkeypatch,
) -> None:
    module_name = "android_app.mobile_runtime.mobile_worker"
    monkeypatch.setenv("AIZIPAI_MOBILE_RUNTIME", "1")
    sys.modules.pop(module_name, None)
    mobile_worker = importlib.import_module(module_name)

    task_name = "ai.pro_brain._evaluate_action_task"
    kotlin_pool = Path(
        "android_app/app/src/main/java/com/example/ai/StrategyProcessPool.kt"
    ).read_text(encoding="utf-8")

    assert task_name in mobile_worker._TASKS
    assert f'"{task_name}"' in kotlin_pool
    assert '"runtime.batch"' in kotlin_pool
    assert "if (request.client != null)" in kotlin_pool
    assert "restartWorkers()" in kotlin_pool
    sys.modules.pop(module_name, None)


def test_mobile_recommendation_does_not_crop_frozen_v81_with_strategy_deadline(
    android_bridge,
    monkeypatch,
    tmp_path,
) -> None:
    seen: dict[str, object] = {}
    buttons = [object()]
    original_cwd = Path.cwd()

    def fake_run_once(**kwargs):
        seen.update(kwargs)
        return {"action_plan": {"action": "wait", "ready": False}, "fatal": False}

    monkeypatch.setattr(android_bridge, "run_once", fake_run_once)
    monkeypatch.setattr(android_bridge, "release_pending_strategy_work", lambda _pool: {})
    try:
        android_bridge.recommend_json(
            "unused-frame",
            False,
            str(tmp_path),
            2344,
            1080,
            precomputed_buttons=buttons,
        )
    finally:
        import os

        os.chdir(original_cwd)

    assert seen["decision_time_budget_seconds"] is None
    assert seen["precomputed_buttons"] is buttons
    assert seen["pending_surface_buttons"] is None


def test_mobile_signature_preserves_semantic_chi_intent_for_guard_commit(
    android_bridge,
    monkeypatch,
    tmp_path,
) -> None:
    original_cwd = Path.cwd()
    plan = {
        "action": "chi",
        "ready": True,
        "execution_mode": "tap_sequence",
        "reason": "semantic chi",
        "target": "button:chi",
        "target_type": "button",
        "target_label": "chi",
        "target_option_id": "semantic_chi_001",
        "target_option_cards": ["七", "八", "九"],
        "clicks": [{"target": "button:chi", "x": 1960, "y": 507}],
        "policy_selected_action": {"type": "chi", "label": None},
        "validation": {
            "passed": True,
            "target_matches_policy": True,
            "checks": ["semantic_chi_selected"],
        },
    }
    monkeypatch.setattr(
        android_bridge,
        "run_once",
        lambda **_kwargs: {
            "action_plan": plan,
            "flow": {"state": "play"},
            "hand": ["七", "九"],
            "sanity_checks": {"controlled_card_count": 20},
            "fatal": False,
        },
    )
    monkeypatch.setattr(android_bridge, "release_pending_strategy_work", lambda _pool: {})
    try:
        encoded = json.loads(
            android_bridge.recommend_json("unused", True, str(tmp_path), 2344, 1080)
        )["_mobile_signature"]
    finally:
        import os

        os.chdir(original_cwd)

    _signature, _context, decoded, reason = android_bridge._decode_mobile_action(encoded)

    assert reason == "ok"
    assert decoded["target_type"] == "button"
    assert decoded["target_option_id"] == "semantic_chi_001"
    assert decoded["target_option_cards"] == ["七", "八", "九"]

    committed = json.loads(
        android_bridge.commit_executed_action_json(encoded, str(tmp_path), True)
    )
    saved_guard = json.loads(
        (tmp_path / "action_guard_1v1-wang.json").read_text(encoding="utf-8")
    )
    assert committed["committed"] is True
    assert saved_guard["pending_response"]["intended_option_id"] == "semantic_chi_001"
    assert saved_guard["pending_response"]["intended_option_cards"] == ["七", "八", "九"]


def test_mobile_pass_preflight_skips_post_action_detectors(
    android_bridge,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        android_bridge,
        "detect_flow_state",
        lambda _image: SimpleNamespace(state="play"),
    )

    def unexpected(*_args, **_kwargs):
        raise AssertionError("post-action detector called during preflight")

    monkeypatch.setattr(android_bridge, "detect_option_candidates", unexpected)
    monkeypatch.setattr(android_bridge, "detect_discard_button", unexpected)
    monkeypatch.setattr(android_bridge, "recognize_pending_cards", unexpected)
    expected = {
        "flow_state": "play",
        "button_names": ["pass"],
        "discard_button_visible": False,
        "opponent_pending_card": "四",
    }

    actual = android_bridge._fresh_context_from_image(
        object(),
        action="pass",
        expected=expected,
        buttons=[SimpleNamespace(name="pass")],
        surface_phase="pre",
    )

    assert android_bridge.evaluate_pre_action(
        action="pass",
        expected=expected,
        actual=actual,
    ) == {"status": "execute", "reason": "action_surface_matches"}


def test_decide_rgba_reuses_exact_frame_buttons_for_recommendation(
    android_bridge,
    monkeypatch,
    tmp_path,
) -> None:
    button = SimpleNamespace(
        name="pass",
        to_dict=lambda: {"name": "pass", "center": [1200, 500]},
    )
    detected = [button]
    seen: dict[str, object] = {"detect_calls": 0}

    monkeypatch.setattr(android_bridge, "register_rgba_frame", lambda *_args, **_kwargs: Path("frame-token"))
    monkeypatch.setattr(android_bridge, "unregister_frame", lambda _token: None)
    monkeypatch.setattr(android_bridge, "read_bgr", lambda _token: object())

    def fake_detect_buttons(_image, **_kwargs):
        seen["detect_calls"] = int(seen["detect_calls"]) + 1
        seen["detect_names"] = _kwargs.get("names")
        return detected

    def fake_visible_hu_plan(buttons):
        seen["priority_buttons"] = buttons
        return None

    def fake_recommend_json(*_args, **kwargs):
        seen["recommend_buttons"] = kwargs.get("precomputed_buttons")
        seen["surface_buttons"] = kwargs.get("pending_surface_buttons")
        return '{"action":"pass","ready":true}'

    monkeypatch.setattr(android_bridge, "detect_buttons", fake_detect_buttons)
    monkeypatch.setattr(android_bridge, "visible_hu_plan", fake_visible_hu_plan)
    monkeypatch.setattr(android_bridge, "recommend_json", fake_recommend_json)
    monkeypatch.setattr(
        android_bridge,
        "priority_action_json",
        lambda _path: (_ for _ in ()).throw(AssertionError("second button scan")),
    )

    result = android_bridge.decide_rgba_json(
        object(),
        2344,
        1080,
        2344 * 4,
        False,
        str(tmp_path),
    )

    assert json.loads(result) == {"action": "pass", "ready": True}
    assert seen["detect_calls"] == 1
    assert seen["detect_names"] is None
    assert seen["priority_buttons"] == [{"name": "pass", "center": [1200, 500]}]
    assert seen["recommend_buttons"] is detected
    assert seen["surface_buttons"] is None


def test_trusted_response_hand_requires_fresh_confirmed_discard_ledger(
    android_bridge,
    tmp_path,
) -> None:
    ledger = tmp_path / "ledger.json"
    guard = tmp_path / "guard.json"
    buttons = [SimpleNamespace(name="chi")]
    now = android_bridge.time.time()
    ledger.write_text(
        json.dumps(
            {
                "trusted": True,
                "shadow_only": True,
                "last_confirmed_action": "discard",
                "updated_at": now,
                "hand_counts": {"二": 2, "七": 1},
            }
        ),
        encoding="utf-8",
    )
    guard.write_text(
        json.dumps(
            {
                "last_executed_signature": ["discard", [["hand:九", 900, 900]]],
                "last_executed_at": now,
                "last_context": {
                    "hand_signature": [["七", 1], ["二", 2], ["九", 1]],
                    "planned_hand_target": {"label": "九"},
                },
            }
        ),
        encoding="utf-8",
    )

    assert android_bridge._trusted_response_hand_labels(ledger, guard, buttons) == ["七", "二", "二"]

    stale = json.loads(ledger.read_text(encoding="utf-8"))
    stale["updated_at"] = android_bridge.time.time() - 121.0
    ledger.write_text(json.dumps(stale), encoding="utf-8")
    assert android_bridge._trusted_response_hand_labels(ledger, guard, buttons) is None
    assert android_bridge._trusted_response_hand_labels(
        ledger,
        guard,
        [SimpleNamespace(name="discard")],
    ) is None


def test_trusted_response_hand_rejects_guard_or_ledger_mismatch(
    android_bridge,
    tmp_path,
) -> None:
    ledger = tmp_path / "ledger.json"
    guard = tmp_path / "guard.json"
    now = android_bridge.time.time()
    ledger.write_text(
        json.dumps(
            {
                "trusted": True,
                "shadow_only": True,
                "last_confirmed_action": "discard",
                "updated_at": now,
                "hand_counts": {"二": 2},
            }
        ),
        encoding="utf-8",
    )
    guard.write_text(
        json.dumps(
            {
                "last_executed_signature": ["discard", []],
                "last_executed_at": now,
                "last_context": {
                    "hand_signature": [["二", 2], ["九", 1]],
                    "planned_hand_target": {"label": "八"},
                },
            }
        ),
        encoding="utf-8",
    )

    assert android_bridge._trusted_response_hand_labels(ledger, guard, [SimpleNamespace(name="chi")]) is None


def test_decide_rgba_preserves_visible_hu_priority_with_single_button_scan(
    android_bridge,
    monkeypatch,
    tmp_path,
) -> None:
    button = SimpleNamespace(
        name="hu",
        to_dict=lambda: {"name": "hu", "center": [900, 500]},
    )
    seen = {"detect_calls": 0}

    monkeypatch.setattr(android_bridge, "register_rgba_frame", lambda *_args, **_kwargs: Path("frame-token"))
    monkeypatch.setattr(android_bridge, "unregister_frame", lambda _token: None)
    monkeypatch.setattr(android_bridge, "read_bgr", lambda _token: object())

    def fake_detect_buttons(_image, **_kwargs):
        seen["detect_calls"] += 1
        seen["detect_names"] = _kwargs.get("names")
        return [button]

    monkeypatch.setattr(android_bridge, "detect_buttons", fake_detect_buttons)
    monkeypatch.setattr(android_bridge, "visible_hu_plan", lambda _buttons: {"action": "hu", "ready": True})
    monkeypatch.setattr(
        android_bridge,
        "recommend_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("HU must not enter policy")),
    )
    monkeypatch.setattr(
        android_bridge,
        "priority_action_json",
        lambda _path: (_ for _ in ()).throw(AssertionError("second button scan")),
    )

    result = json.loads(
        android_bridge.decide_rgba_json(
            object(),
            2344,
            1080,
            2344 * 4,
            True,
            str(tmp_path),
        )
    )

    assert result["action"] == "hu"
    assert result["ready"] is True
    assert result["_mobile_status"] == "execute"
    assert seen["detect_calls"] == 1
    assert seen["detect_names"] is None


def test_decide_rgba_keeps_hu_only_surface_scan_during_committed_chi_transaction(
    android_bridge,
    monkeypatch,
    tmp_path,
) -> None:
    (tmp_path / "action_guard_1v1-wang.json").write_text(
        json.dumps({"pending_response": {"action": "chi"}}),
        encoding="utf-8",
    )
    button = SimpleNamespace(name="hu", to_dict=lambda: {"name": "hu"})
    seen = {}
    monkeypatch.setattr(android_bridge, "register_rgba_frame", lambda *_args, **_kwargs: Path("frame-token"))
    monkeypatch.setattr(android_bridge, "unregister_frame", lambda _token: None)
    monkeypatch.setattr(android_bridge, "read_bgr", lambda _token: object())

    def fake_detect_buttons(_image, **kwargs):
        seen["names"] = kwargs.get("names")
        return [button]

    def fake_recommend_json(*_args, **kwargs):
        seen["precomputed"] = kwargs.get("precomputed_buttons")
        seen["surface"] = kwargs.get("pending_surface_buttons")
        return '{"action":"wait","ready":false}'

    monkeypatch.setattr(android_bridge, "detect_buttons", fake_detect_buttons)
    monkeypatch.setattr(android_bridge, "visible_hu_plan", lambda _buttons: None)
    monkeypatch.setattr(android_bridge, "recommend_json", fake_recommend_json)

    android_bridge.decide_rgba_json(
        object(),
        2344,
        1080,
        2344 * 4,
        True,
        str(tmp_path),
    )

    assert seen["names"] == ("hu",)
    assert seen["precomputed"] is None
    assert seen["surface"] == [button]


def test_fresh_pre_action_frame_preempts_slow_normal_plan_with_hu(
    android_bridge,
    monkeypatch,
) -> None:
    hu = SimpleNamespace(
        name="hu",
        to_dict=lambda: {"name": "hu", "center": [900, 500]},
    )
    monkeypatch.setattr(android_bridge, "read_bgr", lambda _path: object())
    monkeypatch.setattr(android_bridge, "detect_buttons", lambda *_args, **_kwargs: [hu])

    def unexpected_normal_verification(*_args, **_kwargs):
        raise AssertionError("fresh HU frame continued verifying the stale normal plan")

    monkeypatch.setattr(android_bridge, "_fresh_context_from_image", unexpected_normal_verification)
    stale_pass_signature = json.dumps(
        {
            "plan": ["pass", [["button:pass", 2100, 510]]],
            "context": {"flow_state": "play", "button_names": ["pass"]},
            "execution": {"action": "pass", "ready": True},
        }
    )

    result = json.loads(
        android_bridge.verify_before_execution_json(
            "new-frame-after-slow-decision",
            stale_pass_signature,
        )
    )

    assert result["status"] == "preempt_hu"
    assert result["plan"]["action"] == "hu"
    assert result["plan"]["ready"] is True
    assert result["plan"]["clicks"] == [
        {"target": "button:hu", "x": 900, "y": 500, "delay_ms": 0}
    ]
