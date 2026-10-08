"""Chaquopy entry point for the established live planning chain."""

from __future__ import annotations

import json
import os
import base64
import pickle
import sys
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from threading import Lock
from pathlib import Path

from java import jclass


RUNTIME_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RUNTIME_ROOT / "chenzhou_zipai_ai"
for path in (RUNTIME_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("ANDROID_ARGUMENT", "1")
os.environ.setdefault("AIZIPAI_MOBILE_RUNTIME", "1")
os.environ.setdefault("AIZIPAI_PARALLEL_VISION", "1")
os.environ.setdefault("AIZIPAI_VISION_TIMEOUT_SECONDS", "6")
# The shadow recognizer records neural diagnostics but cannot change a label,
# confidence, legality decision, or tap target.  Do not spend foreground mobile
# latency on a result which the production path is contractually forbidden to
# consume.  The independent selected-card verifier remains enabled.
os.environ.setdefault("AIZIPAI_CARD_HYBRID_MODE", "off")


_PROCESS_POOL = jclass("com.example.ai.StrategyProcessPool")
_MOBILE_STRATEGY_WORKERS = max(1, int(_PROCESS_POOL.workerCount()))
# Keep at most two bounded action-evaluation batches per physical worker. A
# single batch per worker leaves cores idle when CHI candidates have uneven EV
# cost; eight total tasks on a four-worker phone is still far below the old
# unbounded one-task-per-rollout queue.
_MOBILE_BATCHES_PER_WORKER = 2
os.environ["AIZIPAI_MOBILE_STRATEGY_WORKERS"] = str(_MOBILE_STRATEGY_WORKERS)
os.environ.setdefault(
    "AIZIPAI_MOBILE_SHARDS_PER_WORKER",
    "8" if _MOBILE_STRATEGY_WORKERS >= 8 else "2",
)
_BATCHABLE_TASKS = {
    "ai.pro_brain._evaluate_action_task",
}


class _AndroidRemoteFuture:
    def __init__(self, request_id: int) -> None:
        self.request_id = int(request_id)

    def result(self, timeout: float | None = None):
        timeout_ms = 45_000 if timeout is None else max(1, int(timeout * 1000))
        try:
            encoded = str(_PROCESS_POOL.awaitResult(self.request_id, timeout_ms))
        except Exception as error:
            # Python rollout code cannot be interrupted reliably by Java thread
            # cancellation. Kill this worker generation so retries never inherit
            # CPU-bound tasks from the timed-out decision.
            _PROCESS_POOL.restartWorkers()
            if "strategy_request_timeout:" in str(error):
                # Exact-sharded search already knows how to turn an exhausted
                # stage into a deadline-safe production fallback. Preserve that
                # contract instead of crashing the whole captured frame.
                raise FutureTimeoutError(str(error)) from error
            raise
        return pickle.loads(base64.b64decode(encoded))

    def cancel(self) -> bool:
        return bool(_PROCESS_POOL.cancel(self.request_id))


class _AndroidProcessExecutor:
    def __init__(self, *, max_workers: int | None = None, **_kwargs) -> None:
        self.max_workers = max_workers
        # The frozen runtime uses 12-worker root searches and 20-worker
        # background validation. Android has one shared physical worker pool,
        # so keep the latency-critical root shards ahead of queued validation.
        self.priority = 10 if max_workers is not None and max_workers <= 12 else 0

    def submit(self, function, *args, **kwargs) -> _AndroidRemoteFuture:
        task_name = f"{function.__module__}.{function.__name__}"
        return self._submit_task(task_name, args, kwargs)

    def _submit_task(
        self,
        task_name: str,
        args: tuple,
        kwargs: dict,
    ) -> _AndroidRemoteFuture:
        payload = base64.b64encode(
            pickle.dumps((args, kwargs), protocol=4)
        ).decode("ascii")
        return _AndroidRemoteFuture(
            _PROCESS_POOL.submitPrioritized(task_name, payload, self.priority)
        )

    def map(
        self,
        function,
        *iterables,
        timeout: float | None = None,
        chunksize: int = 1,
    ):
        if chunksize < 1:
            raise ValueError("chunksize must be >= 1")
        deadline = None if timeout is None else time.perf_counter() + max(0.0, float(timeout))
        calls = [(tuple(args), {}) for args in zip(*iterables)]
        task_name = f"{function.__module__}.{function.__name__}"
        is_batched = task_name in _BATCHABLE_TASKS and len(calls) > 1
        if is_batched:
            requested_batches = (len(calls) + chunksize - 1) // chunksize
            batch_count = min(
                _MOBILE_STRATEGY_WORKERS * _MOBILE_BATCHES_PER_WORKER,
                max(1, requested_batches),
            )
            base_size, extra = divmod(len(calls), batch_count)
            batches = []
            offset = 0
            for index in range(batch_count):
                size = base_size + int(index < extra)
                batches.append(calls[offset : offset + size])
                offset += size
            pending = [
                self._submit_task(
                    "runtime.batch",
                    (task_name, batch),
                    {},
                )
                for batch in batches
            ]
        else:
            pending = [
                self._submit_task(task_name, args, kwargs)
                for args, kwargs in calls
            ]

        def ordered_results():
            try:
                while pending:
                    future = pending.pop(0)
                    remaining = None
                    if deadline is not None:
                        remaining = deadline - time.perf_counter()
                        if remaining <= 0.0:
                            raise TimeoutError("android_executor_map_timeout")
                    result = future.result(timeout=remaining)
                    if is_batched:
                        yield from result
                    else:
                        yield result
            finally:
                for future in pending:
                    future.cancel()

        return ordered_results()

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        del wait
        if cancel_futures:
            _PROCESS_POOL.cancelAll()


def _android_process_pool(**kwargs) -> _AndroidProcessExecutor:
    return _AndroidProcessExecutor(**kwargs)


def _install_android_executor_compat() -> None:
    import ai.dual_discard_validator as validator

    if validator._SHARED_EXECUTORS:
        raise RuntimeError("android_executor_compat_installed_after_executor_creation")
    validator.ProcessPoolExecutor = _android_process_pool


_install_android_executor_compat()

from ai.android_validation_cleanup import (
    install_android_validation_cleanup,
    release_pending_strategy_work,
)

install_android_validation_cleanup()

from control.action_plan import action_plan_contract_error
from tools.live_assistant import (
    _plan_signature,
    _result_context_signature,
    _signature_to_json,
    abort_external_action_guard,
    commit_executed_action_guard,
    prepare_external_action_guard,
    reset_transient_external_action_guard,
    run_once,
)
from tools.inspect_state import inspect_screenshot
from tools.mobile_freshness import (
    FRESH_CONTEXT_KEYS,
    changed_fresh_context_for_action,
    mobile_guard_signature,
    planned_hand_target,
    settlement_ready_click_is_fresh,
)
from tools.mobile_action_verifier import (
    evaluate_discard_selection,
    evaluate_post_action,
    evaluate_pre_action,
    required_pre_surface_features,
    required_surface_features,
)
from tools.confirmed_hand_ledger import load_confirmed_ledger, record_confirmed_action
from tools.mobile_priority import visible_hu_plan
from vision.button_detector import detect_buttons, detect_buttons_from_path
from vision.discard_button_detector import detect_discard_button
from vision.flow_detector import detect_flow_state, detect_flow_state_from_path
from vision.hand_recognizer import hand_target_is_clickable
from vision.hand_target_locator import locate_hand_target
from vision.history_memory import VisionMemory, save_memory
from vision.image_source import read_bgr, register_rgba_frame, unregister_frame
from vision.option_detector import detect_option_candidates
from vision.pending_card_recognizer import recognize_pending_cards
from vision.regions import load_region_config
from vision.viewport import transform_from_config


CONFIG_PATH = PROJECT_ROOT / "config" / "screen_1080x2400.yaml"
RULES_PATH = PROJECT_ROOT / "config" / "rules.yaml"
_WARMED_MODES: set[bool] = set()
_WARMUP_LOCK = Lock()


def prewarm_runtime_profile_json(home_dir: str, profile: str) -> str:
    """Diagnostic-only single-component warmup used by Android instrumentation."""
    home = Path(home_dir)
    home.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if profile == "noop":
        pass
    elif profile == "chdir":
        os.chdir(home)
    elif profile == "rules":
        os.chdir(home)
        from engine.rules import rules_for_room

        rules_for_room(
            RULES_PATH,
            wildcard_enabled=True,
            players=2,
            room_mode="1v1-wang",
        )
    elif profile == "classifier_load":
        os.chdir(home)
        from vision.hand_recognizer import _load_neural_classifier
        from vision.neural_card_classifier import DEFAULT_MODEL_PATH

        _load_neural_classifier(str(DEFAULT_MODEL_PATH))
    elif profile == "classifier_infer":
        os.chdir(home)
        import numpy as np

        from vision.hand_recognizer import _load_neural_classifier
        from vision.neural_card_classifier import DEFAULT_MODEL_PATH

        classifier = _load_neural_classifier(str(DEFAULT_MODEL_PATH))
        classifier.classify_batch([np.full((160, 90, 3), 255, dtype=np.uint8)])
    elif profile in {"buttons", "hand"}:
        os.chdir(home)
        from vision.template_loader import load_templates_from_dirs

        directory = "buttons" if profile == "buttons" else "hand_auto"
        load_templates_from_dirs(
            (PROJECT_ROOT / "data" / "templates" / directory,),
            grayscale=profile == "buttons",
        )
    else:
        raise ValueError(f"unsupported_prewarm_profile:{profile}")
    return json.dumps(
        {
            "ready": True,
            "profile": profile,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        },
        separators=(",", ":"),
    )


def prewarm_runtime_json(home_dir: str) -> str:
    """Warm the main Chaquopy process without reading a game frame or writing gameplay state."""
    home = Path(home_dir)
    home.mkdir(parents=True, exist_ok=True)
    os.chdir(home)
    started = time.perf_counter()
    warmed_now: list[str] = []
    with _WARMUP_LOCK:
        import numpy as np

        import ai.frozen_two_player_strategy  # noqa: F401
        from engine.hu_checker import best_grouping
        from engine.rules import rules_for_room
        from vision.hand_recognizer import _load_neural_classifier
        from vision.neural_card_classifier import DEFAULT_MODEL_PATH
        from vision.template_loader import load_templates_from_dirs

        classifier = _load_neural_classifier(str(DEFAULT_MODEL_PATH))
        classifier.classify_batch([np.full((160, 90, 3), 255, dtype=np.uint8)])
        load_templates_from_dirs((PROJECT_ROOT / "data" / "templates" / "buttons",), grayscale=True)
        # Keep the always-used opening path warm. Option, discard-history, and
        # meld templates are loaded by their recognizers only when that UI is
        # actually present; retaining all four sets here causes severe paging in
        # the Android decision process before strategy evaluation begins.
        load_templates_from_dirs(
            (PROJECT_ROOT / "data" / "templates" / "hand_auto",),
            grayscale=False,
        )
        best_grouping(["一", "二", "三"])
        for wildcard_enabled in (False, True):
            if wildcard_enabled in _WARMED_MODES:
                continue
            mode = "1v1-wang" if wildcard_enabled else "1v1-no-wang"
            rules_for_room(
                RULES_PATH,
                wildcard_enabled=wildcard_enabled,
                players=2,
                room_mode=mode,
            )
            _WARMED_MODES.add(wildcard_enabled)
            warmed_now.append(mode)
    return json.dumps(
        {
            "ready": True,
            "warmed_now": warmed_now,
            "warmed_modes": ["1v1-wang" if mode else "1v1-no-wang" for mode in sorted(_WARMED_MODES)],
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def diagnostics_json() -> str:
    import hashlib

    import cv2
    import numpy as np

    from vision.neural_card_classifier import OnnxCardClassifier

    classifier = OnnxCardClassifier(cache_size=0)
    prediction = classifier.classify_batch(
        [np.full((160, 90, 3), 255, dtype=np.uint8)]
    )[0]
    return json.dumps(
        {
            "opencv": cv2.__version__,
            "backend": classifier.backend_name,
            "label": prediction.label,
            "confidence": prediction.confidence,
            "runner_up_label": prediction.runner_up_label,
            "runner_up_confidence": prediction.runner_up_confidence,
            "model_sha256": hashlib.sha256(classifier.model_path.read_bytes()).hexdigest(),
            "session_count": (
                classifier._android_bridge.cachedSessionCount()
                if classifier._android_bridge is not None
                else 0
            ),
            "discard_executor": "android_process_pool",
            "strategy_workers": int(_PROCESS_POOL.workerCount()),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def latest_strategy_timing_json() -> str:
    """Return compact timing for the most recent frozen-policy decision."""
    from ai.frozen_two_player_strategy import _frozen_policy

    policy = _frozen_policy()
    progressive = getattr(policy, "search", None)

    def stage_payload(name: str) -> dict[str, object] | None:
        stage = getattr(progressive, name, None)
        if stage is None:
            return None
        return {
            "elapsed_ms": round(float(stage.elapsed_ms), 3),
            "simulations": int(stage.simulations),
            "paired_determinizations": int(stage.paired_determinizations),
            "deadline_interruptions": int(stage.deadline_interruptions),
            "candidate_count": len(stage.candidates),
            "selected_label": stage.selected_label,
            "empirical_best_label": stage.empirical_best_label,
            "candidates": [
                {
                    "label": candidate.label,
                    "visits": int(candidate.visits),
                    "average_reward": round(float(candidate.average_reward), 6),
                    "heuristic_value": round(float(candidate.heuristic_value), 6),
                }
                for candidate in stage.candidates
            ],
        }

    events = policy.discard_events()
    event = events[-1] if events else {}
    return json.dumps(
        {
            "decision_elapsed_ms": event.get("elapsed_ms"),
            "decision_simulations": event.get("simulations"),
            "decision_reason": event.get("reason"),
            "production_label": event.get("production_label"),
            "selected_label": event.get("search_selected_label"),
            "paired_advantages": event.get("paired_advantages", []),
            "coverage": stage_payload("last_coverage"),
            "refinement": stage_payload("last_refinement"),
            "selection": stage_payload("last_discard_selection"),
            "confirmation": stage_payload("last_confirmation"),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def worker_pool_diagnostics_json() -> str:
    worker_count = int(_PROCESS_POOL.workerCount())
    results = []
    for index in range(worker_count):
        payload = base64.b64encode(
            pickle.dumps(((index,), {}), protocol=4)
        ).decode("ascii")
        encoded = str(
            _PROCESS_POOL.executeOnWorker(
                index,
                "diagnostic.warmup",
                payload,
                60_000,
            )
        )
        results.append(pickle.loads(base64.b64decode(encoded)))
    parent_perf_counter = time.perf_counter()
    return json.dumps(
        {
            "worker_count": int(_PROCESS_POOL.workerCount()),
            "pids": sorted({int(item["pid"]) for item in results}),
            "all_ready": all(bool(item["ready"]) for item in results),
            "parent_perf_counter": parent_perf_counter,
            "worker_perf_counters": [
                item.get("worker_perf_counter") for item in results
            ],
        },
        separators=(",", ":"),
    )


def _clear_confirmed_round_runtime_caches() -> dict[str, object]:
    from ai.runtime_cache_control import clear_round_strategy_caches

    main = clear_round_strategy_caches()
    return {
        "main": main,
        "workers": "bounded_mobile_lru_preserved",
    }


def worker_pool_cancel_recovery_json() -> str:
    before = json.loads(worker_pool_diagnostics_json())
    pending = []
    for _index in range(int(_PROCESS_POOL.workerCount())):
        payload = base64.b64encode(
            pickle.dumps(((5.0,), {}), protocol=4)
        ).decode("ascii")
        pending.append(
            _AndroidRemoteFuture(
                _PROCESS_POOL.submit("diagnostic.sleep", payload)
            )
        )
    _PROCESS_POOL.cancelAll()
    cancelled = 0
    for future in pending:
        try:
            future.result(timeout=1.0)
        except Exception:
            cancelled += 1
    recovery_started = time.perf_counter()
    recovered = json.loads(worker_pool_diagnostics_json())
    return json.dumps(
        {
            "cancelled": cancelled,
            "recovered_worker_count": len(recovered["pids"]),
            "all_ready": recovered["all_ready"],
            "same_worker_pids": recovered["pids"] == before["pids"],
            "recovery_ms": round((time.perf_counter() - recovery_started) * 1000, 3),
        },
        separators=(",", ":"),
    )


def worker_pool_priority_diagnostics_json() -> str:
    worker_count = int(_PROCESS_POOL.workerCount())
    json.loads(worker_pool_diagnostics_json())
    payload = base64.b64encode(
        pickle.dumps(((0.6,), {}), protocol=4)
    ).decode("ascii")
    background = [
        _AndroidRemoteFuture(
            _PROCESS_POOL.submitPrioritized("diagnostic.sleep", payload, 0)
        )
        for _index in range(worker_count * 2)
    ]
    primary_payload = base64.b64encode(
        pickle.dumps(((0.0,), {}), protocol=4)
    ).decode("ascii")
    started = time.perf_counter()
    primary = _AndroidRemoteFuture(
        _PROCESS_POOL.submitPrioritized(
            "diagnostic.sleep",
            primary_payload,
            10,
        )
    )
    try:
        result = primary.result(timeout=5.0)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    finally:
        _PROCESS_POOL.cancelAll()
        for future in background:
            future.cancel()
    return json.dumps(
        {
            "background_tasks": len(background),
            "primary_elapsed_ms": round(elapsed_ms, 3),
            "primary_pid": int(result["pid"]),
        },
        separators=(",", ":"),
    )


def current_inner_production_diagnostics_json() -> str:
    """Run the reproduced inner production state without vision or frozen search."""

    from ai.pro_brain import choose_action as choose_production_action
    from engine.rules import rules_for_room

    hand = [
        "一", "壹", "壹", "拾", "二", "二", "十", "十", "四", "五", "六",
        "九", "肆", "伍", "玖", "陆", "七", "柒", "柒", "柒", "玖",
    ]
    rules = rules_for_room(
        PROJECT_ROOT / "config" / "rules.yaml",
        wildcard_enabled=True,
        players=2,
        room_mode="1v1-wang",
    )
    started = time.perf_counter()
    decision = choose_production_action(
        {"hand": hand, "legal_actions": [{"type": "DISCARD"}]},
        rules=rules,
    )
    return json.dumps(
        {
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "selected_action": decision.selected_action,
            "selected_label": decision.selected_label,
            "evaluation_count": len(decision.action_evals),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def sequential_production_diagnostics_json() -> str:
    """Compare an outer-like production call followed by the reproduced inner call."""

    from ai.pro_brain import choose_action as choose_production_action
    from engine.rules import rules_for_room

    hand = [
        "一", "壹", "壹", "拾", "二", "二", "十", "十", "四", "五", "六",
        "九", "肆", "伍", "玖", "陆", "七", "柒", "柒", "柒", "玖",
    ]
    rules = rules_for_room(
        PROJECT_ROOT / "config" / "rules.yaml",
        wildcard_enabled=True,
        players=2,
        room_mode="1v1-wang",
    )
    outer_state = {
        "hand": hand,
        "hand_details": [
            {
                "card_id": f"h{index:03d}",
                "label": label,
                "clickable": label != "柒",
            }
            for index, label in enumerate(hand, start=1)
        ],
        "legal_actions": [{"type": "DISCARD"}],
    }
    inner_state = {"hand": hand, "legal_actions": [{"type": "DISCARD"}]}
    outer_started = time.perf_counter()
    outer = choose_production_action(outer_state, rules=rules)
    outer_ms = (time.perf_counter() - outer_started) * 1000.0
    inner_started = time.perf_counter()
    inner = choose_production_action(inner_state, rules=rules)
    inner_ms = (time.perf_counter() - inner_started) * 1000.0
    return json.dumps(
        {
            "outer_ms": round(outer_ms, 3),
            "outer_label": outer.selected_label,
            "inner_ms": round(inner_ms, 3),
            "inner_label": inner.selected_label,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def single_discard_evaluation_diagnostics_json(candidate_index: int) -> str:
    """Time one production discard evaluation from the reproduced opening hand."""

    from ai.pro_brain import (
        EVScorer,
        allocate_hand_structures,
        analyze_hand,
        build_decision_context,
        generate_legal_actions,
    )
    from engine.rules import rules_for_room

    hand = [
        "一", "壹", "壹", "拾", "二", "二", "十", "十", "四", "五", "六",
        "九", "肆", "伍", "玖", "陆", "七", "柒", "柒", "柒", "玖",
    ]
    rules = rules_for_room(
        PROJECT_ROOT / "config" / "rules.yaml",
        wildcard_enabled=True,
        players=2,
        room_mode="1v1-wang",
    )
    context = build_decision_context(
        {"hand": hand, "legal_actions": [{"type": "DISCARD"}]},
        rules=rules,
    )
    allocation = allocate_hand_structures(context)
    analysis = analyze_hand(context, allocation)
    actions, _ = generate_legal_actions(context, allocation, analysis)
    index = int(candidate_index) - 1
    if not 0 <= index < len(actions):
        raise ValueError(f"candidate_index_out_of_range:{candidate_index}")
    action = actions[index]
    started = time.perf_counter()
    evaluated = EVScorer().evaluate(context, action, allocation, analysis)
    return json.dumps(
        {
            "candidate_index": index + 1,
            "label": action.label,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "allowed": evaluated.allowed,
            "ev": evaluated.ev,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def policy_initialized_inner_diagnostics_json() -> str:
    """Measure production after constructing the frozen v8.1 policy object."""

    from ai.frozen_two_player_strategy import _frozen_policy
    from ai.pro_brain import choose_action as choose_production_action
    from engine.rules import rules_for_room

    hand = [
        "一", "壹", "壹", "拾", "二", "二", "十", "十", "四", "五", "六",
        "九", "肆", "伍", "玖", "陆", "七", "柒", "柒", "柒", "玖",
    ]
    rules = rules_for_room(
        PROJECT_ROOT / "config" / "rules.yaml",
        wildcard_enabled=True,
        players=2,
        room_mode="1v1-wang",
    )
    policy_started = time.perf_counter()
    policy = _frozen_policy()
    policy_ms = (time.perf_counter() - policy_started) * 1000.0
    decision_started = time.perf_counter()
    decision = choose_production_action(
        {"hand": hand, "legal_actions": [{"type": "DISCARD"}]},
        rules=rules,
    )
    return json.dumps(
        {
            "policy_ms": round(policy_ms, 3),
            "policy_name": str(policy.name),
            "decision_ms": round((time.perf_counter() - decision_started) * 1000.0, 3),
            "selected_label": decision.selected_label,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def current_frame_production_pair_diagnostics_json(screenshot: str) -> str:
    """Run the two production evaluations from the exact recognized frame state."""

    from ai.full_game_simulator import _production_state_from_public_view
    from ai.pro_brain import choose_action as choose_production_action
    from ai.search_brain import public_view_from_production_state
    from tools.recommend_action import inspect_screenshot, rules_for_room

    state = inspect_screenshot(
        screenshot,
        config_path=CONFIG_PATH,
        expected_total=21,
    )
    rules = rules_for_room(
        RULES_PATH,
        wildcard_enabled=True,
        players=2,
        room_mode="1v1-wang",
    )
    outer_started = time.perf_counter()
    outer = choose_production_action(state, rules=rules)
    outer_ms = (time.perf_counter() - outer_started) * 1000.0
    view = public_view_from_production_state(state, outer, rules=rules)
    if view is None:
        raise RuntimeError("current_frame_public_view_missing")
    inner_state = _production_state_from_public_view(
        view,
        legal_actions=[{"type": "DISCARD"}],
        pending_card=None,
        chi_options=None,
    )
    inner_started = time.perf_counter()
    inner = choose_production_action(inner_state, rules=rules)
    return json.dumps(
        {
            "hand_count": len(state.get("hand") or []),
            "outer_ms": round(outer_ms, 3),
            "outer_label": outer.selected_label,
            "inner_ms": round((time.perf_counter() - inner_started) * 1000.0, 3),
            "inner_label": inner.selected_label,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def current_frame_memory_production_diagnostics_json(
    screenshot: str,
    home_dir: str,
) -> str:
    """Run production on the exact frame after the live memory preparation."""

    from ai.full_game_simulator import _production_state_from_public_view
    from ai.pro_brain import choose_action as choose_production_action
    from ai.search_brain import public_view_from_production_state
    from engine.rules import rules_for_room
    from tools import recommend_action as recommendation
    from vision.history_memory import load_memory, recover_temporally_hidden_hand

    state = recommendation.inspect_screenshot(
        screenshot,
        config_path=CONFIG_PATH,
        expected_total=21,
    )
    memory_path = Path(home_dir) / "vision_memory_1v1-wang.json"
    memory = load_memory(memory_path)
    state = recover_temporally_hidden_hand(state, memory, expected_total=21)
    memory.update_from_snapshot(state)
    state = recommendation._clear_stale_options_when_discard_button_visible(state)
    state = recommendation._repair_chi_options_before_policy(state)
    rules = rules_for_room(
        RULES_PATH,
        wildcard_enabled=True,
        players=2,
        room_mode="1v1-wang",
    )
    state_for_policy = dict(state)
    state_for_policy["memory"] = memory.to_dict()
    state_for_policy["room_players"] = 2
    started = time.perf_counter()
    decision = choose_production_action(state_for_policy, rules=rules)
    outer_ms = (time.perf_counter() - started) * 1000.0
    view = public_view_from_production_state(
        state_for_policy,
        decision,
        rules=rules,
    )
    if view is None:
        raise RuntimeError("memory_prepared_public_view_missing")
    inner_state = _production_state_from_public_view(
        view,
        legal_actions=[{"type": "DISCARD"}],
        pending_card=None,
        chi_options=None,
    )
    inner_started = time.perf_counter()
    inner = choose_production_action(inner_state, rules=rules)
    return json.dumps(
        {
            "hand_count": len(state_for_policy.get("hand") or []),
            "memory_frames": memory.frames_seen,
            "memory_hand_count": len(memory.last_hand),
            "elapsed_ms": round(outer_ms, 3),
            "selected_label": decision.selected_label,
            "evaluation_count": len(decision.action_evals),
            "inner_elapsed_ms": round(
                (time.perf_counter() - inner_started) * 1000.0,
                3,
            ),
            "inner_selected_label": inner.selected_label,
            "inner_evaluation_count": len(inner.action_evals),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def current_frame_inner_candidate_diagnostics_json(
    screenshot: str,
    home_dir: str,
    candidate_index: int,
) -> str:
    """Evaluate one exact inner discard candidate from the saved live state."""

    from ai.full_game_simulator import _production_state_from_public_view
    from ai.pro_brain import (
        EVScorer,
        allocate_hand_structures,
        analyze_hand,
        build_decision_context,
        choose_action as choose_production_action,
        generate_legal_actions,
    )
    from ai.search_brain import public_view_from_production_state
    from engine.rules import rules_for_room
    from tools import recommend_action as recommendation
    from vision.history_memory import load_memory, recover_temporally_hidden_hand

    state = recommendation.inspect_screenshot(
        screenshot,
        config_path=CONFIG_PATH,
        expected_total=21,
    )
    memory = load_memory(Path(home_dir) / "vision_memory_1v1-wang.json")
    state = recover_temporally_hidden_hand(state, memory, expected_total=21)
    memory.update_from_snapshot(state)
    state = recommendation._clear_stale_options_when_discard_button_visible(state)
    state = recommendation._repair_chi_options_before_policy(state)
    rules = rules_for_room(
        RULES_PATH,
        wildcard_enabled=True,
        players=2,
        room_mode="1v1-wang",
    )
    state_for_policy = dict(state)
    state_for_policy["memory"] = memory.to_dict()
    state_for_policy["room_players"] = 2
    outer = choose_production_action(state_for_policy, rules=rules)
    view = public_view_from_production_state(state_for_policy, outer, rules=rules)
    if view is None:
        raise RuntimeError("inner_candidate_public_view_missing")
    inner_state = _production_state_from_public_view(
        view,
        legal_actions=[{"type": "DISCARD"}],
        pending_card=None,
        chi_options=None,
    )
    context = build_decision_context(inner_state, rules=rules)
    allocation = allocate_hand_structures(context)
    analysis = analyze_hand(context, allocation)
    actions, _ = generate_legal_actions(context, allocation, analysis)
    index = int(candidate_index) - 1
    if not 0 <= index < len(actions):
        raise ValueError(f"candidate_index_out_of_range:{candidate_index}")
    action = actions[index]
    started = time.perf_counter()
    evaluated = EVScorer().evaluate(context, action, allocation, analysis)
    return json.dumps(
        {
            "candidate_index": index + 1,
            "label": action.label,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "allowed": evaluated.allowed,
            "ev": evaluated.ev,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def current_frame_vision_diagnostics_json(screenshot: str) -> str:
    """Diagnostic-only recognition timing for the saved device frame."""
    from tools.inspect_state import inspect_screenshot

    timings: dict[str, float] = {}
    state = inspect_screenshot(
        screenshot,
        config_path=CONFIG_PATH,
        expected_total=21,
        timings=timings,
    )
    return json.dumps(
        {
            "hand_count": len(state.get("hand") or []),
            "timings": timings,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def priority_action_json(screenshot: str) -> str:
    buttons = [
        button.to_dict()
        for button in detect_buttons_from_path(
            screenshot,
            config_path=CONFIG_PATH,
            template_dir=PROJECT_ROOT / "data" / "templates" / "buttons",
        )
    ]
    return json.dumps(
        visible_hu_plan(buttons) or {"ready": False},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def probe_actionable_json(screenshot: str) -> str:
    """Inspect only the UI event surface; never classify the full hand or run policy."""

    started = time.perf_counter()
    image = read_bgr(screenshot)
    buttons = detect_buttons(
        image,
        config_path=CONFIG_PATH,
        template_dir=PROJECT_ROOT / "data" / "templates" / "buttons",
    )
    discard_button = detect_discard_button(image, config_path=CONFIG_PATH)
    flow = detect_flow_state(image)
    option_stage = None
    if not buttons and discard_button is None and flow.state == "play":
        candidates = detect_option_candidates(image, config_path=CONFIG_PATH)
        stages = {candidate.region_name for candidate in candidates}
        if "compare_options" in stages:
            option_stage = "compare"
        elif "chi_options" in stages:
            option_stage = "chi"
    button_names = sorted(button.name.lower() for button in buttons)
    actionable = bool(
        button_names
        or discard_button is not None
        or option_stage is not None
        or flow.state == "settlement_ready"
    )
    return json.dumps(
        {
            "actionable": actionable,
            "flow_state": flow.state,
            "button_names": button_names,
            "discard_button_visible": discard_button is not None,
            "option_stage": option_stage,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def probe_actionable_rgba_json(
    pixels: object,
    width: int,
    height: int,
    row_bytes: int,
    home_dir: str,
) -> str:
    token = register_rgba_frame(
        pixels,
        width=int(width),
        height=int(height),
        row_bytes=int(row_bytes),
        token_root=home_dir,
    )
    try:
        return probe_actionable_json(str(token))
    finally:
        unregister_frame(token)


def decide_rgba_json(
    pixels: object,
    width: int,
    height: int,
    row_bytes: int,
    wildcard_enabled: bool,
    home_dir: str,
) -> str:
    """Run the unchanged desktop decision chain on one lossless Android frame."""
    token = register_rgba_frame(
        pixels,
        width=int(width),
        height=int(height),
        row_bytes=int(row_bytes),
        token_root=home_dir,
    )
    try:
        image = read_bgr(token)
        mode = "1v1-wang" if bool(wildcard_enabled) else "1v1-no-wang"
        guard_file = Path(home_dir) / f"action_guard_{mode}.json"
        pending_transaction = False
        try:
            guard_payload = json.loads(guard_file.read_text(encoding="utf-8"))
            pending = guard_payload.get("pending_response")
            pending_transaction = bool(
                isinstance(pending, dict)
                and str(pending.get("action")) in {"chi", "compare"}
            )
        except (OSError, ValueError, TypeError):
            pass
        buttons = detect_buttons(
            image,
            config_path=CONFIG_PATH,
            template_dir=PROJECT_ROOT / "data" / "templates" / "buttons",
            names=("hu",) if pending_transaction else None,
        )
        priority = visible_hu_plan([button.to_dict() for button in buttons])
        if priority is not None and priority.get("ready"):
            priority["_mobile_status"] = "execute"
            priority["_mobile_fatal"] = False
            return json.dumps(priority, ensure_ascii=False, separators=(",", ":"))
        return recommend_json(
            str(token),
            bool(wildcard_enabled),
            home_dir,
            int(width),
            int(height),
            precomputed_buttons=None if pending_transaction else buttons,
            pending_surface_buttons=buttons if pending_transaction else None,
        )
    finally:
        unregister_frame(token)


def recommend_json(
    screenshot: str,
    wildcard_enabled: bool,
    home_dir: str,
    width: int,
    height: int,
    *,
    precomputed_buttons: list | None = None,
    pending_surface_buttons: list | None = None,
) -> str:
    home = Path(home_dir)
    home.mkdir(parents=True, exist_ok=True)
    # Chaquopy assets are importable but are not a real directory which chdir can
    # enter. Use the app-owned files directory as a readable cwd, then let the
    # frozen entry points resolve packaged configs from their absolute module root.
    os.chdir(home)
    mode = "1v1-wang" if bool(wildcard_enabled) else "1v1-no-wang"
    memory_file = home / f"vision_memory_{mode}.json"
    guard_file = home / f"action_guard_{mode}.json"
    trusted_response_hand = _trusted_response_hand_labels(
        home / f"confirmed_hand_ledger_{mode}.json",
        guard_file,
        precomputed_buttons,
    )
    result = run_once(
        device_id="android-local",
        memory_file=memory_file,
        guard_file=guard_file,
        execute_settlement_ready=False,
        execute_play_actions=False,
        seat_role="auto",
        allow_opening_hand_compact=False,
        runtime_screenshot_path=screenshot,
        reuse_runtime_screenshot=True,
        wildcard_enabled=bool(wildcard_enabled),
        players=2,
        room_mode=mode,
        # Device latency is an acceptance metric, not permission to return a
        # different policy after a partial frozen-v8.1 evaluation.
        decision_time_budget_seconds=None,
        precomputed_buttons=precomputed_buttons,
        pending_surface_buttons=pending_surface_buttons,
        precomputed_hand_labels=trusted_response_hand,
    )
    process_cleanup = release_pending_strategy_work(_PROCESS_POOL)
    plan = dict(result.get("action_plan") or {})
    fatal = bool(result.get("fatal"))
    plan.setdefault(
        "action",
        str(result.get("action") or ("error" if fatal else "wait")),
    )
    plan.setdefault(
        "reason",
        str(
            result.get("reason")
            or result.get("error")
            or ("运行链路报告致命错误" if fatal else "等待牌局事件")
        ),
    )
    if plan.get("ready"):
        mobile_status = "execute"
    elif fatal:
        mobile_status = "error"
    elif str(plan.get("action") or "").lower() == "safe_halt":
        mobile_status = "blocked"
    else:
        mobile_status = "wait"
    plan["_mobile_status"] = mobile_status
    plan["_mobile_fatal"] = fatal
    plan["_mobile_process_cleanup"] = process_cleanup
    plan["_mobile_flow_state"] = str(result.get("flow", {}).get("state") or "")
    plan["_mobile_surface_fast_path"] = bool(
        result.get("pending_chi_surface_fast_path")
        or result.get("pending_compare_surface_fast_path")
    )
    if plan.get("execution_mode") == "discard_drag":
        plan = _mobile_discard_drag(plan, width=int(width), height=int(height))

    contract_error = action_plan_contract_error(plan)
    if plan.get("ready") and contract_error is not None:
        raise RuntimeError(f"mobile_action_plan_contract_error:{contract_error}")

    signature = _plan_signature(plan)
    context = _result_context_signature(result)
    planned_target = planned_hand_target(plan)
    if planned_target is not None:
        context["planned_hand_target"] = planned_target
    plan["_mobile_signature"] = json.dumps(
        {
            "plan": _signature_to_json(signature) if signature is not None else None,
            "context": context,
            "execution": {
                "action": plan.get("action"),
                "ready": bool(plan.get("ready")),
                "execution_mode": plan.get("execution_mode", "tap_sequence"),
                "reason": plan.get("reason"),
                "validation": plan.get("validation"),
                "target": plan.get("target"),
                "target_type": plan.get("target_type"),
                "target_label": plan.get("target_label"),
                "target_option_id": plan.get("target_option_id"),
                "target_option_cards": plan.get("target_option_cards") or [],
                "option_center": plan.get("option_center"),
                "_mobile_surface_fast_path": bool(
                    plan.get("_mobile_surface_fast_path")
                ),
                "clicks": plan.get("clicks") or [],
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if plan.get("action") == "settlement_ready" and plan.get("ready"):
        save_memory(VisionMemory(), memory_file)
    return json.dumps(plan, ensure_ascii=False, separators=(",", ":"))


def _trusted_response_hand_labels(
    ledger_path: Path,
    guard_path: Path,
    buttons: list | None,
) -> list[str] | None:
    button_names = {
        str(getattr(button, "name", "")).lower()
        for button in (buttons or [])
    }
    if not button_names.intersection({"chi", "peng", "pao", "pass"}):
        return None
    ledger = load_confirmed_ledger(ledger_path)
    if (
        not ledger.get("trusted")
        or ledger.get("shadow_only") is not True
        or str(ledger.get("last_confirmed_action") or "") != "discard"
        or ledger.get("pending_reconciliation_reason")
        or ledger.get("observation_mismatch")
    ):
        return None
    try:
        guard = json.loads(guard_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    signature = guard.get("last_executed_signature")
    context = guard.get("last_context")
    if (
        not isinstance(signature, list)
        or not signature
        or str(signature[0]).lower() != "discard"
        or not isinstance(context, dict)
        or guard.get("inflight_action")
    ):
        return None
    try:
        ledger_updated_at = float(ledger.get("updated_at") or 0.0)
        guard_updated_at = float(guard.get("last_executed_at") or 0.0)
        age_seconds = time.time() - ledger_updated_at
    except (TypeError, ValueError):
        return None
    if (
        not 0.0 <= age_seconds <= 120.0
        or abs(ledger_updated_at - guard_updated_at) > 2.0
    ):
        return None
    expected_counts: dict[str, int] = {}
    for item in context.get("hand_signature") or []:
        if not isinstance(item, list) or len(item) != 2:
            return None
        try:
            count = int(item[1])
        except (TypeError, ValueError):
            return None
        if count > 0:
            expected_counts[str(item[0])] = count
    target = context.get("planned_hand_target") or {}
    target_label = str(target.get("label") or "")
    if not target_label or expected_counts.get(target_label, 0) <= 0:
        return None
    expected_counts[target_label] -= 1
    if expected_counts[target_label] <= 0:
        expected_counts.pop(target_label, None)
    try:
        ledger_counts = {
            str(label): int(count)
            for label, count in (ledger.get("hand_counts") or {}).items()
            if int(count) > 0
        }
    except (TypeError, ValueError):
        return None
    if ledger_counts != expected_counts:
        return None
    labels = [
        str(label)
        for label, count in sorted(ledger_counts.items())
        for _ in range(max(0, int(count)))
    ]
    return labels or None


def verify_before_execution_json(screenshot: str, mobile_signature: str) -> str:
    """Return an executable current-frame plan or a bounded rejection reason."""

    started = time.perf_counter()
    parsed = json.loads(mobile_signature)
    image = read_bgr(screenshot)
    buttons = detect_buttons(
        image,
        config_path=CONFIG_PATH,
        template_dir=PROJECT_ROOT / "data" / "templates" / "buttons",
    )
    priority = visible_hu_plan([button.to_dict() for button in buttons]) or {"ready": False}
    if priority.get("ready"):
        return json.dumps(
            {
                "status": "execute" if parsed.get("priority") == "hu" else "preempt_hu",
                "reason": "visible_hu_button",
                "plan": priority,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    if parsed.get("priority") == "hu":
        return json.dumps(
            {
                "status": "stale",
                "reason": "visible_hu_button_missing",
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    action = _signature_action(parsed)
    expected = parsed.get("context") or {}
    actual = _fresh_context_from_image(
        image,
        action=action,
        expected=expected,
        buttons=buttons,
        screenshot=screenshot,
        surface_phase="pre",
    )
    signature_json = parsed.get("plan")
    if action == "settlement_ready":
        raw_clicks = signature_json[1] if isinstance(signature_json, list) and len(signature_json) > 1 else None
        flow = detect_flow_state(image).to_dict()
        if not settlement_ready_click_is_fresh(raw_clicks, flow):
            return json.dumps(
                {
                    "status": "stale",
                    "reason": "settlement_ready_button_changed",
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
    target_result = None
    if action == "discard":
        target = expected.get("planned_hand_target")
        if not isinstance(target, dict) or not target.get("label"):
            target_result = {"status": "missing", "reason": "planned_target_missing"}
        else:
            target_result = locate_hand_target(
                image,
                expected_label=str(target["label"]),
                original_x=int(target.get("x", -1)),
                original_y=int(target.get("y", -1)),
                config_path=CONFIG_PATH,
                template_dir=PROJECT_ROOT / "data" / "templates" / "hand_auto",
            )
    decision = evaluate_pre_action(
        action=action,
        expected=expected,
        actual=actual,
        target=target_result,
    )
    status = str(decision.get("status") or "uncertain")
    payload = {
        "status": status,
        "reason": str(decision.get("reason") or "pre_action_uncertain"),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    if status in {"execute", "relocated"}:
        plan = _execution_plan_from_signature(parsed, mobile_signature)
        if plan is None:
            payload.update(status="uncertain", reason="execution_plan_missing")
        else:
            if status == "relocated" and isinstance(target_result, dict):
                plan = _relocate_discard_plan(
                    plan,
                    x=int(target_result["x"]),
                    y=int(target_result["y"]),
                )
            if (
                str(plan.get("execution_mode") or "") == "select_then_discard_button"
                and bool((target_result or {}).get("selected"))
            ):
                plan = _discard_confirmation_only_plan(plan)
                payload["reason"] = "semantic_target_already_selected"
            payload["plan"] = plan
            payload["target"] = target_result
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def verify_discard_selection_json(screenshot: str, mobile_signature: str) -> str:
    """Verify the selected semantic card between tap and discard confirmation."""

    started = time.perf_counter()
    parsed = json.loads(mobile_signature)
    expected = parsed.get("context") or {}
    image = read_bgr(screenshot)
    buttons = detect_buttons(
        image,
        config_path=CONFIG_PATH,
        template_dir=PROJECT_ROOT / "data" / "templates" / "buttons",
    )
    target = expected.get("planned_hand_target")
    if isinstance(target, dict) and target.get("label"):
        target_result = locate_hand_target(
            image,
            expected_label=str(target["label"]),
            original_x=int(target.get("x", -1)),
            original_y=int(target.get("y", -1)),
            config_path=CONFIG_PATH,
            template_dir=PROJECT_ROOT / "data" / "templates" / "hand_auto",
            prefer_selected=True,
        )
    else:
        target_result = {"status": "missing", "reason": "planned_target_missing"}
    actual = _fresh_context_from_image(
        image,
        action="discard",
        expected=expected,
        buttons=buttons,
        screenshot=screenshot,
        surface_phase="pre",
    )
    decision = evaluate_discard_selection(
        expected=expected,
        actual=actual,
        target=target_result,
    )
    decision["target"] = target_result
    decision["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return json.dumps(decision, ensure_ascii=False, separators=(",", ":"))


def verify_discard_selection_rgba_json(
    pixels: object,
    width: int,
    height: int,
    row_bytes: int,
    mobile_signature: str,
    home_dir: str,
) -> str:
    token = register_rgba_frame(
        pixels,
        width=int(width),
        height=int(height),
        row_bytes=int(row_bytes),
        token_root=home_dir,
    )
    try:
        return verify_discard_selection_json(str(token), mobile_signature)
    finally:
        unregister_frame(token)


def verify_before_execution_rgba_json(
    pixels: object,
    width: int,
    height: int,
    row_bytes: int,
    mobile_signature: str,
    home_dir: str,
) -> str:
    token = register_rgba_frame(
        pixels,
        width=int(width),
        height=int(height),
        row_bytes=int(row_bytes),
        token_root=home_dir,
    )
    try:
        return verify_before_execution_json(str(token), mobile_signature)
    finally:
        unregister_frame(token)


def confirm_action_result_json(screenshot: str, mobile_signature: str) -> str:
    """Observe action-specific effects without treating arbitrary change as success."""

    started = time.perf_counter()
    parsed = json.loads(mobile_signature)
    action = "hu" if parsed.get("priority") == "hu" else _signature_action(parsed)
    expected = parsed.get("context") or {}
    if action == "hu" and not expected:
        expected = {
            "flow_state": "play",
            "button_names": ["hu"],
            "discard_button_visible": False,
        }
    actual = _fresh_context_from_image(
        read_bgr(screenshot),
        action=action,
        expected=expected,
        screenshot=screenshot,
    )
    decision = evaluate_post_action(action=action, expected=expected, actual=actual)
    decision["observed_option_stage"] = actual.get("option_stage")
    decision["observed_button_names"] = actual.get("button_names") or []
    decision["surface_features"] = sorted(required_surface_features(action, expected=expected))
    decision["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return json.dumps(decision, ensure_ascii=False, separators=(",", ":"))


def confirm_action_result_rgba_json(
    pixels: object,
    width: int,
    height: int,
    row_bytes: int,
    mobile_signature: str,
    home_dir: str,
) -> str:
    token = register_rgba_frame(
        pixels,
        width=int(width),
        height=int(height),
        row_bytes=int(row_bytes),
        token_root=home_dir,
    )
    try:
        return confirm_action_result_json(str(token), mobile_signature)
    finally:
        unregister_frame(token)


def validate_fresh_plan_json(screenshot: str, mobile_signature: str) -> str:
    """Validate a completed plan against a newly captured frame without replanning."""
    parsed = json.loads(mobile_signature)
    if parsed.get("priority") == "hu":
        current = json.loads(priority_action_json(screenshot))
        fresh = bool(current.get("ready") and current.get("action") == "hu")
        return json.dumps(
            {
                "fresh": fresh,
                "changed": [] if fresh else ["visible_hu_button"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    signature_json = parsed.get("plan")
    if isinstance(signature_json, list) and signature_json[:1] == ["settlement_ready"]:
        flow = detect_flow_state_from_path(screenshot).to_dict()
        raw_clicks = signature_json[1] if len(signature_json) > 1 else None
        fresh = settlement_ready_click_is_fresh(raw_clicks, flow)
        return json.dumps(
            {
                "fresh": fresh,
                "changed": [] if fresh else ["settlement_ready_button"],
                "actual": {"flow": flow},
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    expected = parsed.get("context") or {}
    action = str(signature_json[0] or "") if isinstance(signature_json, list) and signature_json else ""
    actual = _fresh_context_from_screenshot(screenshot, action=action, expected=expected)
    changed = changed_fresh_context_for_action(expected, actual, action)
    if action.lower() == "discard":
        target = expected.get("planned_hand_target")
        if isinstance(target, dict):
            image = read_bgr(screenshot)
            if not hand_target_is_clickable(
                image,
                int(target.get("x", -1)),
                int(target.get("y", -1)),
                config_path=CONFIG_PATH,
            ):
                changed.append("planned_hand_target")
    return json.dumps(
        {
            "fresh": not changed,
            "changed": changed,
            "expected": {key: expected.get(key) for key in FRESH_CONTEXT_KEYS},
            "actual": {key: actual.get(key) for key in FRESH_CONTEXT_KEYS},
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def reset_transient_guard_json(home_dir: str, wildcard_enabled: bool) -> str:
    reset = reset_transient_external_action_guard(
        _mobile_guard_file(home_dir, wildcard_enabled),
    )
    return json.dumps(
        {"reset": bool(reset), "reason": "ok" if reset else "guard_write_failed"},
        separators=(",", ":"),
    )


def validate_fresh_rgba_json(
    pixels: object,
    width: int,
    height: int,
    row_bytes: int,
    mobile_signature: str,
    home_dir: str,
) -> str:
    token = register_rgba_frame(
        pixels,
        width=int(width),
        height=int(height),
        row_bytes=int(row_bytes),
        token_root=home_dir,
    )
    try:
        return validate_fresh_plan_json(str(token), mobile_signature)
    finally:
        unregister_frame(token)


def lossless_frame_probe_json(
    pixels: object,
    width: int,
    height: int,
    row_bytes: int,
    home_dir: str,
) -> str:
    """Device-test hook proving Android bytes arrive as exact BGR pixels."""
    import hashlib

    token = register_rgba_frame(
        pixels,
        width=int(width),
        height=int(height),
        row_bytes=int(row_bytes),
        token_root=home_dir,
    )
    try:
        image = read_bgr(token)
        return json.dumps(
            {
                "shape": list(image.shape),
                "bgr": image.reshape(-1, 3).tolist(),
                "sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            },
            separators=(",", ":"),
        )
    finally:
        unregister_frame(token)


def profile_rgba_vision_json(
    pixels: object,
    width: int,
    height: int,
    row_bytes: int,
    home_dir: str,
) -> str:
    started = time.perf_counter()
    token = register_rgba_frame(
        pixels,
        width=int(width),
        height=int(height),
        row_bytes=int(row_bytes),
        token_root=home_dir,
    )
    registered = time.perf_counter()
    try:
        priority = json.loads(priority_action_json(str(token)))
        priority_done = time.perf_counter()
        vision_timings: dict[str, float] = {}
        state = inspect_screenshot(
            str(token),
            config_path=CONFIG_PATH,
            expected_total=None,
            opponent_priority_pending=False,
            timings=vision_timings,
        )
        inspected = time.perf_counter()
        return json.dumps(
            {
                "register_ms": round((registered - started) * 1000, 3),
                "priority_ms": round((priority_done - registered) * 1000, 3),
                "inspect_ms": round((inspected - priority_done) * 1000, 3),
                "total_ms": round((inspected - started) * 1000, 3),
                "priority_action": priority.get("action") if priority.get("ready") else None,
                "hand_count": len(state.get("hand") or []),
                "buttons": [item.get("name") for item in state.get("buttons") or []],
                "option_count": len(state.get("option_details") or []),
                "vision_timings": vision_timings,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    finally:
        unregister_frame(token)


def commit_executed_action_json(
    mobile_signature: str,
    home_dir: str,
    wildcard_enabled: bool,
) -> str:
    """Commit an Android-confirmed gesture through the desktop guard contract."""
    signature, context, action_plan, reason = _decode_mobile_action(mobile_signature)
    if reason == "priority_hu":
        return json.dumps(
            {"committed": True, "reason": "priority_hu_has_no_deferred_followup"},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    if signature is None or context is None or action_plan is None:
        return json.dumps(
            {"committed": False, "reason": reason},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    committed = commit_executed_action_guard(
        _mobile_guard_file(home_dir, wildcard_enabled),
        signature=mobile_guard_signature(signature),
        context_signature=context,
        action_plan=action_plan,
    )
    if committed:
        mode = "1v1-wang" if bool(wildcard_enabled) else "1v1-no-wang"
        record_confirmed_action(
            Path(home_dir) / f"confirmed_hand_ledger_{mode}.json",
            mode=mode,
            action=str(action_plan.get("action") or ""),
            context=context,
        )
        cache_cleanup = (
            _clear_confirmed_round_runtime_caches()
            if str(action_plan.get("action") or "") == "settlement_ready"
            else None
        )
    else:
        cache_cleanup = None
    return json.dumps(
        {
            "committed": bool(committed),
            "reason": "ok" if committed else "guard_write_failed",
            "round_cache_cleanup": cache_cleanup,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def prepare_external_action_json(
    mobile_signature: str,
    home_dir: str,
    wildcard_enabled: bool,
) -> str:
    signature, context, _action_plan, reason = _decode_mobile_action(mobile_signature)
    if reason == "priority_hu":
        return json.dumps({"armed": True, "reason": "priority_hu"}, separators=(",", ":"))
    if signature is None or context is None:
        return json.dumps({"armed": False, "reason": reason}, separators=(",", ":"))
    outcome = prepare_external_action_guard(
        _mobile_guard_file(home_dir, wildcard_enabled),
        signature=mobile_guard_signature(signature),
        context_signature=context,
    )
    return json.dumps(
        {"armed": outcome == "armed", "reason": outcome},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def abort_external_action_json(
    mobile_signature: str,
    home_dir: str,
    wildcard_enabled: bool,
) -> str:
    signature, context, _action_plan, reason = _decode_mobile_action(mobile_signature)
    if reason == "priority_hu":
        return json.dumps({"aborted": True, "reason": "priority_hu"}, separators=(",", ":"))
    if signature is None or context is None:
        return json.dumps({"aborted": False, "reason": reason}, separators=(",", ":"))
    aborted = abort_external_action_guard(
        _mobile_guard_file(home_dir, wildcard_enabled),
        signature=mobile_guard_signature(signature),
        context_signature=context,
    )
    return json.dumps(
        {"aborted": bool(aborted), "reason": "ok" if aborted else "guard_write_failed"},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _mobile_guard_file(home_dir: str, wildcard_enabled: bool) -> Path:
    home = Path(home_dir)
    home.mkdir(parents=True, exist_ok=True)
    mode = "1v1-wang" if bool(wildcard_enabled) else "1v1-no-wang"
    return home / f"action_guard_{mode}.json"


def _signature_action(parsed: dict) -> str:
    signature_json = parsed.get("plan")
    if not isinstance(signature_json, list) or not signature_json:
        return ""
    return str(signature_json[0] or "").lower()


def _execution_plan_from_signature(parsed: dict, mobile_signature: str) -> dict | None:
    execution = parsed.get("execution")
    if not isinstance(execution, dict):
        return None
    plan = dict(execution)
    plan["clicks"] = [dict(item) for item in execution.get("clicks") or [] if isinstance(item, dict)]
    plan["ready"] = True
    plan.setdefault("validation", {"passed": True, "checks": ["pre_action_verifier"]})
    plan["_mobile_signature"] = mobile_signature
    return plan


def _relocate_discard_plan(plan: dict, *, x: int, y: int) -> dict:
    relocated = dict(plan)
    clicks = [dict(item) for item in plan.get("clicks") or []]
    if not clicks:
        return relocated
    click = clicks[0]
    if str(plan.get("execution_mode") or "") == "drag_sequence":
        click["from_x"] = int(x)
        click["from_y"] = int(y)
        click["x"] = int(x)
    else:
        click["x"] = int(x)
        click["y"] = int(y)
    clicks[0] = click
    relocated["clicks"] = clicks
    relocated["reason"] = f"目标牌已换位，更新点击坐标为({int(x)},{int(y)})"
    validation = dict(relocated.get("validation") or {})
    validation["passed"] = True
    checks = list(validation.get("checks") or [])
    if "target_relocated_on_latest_frame" not in checks:
        checks.append("target_relocated_on_latest_frame")
    validation["checks"] = checks
    relocated["validation"] = validation
    return relocated


def _discard_confirmation_only_plan(plan: dict) -> dict:
    patched = dict(plan)
    clicks = [dict(item) for item in plan.get("clicks") or []]
    if (
        len(clicks) == 2
        and str(clicks[0].get("target") or "").startswith("hand:")
        and str(clicks[1].get("target") or "") == "button:discard"
    ):
        patched["clicks"] = [clicks[1]]
        patched["execution_mode"] = "tap_sequence"
        patched["reason"] = "目标牌已处于选中状态，直接确认出牌"
    return patched


def _decode_mobile_action(mobile_signature: str) -> tuple[tuple | None, dict | None, dict | None, str]:
    try:
        parsed = json.loads(mobile_signature)
    except (json.JSONDecodeError, TypeError):
        return None, None, None, "invalid_mobile_signature"
    if parsed.get("priority") == "hu":
        return None, None, None, "priority_hu"
    signature_json = parsed.get("plan")
    context = parsed.get("context")
    if not isinstance(signature_json, list) or len(signature_json) != 2 or not isinstance(context, dict):
        return None, None, None, "invalid_mobile_signature"
    action = str(signature_json[0] or "")
    raw_clicks = signature_json[1]
    if not action or not isinstance(raw_clicks, list):
        return None, None, None, "invalid_mobile_plan_signature"
    try:
        clicks = tuple(
            (str(item[0]), int(item[1]), int(item[2]))
            for item in raw_clicks
            if isinstance(item, list) and len(item) == 3
        )
    except (TypeError, ValueError):
        clicks = ()
    if len(clicks) != len(raw_clicks):
        return None, None, None, "invalid_mobile_click_signature"
    signature = (action, clicks)
    action_plan = {
        "action": action,
        "ready": True,
        "clicks": [
            {"target": target, "x": x, "y": y}
            for target, x, y in clicks
        ],
    }
    execution = parsed.get("execution")
    if isinstance(execution, dict):
        for key in (
            "execution_mode",
            "reason",
            "validation",
            "target",
            "target_type",
            "target_label",
            "target_option_id",
            "target_option_cards",
            "option_center",
        ):
            if key in execution:
                action_plan[key] = execution[key]
    return signature, context, action_plan, "ok"


def _fresh_context_from_screenshot(
    screenshot: str,
    *,
    action: str = "",
    expected: dict | None = None,
) -> dict:
    return _fresh_context_from_image(
        read_bgr(screenshot),
        action=action,
        expected=expected,
        screenshot=screenshot,
    )


def _fresh_context_from_image(
    image,
    *,
    action: str = "",
    expected: dict | None = None,
    buttons: list | None = None,
    screenshot: str | None = None,
    surface_phase: str = "post",
) -> dict:
    expected = expected or {}
    features = (
        required_pre_surface_features(action, expected=expected)
        if surface_phase == "pre"
        else required_surface_features(action, expected=expected)
    )
    if "full_state" not in features:
        detected_buttons = buttons
        if "buttons" in features and detected_buttons is None:
            detected_buttons = detect_buttons(
                image,
                config_path=CONFIG_PATH,
                template_dir=PROJECT_ROOT / "data" / "templates" / "buttons",
            )
        detected_buttons = detected_buttons or []
        option_stage = None
        if "options" in features:
            candidates = detect_option_candidates(image, config_path=CONFIG_PATH)
            stages = {candidate.region_name for candidate in candidates}
            if "compare_options" in stages:
                option_stage = "compare"
            elif "chi_options" in stages:
                option_stage = "chi"
        pending_action_card = None
        opponent_pending_card = None
        if "opponent_pending_card" in features:
            pending_cards = recognize_pending_cards(image, config_path=CONFIG_PATH)
            pending_action = pending_cards.get("pending_action_card")
            opponent_pending = pending_cards.get("opponent_pending_card")
            pending_action_card = pending_action.name if pending_action is not None else None
            opponent_pending_card = opponent_pending.name if opponent_pending is not None else None
        return {
            "flow_state": detect_flow_state(image).state if "flow" in features else expected.get("flow_state"),
            "controlled_card_count": expected.get("controlled_card_count"),
            "hand_signature": expected.get("hand_signature"),
            "button_names": sorted(button.name.lower() for button in detected_buttons),
            "discard_button_visible": (
                detect_discard_button(image, config_path=CONFIG_PATH) is not None
                if "discard_button" in features
                else bool(expected.get("discard_button_visible"))
            ),
            "pending_action_card": pending_action_card,
            "opponent_pending_card": opponent_pending_card,
            "option_stage": option_stage,
        }

    if screenshot is None:
        raise ValueError("full_state_confirmation_requires_screenshot")
    state = inspect_screenshot(
        screenshot,
        config_path=CONFIG_PATH,
        expected_total=None,
        opponent_priority_pending=False,
    )
    state["flow"] = detect_flow_state_from_path(screenshot).to_dict()
    details = state.get("option_details") or []
    if any(item.get("region_name") == "compare_options" for item in details):
        state["option_stage"] = "compare"
    elif any(item.get("region_name") == "chi_options" for item in details):
        state["option_stage"] = "chi"
    else:
        state["option_stage"] = None
    return _result_context_signature(state)


def _mobile_discard_drag(plan: dict, *, width: int, height: int) -> dict:
    config = load_region_config(CONFIG_PATH)
    transform = transform_from_config(config, width, height)
    play = config.get("play", {})
    drop_y = transform.map_point(0, int(play.get("discard_drop_y", 320)), clip=True)[1]
    duration_ms = int(play.get("drag_duration_ms", 850))
    clicks = []
    for original in plan.get("clicks") or []:
        click = dict(original)
        start_x = int(click["x"])
        start_y = int(click["y"])
        click.update(
            {
                "from_x": start_x,
                "from_y": start_y,
                "x": start_x,
                "y": drop_y,
                "duration_ms": duration_ms,
            }
        )
        clicks.append(click)
    patched = dict(plan)
    patched["clicks"] = clicks
    patched["execution_mode"] = "drag_sequence"
    return patched
