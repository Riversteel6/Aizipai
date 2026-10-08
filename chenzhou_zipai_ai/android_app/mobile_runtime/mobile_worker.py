"""Execute frozen strategy shards inside isolated Android app processes."""

from __future__ import annotations

import base64
import os
import pickle
import sys
import time
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RUNTIME_ROOT / "chenzhou_zipai_ai"
for path in (RUNTIME_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("AIZIPAI_MOBILE_RUNTIME", "1")


_TASKS = {
    "ai.pro_brain._evaluate_action_task": (
        "ai.pro_brain",
        "_evaluate_action_task",
    ),
    "ai.dual_discard_validator._root_search_task": (
        "ai.dual_discard_validator",
        "_root_search_task",
    ),
    "ai.parallel_response_search._response_search_task": (
        "ai.parallel_response_search",
        "_response_search_task",
    ),
}


def _execute_registered_task(task_name: str, args, kwargs):
    module_name, function_name = _TASKS[str(task_name)]
    module = __import__(module_name, fromlist=[function_name])
    return getattr(module, function_name)(*args, **kwargs)

def warmup(home_dir: str) -> bool:
    home = Path(home_dir)
    home.mkdir(parents=True, exist_ok=True)
    os.chdir(home)
    __import__("ai.dual_discard_validator")
    __import__("ai.parallel_response_search")
    __import__("ai.pro_brain")
    # Build the immutable legal-group index before a live response fans out to
    # this process. Otherwise every freshly restarted worker pays the same cold
    # construction cost inside the user's CHI/PASS deadline.
    from engine.hu_checker import best_grouping

    best_grouping(["一", "二", "三"])
    return True


def execute_task(task_name: str, payload: str, home_dir: str) -> str:
    home = Path(home_dir)
    home.mkdir(parents=True, exist_ok=True)
    os.chdir(home)
    args, kwargs = pickle.loads(base64.b64decode(str(payload)))
    if task_name == "diagnostic.echo":
        result = {
            "pid": os.getpid(),
            "value": args[0] if args else None,
            "worker_perf_counter": time.perf_counter(),
        }
    elif task_name == "diagnostic.sleep":
        time.sleep(float(args[0]))
        result = {"pid": os.getpid(), "slept": float(args[0])}
    elif task_name == "diagnostic.warmup":
        warmup(home_dir)
        result = {
            "pid": os.getpid(),
            "ready": True,
            "worker_perf_counter": time.perf_counter(),
        }
    elif task_name == "runtime.batch":
        batched_task_name, calls = args
        if str(batched_task_name) not in _TASKS:
            raise ValueError(f"unsupported_batched_strategy_task:{batched_task_name}")
        if str(batched_task_name) == "ai.pro_brain._evaluate_action_task":
            from ai.pro_brain import _evaluate_action_batch_task

            if any(call_kwargs for _call_args, call_kwargs in calls):
                raise ValueError("batched_action_evaluation_kwargs_unsupported")
            result = _evaluate_action_batch_task(
                [tuple(call_args)[0] for call_args, _call_kwargs in calls]
            )
        else:
            result = [
                _execute_registered_task(
                    str(batched_task_name),
                    tuple(call_args),
                    dict(call_kwargs),
                )
                for call_args, call_kwargs in calls
            ]
    else:
        result = _execute_registered_task(str(task_name), args, kwargs)
    return base64.b64encode(pickle.dumps(result, protocol=4)).decode("ascii")
