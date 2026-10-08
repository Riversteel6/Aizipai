"""Android adapter for deadline-bounded discard validation cleanup."""

from __future__ import annotations

import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable


class _DeferredValidationFuture:
    """Start Android validation only when the foreground anchor asks for it."""

    def __init__(
        self,
        executor: ThreadPoolExecutor,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        self._executor = executor
        self._function = function
        self._args = args
        self._kwargs = kwargs
        self._lock = threading.Lock()
        self._actual: Future | None = None
        self._cancelled = False

    def _start(self) -> Future:
        with self._lock:
            if self._actual is None:
                if self._cancelled:
                    cancelled = Future()
                    cancelled.cancel()
                    self._actual = cancelled
                else:
                    self._actual = self._executor.submit(
                        self._function,
                        *self._args,
                        **self._kwargs,
                    )
            return self._actual

    def result(self, timeout: float | None = None):
        return self._start().result(timeout=timeout)

    def cancel(self) -> bool:
        with self._lock:
            if self._actual is None:
                self._cancelled = True
                return True
            return self._actual.cancel()

    def done(self) -> bool:
        with self._lock:
            return self._cancelled or (
                self._actual is not None and self._actual.done()
            )


class AnchorFirstValidationExecutor:
    """Defer Android validation until the latency-critical anchor is complete."""

    def __init__(self, max_workers: int = 1, **_kwargs) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: list[_DeferredValidationFuture] = []

    def submit(self, function, /, *args, **kwargs) -> _DeferredValidationFuture:
        future = _DeferredValidationFuture(
            self._executor,
            function,
            args,
            kwargs,
        )
        self._futures.append(future)
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        del wait
        if cancel_futures:
            for future in self._futures:
                future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=cancel_futures)


def install_android_validation_cleanup() -> None:
    import ai.dual_validated_candidate as candidate

    candidate.ThreadPoolExecutor = AnchorFirstValidationExecutor


def release_pending_strategy_work(process_pool) -> dict[str, int | str]:
    """Hard-reset expired Android shards before a decision can be retried."""

    metrics = json.loads(str(process_pool.metricsJson()))
    pending = max(0, int(metrics.get("pending", 0)))
    cleanup = "none"
    if pending:
        process_pool.restartWorkers()
        cleanup = "restarted_generation"
    return {
        "pending_before_cleanup": pending,
        "cleanup": cleanup,
    }
