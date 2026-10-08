from threading import Event
from time import perf_counter

from ai.android_validation_cleanup import (
    AnchorFirstValidationExecutor,
    release_pending_strategy_work,
)


def test_validation_waits_until_foreground_anchor_requests_result():
    started = Event()
    executor = AnchorFirstValidationExecutor(max_workers=1)
    future = executor.submit(lambda: started.set() or "done")

    assert not started.wait(0.05)
    assert future.result(timeout=1.0) == "done"
    assert started.is_set()
    executor.shutdown(wait=True)


def test_expired_validation_cleanup_does_not_wait_for_running_work():
    release = Event()
    executor = AnchorFirstValidationExecutor(max_workers=1)
    future = executor.submit(release.wait, 1.0)
    runner = Event()

    def request_result():
        try:
            future.result(timeout=1.0)
        finally:
            runner.set()

    from threading import Thread

    Thread(target=request_result, daemon=True).start()
    assert not runner.wait(0.05)

    started = perf_counter()
    executor.shutdown(wait=True, cancel_futures=True)
    elapsed = perf_counter() - started
    release.set()

    assert elapsed < 0.2


class FakeStrategyProcessPool:
    def __init__(self, pending: int):
        self.pending = pending
        self.cancel_calls = 0
        self.restart_calls = 0

    def metricsJson(self) -> str:
        return f'{{"pending":{self.pending}}}'

    def cancelAll(self) -> None:
        self.cancel_calls += 1

    def restartWorkers(self) -> None:
        self.restart_calls += 1


def test_pending_strategy_work_restarts_workers_to_clear_running_python_tasks():
    pool = FakeStrategyProcessPool(pending=3)

    result = release_pending_strategy_work(pool)

    assert result == {
        "pending_before_cleanup": 3,
        "cleanup": "restarted_generation",
    }
    assert pool.cancel_calls == 0
    assert pool.restart_calls == 1


def test_idle_strategy_pool_is_left_untouched():
    pool = FakeStrategyProcessPool(pending=0)

    result = release_pending_strategy_work(pool)

    assert result == {
        "pending_before_cleanup": 0,
        "cleanup": "none",
    }
    assert pool.cancel_calls == 0
    assert pool.restart_calls == 0
