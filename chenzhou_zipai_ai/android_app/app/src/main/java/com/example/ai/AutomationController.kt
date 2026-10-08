package com.example.ai

import android.accessibilityservice.AccessibilityService
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import android.view.Display
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

internal data class DecisionWorkerFailureStatus(
    val title: String,
    val reason: String,
)

internal fun decisionWorkerFailureStatus(message: String?): DecisionWorkerFailureStatus =
    if (message.orEmpty().startsWith("decision_worker_timeout:")) {
        DecisionWorkerFailureStatus(
            title = "策略搜索超时",
            reason = "已清理本轮搜索任务，正在自动重试",
        )
    } else {
        DecisionWorkerFailureStatus(
            title = "字牌AI正在恢复",
            reason = "识图决策进程异常，正在自动重建",
        )
    }

class AutomationController(private val service: AccessibilityService) : AutoCloseable {
    private companion object {
        const val TAG = "AizipaiRuntime"
        const val UNCHANGED_FRAME_DELAY_MS = 350L
        const val SCREENSHOT_THROTTLE_BACKOFF_MS = 350L
        const val MIN_SCREENSHOT_INTERVAL_MS = 350L
        const val DECISION_TIMEOUT_MS = 27_000L
        const val ACTION_CONFIRMATION_DELAY_MS = 180L
    }
    private val gate = RunGeneration()
    private val worker = Executors.newSingleThreadScheduledExecutor { runnable ->
        Thread(runnable, "aizipai-runtime").apply { isDaemon = true }
    }
    private val bridge = PythonDecisionBridge(service)
    private val gestures = GestureExecutor(service)
    private val eventGate = FrameEventGate()
    private val executionGate = ActionConfirmationGate()
    private val postActionGate = PostActionConfirmationGate()
    private val settlementEpochTracker = SettlementEpochTracker()
    private val screenshotCadence = ScreenshotCadence(MIN_SCREENSHOT_INTERVAL_MS)
    private val decisionWatchdog = DecisionWatchdog(DECISION_TIMEOUT_MS) { token ->
        interruptDecisionWorker(token, "决策超过27秒仍未返回")
    }
    private val screenshotSequence = AtomicInteger()
    @Volatile private var scheduled: Future<*>? = null
    @Volatile private var mode: GameMode = GameMode.NO_WANG
    private var consecutiveFailures = 0
    private var skippedFrames = 0
    private var heldWarningReason: String? = null

    init {
        worker.execute {
            runCatching { bridge.prewarm() }
                .onSuccess { result -> Log.i(TAG, "runtime_prewarm_ready result=$result") }
                .onFailure { error -> Log.w(TAG, "runtime_prewarm_failed", error) }
        }
    }

    fun start(selectedMode: GameMode): Boolean {
        if (gate.isRunning()) return false
        mode = selectedMode
        consecutiveFailures = 0
        eventGate.reset()
        executionGate.reset()
        screenshotCadence.reset()
        skippedFrames = 0
        heldWarningReason = null
        val token = gate.start()
        settlementEpochTracker.resetSession("$token-${SystemClock.elapsedRealtimeNanos()}")
        Log.i(TAG, "runtime_started token=$token mode=${selectedMode.storedValue}")
        RuntimeStatusCenter.update(
            RuntimePhase.STARTING,
            "字牌AI正在启动",
            "准备读取游戏画面",
            selectedMode,
        )
        scheduled = worker.schedule(
            { resetTransientGuardAndCapture(token, selectedMode, 1) },
            0,
            TimeUnit.MILLISECONDS,
        )
        return true
    }

    fun stop(): Boolean {
        if (!gate.isRunning()) return false
        gate.stop()
        Log.i(TAG, "runtime_stopped")
        scheduled?.cancel(true)
        scheduled = null
        decisionWatchdog.cancel()
        bridge.restartDecisionWorker("runtime_stopped")
        StrategyProcessPool.cancelAll()
        return true
    }

    fun isRunning(): Boolean = gate.isRunning()

    private fun schedule(token: Long, delayMs: Long) {
        if (!gate.isCurrent(token)) return
        scheduled = worker.schedule({ capture(token) }, delayMs, TimeUnit.MILLISECONDS)
    }

    private fun deferScreenshot(token: Long, retry: () -> Unit): Boolean {
        if (!gate.isCurrent(token)) return true
        val now = SystemClock.elapsedRealtime()
        val delayMs = screenshotCadence.delayBeforeRequest(now)
        if (delayMs > 0) {
            scheduled = worker.schedule(retry, delayMs, TimeUnit.MILLISECONDS)
            return true
        }
        screenshotCadence.markRequested(now)
        return false
    }

    private fun capture(token: Long) {
        if (!gate.isCurrent(token)) return
        if (deferScreenshot(token) { capture(token) }) return
        val requestedAt = SystemClock.elapsedRealtime()
        service.takeScreenshot(
            Display.DEFAULT_DISPLAY,
            worker,
            object : AccessibilityService.TakeScreenshotCallback {
                override fun onSuccess(result: AccessibilityService.ScreenshotResult) {
                    val buffer = result.hardwareBuffer
                    val bitmap = try {
                        Bitmap.wrapHardwareBuffer(buffer, result.colorSpace)?.copy(Bitmap.Config.ARGB_8888, false)
                    } finally {
                        buffer.close()
                    }
                    if (bitmap == null) {
                        handleFailure(token, "截图缓冲区不可用", restartDecisionWorker = false)
                        return
                    }
                    Log.i(
                        TAG,
                        "screenshot_ready elapsed_ms=${SystemClock.elapsedRealtime() - requestedAt} " +
                            "size=${bitmap.width}x${bitmap.height}",
                    )
                    processFrame(token, bitmap, requestedAt)
                }

                override fun onFailure(errorCode: Int) {
                    Log.w(TAG, "screenshot_failed code=$errorCode token=$token")
                    if (errorCode == AccessibilityService.ERROR_TAKE_SCREENSHOT_INTERVAL_TIME_SHORT) {
                        RuntimeStatusCenter.update(
                            RuntimePhase.OBSERVING,
                            "字牌AI运行中",
                            "系统限制截图频率，正在自动继续",
                            mode,
                            severity = RuntimeSeverity.WARNING,
                        )
                        schedule(token, SCREENSHOT_THROTTLE_BACKOFF_MS)
                        return
                    }
                    handleFailure(token, "系统截图失败：$errorCode", restartDecisionWorker = false)
                }
            },
        )
    }

    private fun processFrame(
        token: Long,
        bitmap: Bitmap,
        requestedAt: Long,
        transitionFrame: Boolean = false,
    ) {
        var bitmapOwnershipTransferred = false
        if (!gate.isCurrent(token)) {
            bitmap.recycle()
            return
        }
        if (bitmap.width <= bitmap.height) {
            bitmap.recycle()
            consecutiveFailures = 0
            RuntimeStatusCenter.update(
                RuntimePhase.OBSERVING,
                "等待进入游戏",
                "切换到横屏游戏后会自动开始",
                mode,
            )
            schedule(token, 500)
            return
        }
        val fingerprintStartedAt = SystemClock.elapsedRealtime()
        val frameDecision = eventGate.decide(
            FrameFingerprint.fromBitmap(bitmap),
            fingerprintStartedAt,
        )
        if (frameDecision == FrameGateDecision.SKIP) {
            consecutiveFailures = 0
            skippedFrames += 1
            if (skippedFrames % 50 == 0) {
                Log.i(
                    TAG,
                    "light_frames_skipped count=$skippedFrames " +
                        "fingerprint_ms=${SystemClock.elapsedRealtime() - fingerprintStartedAt}",
                )
            }
            bitmap.recycle()
            val warning = heldWarningReason
            RuntimeStatusCenter.update(
                RuntimePhase.OBSERVING,
                if (warning == null) "字牌AI运行中" else "开局手牌仍未确认",
                warning ?: "轻量监控中，等待牌局变化",
                mode,
                severity = if (warning == null) RuntimeSeverity.NORMAL else RuntimeSeverity.WARNING,
            )
            schedule(token, UNCHANGED_FRAME_DELAY_MS)
            return
        }
        try {
            screenshotSequence.incrementAndGet()
            val frameCopyStartedAt = SystemClock.elapsedRealtime()
            val frame = LosslessFrameFactory.fromBitmap(bitmap)
            val frameCopiedAt = SystemClock.elapsedRealtime()
            if (frameDecision == FrameGateDecision.PROBE) {
                val probeStartedAt = SystemClock.elapsedRealtime()
                val probe = bridge.probe(frame)
                settlementEpochTracker.observe(probe.flowState)
                Log.i(
                    TAG,
                    "light_event_probe actionable=${probe.actionable} " +
                        "python_ms=${probe.elapsedMs} " +
                        "elapsed_ms=${SystemClock.elapsedRealtime() - probeStartedAt}",
                )
                if (!probe.actionable) {
                    consecutiveFailures = 0
                    RuntimeStatusCenter.update(
                        RuntimePhase.OBSERVING,
                        "字牌AI运行中",
                        probe.summary,
                        mode,
                    )
                    schedule(token, UNCHANGED_FRAME_DELAY_MS)
                    return
                }
            }
            RuntimeStatusCenter.update(
                RuntimePhase.THINKING,
                "正在识别和思考",
                "当前画面已读取",
                mode,
            )
            val decisionStartedAt = SystemClock.elapsedRealtime()
            val watchdogTicket = decisionWatchdog.arm(token)
            val outcome = try {
                bridge.decide(frame, mode)
            } finally {
                decisionWatchdog.disarm(watchdogTicket)
            }
            settlementEpochTracker.observe(outcome.flowState)
            val plan = outcome.plan?.let(settlementEpochTracker::decorate)
            val decisionElapsed = SystemClock.elapsedRealtime() - decisionStartedAt
            Log.i(
                TAG,
                "decision_finished elapsed_ms=${SystemClock.elapsedRealtime() - requestedAt} " +
                    "frame_copy_ms=${frameCopiedAt - frameCopyStartedAt} decision_ms=$decisionElapsed " +
                    "action=${outcome.action} disposition=${outcome.disposition}",
            )
            if (!gate.isCurrent(token)) return
            if (plan != null) {
                heldWarningReason = null
                RuntimeStatusCenter.update(
                    RuntimePhase.VERIFYING,
                    "准备执行${humanAction(plan.action)}",
                    "策略耗时 ${decisionElapsed}ms，正在核对最新画面",
                    mode,
                )
                if (canExecuteSurfacePlanOnCurrentFrame(plan, transitionFrame)) {
                    bitmapOwnershipTransferred = true
                    verifyFrameAndExecute(token, plan, bitmap, requestedAt)
                } else {
                    verifyAndExecute(token, plan, requestedAt)
                }
                return
            } else if (outcome.disposition == DecisionDisposition.ERROR) {
                heldWarningReason = null
                Log.e(
                    TAG,
                    "decision_error action=${outcome.action} reason=${outcome.reason}",
                )
                continueObserving(
                    token,
                    title = "决策暂时不可用",
                    reason = outcome.reason,
                    delayMs = 1_500,
                )
                return
            } else {
                executionGate.onNoPlan()
                Log.i(
                    TAG,
                    "decision_no_execution disposition=${outcome.disposition} " +
                        "action=${outcome.action} reason=${outcome.reason}",
                )
                RuntimeStatusCenter.update(
                    RuntimePhase.OBSERVING,
                    if (outcome.disposition == DecisionDisposition.BLOCKED) {
                        "当前画面暂不操作"
                    } else {
                        "字牌AI运行中"
                    },
                    outcome.reason,
                    mode,
                    severity = if (outcome.disposition == DecisionDisposition.BLOCKED) {
                        RuntimeSeverity.WARNING
                    } else {
                        RuntimeSeverity.NORMAL
                    },
                )
            }
            if (outcome.action == "wait_opening_locked_recheck") {
                heldWarningReason = outcome.reason
                eventGate.forceNext()
                schedule(token, 120)
                return
            }
            if (outcome.action == "wait_chi_candidate_recheck") {
                heldWarningReason = outcome.reason
                eventGate.forceNext()
                schedule(token, 120)
                return
            }
            if (
                outcome.reasonCode == "opening_compact_exhausted" ||
                outcome.reasonCode == "opening_compact_no_progress"
            ) {
                heldWarningReason = outcome.reason
                eventGate.suppressFallbackUntilChange()
            } else if (outcome.disposition != DecisionDisposition.BLOCKED) {
                heldWarningReason = null
            }
            consecutiveFailures = 0
            val elapsed = SystemClock.elapsedRealtime() - requestedAt
            schedule(token, maxOf(80, 240 - elapsed))
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
        } catch (error: DecisionWorkerUnavailableException) {
            Log.e(TAG, "decision_worker_unavailable token=$token", error)
            val status = decisionWorkerFailureStatus(error.message)
            recoverDecisionWorker(token, status.title, status.reason)
        } catch (error: Throwable) {
            Log.e(TAG, "frame_processing_failed token=$token", error)
            handleFailure(token, "画面处理异常", restartDecisionWorker = true)
        } finally {
            if (!bitmapOwnershipTransferred) bitmap.recycle()
        }
    }

    private fun verifyAndExecute(token: Long, plan: DecisionPlan, decisionRequestedAt: Long) {
        if (!gate.isCurrent(token)) return
        if (deferScreenshot(token) { verifyAndExecute(token, plan, decisionRequestedAt) }) return
        service.takeScreenshot(
            Display.DEFAULT_DISPLAY,
            worker,
            object : AccessibilityService.TakeScreenshotCallback {
                override fun onSuccess(result: AccessibilityService.ScreenshotResult) {
                    val buffer = result.hardwareBuffer
                    val bitmap = try {
                        Bitmap.wrapHardwareBuffer(buffer, result.colorSpace)?.copy(Bitmap.Config.ARGB_8888, false)
                    } finally {
                        buffer.close()
                    }
                    if (bitmap == null) {
                        handleFailure(token, "核验截图缓冲区不可用", restartDecisionWorker = false)
                        return
                    }
                    verifyFrameAndExecute(token, plan, bitmap, decisionRequestedAt)
                }

                override fun onFailure(errorCode: Int) {
                    Log.w(TAG, "verification_screenshot_failed code=$errorCode token=$token")
                    if (errorCode == AccessibilityService.ERROR_TAKE_SCREENSHOT_INTERVAL_TIME_SHORT) {
                        RuntimeStatusCenter.update(
                            RuntimePhase.VERIFYING,
                            "正在核对最新画面",
                            "系统限制截图频率，稍后继续核对",
                            mode,
                            severity = RuntimeSeverity.WARNING,
                        )
                        scheduled = worker.schedule(
                            { verifyAndExecute(token, plan, decisionRequestedAt) },
                            SCREENSHOT_THROTTLE_BACKOFF_MS,
                            TimeUnit.MILLISECONDS,
                        )
                        return
                    }
                    handleFailure(token, "核验截图失败：$errorCode", restartDecisionWorker = false)
                }
            },
        )
    }

    private fun verifyFrameAndExecute(
        token: Long,
        plan: DecisionPlan,
        bitmap: Bitmap,
        decisionRequestedAt: Long,
    ) {
        try {
            if (!gate.isCurrent(token)) return
            if (bitmap.width <= bitmap.height) {
                RuntimeStatusCenter.update(
                    RuntimePhase.OBSERVING,
                    "等待返回游戏",
                    "动作已取消，回到横屏游戏后继续",
                    mode,
                )
                schedule(token, 500)
                return
            }
            val verifyStartedAt = SystemClock.elapsedRealtime()
            screenshotSequence.incrementAndGet()
            val frame = LosslessFrameFactory.fromBitmap(bitmap)
            val verification = bridge.verifyBeforeExecution(frame, plan)
            val executablePlan = when (verification.status) {
                PreActionStatus.EXECUTE,
                PreActionStatus.RELOCATED,
                PreActionStatus.PREEMPT_HU,
                -> verification.plan
                PreActionStatus.STALE -> {
                    Log.i(
                        TAG,
                        "stale_plan_rejected action=${plan.action} reason=${verification.reason}",
                    )
                    consecutiveFailures = 0
                    RuntimeStatusCenter.update(
                        RuntimePhase.OBSERVING,
                        "牌面已变化",
                        "旧动作已取消，立即重新观察",
                        mode,
                    )
                    eventGate.forceNext()
                    schedule(token, 0)
                    return
                }
                PreActionStatus.UNCERTAIN -> {
                    Log.w(
                        TAG,
                        "pre_action_uncertain action=${plan.action} reason=${verification.reason}",
                    )
                    continueObserving(
                        token,
                        title = "目标牌暂时看不清",
                        reason = "未点击，正在重新读取当前画面：${verification.reason}",
                        delayMs = 120,
                    )
                    return
                }
            }
            if (executablePlan == null) {
                Log.w(TAG, "pre_action_plan_missing status=${verification.status}")
                continueObserving(
                    token,
                    title = "动作已取消",
                    reason = "当前帧没有可安全执行的动作",
                    delayMs = 120,
                )
                return
            }
            val verifyElapsed = SystemClock.elapsedRealtime() - verifyStartedAt
            when (executionGate.decide(executablePlan, SystemClock.elapsedRealtime())) {
                ConfirmationDecision.WAIT -> {
                    RuntimeStatusCenter.update(
                        RuntimePhase.VERIFYING,
                        "等待游戏响应",
                        "相同动作刚刚执行，暂不重复点击",
                        mode,
                    )
                    schedule(token, 120)
                    return
                }
                ConfirmationDecision.HOLD -> {
                    continueObserving(
                        token,
                        title = "等待游戏画面变化",
                        reason = "同一动作尚未确认，已停止重复点击",
                        delayMs = UNCHANGED_FRAME_DELAY_MS,
                    )
                    return
                }
                ConfirmationDecision.EXECUTE -> Unit
            }
            val arm = bridge.prepareExecution(executablePlan, mode)
            if (!arm.armed) {
                if (arm.reason == "duplicate_execution_same_context") {
                    Log.i(TAG, "duplicate_execution_rejected action=${executablePlan.action}")
                    consecutiveFailures = 0
                    RuntimeStatusCenter.update(
                        RuntimePhase.OBSERVING,
                        "等待游戏响应",
                        "上一动作的画面尚未变化",
                        mode,
                        severity = RuntimeSeverity.WARNING,
                    )
                    schedule(token, 120)
                } else {
                    continueObserving(
                        token,
                        title = "动作暂缓",
                        reason = "动作执行前账本校验失败：${arm.reason}",
                        resetTransientGuard = true,
                    )
                }
                return
            }
            RuntimeStatusCenter.update(
                RuntimePhase.EXECUTING,
                "正在执行${humanAction(executablePlan.action)}",
                if (verification.status == PreActionStatus.RELOCATED) "目标牌换位已重新定位" else "最新画面核对完成",
                mode,
            )
            if (executablePlan.executionMode == "select_then_discard_button") {
                executeSelectThenDiscard(
                    token,
                    executablePlan,
                    decisionRequestedAt,
                    verifyElapsed,
                )
                return
            }
            val gestureStartedAt = SystemClock.elapsedRealtime()
            val executed = gestures.execute(executablePlan) { gate.isCurrent(token) }
            if (!executed) {
                val aborted = runCatching { bridge.abortExecution(executablePlan, mode) }.getOrDefault(false)
                if (!aborted) {
                    Log.e(TAG, "execution_guard_abort_failed action=${executablePlan.action}")
                    continueObserving(
                        token,
                        title = "动作已取消",
                        reason = "动作账本撤销失败",
                        resetTransientGuard = true,
                    )
                    return
                }
            }
            if (!executed && gate.isCurrent(token)) {
                Log.w(TAG, "gesture_failed action=${executablePlan.action} signature=${executablePlan.signature.hashCode()}")
                handleFailure(token, "系统未接受无障碍手势", restartDecisionWorker = false)
                return
            }
            if (!executed) return
            completeExecutedGesture(
                token,
                executablePlan,
                decisionRequestedAt,
                verifyElapsed,
                gestureStartedAt,
            )
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
        } catch (error: Throwable) {
            Log.e(TAG, "plan_verification_failed token=$token", error)
            handleFailure(token, "动作执行前核验异常", restartDecisionWorker = true)
        } finally {
            bitmap.recycle()
        }
    }

    private fun executeSelectThenDiscard(
        token: Long,
        plan: DecisionPlan,
        decisionRequestedAt: Long,
        verifyElapsed: Long,
    ) {
        val selectionPlan = plan.copy(
            executionMode = "tap_sequence",
            steps = listOf(plan.steps.first()),
        )
        val gestureStartedAt = SystemClock.elapsedRealtime()
        val selected = gestures.execute(selectionPlan) { gate.isCurrent(token) }
        if (!selected) {
            abortArmedExecution(token, plan, "系统未接受选牌手势")
            return
        }
        RuntimeStatusCenter.update(
            RuntimePhase.VERIFYING,
            "已选中目标牌",
            "正在核对牌值和选中状态",
            mode,
        )
        scheduled = worker.schedule(
            {
                verifyDiscardSelection(
                    token,
                    plan,
                    decisionRequestedAt,
                    verifyElapsed,
                    gestureStartedAt,
                    attempt = 1,
                )
            },
            80,
            TimeUnit.MILLISECONDS,
        )
    }

    private fun verifyDiscardSelection(
        token: Long,
        plan: DecisionPlan,
        decisionRequestedAt: Long,
        verifyElapsed: Long,
        gestureStartedAt: Long,
        attempt: Int,
    ) {
        if (!gate.isCurrent(token)) return
        if (
            deferScreenshot(token) {
                verifyDiscardSelection(
                    token,
                    plan,
                    decisionRequestedAt,
                    verifyElapsed,
                    gestureStartedAt,
                    attempt,
                )
            }
        ) return
        service.takeScreenshot(
            Display.DEFAULT_DISPLAY,
            worker,
            object : AccessibilityService.TakeScreenshotCallback {
                override fun onSuccess(result: AccessibilityService.ScreenshotResult) {
                    val buffer = result.hardwareBuffer
                    val bitmap = try {
                        Bitmap.wrapHardwareBuffer(buffer, result.colorSpace)
                            ?.copy(Bitmap.Config.ARGB_8888, false)
                    } finally {
                        buffer.close()
                    }
                    if (bitmap == null) {
                        retryDiscardSelectionVerification(
                            token,
                            plan,
                            decisionRequestedAt,
                            verifyElapsed,
                            gestureStartedAt,
                            attempt,
                            "选牌核验截图不可用",
                        )
                        return
                    }
                    try {
                        val frame = LosslessFrameFactory.fromBitmap(bitmap)
                        val verification = bridge.verifyDiscardSelection(frame, plan)
                        when (verification.status) {
                            DiscardSelectionStatus.SELECTED -> {
                                val confirmPlan = plan.copy(
                                    executionMode = "tap_sequence",
                                    steps = listOf(plan.steps.last()),
                                )
                                val confirmed = gestures.execute(confirmPlan) { gate.isCurrent(token) }
                                if (!confirmed) {
                                    abortArmedExecution(token, plan, "系统未接受出牌确认手势")
                                    return
                                }
                                completeExecutedGesture(
                                    token,
                                    plan,
                                    decisionRequestedAt,
                                    verifyElapsed,
                                    gestureStartedAt,
                                )
                            }
                            DiscardSelectionStatus.PENDING -> {
                                retryDiscardSelectionVerification(
                                    token,
                                    plan,
                                    decisionRequestedAt,
                                    verifyElapsed,
                                    gestureStartedAt,
                                    attempt,
                                    verification.reason,
                                )
                            }
                            DiscardSelectionStatus.PREEMPT_HU -> {
                                abortArmedExecution(token, plan, "胡按钮出现，取消当前出牌")
                            }
                            DiscardSelectionStatus.STALE,
                            DiscardSelectionStatus.UNCERTAIN,
                            -> abortArmedExecution(
                                token,
                                plan,
                                "选牌核验未通过：${verification.reason}",
                            )
                        }
                    } catch (error: Throwable) {
                        Log.e(TAG, "discard_selection_verification_failed", error)
                        retryDiscardSelectionVerification(
                            token,
                            plan,
                            decisionRequestedAt,
                            verifyElapsed,
                            gestureStartedAt,
                            attempt,
                            "选牌核验异常",
                        )
                    } finally {
                        bitmap.recycle()
                    }
                }

                override fun onFailure(errorCode: Int) {
                    retryDiscardSelectionVerification(
                        token,
                        plan,
                        decisionRequestedAt,
                        verifyElapsed,
                        gestureStartedAt,
                        attempt,
                        "选牌核验截图失败：$errorCode",
                    )
                }
            },
        )
    }

    private fun retryDiscardSelectionVerification(
        token: Long,
        plan: DecisionPlan,
        decisionRequestedAt: Long,
        verifyElapsed: Long,
        gestureStartedAt: Long,
        completedAttempt: Int,
        reason: String,
    ) {
        if (!gate.isCurrent(token)) return
        if (completedAttempt >= 3) {
            abortArmedExecution(token, plan, "选牌结果无法确认：$reason")
            return
        }
        RuntimeStatusCenter.update(
            RuntimePhase.VERIFYING,
            "正在确认选中的牌",
            "等待选中状态稳定（${completedAttempt + 1}/3）",
            mode,
            severity = RuntimeSeverity.WARNING,
        )
        scheduled = worker.schedule(
            {
                verifyDiscardSelection(
                    token,
                    plan,
                    decisionRequestedAt,
                    verifyElapsed,
                    gestureStartedAt,
                    completedAttempt + 1,
                )
            },
            SCREENSHOT_THROTTLE_BACKOFF_MS,
            TimeUnit.MILLISECONDS,
        )
    }

    private fun abortArmedExecution(token: Long, plan: DecisionPlan, reason: String) {
        if (!gate.isCurrent(token)) return
        val aborted = runCatching { bridge.abortExecution(plan, mode) }.getOrDefault(false)
        if (!aborted) {
            continueObserving(
                token,
                title = "动作已取消",
                reason = "动作账本撤销失败",
                resetTransientGuard = true,
            )
            return
        }
        Log.w(TAG, "armed_execution_aborted action=${plan.action} reason=$reason")
        RuntimeStatusCenter.update(
            RuntimePhase.OBSERVING,
            "本次出牌未确认",
            reason,
            mode,
            severity = RuntimeSeverity.WARNING,
        )
        eventGate.forceNext()
        schedule(token, 0)
    }

    private fun completeExecutedGesture(
        token: Long,
        executablePlan: DecisionPlan,
        decisionRequestedAt: Long,
        verifyElapsed: Long,
        gestureStartedAt: Long,
    ) {
        executionGate.markExecuted(executablePlan, SystemClock.elapsedRealtime())
        Log.i(
            TAG,
            "gesture_completed action=${executablePlan.action} signature=${executablePlan.signature.hashCode()} " +
                "verify_ms=$verifyElapsed gesture_ms=${SystemClock.elapsedRealtime() - gestureStartedAt} " +
                "total_elapsed_ms=${SystemClock.elapsedRealtime() - decisionRequestedAt}",
        )
        RuntimeStatusCenter.update(
            RuntimePhase.VERIFYING,
            "已点击${humanAction(executablePlan.action)}",
            "正在确认游戏是否响应",
            mode,
        )
        if (!requiresVisualPostConfirmation(executablePlan.action)) {
            val commitStartedAt = SystemClock.elapsedRealtime()
            if (!bridge.commitExecuted(executablePlan, mode)) {
                Log.e(TAG, "execution_guard_commit_failed action=${executablePlan.action}")
                continueObserving(
                    token,
                    title = "整理动作已发出",
                    reason = "动作状态提交失败，重新读取当前布局",
                    resetTransientGuard = true,
                )
                return
            }
            Log.i(
                TAG,
                "action_confirmed action=${executablePlan.action} checks=0 " +
                    "commit_ms=${SystemClock.elapsedRealtime() - commitStartedAt} " +
                    "total_elapsed_ms=${SystemClock.elapsedRealtime() - decisionRequestedAt} " +
                    "confirmation=gesture_then_reobserve",
            )
            eventGate.forceNext()
            consecutiveFailures = 0
            schedule(token, 0)
            return
        }
        scheduled = worker.schedule(
            { confirmExecutedAction(token, executablePlan, decisionRequestedAt, 1, 0) },
            ACTION_CONFIRMATION_DELAY_MS,
            TimeUnit.MILLISECONDS,
        )
    }

    private fun confirmExecutedAction(
        token: Long,
        plan: DecisionPlan,
        decisionRequestedAt: Long,
        check: Int,
        consecutiveConfirmedSignals: Int,
    ) {
        if (!gate.isCurrent(token)) return
        if (
            deferScreenshot(token) {
                confirmExecutedAction(
                    token,
                    plan,
                    decisionRequestedAt,
                    check,
                    consecutiveConfirmedSignals,
                )
            }
        ) return
        service.takeScreenshot(
            Display.DEFAULT_DISPLAY,
            worker,
            object : AccessibilityService.TakeScreenshotCallback {
                override fun onSuccess(result: AccessibilityService.ScreenshotResult) {
                    val buffer = result.hardwareBuffer
                    val bitmap = try {
                        Bitmap.wrapHardwareBuffer(buffer, result.colorSpace)
                            ?.copy(Bitmap.Config.ARGB_8888, false)
                    } finally {
                        buffer.close()
                    }
                    if (bitmap == null) {
                        retryActionConfirmation(token, plan, decisionRequestedAt, check, 0)
                        return
                    }
                    var bitmapOwnershipTransferred = false
                    try {
                        val frame = LosslessFrameFactory.fromBitmap(bitmap)
                        val observation = bridge.observeActionResult(frame, plan)
                        val nextConfirmedSignals = if (
                            observation.status == ActionResultStatus.CONFIRMED_SIGNAL
                        ) {
                            consecutiveConfirmedSignals + 1
                        } else {
                            0
                        }
                        when (
                            postActionGate.decide(
                                observation.status,
                                check,
                                nextConfirmedSignals,
                                strongTransition = observation.strongTransition,
                            )
                        ) {
                            PostActionConfirmationDecision.CONFIRMED -> {
                                val commitStartedAt = SystemClock.elapsedRealtime()
                                if (!bridge.commitExecuted(plan, mode)) {
                                    Log.e(TAG, "execution_guard_commit_failed action=${plan.action}")
                                    continueObserving(
                                        token,
                                        title = "画面已响应",
                                        reason = "动作账本提交失败",
                                        resetTransientGuard = true,
                                    )
                                    return
                                }
                                Log.i(
                                    TAG,
                                    "action_confirmed action=${plan.action} checks=$check " +
                                        "commit_ms=${SystemClock.elapsedRealtime() - commitStartedAt} " +
                                        "total_elapsed_ms=${SystemClock.elapsedRealtime() - decisionRequestedAt}",
                                )
                                eventGate.forceNext()
                                consecutiveFailures = 0
                                RuntimeStatusCenter.update(
                                    RuntimePhase.OBSERVING,
                                    "已确认${humanAction(plan.action)}",
                                    "游戏画面已经响应",
                                    mode,
                                )
                                if (
                                    canReuseConfirmedTransition(
                                        plan.action,
                                        observation.strongTransition,
                                    )
                                ) {
                                    bitmapOwnershipTransferred = true
                                    processFrame(
                                        token,
                                        bitmap,
                                        SystemClock.elapsedRealtime(),
                                        transitionFrame = true,
                                    )
                                } else {
                                    schedule(token, 0)
                                }
                            }
                            PostActionConfirmationDecision.RETRY -> {
                                retryActionConfirmation(
                                    token,
                                    plan,
                                    decisionRequestedAt,
                                    check,
                                    nextConfirmedSignals,
                                )
                            }
                            PostActionConfirmationDecision.FAILED -> {
                                val aborted = runCatching { bridge.abortExecution(plan, mode) }
                                    .getOrDefault(false)
                                if (!aborted) {
                                    continueObserving(
                                        token,
                                        title = "点击结果未确认",
                                        reason = "动作账本撤销失败",
                                        resetTransientGuard = true,
                                    )
                                    return
                                }
                                Log.w(
                                    TAG,
                                    "action_not_confirmed action=${plan.action} checks=$check",
                                )
                                RuntimeStatusCenter.update(
                                    RuntimePhase.OBSERVING,
                                    "点击未生效",
                                    "已撤销本次动作，重新读取当前画面",
                                    mode,
                                    severity = RuntimeSeverity.WARNING,
                                )
                                eventGate.forceNext()
                                schedule(token, 0)
                            }
                        }
                    } catch (error: Throwable) {
                        Log.e(TAG, "action_confirmation_failed action=${plan.action}", error)
                        retryActionConfirmation(token, plan, decisionRequestedAt, check, 0)
                    } finally {
                        if (!bitmapOwnershipTransferred) bitmap.recycle()
                    }
                }

                override fun onFailure(errorCode: Int) {
                    Log.w(
                        TAG,
                        "action_confirmation_screenshot_failed action=${plan.action} code=$errorCode",
                    )
                    retryActionConfirmation(token, plan, decisionRequestedAt, check, 0)
                }
            },
        )
    }

    private fun retryActionConfirmation(
        token: Long,
        plan: DecisionPlan,
        decisionRequestedAt: Long,
        completedCheck: Int,
        consecutiveConfirmedSignals: Int,
    ) {
        if (!gate.isCurrent(token)) return
        if (
            postActionGate.decide(
                status = ActionResultStatus.PENDING,
                check = completedCheck,
                consecutiveConfirmedSignals = consecutiveConfirmedSignals,
            ) == PostActionConfirmationDecision.FAILED
        ) {
            val aborted = runCatching { bridge.abortExecution(plan, mode) }.getOrDefault(false)
            if (!aborted) {
                continueObserving(
                    token,
                    title = "点击结果未确认",
                    reason = "动作账本撤销失败",
                    resetTransientGuard = true,
                )
                return
            }
            eventGate.forceNext()
            schedule(token, 0)
            return
        }
        RuntimeStatusCenter.update(
            RuntimePhase.VERIFYING,
            "正在确认${humanAction(plan.action)}",
            "等待游戏画面变化（${completedCheck + 1}/4）",
            mode,
        )
        scheduled = worker.schedule(
            {
                confirmExecutedAction(
                    token,
                    plan,
                    decisionRequestedAt,
                    completedCheck + 1,
                    consecutiveConfirmedSignals,
                )
            },
            SCREENSHOT_THROTTLE_BACKOFF_MS,
            TimeUnit.MILLISECONDS,
        )
    }

    private fun handleFailure(
        token: Long,
        reason: String,
        restartDecisionWorker: Boolean,
    ) {
        if (!gate.isCurrent(token)) return
        consecutiveFailures += 1
        RuntimeStatusCenter.update(
            RuntimePhase.OBSERVING,
            "画面读取失败",
            "正在自动重试（$consecutiveFailures/3）",
            mode,
            severity = RuntimeSeverity.WARNING,
        )
        if (consecutiveFailures >= 3) {
            Log.w(
                TAG,
                "runtime_recovery_cycle token=$token failures=$consecutiveFailures",
            )
            RuntimeStatusCenter.update(
                RuntimePhase.OBSERVING,
                "字牌AI正在恢复",
                reason,
                mode,
                severity = RuntimeSeverity.WARNING,
            )
            if (restartDecisionWorker) {
                bridge.restartDecisionWorker("consecutive_frame_failures:$reason")
                StrategyProcessPool.cancelAll()
            }
            consecutiveFailures = 0
            eventGate.forceNext()
            schedule(token, 1_000)
            return
        }
        schedule(token, 500L * consecutiveFailures)
    }

    private fun continueObserving(
        token: Long,
        title: String,
        reason: String,
        delayMs: Long = 1_000,
        resetTransientGuard: Boolean = false,
    ) {
        if (!gate.isCurrent(token)) return
        Log.w(TAG, "runtime_holding token=$token reason=$reason")
        if (resetTransientGuard) {
            runCatching { bridge.resetTransientGuard(mode) }
                .onFailure { Log.e(TAG, "transient_guard_recovery_failed", it) }
        }
        RuntimeStatusCenter.update(
            RuntimePhase.OBSERVING,
            title,
            "$reason；脚本仍在运行",
            mode,
            severity = RuntimeSeverity.WARNING,
        )
        schedule(token, delayMs)
    }

    private fun recoverDecisionWorker(token: Long, title: String, reason: String) {
        if (!gate.isCurrent(token)) return
        Log.e(TAG, "decision_worker_recovery token=$token reason=$reason")
        RuntimeStatusCenter.update(
            RuntimePhase.OBSERVING,
            title,
            reason,
            mode,
            severity = RuntimeSeverity.WARNING,
        )
        bridge.restartDecisionWorker(reason)
        eventGate.forceNext()
        consecutiveFailures = 0
        schedule(token, 250)
    }

    private fun resetTransientGuardAndCapture(token: Long, selectedMode: GameMode, attempt: Int) {
        if (!gate.isCurrent(token)) return
        val reset = runCatching { bridge.resetTransientGuard(selectedMode) }
        if (reset.getOrDefault(false)) {
            Log.i(TAG, "transient_guard_reset token=$token attempts=$attempt")
            capture(token)
            return
        }
        Log.e(TAG, "transient_guard_reset_failed token=$token attempts=$attempt", reset.exceptionOrNull())
        RuntimeStatusCenter.update(
            RuntimePhase.OBSERVING,
            "字牌AI正在恢复",
            "启动前动作状态清理失败，脚本仍在重试",
            selectedMode,
            severity = RuntimeSeverity.WARNING,
        )
        bridge.restartDecisionWorker("transient_guard_reset_failed")
        scheduled = worker.schedule(
            { resetTransientGuardAndCapture(token, selectedMode, attempt + 1) },
            minOf(3_000L, 500L * attempt),
            TimeUnit.MILLISECONDS,
        )
    }

    private fun interruptDecisionWorker(token: Long, reason: String) {
        if (!gate.isCurrent(token)) return
        Log.e(TAG, "decision_worker_watchdog token=$token reason=$reason")
        RuntimeStatusCenter.update(
            RuntimePhase.OBSERVING,
            "字牌AI正在恢复",
            reason,
            mode,
            severity = RuntimeSeverity.WARNING,
        )
        bridge.restartDecisionWorker(reason)
    }

    override fun close() {
        gate.stop()
        scheduled?.cancel(true)
        StrategyProcessPool.cancelAll()
        decisionWatchdog.close()
        bridge.close()
        worker.shutdownNow()
    }
}
