package com.example.ai

import android.content.Context
import org.json.JSONObject

class PythonDecisionBridge(private val context: Context) : AutoCloseable {
    private val local = LocalPythonDecisionBridge(context)

    fun prewarm(): JSONObject = local.prewarm()

    fun resetTransientGuard(mode: GameMode): Boolean =
        local.resetTransientGuard(mode)

    fun decide(frame: LosslessFrame, mode: GameMode): DecisionOutcome = DecisionOutcome.parse(
        local.decideRaw(frame, mode),
        frame.width,
        frame.height,
    )

    fun probe(frame: LosslessFrame): ActionableProbeResult = local.probe(frame)

    fun decide(framePath: String, mode: GameMode, width: Int, height: Int): DecisionOutcome {
        return DecisionOutcome.parse(local.decideRaw(framePath, mode, width, height), width, height)
    }

    fun isFresh(frame: LosslessFrame, plan: DecisionPlan): Boolean =
        local.isFresh(frame, plan)

    fun verifyBeforeExecution(frame: LosslessFrame, plan: DecisionPlan): PreActionVerification =
        local.verifyBeforeExecution(frame, plan)

    fun observeActionResult(frame: LosslessFrame, plan: DecisionPlan): ActionResultObservation =
        local.observeActionResult(frame, plan)

    fun verifyDiscardSelection(
        frame: LosslessFrame,
        plan: DecisionPlan,
    ): DiscardSelectionVerification = local.verifyDiscardSelection(frame, plan)

    fun commitExecuted(plan: DecisionPlan, mode: GameMode): Boolean =
        local.commitExecuted(plan, mode)

    fun prepareExecution(plan: DecisionPlan, mode: GameMode): ExecutionArmResult =
        local.prepareExecution(plan, mode)

    fun abortExecution(plan: DecisionPlan, mode: GameMode): Boolean =
        local.abortExecution(plan, mode)

    fun restartDecisionWorker(reason: String) {
        android.util.Log.w("AizipaiRuntime", "local_decision_reset reason=$reason")
        StrategyProcessPool.cancelAll()
    }

    override fun close() {
        StrategyProcessPool.cancelAll()
    }
}
