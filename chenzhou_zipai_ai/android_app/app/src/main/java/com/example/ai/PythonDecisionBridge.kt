package com.example.ai

import android.content.Context
import android.util.Log
import com.chaquo.python.PyException
import com.chaquo.python.Python
import org.json.JSONObject

data class ExecutionArmResult(val armed: Boolean, val reason: String)
data class ActionableProbeResult(
    val actionable: Boolean,
    val summary: String,
    val elapsedMs: Long,
    val flowState: String,
)

class VisionDecisionTimeoutException(cause: Throwable) : RuntimeException(cause)

class LocalPythonDecisionBridge(private val context: Context) {
    private companion object {
        const val TAG = "AizipaiRuntime"
    }
    init {
        StrategyProcessPool.initialize(context.applicationContext)
    }

    fun prewarm(): JSONObject {
        val module = Python.getInstance().getModule("mobile_bridge")
        val result = JSONObject(
            module.callAttr("prewarm_runtime_json", context.filesDir.absolutePath).toString(),
        )
        // Start and warm isolated workers in a bounded sequence. Starting all
        // Python processes at once makes Android spend far longer contending on
        // APK imports than it saves, and can trap every retry on the cold path.
        result.put(
            "strategy_workers",
            JSONObject(module.callAttr("worker_pool_diagnostics_json").toString()),
        )
        return result
    }

    fun resetTransientGuard(mode: GameMode): Boolean {
        val raw = Python.getInstance()
            .getModule("mobile_bridge")
            .callAttr(
                "reset_transient_guard_json",
                context.filesDir.absolutePath,
                mode.wildcardEnabled,
            )
            .toString()
        return JSONObject(raw).optBoolean("reset", false)
    }

    fun decide(framePath: String, mode: GameMode, width: Int, height: Int): DecisionPlan? {
        return parseDecision(decideRaw(framePath, mode, width, height), width, height)
    }

    fun decideRaw(framePath: String, mode: GameMode, width: Int, height: Int): String {
        val module = Python.getInstance().getModule("mobile_bridge")
        val priorityRaw = module.callAttr("priority_action_json", framePath).toString()
        if (DecisionPlan.parse(priorityRaw, width, height) != null) return priorityRaw
        return module
            .callAttr(
                "recommend_json",
                framePath,
                mode.wildcardEnabled,
                context.filesDir.absolutePath,
                width,
                height,
            )
            .toString()
    }

    fun decide(frame: LosslessFrame, mode: GameMode): DecisionPlan? {
        return parseDecision(decideRaw(frame, mode), frame.width, frame.height)
    }

    fun decideRaw(frame: LosslessFrame, mode: GameMode): String {
        return try {
            Python.getInstance()
                .getModule("mobile_bridge")
                .callAttr(
                    "decide_rgba_json",
                    frame.pixels,
                    frame.width,
                    frame.height,
                    frame.rowBytes,
                    mode.wildcardEnabled,
                    context.filesDir.absolutePath,
                )
                .toString()
        } catch (error: PyException) {
            if (error.message.orEmpty().contains("vision_pipeline_timeout")) {
                throw VisionDecisionTimeoutException(error)
            }
            throw error
        }
    }

    fun probe(frame: LosslessFrame): ActionableProbeResult {
        val raw = Python.getInstance()
            .getModule("mobile_bridge")
            .callAttr(
                "probe_actionable_rgba_json",
                frame.pixels,
                frame.width,
                frame.height,
                frame.rowBytes,
                context.filesDir.absolutePath,
            )
            .toString()
        val result = JSONObject(raw)
        val summary = when {
            (result.optJSONArray("button_names")?.length() ?: 0) > 0 -> "检测到操作按钮"
            result.optBoolean("discard_button_visible", false) -> "检测到出牌回合"
            result.optString("option_stage").isNotBlank() -> "检测到候选牌"
            result.optString("flow_state") == "settlement_ready" -> "检测到准备按钮"
            else -> "画面无待处理事件"
        }
        return ActionableProbeResult(
            actionable = result.optBoolean("actionable", false),
            summary = summary,
            elapsedMs = result.optLong("elapsed_ms", 0L),
            flowState = result.optString("flow_state"),
        )
    }

    private fun parseDecision(raw: String, width: Int, height: Int): DecisionPlan? {
        val plan = DecisionPlan.parse(raw, width, height)
        if (plan == null) {
            val payload = runCatching { JSONObject(raw) }.getOrNull()
            Log.i(
                TAG,
                "decision_not_ready action=${payload?.optString("action", "unknown")} " +
                    "reason=${payload?.optString("reason", "unknown")}",
            )
        }
        return plan
    }

    fun isFresh(framePath: String, plan: DecisionPlan): Boolean {
        val raw = Python.getInstance()
            .getModule("mobile_bridge")
            .callAttr("validate_fresh_plan_json", framePath, plan.signature)
            .toString()
        return JSONObject(raw).optBoolean("fresh", false)
    }

    fun isFresh(frame: LosslessFrame, plan: DecisionPlan): Boolean {
        val raw = Python.getInstance()
            .getModule("mobile_bridge")
            .callAttr(
                "validate_fresh_rgba_json",
                frame.pixels,
                frame.width,
                frame.height,
                frame.rowBytes,
                plan.signature,
                context.filesDir.absolutePath,
            )
            .toString()
        return JSONObject(raw).optBoolean("fresh", false)
    }

    fun verifyBeforeExecution(frame: LosslessFrame, plan: DecisionPlan): PreActionVerification {
        val raw = Python.getInstance()
            .getModule("mobile_bridge")
            .callAttr(
                "verify_before_execution_rgba_json",
                frame.pixels,
                frame.width,
                frame.height,
                frame.rowBytes,
                plan.signature,
                context.filesDir.absolutePath,
            )
            .toString()
        return PreActionVerification.parse(raw, frame.width, frame.height)
    }

    fun observeActionResult(frame: LosslessFrame, plan: DecisionPlan): ActionResultObservation {
        val raw = Python.getInstance()
            .getModule("mobile_bridge")
            .callAttr(
                "confirm_action_result_rgba_json",
                frame.pixels,
                frame.width,
                frame.height,
                frame.rowBytes,
                plan.signature,
                context.filesDir.absolutePath,
            )
            .toString()
        return ActionResultObservation.parse(raw)
    }

    fun verifyDiscardSelection(
        frame: LosslessFrame,
        plan: DecisionPlan,
    ): DiscardSelectionVerification {
        val raw = Python.getInstance()
            .getModule("mobile_bridge")
            .callAttr(
                "verify_discard_selection_rgba_json",
                frame.pixels,
                frame.width,
                frame.height,
                frame.rowBytes,
                plan.signature,
                context.filesDir.absolutePath,
            )
            .toString()
        return DiscardSelectionVerification.parse(raw)
    }

    fun commitExecuted(plan: DecisionPlan, mode: GameMode): Boolean {
        val raw = Python.getInstance()
            .getModule("mobile_bridge")
            .callAttr(
                "commit_executed_action_json",
                plan.signature,
                context.filesDir.absolutePath,
                mode.wildcardEnabled,
            )
            .toString()
        val result = JSONObject(raw)
        if (!result.isNull("round_cache_cleanup")) {
            Log.i(TAG, "round_cache_cleanup result=${result.getJSONObject("round_cache_cleanup")}")
        }
        return result.optBoolean("committed", false)
    }

    fun prepareExecution(plan: DecisionPlan, mode: GameMode): ExecutionArmResult {
        val result = JSONObject(
            Python.getInstance()
                .getModule("mobile_bridge")
                .callAttr(
                    "prepare_external_action_json",
                    plan.signature,
                    context.filesDir.absolutePath,
                    mode.wildcardEnabled,
                )
                .toString(),
        )
        return ExecutionArmResult(
            armed = result.optBoolean("armed", false),
            reason = result.optString("reason", "unknown"),
        )
    }

    fun abortExecution(plan: DecisionPlan, mode: GameMode): Boolean {
        val raw = Python.getInstance()
            .getModule("mobile_bridge")
            .callAttr(
                "abort_external_action_json",
                plan.signature,
                context.filesDir.absolutePath,
                mode.wildcardEnabled,
            )
            .toString()
        return JSONObject(raw).optBoolean("aborted", false)
    }
}
