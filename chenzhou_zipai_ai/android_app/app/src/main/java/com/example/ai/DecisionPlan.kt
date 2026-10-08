package com.example.ai

import org.json.JSONObject

data class GestureStep(
    val target: String,
    val x: Int,
    val y: Int,
    val fromX: Int?,
    val fromY: Int?,
    val delayMs: Long,
    val durationMs: Long,
)

data class DecisionPlan(
    val action: String,
    val executionMode: String,
    val steps: List<GestureStep>,
    val signature: String,
    val surfaceFastPath: Boolean = false,
) {
    companion object {
        fun parse(raw: String, width: Int, height: Int): DecisionPlan? =
            parseObject(JSONObject(raw), width, height)

        fun parseObject(root: JSONObject, width: Int, height: Int): DecisionPlan? {
            if (!root.optBoolean("ready", false)) return null
            if (!root.optJSONObject("validation")?.optBoolean("passed", false).orFalse()) return null
            val action = root.optString("action").lowercase()
            val mode = root.optString("execution_mode", "tap_sequence")
            if (
                action.isBlank() ||
                mode !in setOf("tap_sequence", "drag_sequence", "select_then_discard_button")
            ) return null
            val clicks = root.optJSONArray("clicks") ?: return null
            val steps = buildList {
                for (index in 0 until clicks.length()) {
                    val click = clicks.optJSONObject(index) ?: return null
                    val x = click.optInt("x", -1)
                    val y = click.optInt("y", -1)
                    if (x !in 0 until width || y !in 0 until height) return null
                    val fromX = if (click.has("from_x")) click.optInt("from_x", -1) else null
                    val fromY = if (click.has("from_y")) click.optInt("from_y", -1) else null
                    if (mode == "drag_sequence") {
                        if (fromX == null || fromY == null) return null
                        if (fromX !in 0 until width || fromY !in 0 until height) return null
                    }
                    add(
                        GestureStep(
                            target = click.optString("target"),
                            x = x,
                            y = y,
                            fromX = fromX,
                            fromY = fromY,
                            delayMs = click.optLong("delay_ms", 80).coerceIn(0, 5_000),
                            durationMs = click.optLong("duration_ms", 520).coerceIn(1, 5_000),
                        )
                    )
                }
            }
            if (
                mode == "select_then_discard_button" &&
                (
                    action != "discard" ||
                    steps.size != 2 ||
                    !steps[0].target.startsWith("hand:") ||
                    steps[1].target != "button:discard"
                )
            ) return null
            val signature = root.optString("_mobile_signature")
            return if (steps.isEmpty() || signature.isBlank()) null else DecisionPlan(
                action,
                mode,
                steps,
                signature,
                surfaceFastPath = root.optBoolean("_mobile_surface_fast_path", false),
            )
        }
    }
}

enum class DiscardSelectionStatus {
    SELECTED,
    PENDING,
    PREEMPT_HU,
    STALE,
    UNCERTAIN,
}

data class DiscardSelectionVerification(
    val status: DiscardSelectionStatus,
    val reason: String,
    val elapsedMs: Long,
) {
    companion object {
        fun parse(raw: String): DiscardSelectionVerification {
            val root = JSONObject(raw)
            val status = when (root.optString("status").lowercase()) {
                "selected" -> DiscardSelectionStatus.SELECTED
                "pending" -> DiscardSelectionStatus.PENDING
                "preempt_hu" -> DiscardSelectionStatus.PREEMPT_HU
                "stale" -> DiscardSelectionStatus.STALE
                else -> DiscardSelectionStatus.UNCERTAIN
            }
            return DiscardSelectionVerification(
                status = status,
                reason = root.optString("reason", "discard_selection_uncertain"),
                elapsedMs = root.optLong("elapsed_ms", 0L),
            )
        }
    }
}

enum class PreActionStatus {
    EXECUTE,
    RELOCATED,
    PREEMPT_HU,
    STALE,
    UNCERTAIN,
}

data class PreActionVerification(
    val status: PreActionStatus,
    val reason: String,
    val plan: DecisionPlan?,
    val elapsedMs: Long,
) {
    companion object {
        fun parse(raw: String, width: Int, height: Int): PreActionVerification {
            val root = JSONObject(raw)
            val status = when (root.optString("status").lowercase()) {
                "execute" -> PreActionStatus.EXECUTE
                "relocated" -> PreActionStatus.RELOCATED
                "preempt_hu" -> PreActionStatus.PREEMPT_HU
                "stale" -> PreActionStatus.STALE
                else -> PreActionStatus.UNCERTAIN
            }
            val plan = root.optJSONObject("plan")?.let {
                DecisionPlan.parseObject(it, width, height)
            }
            return PreActionVerification(
                status = status,
                reason = root.optString("reason", "pre_action_uncertain"),
                plan = plan,
                elapsedMs = root.optLong("elapsed_ms", 0L),
            )
        }
    }
}

enum class ActionResultStatus {
    CONFIRMED_SIGNAL,
    PENDING,
    UNCERTAIN,
}

data class ActionResultObservation(
    val status: ActionResultStatus,
    val reason: String,
    val elapsedMs: Long,
    val strongTransition: Boolean,
) {
    companion object {
        fun parse(raw: String): ActionResultObservation {
            val root = JSONObject(raw)
            val status = when (root.optString("status").lowercase()) {
                "confirmed_signal" -> ActionResultStatus.CONFIRMED_SIGNAL
                "pending" -> ActionResultStatus.PENDING
                else -> ActionResultStatus.UNCERTAIN
            }
            return ActionResultObservation(
                status = status,
                reason = root.optString("reason", "action_result_uncertain"),
                elapsedMs = root.optLong("elapsed_ms", 0L),
                strongTransition = root.optBoolean("strong_transition", false),
            )
        }
    }
}

enum class DecisionDisposition {
    EXECUTE,
    WAIT,
    BLOCKED,
    ERROR,
}

data class DecisionOutcome(
    val disposition: DecisionDisposition,
    val action: String,
    val reason: String,
    val fatal: Boolean,
    val plan: DecisionPlan?,
    val reasonCode: String = "",
    val flowState: String = "",
) {
    companion object {
        fun parse(raw: String, width: Int, height: Int): DecisionOutcome {
            val root = JSONObject(raw)
            val action = root.optString("action", "wait").lowercase()
            val reason = root.optString("reason", "等待牌局事件")
            val reasonCode = root.optJSONObject("validation")?.optString("reason_code").orEmpty()
            val flowState = root.optString("_mobile_flow_state")
            val fatal = root.optBoolean("_mobile_fatal", root.optBoolean("fatal", false))
            val plan = DecisionPlan.parse(raw, width, height)
            val explicit = root.optString("_mobile_status").lowercase()
            val disposition = when {
                plan != null -> DecisionDisposition.EXECUTE
                explicit == "error" || fatal -> DecisionDisposition.ERROR
                explicit == "blocked" || action == "safe_halt" -> DecisionDisposition.BLOCKED
                else -> DecisionDisposition.WAIT
            }
            return DecisionOutcome(disposition, action, reason, fatal, plan, reasonCode, flowState)
        }
    }
}

private fun Boolean?.orFalse(): Boolean = this == true
