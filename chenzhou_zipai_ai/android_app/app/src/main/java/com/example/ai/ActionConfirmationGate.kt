package com.example.ai

enum class ConfirmationDecision {
    EXECUTE,
    WAIT,
    HOLD,
}

enum class PostActionConfirmationDecision {
    CONFIRMED,
    RETRY,
    FAILED,
}

internal fun requiresVisualPostConfirmation(action: String): Boolean =
    action.lowercase() != "compact_hand"

internal fun canReuseConfirmedTransition(action: String, strongTransition: Boolean): Boolean =
    strongTransition && action.lowercase() in setOf("expand_chi_options", "chi", "chi_option")

internal fun canExecuteSurfacePlanOnCurrentFrame(
    plan: DecisionPlan,
    transitionFrame: Boolean,
): Boolean =
    transitionFrame &&
        plan.surfaceFastPath &&
        plan.action.lowercase() in setOf("chi_option", "compare_option")

class PostActionConfirmationGate(private val maximumChecks: Int = 4) {
    init {
        require(maximumChecks >= 1)
    }

    fun decide(
        status: ActionResultStatus,
        check: Int,
        consecutiveConfirmedSignals: Int,
        strongTransition: Boolean = false,
    ): PostActionConfirmationDecision = when {
        status == ActionResultStatus.CONFIRMED_SIGNAL && strongTransition ->
            PostActionConfirmationDecision.CONFIRMED
        status == ActionResultStatus.CONFIRMED_SIGNAL && consecutiveConfirmedSignals >= 2 ->
            PostActionConfirmationDecision.CONFIRMED
        check < maximumChecks -> PostActionConfirmationDecision.RETRY
        else -> PostActionConfirmationDecision.FAILED
    }
}

class ActionConfirmationGate {
    private var lastSignature: String? = null
    private var lastExecutedAt = 0L
    private var attempts = 0
    private var emptyFrames = 0

    fun decide(plan: DecisionPlan, nowMs: Long): ConfirmationDecision {
        emptyFrames = 0
        if (lastSignature != plan.signature) return ConfirmationDecision.EXECUTE
        val waitMs = if (plan.action == "expand_chi_options") 1_500L else 900L
        if (nowMs - lastExecutedAt < waitMs) return ConfirmationDecision.WAIT
        val retryable = plan.executionMode == "tap_sequence" && plan.steps.size == 1
        return if (retryable && attempts < 2) ConfirmationDecision.EXECUTE else ConfirmationDecision.HOLD
    }

    fun markExecuted(plan: DecisionPlan, nowMs: Long) {
        if (lastSignature == plan.signature) {
            attempts += 1
        } else {
            lastSignature = plan.signature
            attempts = 1
        }
        lastExecutedAt = nowMs
    }

    fun onNoPlan() {
        emptyFrames += 1
        if (emptyFrames >= 2) reset()
    }

    fun reset() {
        lastSignature = null
        lastExecutedAt = 0L
        attempts = 0
        emptyFrames = 0
    }
}
