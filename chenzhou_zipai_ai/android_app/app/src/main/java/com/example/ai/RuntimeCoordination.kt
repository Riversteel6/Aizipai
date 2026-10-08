package com.example.ai

import org.json.JSONObject

class SettlementEpochTracker {
    private var sessionId = "not-started"
    private var generation = 0L
    private var inSettlement = false

    fun resetSession(id: String) {
        sessionId = id
        generation = 0L
        inSettlement = false
    }

    fun observe(flowState: String) {
        if (flowState.isBlank()) return
        val settlement = flowState == "settlement_ready"
        if (settlement && !inSettlement) generation += 1
        inSettlement = settlement
    }

    fun decorate(plan: DecisionPlan): DecisionPlan {
        if (plan.action != "settlement_ready") return plan
        val root = runCatching { JSONObject(plan.signature) }.getOrNull() ?: return plan
        val context = root.optJSONObject("context") ?: JSONObject().also { root.put("context", it) }
        context.put("settlement_epoch", "$sessionId:$generation")
        return plan.copy(signature = root.toString())
    }
}

class ScreenshotCadence(private val minIntervalMs: Long) {
    private var lastRequestAt: Long? = null

    fun delayBeforeRequest(now: Long): Long {
        val previous = lastRequestAt ?: return 0L
        return (minIntervalMs - (now - previous)).coerceAtLeast(0L)
    }

    fun markRequested(now: Long) {
        lastRequestAt = now
    }

    fun reset() {
        lastRequestAt = null
    }
}
