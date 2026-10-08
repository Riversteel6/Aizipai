package com.example.ai

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Test

class SettlementEpochTrackerTest {
    @Test
    fun repeatedFramesInOneSettlementKeepOneEpoch() {
        val tracker = SettlementEpochTracker()
        tracker.resetSession("session-a")

        tracker.observe("settlement_ready")
        val first = tracker.decorate(settlementPlan())
        tracker.observe("settlement_ready")
        val repeated = tracker.decorate(settlementPlan())

        assertEquals(epoch(first), epoch(repeated))
    }

    @Test
    fun leavingAndReenteringSettlementCreatesANewEpoch() {
        val tracker = SettlementEpochTracker()
        tracker.resetSession("session-a")

        tracker.observe("settlement_ready")
        val first = tracker.decorate(settlementPlan())
        tracker.observe("play")
        tracker.observe("settlement_ready")
        val nextRound = tracker.decorate(settlementPlan())

        assertNotEquals(epoch(first), epoch(nextRound))
    }

    @Test
    fun restartingRuntimeCannotReuseAnOldSettlementEpoch() {
        val tracker = SettlementEpochTracker()
        tracker.resetSession("session-a")
        tracker.observe("settlement_ready")
        val beforeRestart = tracker.decorate(settlementPlan())

        tracker.resetSession("session-b")
        tracker.observe("settlement_ready")
        val afterRestart = tracker.decorate(settlementPlan())

        assertNotEquals(epoch(beforeRestart), epoch(afterRestart))
    }

    @Test
    fun nonSettlementPlansRemainByteForByteUnchanged() {
        val tracker = SettlementEpochTracker()
        tracker.resetSession("session-a")
        tracker.observe("play")
        val pass = DecisionPlan("pass", "tap_sequence", emptyList(), "{\"context\":{\"flow_state\":\"play\"}}")

        assertEquals(pass, tracker.decorate(pass))
    }

    private fun settlementPlan() = DecisionPlan(
        action = "settlement_ready",
        executionMode = "tap_sequence",
        steps = emptyList(),
        signature = "{\"context\":{\"flow_state\":\"settlement_ready\"}}",
    )

    private fun epoch(plan: DecisionPlan): String = JSONObject(plan.signature)
        .getJSONObject("context")
        .getString("settlement_epoch")
}
