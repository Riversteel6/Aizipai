package com.example.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertFalse
import org.junit.Test

class DecisionPlanTest {
    @Test
    fun parsesValidatedTapPlan() {
        val plan = DecisionPlan.parse(
            """{"action":"hu","ready":true,"execution_mode":"tap_sequence","validation":{"passed":true},"clicks":[{"target":"button:hu","x":1800,"y":420}],"_mobile_signature":"hu-1"}""",
            2344,
            1080,
        )

        assertNotNull(plan)
        assertEquals("hu", plan!!.action)
    }

    @Test
    fun parsesRelocatedPlanReturnedByPreActionVerifier() {
        val verification = PreActionVerification.parse(
            """{"status":"relocated","reason":"relocated","elapsed_ms":42,"plan":{"action":"discard","ready":true,"execution_mode":"tap_sequence","validation":{"passed":true},"clicks":[{"target":"hand:一","x":640,"y":920}],"_mobile_signature":"discard-1"}}""",
            2344,
            1080,
        )

        assertEquals(PreActionStatus.RELOCATED, verification.status)
        assertNotNull(verification.plan)
        assertEquals(640, verification.plan!!.steps.single().x)
        assertEquals("discard-1", verification.plan!!.signature)
    }

    @Test
    fun parsesOnlyWellFormedSelectThenDiscardPlan() {
        val valid = DecisionPlan.parse(
            """{"action":"discard","ready":true,"execution_mode":"select_then_discard_button","validation":{"passed":true},"clicks":[{"target":"hand:九","x":640,"y":920},{"target":"button:discard","x":1172,"y":500}],"_mobile_signature":"discard-2"}""",
            2344,
            1080,
        )
        val missingConfirmation = DecisionPlan.parse(
            """{"action":"discard","ready":true,"execution_mode":"select_then_discard_button","validation":{"passed":true},"clicks":[{"target":"hand:九","x":640,"y":920}],"_mobile_signature":"discard-3"}""",
            2344,
            1080,
        )

        assertNotNull(valid)
        assertEquals(2, valid!!.steps.size)
        assertNull(missingConfirmation)
    }

    @Test
    fun rejectsOutOfBoundsAndUnvalidatedPlans() {
        assertNull(
            DecisionPlan.parse(
                """{"action":"pass","ready":true,"execution_mode":"tap_sequence","validation":{"passed":true},"clicks":[{"target":"button:pass","x":9999,"y":420}],"_mobile_signature":"pass-1"}""",
                2344,
                1080,
            )
        )
        assertNull(
            DecisionPlan.parse(
                """{"action":"hu","ready":true,"validation":{"passed":false},"clicks":[{"target":"button:hu","x":1800,"y":420}],"_mobile_signature":"hu-2"}""",
                2344,
                1080,
            )
        )
    }

    @Test
    fun preservesNonExecutingDecisionReason() {
        val outcome = DecisionOutcome.parse(
            """{"action":"wait_response_options","ready":false,"reason":"等待吃牌候选展开","_mobile_status":"wait","_mobile_fatal":false,"_mobile_flow_state":"play"}""",
            2344,
            1080,
        )

        assertEquals(DecisionDisposition.WAIT, outcome.disposition)
        assertEquals("wait_response_options", outcome.action)
        assertEquals("等待吃牌候选展开", outcome.reason)
        assertEquals("play", outcome.flowState)
        assertFalse(outcome.fatal)
        assertNull(outcome.plan)
    }

    @Test
    fun preservesBlockedAndFatalOutcomes() {
        val blocked = DecisionOutcome.parse(
            """{"action":"safe_halt","ready":false,"reason":"当前状态不可信","validation":{"reason_code":"opening_compact_exhausted"},"_mobile_status":"blocked"}""",
            2344,
            1080,
        )
        val fatal = DecisionOutcome.parse(
            """{"action":"safe_halt","ready":false,"reason":"动作合同损坏","_mobile_status":"error","_mobile_fatal":true}""",
            2344,
            1080,
        )

        assertEquals(DecisionDisposition.BLOCKED, blocked.disposition)
        assertEquals("当前状态不可信", blocked.reason)
        assertEquals("opening_compact_exhausted", blocked.reasonCode)
        assertEquals(DecisionDisposition.ERROR, fatal.disposition)
        assertEquals("动作合同损坏", fatal.reason)
    }
}
