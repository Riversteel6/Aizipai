package com.example.ai

import org.junit.Assert.assertEquals
import org.junit.Test

class ActionConfirmationGateTest {
    private fun plan(signature: String = "same", steps: Int = 1) = DecisionPlan(
        action = "pass",
        executionMode = "tap_sequence",
        steps = List(steps) { GestureStep("button:pass", 10, 10, null, null, 0, 60) },
        signature = signature,
    )

    @Test
    fun aSingleTapGetsOneBoundedRetry() {
        val gate = ActionConfirmationGate()
        val plan = plan()
        assertEquals(ConfirmationDecision.EXECUTE, gate.decide(plan, 0))
        gate.markExecuted(plan, 0)
        assertEquals(ConfirmationDecision.WAIT, gate.decide(plan, 899))
        assertEquals(ConfirmationDecision.EXECUTE, gate.decide(plan, 900))
        gate.markExecuted(plan, 900)
        assertEquals(ConfirmationDecision.HOLD, gate.decide(plan, 1_800))
    }

    @Test
    fun multiStepPlansNeverBlindlyReplay() {
        val gate = ActionConfirmationGate()
        val plan = plan(steps = 2)
        gate.markExecuted(plan, 0)

        assertEquals(ConfirmationDecision.HOLD, gate.decide(plan, 900))
    }

    @Test
    fun oneMissedFrameDoesNotForgetExecution() {
        val gate = ActionConfirmationGate()
        val plan = plan()
        gate.markExecuted(plan, 0)
        gate.onNoPlan()
        assertEquals(ConfirmationDecision.WAIT, gate.decide(plan, 500))
        gate.onNoPlan()
        gate.onNoPlan()
        assertEquals(ConfirmationDecision.EXECUTE, gate.decide(plan, 500))
    }
}

class PostActionConfirmationGateTest {
    @Test
    fun recoveryDragReobservesLayoutInsteadOfWaitingForGameplayConfirmation() {
        assertEquals(false, requiresVisualPostConfirmation("compact_hand"))
        assertEquals(true, requiresVisualPostConfirmation("discard"))
        assertEquals(true, requiresVisualPostConfirmation("settlement_ready"))
    }

    @Test
    fun requiresTwoConsecutiveActionSpecificSignals() {
        val gate = PostActionConfirmationGate(maximumChecks = 4)

        assertEquals(
            PostActionConfirmationDecision.RETRY,
            gate.decide(ActionResultStatus.CONFIRMED_SIGNAL, check = 1, consecutiveConfirmedSignals = 1),
        )
        assertEquals(
            PostActionConfirmationDecision.CONFIRMED,
            gate.decide(ActionResultStatus.CONFIRMED_SIGNAL, check = 2, consecutiveConfirmedSignals = 2),
        )
    }

    @Test
    fun exactNextSurfaceConfirmsOnItsFirstStrongSignal() {
        val gate = PostActionConfirmationGate(maximumChecks = 4)

        assertEquals(
            PostActionConfirmationDecision.CONFIRMED,
            gate.decide(
                ActionResultStatus.CONFIRMED_SIGNAL,
                check = 1,
                consecutiveConfirmedSignals = 1,
                strongTransition = true,
            ),
        )
    }

    @Test
    fun onlyExactChiTransitionsReuseTheirFreshFrame() {
        assertEquals(true, canReuseConfirmedTransition("expand_chi_options", true))
        assertEquals(true, canReuseConfirmedTransition("chi", true))
        assertEquals(true, canReuseConfirmedTransition("chi_option", true))
        assertEquals(false, canReuseConfirmedTransition("pass", true))
        assertEquals(false, canReuseConfirmedTransition("chi_option", false))
    }

    @Test
    fun onlyTaggedSurfacePlansCanExecuteOnTheTransitionFrame() {
        val fast = DecisionPlan(
            action = "chi_option",
            executionMode = "tap_sequence",
            steps = listOf(GestureStep("chi:七八九", 10, 10, null, null, 0, 60)),
            signature = "fast",
            surfaceFastPath = true,
        )
        assertEquals(true, canExecuteSurfacePlanOnCurrentFrame(fast, transitionFrame = true))
        assertEquals(false, canExecuteSurfacePlanOnCurrentFrame(fast, transitionFrame = false))
        assertEquals(
            false,
            canExecuteSurfacePlanOnCurrentFrame(
                fast.copy(action = "pass"),
                transitionFrame = true,
            ),
        )
    }

    @Test
    fun pendingOrUncertainNeverConfirmsAction() {
        val gate = PostActionConfirmationGate(maximumChecks = 4)

        assertEquals(
            PostActionConfirmationDecision.RETRY,
            gate.decide(ActionResultStatus.PENDING, check = 1, consecutiveConfirmedSignals = 0),
        )
        assertEquals(
            PostActionConfirmationDecision.FAILED,
            gate.decide(ActionResultStatus.UNCERTAIN, check = 4, consecutiveConfirmedSignals = 0),
        )
    }
}
