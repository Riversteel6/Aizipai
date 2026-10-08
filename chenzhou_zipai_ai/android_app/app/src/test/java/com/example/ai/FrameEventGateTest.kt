package com.example.ai

import org.junit.Assert.assertFalse
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FrameEventGateTest {
    @Test
    fun unchangedFramesSkipUntilFallbackDeadline() {
        val gate = FrameEventGate(fallbackIntervalMs = 2_500)
        val frame = IntArray(800) { 0x202020 }
        assertEquals(FrameGateDecision.FULL, gate.decide(frame, 100))
        assertEquals(FrameGateDecision.SKIP, gate.decide(frame, 2_599))
        assertEquals(FrameGateDecision.PROBE, gate.decide(frame, 2_600))
    }

    @Test
    fun materialRegionChangeTriggersImmediately() {
        val gate = FrameEventGate()
        val before = IntArray(800) { 0x202020 }
        val after = before.copyOf().apply {
            for (index in 0 until 24) this[index] = 0xf0f0f0
        }
        assertEquals(FrameGateDecision.FULL, gate.decide(before, 100))
        assertEquals(FrameGateDecision.FULL, gate.decide(after, 200))
    }

    @Test
    fun oneCardHandMovementTriggersImmediately() {
        val gate = FrameEventGate()
        val before = IntArray(800) { 0x202020 }
        val after = before.copyOf().apply {
            for (index in 200 until 212) this[index] = 0xf0f0f0
        }

        assertEquals(FrameGateDecision.FULL, gate.decide(before, 100))
        assertEquals(FrameGateDecision.FULL, gate.decide(after, 200))
    }

    @Test
    fun tinyNoiseDoesNotTriggerFullRecognition() {
        val gate = FrameEventGate()
        val before = IntArray(800) { 0x202020 }
        val after = before.copyOf().apply {
            for (index in 0 until 8) this[index] = 0xf0f0f0
        }
        assertEquals(FrameGateDecision.FULL, gate.decide(before, 100))
        assertEquals(FrameGateDecision.SKIP, gate.decide(after, 200))
    }

    @Test
    fun actionCompletionForcesNextFrame() {
        val gate = FrameEventGate()
        val frame = IntArray(800) { 0x202020 }
        assertEquals(FrameGateDecision.FULL, gate.decide(frame, 100))
        assertEquals(FrameGateDecision.SKIP, gate.decide(frame, 200))
        gate.forceNext()
        assertEquals(FrameGateDecision.FULL, gate.decide(frame, 201))
    }

    @Test
    fun suppressedFallbackWaitsForMaterialChange() {
        val gate = FrameEventGate(fallbackIntervalMs = 2_500)
        val before = IntArray(800) { 0x202020 }
        val changed = before.copyOf().apply {
            for (index in 200 until 230) this[index] = 0xf0f0f0
        }

        assertEquals(FrameGateDecision.FULL, gate.decide(before, 100))
        gate.suppressFallbackUntilChange()
        assertEquals(FrameGateDecision.SKIP, gate.decide(before, 20_000))
        assertEquals(FrameGateDecision.FULL, gate.decide(changed, 20_100))
        assertEquals(FrameGateDecision.PROBE, gate.decide(changed, 22_600))
    }
}
