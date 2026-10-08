package com.example.ai

import org.junit.Assert.assertEquals
import org.junit.Test

class ScreenshotCadenceTest {
    @Test
    fun firstRequestRunsImmediately() {
        val cadence = ScreenshotCadence(minIntervalMs = 350)

        assertEquals(0L, cadence.delayBeforeRequest(1_000))
    }

    @Test
    fun immediateVerificationWaitsOnlyForRemainingInterval() {
        val cadence = ScreenshotCadence(minIntervalMs = 350)
        cadence.markRequested(1_000)

        assertEquals(110L, cadence.delayBeforeRequest(1_240))
        assertEquals(0L, cadence.delayBeforeRequest(1_350))
    }

    @Test
    fun resetRemovesPreviousRuntimeTiming() {
        val cadence = ScreenshotCadence(minIntervalMs = 350)
        cadence.markRequested(1_000)

        cadence.reset()

        assertEquals(0L, cadence.delayBeforeRequest(1_001))
    }
}
