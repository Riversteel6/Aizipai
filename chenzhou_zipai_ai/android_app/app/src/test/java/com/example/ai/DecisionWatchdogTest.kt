package com.example.ai

import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DecisionWatchdogTest {
    @Test
    fun completedDecisionDoesNotTriggerRecovery() {
        val fired = CountDownLatch(1)
        val watchdog = DecisionWatchdog(40) { fired.countDown() }
        try {
            val ticket = watchdog.arm(7)
            watchdog.disarm(ticket)
            assertFalse(fired.await(100, TimeUnit.MILLISECONDS))
        } finally {
            watchdog.close()
        }
    }

    @Test
    fun stalledDecisionTriggersRecoveryWithRunToken() {
        val fired = CountDownLatch(1)
        val observedToken = AtomicLong(-1)
        val watchdog = DecisionWatchdog(30) { token ->
            observedToken.set(token)
            fired.countDown()
        }
        try {
            watchdog.arm(19)
            assertTrue(fired.await(500, TimeUnit.MILLISECONDS))
            assertEquals(19, observedToken.get())
        } finally {
            watchdog.close()
        }
    }
}
