package com.example.ai

import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RunGenerationTest {
    @Test
    fun duplicateStartKeepsSingleGeneration() {
        val gate = RunGeneration()
        val first = gate.start()
        val duplicate = gate.start()

        assertTrue(gate.isCurrent(first))
        assertTrue(first == duplicate)
    }

    @Test
    fun stopInvalidatesLateWork() {
        val gate = RunGeneration()
        val old = gate.start()
        gate.stop()

        assertFalse(gate.isCurrent(old))
        val next = gate.start()
        assertNotEquals(old, next)
        assertTrue(gate.isCurrent(next))
    }
}
