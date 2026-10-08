package com.example.ai

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AccessibilityServiceStateTest {
    private val expectedPackage = "com.example.ai"
    private val expectedClass = "com.example.ai.AiAccessibilityService"

    @Test
    fun acceptsFullAndAbbreviatedComponentNames() {
        assertTrue(
            containsAccessibilityService(
                "com.example.ai/com.example.ai.AiAccessibilityService",
                expectedPackage,
                expectedClass,
            ),
        )
        assertTrue(
            containsAccessibilityService(
                "com.example.ai/.AiAccessibilityService",
                expectedPackage,
                expectedClass,
            ),
        )
    }

    @Test
    fun rejectsMissingMalformedAndDifferentServices() {
        assertFalse(containsAccessibilityService(null, expectedPackage, expectedClass))
        assertFalse(containsAccessibilityService("not-a-component", expectedPackage, expectedClass))
        assertFalse(containsAccessibilityService("other.app/.Service", expectedPackage, expectedClass))
    }
}
