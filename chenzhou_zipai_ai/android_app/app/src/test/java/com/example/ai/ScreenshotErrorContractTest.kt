package com.example.ai

import android.accessibilityservice.AccessibilityService
import org.junit.Assert.assertEquals
import org.junit.Test

class ScreenshotErrorContractTest {
    @Test
    fun intervalTooShortErrorCodeMatchesAndroidContract() {
        assertEquals(3, AccessibilityService.ERROR_TAKE_SCREENSHOT_INTERVAL_TIME_SHORT)
    }
}
