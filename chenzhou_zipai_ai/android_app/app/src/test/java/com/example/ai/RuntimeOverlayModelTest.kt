package com.example.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RuntimeOverlayModelTest {
    @Test
    fun placementMatchesTheUserMarkedAreaAtGameResolution() {
        assertEquals(
            RuntimeOverlayBounds(x = 331, y = 3, width = 350, height = 86),
            RuntimeOverlayModel.boundsFor(2344, 1080),
        )
        assertEquals(
            RuntimeOverlayBounds(x = 196, y = 2, width = 207, height = 51),
            RuntimeOverlayModel.boundsFor(1389, 645),
        )
    }

    @Test
    fun overlayIsVisibleOnlyWhileRuntimeIsRunning() {
        val stopped = RuntimeSnapshot(RuntimePhase.STOPPED, "已停止", "不会操作")
        val observing = RuntimeSnapshot(RuntimePhase.OBSERVING, "运行中", "等待牌局变化")

        assertFalse(RuntimeOverlayModel.presentation(stopped, stopped.updatedAtMs).visible)
        assertTrue(RuntimeOverlayModel.presentation(observing, observing.updatedAtMs).visible)
        assertFalse(RuntimeOverlayModel.presentation(observing, observing.updatedAtMs).warning)
    }

    @Test
    fun explicitFailureAndLongCriticalStepAreYellowWithoutStoppingRuntime() {
        val warning = RuntimeSnapshot(
            RuntimePhase.OBSERVING,
            "点击未生效",
            "正在重新读取画面",
            severity = RuntimeSeverity.WARNING,
        )
        val thinking = RuntimeSnapshot(RuntimePhase.THINKING, "正在识别和思考", "当前画面已读取")

        assertTrue(RuntimeOverlayModel.presentation(warning, warning.updatedAtMs).warning)
        val slow = RuntimeOverlayModel.presentation(thinking, thinking.updatedAtMs + 4_001L)
        assertTrue(slow.warning)
        assertEquals("耗时偏长，仍停在此步骤", slow.detail)
        assertTrue(thinking.running)
    }

    @Test
    fun overlayTouchesTooFewCandidateFingerprintSamplesToTriggerAFrame() {
        val bounds = RuntimeOverlayModel.boundsFor(2344, 1080)
        var overlap = 0
        val left = (2344 * 0.18).toInt()
        val top = (1080 * 0.01).toInt()
        val right = (2344 * 0.995).toInt()
        val bottom = (1080 * 0.36).toInt()
        repeat(10) { row ->
            val y = (top + (row + 0.5) * (bottom - top) / 10.0).toInt()
            repeat(20) { column ->
                val x = (left + (column + 0.5) * (right - left) / 20.0).toInt()
                if (x in bounds.x until bounds.x + bounds.width && y in bounds.y until bounds.y + bounds.height) {
                    overlap += 1
                }
            }
        }

        assertEquals(6, overlap)
        assertTrue(overlap / 200.0 < 0.035)
    }
}
