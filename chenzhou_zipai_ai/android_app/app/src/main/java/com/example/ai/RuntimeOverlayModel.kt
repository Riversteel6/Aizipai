package com.example.ai

import kotlin.math.roundToInt

internal data class RuntimeOverlayBounds(
    val x: Int,
    val y: Int,
    val width: Int,
    val height: Int,
)

internal data class RuntimeOverlayPresentation(
    val visible: Boolean,
    val warning: Boolean,
    val title: String,
    val detail: String,
    val nextRefreshDelayMs: Long? = null,
)

internal object RuntimeOverlayModel {
    private const val DESIGN_WIDTH = 2344.0
    private const val DESIGN_HEIGHT = 1080.0
    private const val DESIGN_X = 331.0
    private const val DESIGN_Y = 3.0
    private const val DESIGN_OVERLAY_WIDTH = 350.0
    private const val DESIGN_OVERLAY_HEIGHT = 86.0

    fun boundsFor(screenWidth: Int, screenHeight: Int): RuntimeOverlayBounds {
        val scaleX = screenWidth / DESIGN_WIDTH
        val scaleY = screenHeight / DESIGN_HEIGHT
        return RuntimeOverlayBounds(
            x = (DESIGN_X * scaleX).roundToInt(),
            y = (DESIGN_Y * scaleY).roundToInt(),
            width = (DESIGN_OVERLAY_WIDTH * scaleX).roundToInt().coerceAtLeast(1),
            height = (DESIGN_OVERLAY_HEIGHT * scaleY).roundToInt().coerceAtLeast(1),
        )
    }

    fun presentation(snapshot: RuntimeSnapshot, nowMs: Long): RuntimeOverlayPresentation {
        if (!snapshot.running) {
            return RuntimeOverlayPresentation(false, false, snapshot.title, snapshot.detail)
        }
        val warningAfterMs = warningAfterMs(snapshot.phase)
        val ageMs = (nowMs - snapshot.updatedAtMs).coerceAtLeast(0L)
        val slow = warningAfterMs != null && ageMs >= warningAfterMs
        val explicitWarning = snapshot.severity == RuntimeSeverity.WARNING
        return RuntimeOverlayPresentation(
            visible = true,
            warning = explicitWarning || slow,
            title = snapshot.title,
            detail = if (slow && !explicitWarning) "耗时偏长，仍停在此步骤" else snapshot.detail,
            nextRefreshDelayMs = if (!explicitWarning && warningAfterMs != null && !slow) {
                warningAfterMs - ageMs
            } else {
                null
            },
        )
    }

    private fun warningAfterMs(phase: RuntimePhase): Long? = when (phase) {
        RuntimePhase.STARTING -> 6_000L
        RuntimePhase.THINKING -> 4_000L
        RuntimePhase.VERIFYING -> 3_000L
        RuntimePhase.EXECUTING -> 2_500L
        else -> null
    }
}
