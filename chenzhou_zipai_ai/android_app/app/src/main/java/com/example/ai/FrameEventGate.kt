package com.example.ai

import android.graphics.Bitmap

enum class FrameGateDecision {
    FULL,
    PROBE,
    SKIP,
}

class FrameEventGate(
    private val fallbackIntervalMs: Long = 2_500L,
    private val regionThresholds: DoubleArray = doubleArrayOf(0.045, 0.045, 0.035, 0.08),
) {
    private var baseline: IntArray? = null
    private var lastFullFrameAtMs = 0L
    private var forceNextFrame = true
    private var suppressFallbackProbe = false

    fun decide(signature: IntArray, nowMs: Long): FrameGateDecision {
        val previous = baseline
        val forced = forceNextFrame
        val due = lastFullFrameAtMs == 0L || nowMs - lastFullFrameAtMs >= fallbackIntervalMs
        val changed = previous == null || materiallyChanged(previous, signature, regionThresholds)
        return if (forced || changed) {
            baseline = signature.copyOf()
            lastFullFrameAtMs = nowMs
            forceNextFrame = false
            suppressFallbackProbe = false
            FrameGateDecision.FULL
        } else if (due && !suppressFallbackProbe) {
            baseline = signature.copyOf()
            lastFullFrameAtMs = nowMs
            FrameGateDecision.PROBE
        } else {
            FrameGateDecision.SKIP
        }
    }

    fun shouldProcess(signature: IntArray, nowMs: Long): Boolean =
        decide(signature, nowMs) != FrameGateDecision.SKIP

    fun forceNext() {
        forceNextFrame = true
    }

    fun suppressFallbackUntilChange() {
        suppressFallbackProbe = true
    }

    fun reset() {
        baseline = null
        lastFullFrameAtMs = 0L
        forceNextFrame = true
        suppressFallbackProbe = false
    }

    companion object {
        private const val REGION_SAMPLE_COUNT = 200

        internal fun materiallyChanged(
            before: IntArray,
            after: IntArray,
            thresholds: DoubleArray,
        ): Boolean {
            if (before.size != after.size || before.isEmpty()) return true
            if (before.size != thresholds.size * REGION_SAMPLE_COUNT) return true
            return thresholds.indices.any { region ->
                val start = region * REGION_SAMPLE_COUNT
                changedRatio(
                    before.copyOfRange(start, start + REGION_SAMPLE_COUNT),
                    after.copyOfRange(start, start + REGION_SAMPLE_COUNT),
                ) >= thresholds[region]
            }
        }

        internal fun changedRatio(before: IntArray, after: IntArray): Double {
            if (before.size != after.size || before.isEmpty()) return 1.0
            var changed = 0
            for (index in before.indices) {
                val left = before[index]
                val right = after[index]
                val red = kotlin.math.abs((left shr 16 and 0xff) - (right shr 16 and 0xff))
                val green = kotlin.math.abs((left shr 8 and 0xff) - (right shr 8 and 0xff))
                val blue = kotlin.math.abs((left and 0xff) - (right and 0xff))
                if (maxOf(red, green, blue) >= 32) changed += 1
            }
            return changed.toDouble() / before.size
        }
    }
}

object FrameFingerprint {
    private data class RelativeRegion(
        val left: Double,
        val top: Double,
        val right: Double,
        val bottom: Double,
    )

    private val regions = listOf(
        RelativeRegion(0.70, 0.24, 0.995, 0.72), // HU/吃/碰/过和出牌按钮
        RelativeRegion(0.12, 0.58, 0.96, 0.995), // 自己手牌和抓牌
        RelativeRegion(0.18, 0.01, 0.995, 0.36), // 比牌/吃牌候选
        RelativeRegion(0.32, 0.24, 0.68, 0.76), // 准备、结算和中间来牌
    )

    fun fromBitmap(bitmap: Bitmap): IntArray {
        val samples = IntArray(regions.size * 20 * 10)
        var output = 0
        for (region in regions) {
            val left = (bitmap.width * region.left).toInt().coerceIn(0, bitmap.width - 1)
            val top = (bitmap.height * region.top).toInt().coerceIn(0, bitmap.height - 1)
            val right = (bitmap.width * region.right).toInt().coerceIn(left + 1, bitmap.width)
            val bottom = (bitmap.height * region.bottom).toInt().coerceIn(top + 1, bitmap.height)
            for (row in 0 until 10) {
                val y = (top + (row + 0.5) * (bottom - top) / 10.0).toInt()
                    .coerceIn(top, bottom - 1)
                for (column in 0 until 20) {
                    val x = (left + (column + 0.5) * (right - left) / 20.0).toInt()
                        .coerceIn(left, right - 1)
                    samples[output++] = bitmap.getPixel(x, y) and 0x00ffffff
                }
            }
        }
        return samples
    }
}
