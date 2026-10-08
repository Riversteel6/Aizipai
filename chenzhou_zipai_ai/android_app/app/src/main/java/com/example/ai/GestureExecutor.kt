package com.example.ai

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.graphics.Path
import android.os.Handler
import android.os.Looper
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class GestureExecutor(private val service: AccessibilityService) {
    fun execute(plan: DecisionPlan, stillCurrent: () -> Boolean): Boolean {
        for (step in plan.steps) {
            if (!stillCurrent()) return false
            if (step.delayMs > 0 && !sleepWhileCurrent(step.delayMs, stillCurrent)) return false
            val path = Path()
            val duration: Long
            if (plan.executionMode == "drag_sequence") {
                path.moveTo(step.fromX!!.toFloat(), step.fromY!!.toFloat())
                path.lineTo(step.x.toFloat(), step.y.toFloat())
                duration = step.durationMs
            } else {
                path.moveTo(step.x.toFloat(), step.y.toFloat())
                duration = 60
            }
            val gesture = GestureDescription.Builder()
                .addStroke(GestureDescription.StrokeDescription(path, 0, duration))
                .build()
            val latch = CountDownLatch(1)
            var completed = false
            val accepted = service.dispatchGesture(
                gesture,
                object : AccessibilityService.GestureResultCallback() {
                    override fun onCompleted(gestureDescription: GestureDescription?) {
                        completed = true
                        latch.countDown()
                    }

                    override fun onCancelled(gestureDescription: GestureDescription?) {
                        latch.countDown()
                    }
                },
                Handler(Looper.getMainLooper()),
            )
            if (!accepted || !latch.await(duration + 1_500, TimeUnit.MILLISECONDS) || !completed) return false
        }
        return true
    }

    private fun sleepWhileCurrent(milliseconds: Long, stillCurrent: () -> Boolean): Boolean {
        val deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(milliseconds)
        while (System.nanoTime() < deadline) {
            if (!stillCurrent()) return false
            Thread.sleep(minOf(25, milliseconds))
        }
        return stillCurrent()
    }
}
