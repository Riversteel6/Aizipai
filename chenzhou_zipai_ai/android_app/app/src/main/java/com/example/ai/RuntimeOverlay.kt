package com.example.ai

import android.accessibilityservice.AccessibilityService
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.drawable.GradientDrawable
import android.os.Handler
import android.os.Looper
import android.text.TextUtils
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.WindowManager
import android.widget.TextView

class RuntimeOverlay(private val service: AccessibilityService) : AutoCloseable {
    private companion object {
        const val TAG = "AizipaiOverlay"
        const val NORMAL_COLOR = 0xff54e57a.toInt()
        const val WARNING_COLOR = 0xffffd54f.toInt()
    }

    private val windowManager = service.getSystemService(WindowManager::class.java)
    private val mainHandler = Handler(Looper.getMainLooper())
    private val textView = TextView(service).apply {
        gravity = Gravity.CENTER_VERTICAL or Gravity.START
        includeFontPadding = false
        isClickable = false
        isFocusable = false
        maxLines = 2
        ellipsize = TextUtils.TruncateAt.END
        setTextColor(NORMAL_COLOR)
        setShadowLayer(2f, 1f, 1f, Color.BLACK)
    }
    private var attached = false
    private var refreshRunnable: Runnable? = null
    private val listener: (RuntimeSnapshot) -> Unit = { snapshot ->
        mainHandler.post { render(snapshot) }
    }

    init {
        RuntimeStatusCenter.addListener(listener)
    }

    private fun render(snapshot: RuntimeSnapshot) {
        cancelRefresh()
        val metrics = service.resources.displayMetrics
        val presentation = RuntimeOverlayModel.presentation(snapshot, System.currentTimeMillis())
        if (!presentation.visible || metrics.widthPixels <= metrics.heightPixels) {
            hide()
            return
        }

        val bounds = RuntimeOverlayModel.boundsFor(metrics.widthPixels, metrics.heightPixels)
        textView.text = "${presentation.title}\n${presentation.detail}"
        textView.setTextColor(if (presentation.warning) WARNING_COLOR else NORMAL_COLOR)
        textView.setTextSize(TypedValue.COMPLEX_UNIT_PX, bounds.height * 0.24f)
        textView.setPadding(
            (bounds.width * 0.025f).toInt(),
            0,
            (bounds.width * 0.025f).toInt(),
            0,
        )
        textView.background = GradientDrawable().apply {
            shape = GradientDrawable.RECTANGLE
            cornerRadius = bounds.height * 0.06f
            setColor(0x990d1510.toInt())
        }

        val params = layoutParams(bounds)
        runCatching {
            if (attached) {
                windowManager.updateViewLayout(textView, params)
            } else {
                windowManager.addView(textView, params)
                attached = true
            }
        }.onFailure { error ->
            Log.e(TAG, "runtime_overlay_render_failed", error)
            runCatching { windowManager.removeViewImmediate(textView) }
            attached = false
        }

        presentation.nextRefreshDelayMs?.let { delayMs ->
            val runnable = Runnable { render(RuntimeStatusCenter.snapshot.value) }
            refreshRunnable = runnable
            mainHandler.postDelayed(runnable, delayMs.coerceAtLeast(50L))
        }
    }

    private fun layoutParams(bounds: RuntimeOverlayBounds) = WindowManager.LayoutParams(
        bounds.width,
        bounds.height,
        WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
        WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
            WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
            WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or
            WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
        PixelFormat.TRANSLUCENT,
    ).apply {
        gravity = Gravity.TOP or Gravity.START
        x = bounds.x
        y = bounds.y
        title = "字牌AI运行摘要"
    }

    private fun hide() {
        if (!attached) return
        runCatching { windowManager.removeView(textView) }
            .onFailure { error -> Log.w(TAG, "runtime_overlay_hide_failed", error) }
        attached = false
    }

    private fun cancelRefresh() {
        refreshRunnable?.let(mainHandler::removeCallbacks)
        refreshRunnable = null
    }

    override fun close() {
        RuntimeStatusCenter.removeListener(listener)
        mainHandler.post {
            cancelRefresh()
            hide()
        }
    }
}
