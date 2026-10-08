package com.example.ai

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.AccessibilityServiceInfo
import android.util.Log
import android.view.KeyEvent
import android.view.accessibility.AccessibilityEvent

class AiAccessibilityService : AccessibilityService() {
    private lateinit var controller: AutomationController
    private lateinit var notifier: RuntimeNotifier
    private lateinit var overlay: RuntimeOverlay

    override fun onServiceConnected() {
        super.onServiceConnected()
        serviceInfo = serviceInfo.apply {
            flags = flags or AccessibilityServiceInfo.FLAG_REQUEST_FILTER_KEY_EVENTS
        }
        controller = AutomationController(this)
        notifier = RuntimeNotifier(this)
        overlay = RuntimeOverlay(this)
        activeService = this
        RuntimeStatusCenter.update(
            RuntimePhase.READY,
            "服务已连接",
            "等待启动",
            ModeStore.load(this),
        )
        Log.i(TAG, "accessibility_service_connected flags=${serviceInfo.flags}")
        if (RuntimeRunStore.shouldResume(this)) {
            Log.i(TAG, "runtime_auto_resume_requested")
            startInternal()
        }
    }

    override fun onKeyEvent(event: KeyEvent): Boolean {
        if (event.keyCode != KeyEvent.KEYCODE_VOLUME_UP && event.keyCode != KeyEvent.KEYCODE_VOLUME_DOWN) {
            return super.onKeyEvent(event)
        }
        if (event.action == KeyEvent.ACTION_DOWN && event.repeatCount == 0) {
            Log.i(TAG, "volume_key keyCode=${event.keyCode}")
            when (event.keyCode) {
                KeyEvent.KEYCODE_VOLUME_UP -> startInternal()
                KeyEvent.KEYCODE_VOLUME_DOWN -> stopInternal()
            }
        }
        return true
    }

    private fun startInternal(): Boolean {
        if (!BackgroundRunProtection.isEnabled(this)) {
            RuntimeRunStore.setDesiredRunning(this, false)
            RuntimeStatusCenter.update(
                RuntimePhase.PAUSED,
                "无法启动",
                "请先在字牌AI页面开启后台运行保护",
                ModeStore.load(this),
            )
            return false
        }
        RuntimeRunStore.setDesiredRunning(this, true)
        if (!RuntimeKeepAliveService.start(this)) {
            RuntimeRunStore.setDesiredRunning(this, false)
            RuntimeStatusCenter.update(
                RuntimePhase.ERROR,
                "无法启动",
                "前台保活服务启动失败",
                ModeStore.load(this),
            )
            return false
        }
        val selectedMode = ModeStore.load(this)
        val started = controller.start(selectedMode)
        if (started) {
            notifier.signalStarted()
        } else {
            RuntimeStatusCenter.update(
                RuntimePhase.OBSERVING,
                "字牌AI已在运行",
                "正在观察牌局",
                selectedMode,
            )
        }
        return started
    }

    private fun stopInternal(): Boolean {
        RuntimeRunStore.setDesiredRunning(this, false)
        val stopped = controller.stop()
        RuntimeStatusCenter.update(
            RuntimePhase.STOPPED,
            "字牌AI已停止",
            if (stopped) "不会再截图或点击" else "当前本来就是停止状态",
            ModeStore.load(this),
        )
        RuntimeKeepAliveService.stop(this)
        notifier.signalStopped()
        return stopped
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) = Unit

    override fun onInterrupt() {
        RuntimeRunStore.setDesiredRunning(this, false)
        if (::controller.isInitialized) controller.stop()
        RuntimeKeepAliveService.stop(this)
        RuntimeStatusCenter.update(
            RuntimePhase.PAUSED,
            "字牌AI已暂停",
            "无障碍服务被系统中断",
            ModeStore.load(this),
        )
    }

    override fun onDestroy() {
        val wasCurrentService = activeService === this
        if (wasCurrentService) activeService = null
        if (::controller.isInitialized) controller.close()
        if (::notifier.isInitialized) notifier.close()
        if (::overlay.isInitialized) overlay.close()
        RuntimeKeepAliveService.stop(this)
        if (wasCurrentService) {
            RuntimeStatusCenter.update(
                RuntimePhase.SERVICE_REQUIRED,
                "无障碍服务未连接",
                "请重新开启“字牌AI”无障碍服务",
                ModeStore.load(this),
            )
        }
        super.onDestroy()
    }

    companion object {
        private const val TAG = "AizipaiRuntime"
        @Volatile private var activeService: AiAccessibilityService? = null

        private fun current(): AiAccessibilityService? = activeService

        fun isConnected(): Boolean = current()?.let { it::controller.isInitialized } == true

        fun startRuntime(): Boolean {
            val service = current()
            if (service == null || !service::controller.isInitialized) {
                RuntimeStatusCenter.update(
                    RuntimePhase.SERVICE_REQUIRED,
                    "无法启动",
                    "请先开启“字牌AI”无障碍服务",
                )
                return false
            }
            return service.startInternal()
        }

        fun stopRuntime(): Boolean {
            val service = current()
            if (service == null || !service::controller.isInitialized) {
                RuntimeStatusCenter.update(
                    RuntimePhase.SERVICE_REQUIRED,
                    "当前未运行",
                    "无障碍服务尚未连接",
                )
                return false
            }
            return service.stopInternal()
        }

        fun refreshRuntimeFeedback() {
            val service = current() ?: return
            val phase = if (service.controller.isRunning()) RuntimePhase.OBSERVING else RuntimePhase.READY
            RuntimeStatusCenter.update(
                phase,
                if (phase == RuntimePhase.OBSERVING) "字牌AI运行中" else "服务已连接",
                if (phase == RuntimePhase.OBSERVING) "正在观察牌局" else "等待启动",
                ModeStore.load(service),
            )
            RuntimeStatusCenter.refresh()
        }
    }
}
