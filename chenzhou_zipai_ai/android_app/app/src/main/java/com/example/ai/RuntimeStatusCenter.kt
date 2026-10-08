package com.example.ai

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.concurrent.CopyOnWriteArraySet

enum class RuntimePhase {
    SERVICE_REQUIRED,
    READY,
    STARTING,
    OBSERVING,
    THINKING,
    VERIFYING,
    EXECUTING,
    STOPPED,
    PAUSED,
    ERROR,
}

enum class RuntimeSeverity {
    NORMAL,
    WARNING,
}

data class RuntimeSnapshot(
    val phase: RuntimePhase,
    val title: String,
    val detail: String,
    val mode: GameMode? = null,
    val severity: RuntimeSeverity = RuntimeSeverity.NORMAL,
    val updatedAtMs: Long = System.currentTimeMillis(),
) {
    val running: Boolean
        get() = phase in setOf(
            RuntimePhase.STARTING,
            RuntimePhase.OBSERVING,
            RuntimePhase.THINKING,
            RuntimePhase.VERIFYING,
            RuntimePhase.EXECUTING,
        )
}

object RuntimeStatusCenter {
    private val mutableSnapshot = MutableStateFlow(
        RuntimeSnapshot(
            phase = RuntimePhase.SERVICE_REQUIRED,
            title = "无障碍服务未连接",
            detail = "请先开启“字牌AI”无障碍服务",
        ),
    )
    private val listeners = CopyOnWriteArraySet<(RuntimeSnapshot) -> Unit>()

    val snapshot: StateFlow<RuntimeSnapshot> = mutableSnapshot.asStateFlow()

    fun update(
        phase: RuntimePhase,
        title: String,
        detail: String,
        mode: GameMode? = mutableSnapshot.value.mode,
        severity: RuntimeSeverity = RuntimeSeverity.NORMAL,
    ) {
        val previous = mutableSnapshot.value
        if (
            previous.phase == phase &&
            previous.title == title &&
            previous.detail == detail &&
            previous.mode == mode &&
            previous.severity == severity
        ) return
        val next = RuntimeSnapshot(
            phase = phase,
            title = title,
            detail = detail,
            mode = mode,
            severity = severity,
        )
        mutableSnapshot.value = next
        listeners.forEach { it(next) }
    }

    fun refresh() {
        val current = mutableSnapshot.value
        listeners.forEach { it(current) }
    }

    fun addListener(listener: (RuntimeSnapshot) -> Unit) {
        listeners += listener
        listener(mutableSnapshot.value)
    }

    fun removeListener(listener: (RuntimeSnapshot) -> Unit) {
        listeners -= listener
    }
}

internal fun humanAction(action: String?): String = when (action?.lowercase()) {
    "hu" -> "胡"
    "peng" -> "碰"
    "pass" -> "过"
    "chi", "chi_option" -> "吃牌候选"
    "compare_option" -> "比牌候选"
    "expand_chi_options" -> "展开吃牌"
    "discard" -> "出牌"
    "compact_hand" -> "整理手牌"
    "settlement_ready" -> "准备下一局"
    null, "", "none" -> "等待"
    else -> action
}
