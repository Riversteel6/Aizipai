package com.example.ai

import org.junit.Assert.assertEquals
import org.junit.Test

class AutomationFailurePresentationTest {
    @Test
    fun strategyTimeoutIsNotReportedAsFrameReadFailure() {
        val status = decisionWorkerFailureStatus("decision_worker_timeout:decide_rgba")

        assertEquals("策略搜索超时", status.title)
        assertEquals("已清理本轮搜索任务，正在自动重试", status.reason)
    }

    @Test
    fun nonTimeoutWorkerFailureKeepsRecoveryMessage() {
        val status = decisionWorkerFailureStatus("decision_worker_bind_timeout")

        assertEquals("字牌AI正在恢复", status.title)
        assertEquals("识图决策进程异常，正在自动重建", status.reason)
    }
}
