package com.example.ai

import org.junit.Assert.assertEquals
import org.junit.Test

class RuntimeStatusCenterTest {
    @Test
    fun actionsUseHumanReadableNames() {
        assertEquals("胡", humanAction("hu"))
        assertEquals("展开吃牌", humanAction("expand_chi_options"))
        assertEquals("吃牌候选", humanAction("chi_option"))
        assertEquals("等待", humanAction(null))
    }

    @Test
    fun identicalStatusDoesNotNotifyTwice() {
        var updates = 0
        val listener: (RuntimeSnapshot) -> Unit = { updates += 1 }
        RuntimeStatusCenter.addListener(listener)
        val before = updates
        RuntimeStatusCenter.update(RuntimePhase.READY, "服务已连接", "等待启动", GameMode.WANG)
        val afterFirst = updates
        RuntimeStatusCenter.update(RuntimePhase.READY, "服务已连接", "等待启动", GameMode.WANG)
        RuntimeStatusCenter.removeListener(listener)

        assertEquals(before + 1, afterFirst)
        assertEquals(afterFirst, updates)
    }
}
