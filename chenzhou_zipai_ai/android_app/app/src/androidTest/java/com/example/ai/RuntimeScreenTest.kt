package com.example.ai

import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithText
import org.junit.Rule
import org.junit.Test

class RuntimeScreenTest {
    @get:Rule
    val composeRule = createAndroidComposeRule<MainActivity>()

    @Test
    fun setupAndRuntimeControlsAreVisible() {
        val volumeHint = composeRule.activity.getString(R.string.volume_hint)
        listOf(
            composeRule.activity.getString(R.string.app_version_label),
            "无障碍服务",
            "运行摘要通知",
            "1v1 无王",
            "1v1 有王",
            "保存模式",
            "启动",
            "停止",
            volumeHint,
        ).forEach { label ->
            composeRule.onNodeWithText(label).assertExists()
        }
    }
}
