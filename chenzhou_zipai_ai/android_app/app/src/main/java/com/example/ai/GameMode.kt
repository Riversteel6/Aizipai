package com.example.ai

import android.content.Context

enum class GameMode(val storedValue: String, val wildcardEnabled: Boolean) {
    NO_WANG("no_wang", false),
    WANG("wang", true),
}

object ModeStore {
    private const val PREFERENCES = "aizipai_settings"
    private const val KEY_MODE = "game_mode"

    fun load(context: Context): GameMode {
        val value = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
            .getString(KEY_MODE, GameMode.NO_WANG.storedValue)
        return GameMode.entries.firstOrNull { it.storedValue == value } ?: GameMode.NO_WANG
    }

    fun save(context: Context, mode: GameMode) {
        context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_MODE, mode.storedValue)
            .apply()
    }
}
