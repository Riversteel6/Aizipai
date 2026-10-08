package com.example.ai

import android.content.Context

object RuntimeRunStore {
    private const val PREFERENCES = "runtime_run_state"
    private const val DESIRED_RUNNING = "desired_running"

    fun setDesiredRunning(context: Context, running: Boolean): Boolean =
        context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(DESIRED_RUNNING, running)
            .commit()

    fun shouldResume(context: Context): Boolean =
        context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
            .getBoolean(DESIRED_RUNNING, false)
}
