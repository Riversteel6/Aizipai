package com.example.ai

import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import android.util.Log
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat

class RuntimeKeepAliveService : Service() {
    private val statusListener: (RuntimeSnapshot) -> Unit = { snapshot ->
        if (RuntimeRunStore.shouldResume(this)) publishForeground(snapshot)
    }

    override fun onCreate() {
        super.onCreate()
        RuntimeNotifier.ensureChannel(this)
        RuntimeStatusCenter.addListener(statusListener)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (!RuntimeRunStore.shouldResume(this)) {
            stopSelf(startId)
            return START_NOT_STICKY
        }
        publishForeground(RuntimeStatusCenter.snapshot.value)
        Log.i(TAG, "runtime_keep_alive_started")
        return START_STICKY
    }

    override fun onDestroy() {
        RuntimeStatusCenter.removeListener(statusListener)
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_DETACH)
        Log.i(TAG, "runtime_keep_alive_stopped")
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun publishForeground(snapshot: RuntimeSnapshot) {
        val type = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE
        } else {
            0
        }
        ServiceCompat.startForeground(
            this,
            RuntimeNotifier.NOTIFICATION_ID,
            RuntimeNotifier.buildNotification(this, snapshot),
            type,
        )
    }

    companion object {
        private const val TAG = "AizipaiKeepAlive"

        fun start(context: Context): Boolean = runCatching {
            ContextCompat.startForegroundService(
                context,
                Intent(context, RuntimeKeepAliveService::class.java),
            )
        }.onFailure { Log.e(TAG, "runtime_keep_alive_start_failed", it) }.isSuccess

        fun stop(context: Context) {
            context.stopService(Intent(context, RuntimeKeepAliveService::class.java))
        }
    }
}
