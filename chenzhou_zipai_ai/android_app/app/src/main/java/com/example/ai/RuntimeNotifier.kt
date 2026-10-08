package com.example.ai

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log
import androidx.core.content.ContextCompat

class RuntimeNotifier(private val context: Context) : AutoCloseable {
    companion object {
        const val CHANNEL_ID = "aizipai_runtime"
        const val NOTIFICATION_ID = 8101
        const val ACTION_START = "com.example.ai.action.START"
        const val ACTION_STOP = "com.example.ai.action.STOP"

        fun notificationsAllowed(context: Context): Boolean {
            val permissionGranted = Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU ||
                ContextCompat.checkSelfPermission(
                    context,
                    Manifest.permission.POST_NOTIFICATIONS,
                ) == PackageManager.PERMISSION_GRANTED
            return permissionGranted &&
                context.getSystemService(NotificationManager::class.java).areNotificationsEnabled()
        }

        fun ensureChannel(context: Context) {
            context.getSystemService(NotificationManager::class.java).createNotificationChannel(
                NotificationChannel(
                    CHANNEL_ID,
                    "字牌AI运行摘要",
                    NotificationManager.IMPORTANCE_LOW,
                ).apply {
                    description = "显示字牌AI当前运行阶段和最近一次动作"
                    setShowBadge(false)
                },
            )
        }

        fun buildNotification(context: Context, snapshot: RuntimeSnapshot): Notification {
            val openApp = PendingIntent.getActivity(
                context,
                0,
                Intent(context, MainActivity::class.java),
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
            )
            val modeLabel = when (snapshot.mode) {
                GameMode.WANG -> "1v1 有王"
                GameMode.NO_WANG -> "1v1 无王"
                null -> null
            }
            return Notification.Builder(context, CHANNEL_ID)
                .setSmallIcon(android.R.drawable.stat_notify_sync)
                .setContentTitle(snapshot.title)
                .setContentText(snapshot.detail)
                .setContentIntent(openApp)
                .setCategory(Notification.CATEGORY_SERVICE)
                .setOnlyAlertOnce(true)
                .setOngoing(snapshot.running)
                .setShowWhen(true)
                .setWhen(snapshot.updatedAtMs)
                .apply {
                    if (modeLabel != null) setSubText(modeLabel)
                    if (snapshot.running) {
                        addAction(
                            Notification.Action.Builder(
                                null,
                                "停止",
                                actionIntent(context, ACTION_STOP, 2),
                            ).build(),
                        )
                    } else if (snapshot.phase != RuntimePhase.SERVICE_REQUIRED) {
                        addAction(
                            Notification.Action.Builder(
                                null,
                                "启动",
                                actionIntent(context, ACTION_START, 1),
                            ).build(),
                        )
                    }
                }
                .build()
        }

        private fun actionIntent(context: Context, action: String, requestCode: Int): PendingIntent =
            PendingIntent.getBroadcast(
                context,
                requestCode,
                Intent(context, RuntimeActionReceiver::class.java).setAction(action),
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
            )
    }

    private val manager = context.getSystemService(NotificationManager::class.java)
    private val listener: (RuntimeSnapshot) -> Unit = ::publish

    init {
        ensureChannel(context)
        RuntimeStatusCenter.addListener(listener)
    }

    fun signalStarted() = vibrate(longArrayOf(0, 80))

    fun signalStopped() = vibrate(longArrayOf(0, 70, 90, 70))

    private fun publish(snapshot: RuntimeSnapshot) {
        Log.i(
            "AizipaiStatus",
            "phase=${snapshot.phase} title=${snapshot.title} detail=${snapshot.detail} " +
                "mode=${snapshot.mode?.storedValue}",
        )
        if (!notificationsAllowed(context)) return
        manager.notify(NOTIFICATION_ID, buildNotification(context, snapshot))
    }

    private fun vibrate(pattern: LongArray) {
        val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            context.getSystemService(VibratorManager::class.java).defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            context.getSystemService(Vibrator::class.java)
        }
        if (vibrator.hasVibrator()) {
            vibrator.vibrate(VibrationEffect.createWaveform(pattern, -1))
        }
    }

    override fun close() {
        RuntimeStatusCenter.removeListener(listener)
    }

}

class RuntimeActionReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        when (intent.action) {
            "com.example.ai.action.START" -> AiAccessibilityService.startRuntime()
            "com.example.ai.action.STOP" -> AiAccessibilityService.stopRuntime()
        }
    }
}
