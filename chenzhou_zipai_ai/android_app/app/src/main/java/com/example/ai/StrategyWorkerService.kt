package com.example.ai

import android.app.Service
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.Message
import android.os.Messenger
import android.os.Process
import android.util.Log
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.atomic.AtomicLong

open class StrategyWorkerService : Service() {
    private companion object {
        const val TAG = "AizipaiStrategyWorker"
    }

    private val workerLock = Any()
    private val generation = AtomicLong()
    @Volatile private var worker: ExecutorService = newWorkerExecutor(generation.get())
    private val messenger by lazy {
        Messenger(
            Handler(Looper.getMainLooper()) { message ->
                when (message.what) {
                    StrategyWorkerProtocol.EXECUTE -> handleExecute(message)
                    StrategyWorkerProtocol.CANCEL_ALL -> {
                        rotateWorkerGeneration()
                        true
                    }
                    StrategyWorkerProtocol.RESTART_PROCESS -> {
                        Log.e(TAG, "worker_process_restart pid=${Process.myPid()}")
                        Process.killProcess(Process.myPid())
                        true
                    }
                    else -> false
                }
            },
        )
    }

    override fun onCreate() {
        super.onCreate()
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
        Log.i(TAG, "worker_started pid=${Process.myPid()}")
        currentWorker().first.execute {
            try {
                Python.getInstance().getModule("mobile_worker")
                    .callAttr("warmup", filesDir.absolutePath)
                Log.i(TAG, "worker_ready pid=${Process.myPid()}")
            } catch (error: Throwable) {
                Log.e(TAG, "worker_warmup_failed pid=${Process.myPid()}", error)
            }
        }
    }

    override fun onBind(intent: Intent?): IBinder = messenger.binder

    private fun handleExecute(message: Message): Boolean {
        val replyTo = message.replyTo ?: return true
        val requestId = message.data.getLong(StrategyWorkerProtocol.REQUEST_ID, -1L)
        val taskName = message.data.getString(StrategyWorkerProtocol.TASK_NAME).orEmpty()
        val payload = message.data.getString(StrategyWorkerProtocol.PAYLOAD).orEmpty()
        if (requestId < 0 || taskName.isBlank() || payload.isBlank()) {
            sendResult(replyTo, requestId, null, "invalid_worker_request")
            return true
        }
        Log.i(TAG, "worker_task_received id=$requestId task=$taskName pid=${Process.myPid()}")
        val (executor, taskGeneration) = currentWorker()
        try {
            executor.execute {
                val startedAt = android.os.SystemClock.elapsedRealtime()
                Log.i(TAG, "worker_task_started id=$requestId task=$taskName pid=${Process.myPid()}")
                try {
                    val result = Python.getInstance()
                        .getModule("mobile_worker")
                        .callAttr(
                            "execute_task",
                            taskName,
                            payload,
                            filesDir.absolutePath,
                        )
                        .toString()
                    if (generation.get() == taskGeneration) {
                        Log.i(
                            TAG,
                            "worker_task_finished id=$requestId task=$taskName " +
                                "pid=${Process.myPid()} elapsed_ms=" +
                                (android.os.SystemClock.elapsedRealtime() - startedAt),
                        )
                        sendResult(replyTo, requestId, result, null)
                    }
                } catch (error: Throwable) {
                    if (generation.get() == taskGeneration) {
                        Log.e(TAG, "worker_task_failed id=$requestId task=$taskName", error)
                        sendResult(replyTo, requestId, null, error.stackTraceToString())
                    }
                }
            }
        } catch (_: RejectedExecutionException) {
            // A simultaneous CANCEL_ALL owns this request. The parent has already
            // cancelled its future, so a late response must not be synthesized.
        }
        return true
    }

    private fun currentWorker(): Pair<ExecutorService, Long> = synchronized(workerLock) {
        worker to generation.get()
    }

    private fun rotateWorkerGeneration() {
        val retired = synchronized(workerLock) {
            val nextGeneration = generation.incrementAndGet()
            val previous = worker
            worker = newWorkerExecutor(nextGeneration)
            previous
        }
        retired.shutdownNow()
        Log.i(TAG, "worker_generation_cancelled pid=${Process.myPid()} generation=${generation.get()}")
    }

    private fun newWorkerExecutor(workerGeneration: Long): ExecutorService =
        Executors.newSingleThreadExecutor { runnable ->
            Thread(
                runnable,
                "aizipai-strategy-worker-$workerGeneration",
            ).apply { isDaemon = true }
        }

    private fun sendResult(
        replyTo: Messenger,
        requestId: Long,
        payload: String?,
        error: String?,
    ) {
        val response = Message.obtain(null, StrategyWorkerProtocol.RESULT).apply {
            data = Bundle().apply {
                putLong(StrategyWorkerProtocol.REQUEST_ID, requestId)
                putBoolean(StrategyWorkerProtocol.OK, error == null)
                if (payload != null) putString(StrategyWorkerProtocol.PAYLOAD, payload)
                if (error != null) putString(StrategyWorkerProtocol.ERROR, error)
            }
        }
        try {
            replyTo.send(response)
        } catch (error: Throwable) {
            Log.w(TAG, "worker_result_delivery_failed id=$requestId", error)
        }
    }

    override fun onDestroy() {
        synchronized(workerLock) { worker }.shutdownNow()
        super.onDestroy()
    }
}

class StrategyWorkerService0 : StrategyWorkerService()
class StrategyWorkerService1 : StrategyWorkerService()
class StrategyWorkerService2 : StrategyWorkerService()
class StrategyWorkerService3 : StrategyWorkerService()
class StrategyWorkerService4 : StrategyWorkerService()
class StrategyWorkerService5 : StrategyWorkerService()
class StrategyWorkerService6 : StrategyWorkerService()
class StrategyWorkerService7 : StrategyWorkerService()
