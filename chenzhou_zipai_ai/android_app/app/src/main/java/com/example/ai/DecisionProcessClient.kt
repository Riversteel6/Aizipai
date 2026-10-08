package com.example.ai

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.Message
import android.os.Messenger
import android.util.Log
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import java.util.concurrent.atomic.AtomicLong

class DecisionWorkerUnavailableException(message: String, cause: Throwable? = null) :
    RuntimeException(message, cause)

class DecisionProcessClient(context: Context) : AutoCloseable {
    private companion object {
        const val TAG = "AizipaiDecisionClient"
        const val BIND_TIMEOUT_MS = 10_000L
        const val CALL_TIMEOUT_MS = 25_000L
    }

    @Suppress("PLATFORM_CLASS_MAPPED_TO_KOTLIN")
    private val monitor = java.lang.Object()
    private val appContext = context.applicationContext
    private val intent = Intent(appContext, DecisionWorkerService::class.java)
    private val requestIds = AtomicLong()
    private val pending = ConcurrentHashMap<Long, CompletableFuture<String>>()
    private val transportExecutor: ExecutorService = Executors.newCachedThreadPool { runnable ->
        Thread(runnable, "aizipai-decision-transport").apply { isDaemon = true }
    }
    private val callbackExecutor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "aizipai-decision-binding").apply { isDaemon = true }
    }
    private val replyThread = HandlerThread("aizipai-decision-replies").apply { start() }
    private val replyMessenger = Messenger(
        Handler(replyThread.looper) { message ->
            if (message.what != DecisionWorkerProtocol.RESULT) return@Handler false
            val requestId = message.data.getLong(DecisionWorkerProtocol.REQUEST_ID, -1L)
            val future = pending[requestId] ?: return@Handler true
            if (message.data.getBoolean(DecisionWorkerProtocol.OK, false)) {
                future.complete(message.data.getString(DecisionWorkerProtocol.PAYLOAD).orEmpty())
            } else {
                future.completeExceptionally(
                    DecisionWorkerUnavailableException(
                        message.data.getString(DecisionWorkerProtocol.ERROR) ?: "decision_worker_failed",
                    ),
                )
            }
            true
        },
    )

    @Volatile private var remote: Messenger? = null
    @Volatile private var connection: ServiceConnection? = null
    @Volatile private var bindRequested = false

    fun call(operation: String, arguments: Bundle = Bundle(), timeoutMs: Long = CALL_TIMEOUT_MS): String {
        val requestId = requestIds.incrementAndGet()
        val future = CompletableFuture<String>()
        pending[requestId] = future
        val message = Message.obtain(null, DecisionWorkerProtocol.EXECUTE).apply {
            replyTo = replyMessenger
            data = Bundle(arguments).apply {
                putLong(DecisionWorkerProtocol.REQUEST_ID, requestId)
                putString(DecisionWorkerProtocol.OPERATION, operation)
            }
        }
        val startedAt = android.os.SystemClock.elapsedRealtime()
        Log.i(TAG, "decision_call_started id=$requestId operation=$operation")
        val transaction = transportExecutor.submit<String> {
            connectedMessenger().send(message)
            future.get()
        }
        try {
            val result = transaction.get(timeoutMs.coerceAtLeast(1L), TimeUnit.MILLISECONDS)
            Log.i(
                TAG,
                "decision_call_finished id=$requestId operation=$operation " +
                    "elapsed_ms=${android.os.SystemClock.elapsedRealtime() - startedAt}",
            )
            return result
        } catch (error: TimeoutException) {
            transaction.cancel(true)
            Log.e(
                TAG,
                "decision_call_timeout id=$requestId operation=$operation " +
                    "elapsed_ms=${android.os.SystemClock.elapsedRealtime() - startedAt}",
            )
            throw DecisionWorkerUnavailableException("decision_worker_timeout:$operation", error)
        } catch (error: InterruptedException) {
            Thread.currentThread().interrupt()
            transaction.cancel(true)
            throw DecisionWorkerUnavailableException("decision_worker_interrupted:$operation", error)
        } catch (error: ExecutionException) {
            val cause = (error.cause as? ExecutionException)?.cause ?: error.cause ?: error
            throw DecisionWorkerUnavailableException("decision_worker_call_failed:$operation", cause)
        } finally {
            pending.remove(requestId)
        }
    }

    fun restart(reason: String) {
        Log.e(TAG, "decision_worker_restarting reason=$reason")
        val target = remote
        if (target != null) {
            val request = transportExecutor.submit {
                target.send(Message.obtain(null, DecisionWorkerProtocol.RESTART))
            }
            runCatching { request.get(250, TimeUnit.MILLISECONDS) }
                .onFailure { request.cancel(true) }
        }
        resetBinding(DecisionWorkerUnavailableException("decision_worker_restarting:$reason"))
    }

    private fun connectedMessenger(): Messenger {
        remote?.let { return it }
        val deadline = System.currentTimeMillis() + BIND_TIMEOUT_MS
        synchronized(monitor) {
            remote?.let { return it }
            if (!bindRequested) {
                val nextConnection = newConnection()
                connection = nextConnection
                bindRequested = true
                if (!appContext.bindService(intent, Context.BIND_AUTO_CREATE, callbackExecutor, nextConnection)) {
                    bindRequested = false
                    connection = null
                    throw DecisionWorkerUnavailableException("decision_worker_bind_rejected")
                }
            }
            while (remote == null && bindRequested) {
                val remaining = deadline - System.currentTimeMillis()
                if (remaining <= 0) break
                monitor.wait(remaining)
            }
            return remote ?: throw DecisionWorkerUnavailableException("decision_worker_bind_timeout")
        }
    }

    private fun newConnection(): ServiceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            synchronized(monitor) {
                if (connection !== this) return
                remote = service?.let(::Messenger)
                monitor.notifyAll()
            }
        }

        override fun onServiceDisconnected(name: ComponentName?) = disconnected("disconnected")
        override fun onBindingDied(name: ComponentName?) = disconnected("binding_died")
        override fun onNullBinding(name: ComponentName?) = disconnected("null_binding")

        private fun disconnected(reason: String) {
            resetBinding(DecisionWorkerUnavailableException("decision_worker_$reason"), this)
        }
    }

    private fun resetBinding(error: Throwable, expected: ServiceConnection? = null) {
        val oldConnection = synchronized(monitor) {
            if (expected != null && connection !== expected) return
            val previous = connection
            remote = null
            connection = null
            bindRequested = false
            monitor.notifyAll()
            previous
        }
        pending.values.forEach { it.completeExceptionally(error) }
        if (oldConnection != null) runCatching { appContext.unbindService(oldConnection) }
    }

    override fun close() {
        restart("client_closed")
        transportExecutor.shutdownNow()
        callbackExecutor.shutdownNow()
        replyThread.quitSafely()
    }
}
