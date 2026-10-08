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
import org.json.JSONObject
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutionException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.PriorityQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicBoolean

object StrategyProcessPool {
    private const val TAG = "AizipaiStrategyPool"
    // During a foreground decision the coordinator is waiting on independent
    // rollout results, so allow one isolated worker per target CPU core.
    private const val MAX_WORKERS = 4
    private const val BIND_TIMEOUT_MS = 15_000L
    private const val DEFAULT_RESULT_TIMEOUT_MS = 45_000L

    private val lock = Any()
    private val requestIds = AtomicLong()
    private val workersStarted = AtomicBoolean()
    private val submittedCount = AtomicLong()
    private val completedCount = AtomicLong()
    private val totalLatencyMs = AtomicLong()
    private val maxLatencyMs = AtomicLong()
    private val firstSubmitAtMs = AtomicLong()
    private val lastCompleteAtMs = AtomicLong()
    private val pending = ConcurrentHashMap<Long, PendingRequest>()
    private val schedulerLock = Any()
    private val queued = PriorityQueue<QueuedRequest>(
        compareByDescending<QueuedRequest> { it.priority }
            .thenBy { it.requestId },
    )
    private val activeByClient = mutableMapOf<WorkerClient, Long>()
    private val callbackExecutor: ExecutorService = Executors.newCachedThreadPool { runnable ->
        Thread(runnable, "aizipai-worker-binding").apply { isDaemon = true }
    }
    private val replyThread = HandlerThread("aizipai-worker-replies").apply { start() }
    private val replyMessenger = Messenger(
        Handler(replyThread.looper) { message ->
            if (message.what != StrategyWorkerProtocol.RESULT) return@Handler false
            val requestId = message.data.getLong(StrategyWorkerProtocol.REQUEST_ID, -1L)
            val request = pending[requestId]
            if (request != null) {
                if (message.data.getBoolean(StrategyWorkerProtocol.OK, false)) {
                    request.future.complete(
                        message.data.getString(StrategyWorkerProtocol.PAYLOAD).orEmpty(),
                    )
                } else {
                    request.future.completeExceptionally(
                        IllegalStateException(
                            message.data.getString(StrategyWorkerProtocol.ERROR)
                                ?: "strategy_worker_failed",
                        ),
                    )
                }
                val latency = System.currentTimeMillis() - request.submittedAtMs
                completedCount.incrementAndGet()
                totalLatencyMs.addAndGet(latency)
                maxLatencyMs.accumulateAndGet(latency, ::maxOf)
                lastCompleteAtMs.set(System.currentTimeMillis())
            }
            releaseWorker(requestId)
            true
        },
    )

    @Volatile private var clients: List<WorkerClient> = emptyList()

    @JvmStatic
    fun initialize(context: Context) {
        if (clients.isNotEmpty()) return
        synchronized(lock) {
            if (clients.isNotEmpty()) return
            val appContext = context.applicationContext
            val serviceClasses = listOf(
                StrategyWorkerService0::class.java,
                StrategyWorkerService1::class.java,
                StrategyWorkerService2::class.java,
                StrategyWorkerService3::class.java,
                StrategyWorkerService4::class.java,
                StrategyWorkerService5::class.java,
                StrategyWorkerService6::class.java,
                StrategyWorkerService7::class.java,
            )
            val workerCount = Runtime.getRuntime().availableProcessors()
                .coerceIn(2, MAX_WORKERS)
            clients = serviceClasses.take(workerCount).map { WorkerClient(appContext, it) }
            Log.i(TAG, "strategy_pool_initialized workers=$workerCount")
        }
    }

    @JvmStatic
    fun submit(taskName: String, payload: String): Long =
        submitPrioritized(taskName, payload, 0)

    @JvmStatic
    fun prebindWorkers() {
        val currentClients = clients
        check(currentClients.isNotEmpty()) { "strategy_process_pool_not_initialized" }
        if (workersStarted.compareAndSet(false, true)) {
            Log.i(TAG, "strategy_workers_prebinding workers=${currentClients.size}")
        }
        currentClients.forEach { it.prebind() }
    }

    @JvmStatic
    fun executeOnWorker(
        workerIndex: Int,
        taskName: String,
        payload: String,
        timeoutMs: Long = DEFAULT_RESULT_TIMEOUT_MS,
    ): String {
        require(taskName in allowedTasks()) { "unsupported_strategy_task:$taskName" }
        require(payload.isNotBlank()) { "empty_strategy_payload" }
        val client = clients.getOrNull(workerIndex)
            ?: throw IllegalArgumentException("invalid_strategy_worker:$workerIndex")
        val requestId = requestIds.incrementAndGet()
        val request = PendingRequest(taskName, payload, Int.MAX_VALUE)
        pending[requestId] = request
        submittedCount.incrementAndGet()
        firstSubmitAtMs.compareAndSet(0L, request.submittedAtMs)
        synchronized(schedulerLock) {
            check(activeByClient[client] == null) { "strategy_worker_busy:$workerIndex" }
            request.client = client
            activeByClient[client] = requestId
        }
        if (workersStarted.compareAndSet(false, true)) {
            Log.i(TAG, "strategy_workers_starting_sequentially workers=${clients.size}")
        }
        try {
            client.send(requestId, taskName, payload)
        } catch (error: Throwable) {
            pending.remove(requestId)
            synchronized(schedulerLock) {
                if (activeByClient[client] == requestId) activeByClient.remove(client)
            }
            throw error
        }
        return try {
            awaitResult(requestId, timeoutMs)
        } catch (error: Throwable) {
            restartWorkers()
            throw error
        }
    }

    @JvmStatic
    fun submitPrioritized(taskName: String, payload: String, priority: Int): Long {
        require(taskName in allowedTasks()) { "unsupported_strategy_task:$taskName" }
        require(payload.isNotBlank()) { "empty_strategy_payload" }
        val currentClients = clients
        check(currentClients.isNotEmpty()) { "strategy_process_pool_not_initialized" }
        if (workersStarted.compareAndSet(false, true)) {
            Log.i(TAG, "strategy_workers_starting_on_first_task workers=${currentClients.size}")
        }
        val requestId = requestIds.incrementAndGet()
        val request = PendingRequest(taskName, payload, priority)
        pending[requestId] = request
        submittedCount.incrementAndGet()
        firstSubmitAtMs.compareAndSet(0L, request.submittedAtMs)
        synchronized(schedulerLock) {
            queued.add(QueuedRequest(requestId, priority))
        }
        Log.i(TAG, "task_submitted id=$requestId priority=$priority task=$taskName")
        dispatchPending()
        return requestId
    }

    @JvmStatic
    fun awaitResult(requestId: Long, timeoutMs: Long = DEFAULT_RESULT_TIMEOUT_MS): String {
        val request = pending[requestId]
            ?: throw IllegalStateException("unknown_strategy_request:$requestId")
        return try {
            request.future.get(timeoutMs.coerceAtLeast(1L), TimeUnit.MILLISECONDS)
        } catch (error: InterruptedException) {
            Thread.currentThread().interrupt()
            request.future.cancel(true)
            throw IllegalStateException("strategy_request_interrupted:$requestId", error)
        } catch (error: TimeoutException) {
            request.future.cancel(true)
            throw IllegalStateException("strategy_request_timeout:$requestId", error)
        } catch (error: ExecutionException) {
            throw IllegalStateException("strategy_request_failed:$requestId", error.cause ?: error)
        } finally {
            pending.remove(requestId)
        }
    }

    @JvmStatic
    fun cancel(requestId: Long): Boolean {
        val request = pending.remove(requestId) ?: return false
        val cancelled = request.future.cancel(true)
        if (request.client != null) {
            // A running Chaquopy/Python task does not reliably honor Java thread
            // interruption. Retire the whole generation before a retry can queue
            // behind it.
            restartWorkers()
        } else {
            dispatchPending()
        }
        return cancelled
    }

    @JvmStatic
    fun cancelAll() {
        Log.i(TAG, "cancel_all pending=${pending.size}")
        pending.entries.toList().forEach { (requestId, request) ->
            if (pending.remove(requestId, request)) request.future.cancel(true)
        }
        synchronized(schedulerLock) {
            queued.clear()
            activeByClient.clear()
        }
        clients.forEach { it.cancelAll() }
    }

    @JvmStatic
    fun restartWorkers() {
        pending.entries.toList().forEach { (requestId, request) ->
            if (pending.remove(requestId, request)) request.future.cancel(true)
        }
        synchronized(schedulerLock) {
            queued.clear()
            activeByClient.clear()
        }
        clients.forEach { it.restartProcess() }
    }

    @JvmStatic
    fun workerCount(): Int = clients.size

    @JvmStatic
    fun metricsJson(): String {
        val completed = completedCount.get()
        val first = firstSubmitAtMs.get()
        val last = lastCompleteAtMs.get()
        return JSONObject()
            .put("submitted", submittedCount.get())
            .put("completed", completed)
            .put(
                "average_latency_ms",
                if (completed == 0L) 0.0 else totalLatencyMs.get().toDouble() / completed,
            )
            .put("max_latency_ms", maxLatencyMs.get())
            .put("task_wall_span_ms", if (first == 0L || last < first) 0L else last - first)
            .put("pending", pending.size)
            .toString()
    }

    @JvmStatic
    fun resetMetrics() {
        check(pending.isEmpty()) { "cannot_reset_strategy_metrics_with_pending_tasks" }
        submittedCount.set(0L)
        completedCount.set(0L)
        totalLatencyMs.set(0L)
        maxLatencyMs.set(0L)
        firstSubmitAtMs.set(0L)
        lastCompleteAtMs.set(0L)
    }

    private fun allowedTasks(): Set<String> = setOf(
        "ai.pro_brain._evaluate_action_task",
        "ai.dual_discard_validator._root_search_task",
        "ai.parallel_response_search._response_search_task",
        "runtime.batch",
        "diagnostic.echo",
        "diagnostic.sleep",
        "diagnostic.warmup",
    )

    private fun dispatchPending() {
        while (true) {
            val dispatches = synchronized(schedulerLock) {
                clients
                    .filterNot(activeByClient::containsKey)
                    .mapNotNull { client -> nextDispatchLocked(client) }
            }
            if (dispatches.isEmpty()) return
            dispatches.forEach { dispatch ->
                callbackExecutor.execute {
                    try {
                        dispatch.client.send(
                            dispatch.requestId,
                            dispatch.request.taskName,
                            dispatch.request.payload,
                        )
                    } catch (error: Throwable) {
                        Log.w(
                            TAG,
                            "task_dispatch_failed id=${dispatch.requestId} " +
                                "task=${dispatch.request.taskName}",
                            error,
                        )
                        if (pending.remove(dispatch.requestId, dispatch.request)) {
                            dispatch.request.future.completeExceptionally(error)
                        }
                        synchronized(schedulerLock) {
                            if (activeByClient[dispatch.client] == dispatch.requestId) {
                                activeByClient.remove(dispatch.client)
                            }
                        }
                        dispatchPending()
                    }
                }
            }
        }
    }

    private fun nextDispatchLocked(client: WorkerClient): Dispatch? {
        while (queued.isNotEmpty()) {
            val item = queued.remove()
            val request = pending[item.requestId] ?: continue
            if (request.future.isDone || request.client != null) continue
            request.client = client
            activeByClient[client] = item.requestId
            return Dispatch(item.requestId, request, client)
        }
        return null
    }

    private fun releaseWorker(requestId: Long) {
        synchronized(schedulerLock) {
            val client = activeByClient.entries
                .firstOrNull { it.value == requestId }
                ?.key
            if (client != null) activeByClient.remove(client)
        }
        Log.i(TAG, "task_released id=$requestId pending=${pending.size}")
        dispatchPending()
    }

    private fun failPending(client: WorkerClient, error: Throwable) {
        pending.entries.toList().forEach { (requestId, request) ->
            if (request.client === client && pending.remove(requestId, request)) {
                request.future.completeExceptionally(error)
            }
        }
        synchronized(schedulerLock) {
            activeByClient.remove(client)
        }
        dispatchPending()
    }

    private data class PendingRequest(
        val taskName: String,
        val payload: String,
        val priority: Int,
        val future: CompletableFuture<String> = CompletableFuture(),
        val submittedAtMs: Long = System.currentTimeMillis(),
    ) {
        @Volatile var client: WorkerClient? = null
    }

    private data class QueuedRequest(
        val requestId: Long,
        val priority: Int,
    )

    private data class Dispatch(
        val requestId: Long,
        val request: PendingRequest,
        val client: WorkerClient,
    )

    private class WorkerClient(
        private val context: Context,
        serviceClass: Class<out StrategyWorkerService>,
    ) {
        @Suppress("PLATFORM_CLASS_MAPPED_TO_KOTLIN")
        private val monitor = java.lang.Object()
        private val intent = Intent(context, serviceClass)
        @Volatile private var remote: Messenger? = null
        private var bindRequested = false
        private var activeConnection: ServiceConnection? = null

        fun prebind() {
            callbackExecutor.execute {
                try {
                    connectedMessenger()
                } catch (error: Throwable) {
                    Log.w(TAG, "strategy_worker_prebind_failed", error)
                }
            }
        }

        fun send(requestId: Long, taskName: String, payload: String) {
            val target = connectedMessenger()
            val message = Message.obtain(null, StrategyWorkerProtocol.EXECUTE).apply {
                replyTo = replyMessenger
                data = Bundle().apply {
                    putLong(StrategyWorkerProtocol.REQUEST_ID, requestId)
                    putString(StrategyWorkerProtocol.TASK_NAME, taskName)
                    putString(StrategyWorkerProtocol.PAYLOAD, payload)
                }
            }
            try {
                target.send(message)
            } catch (error: Throwable) {
                markDisconnected()
                failPending(this, IllegalStateException("strategy_worker_send_failed", error))
                throw IllegalStateException("strategy_worker_send_failed", error)
            }
        }

        fun cancelAll() {
            val target = remote ?: return
            try {
                target.send(Message.obtain(null, StrategyWorkerProtocol.CANCEL_ALL))
            } catch (error: Throwable) {
                Log.d(TAG, "strategy_worker_already_gone", error)
                resetBinding()
            }
        }

        fun restartProcess() {
            val target = remote ?: return
            try {
                target.send(Message.obtain(null, StrategyWorkerProtocol.RESTART_PROCESS))
            } catch (error: Throwable) {
                Log.d(TAG, "strategy_worker_already_gone", error)
            }
        }

        private fun connectedMessenger(): Messenger {
            remote?.let { return it }
            val deadline = System.currentTimeMillis() + BIND_TIMEOUT_MS
            synchronized(monitor) {
                remote?.let { return it }
                if (!bindRequested) {
                    val connection = newConnection()
                    activeConnection = connection
                    bindRequested = true
                    val accepted = context.bindService(
                        intent,
                        Context.BIND_AUTO_CREATE,
                        callbackExecutor,
                        connection,
                    )
                    if (!accepted) {
                        activeConnection = null
                        bindRequested = false
                        throw IllegalStateException("strategy_worker_bind_rejected")
                    }
                }
                while (remote == null && bindRequested) {
                    val remaining = deadline - System.currentTimeMillis()
                    if (remaining <= 0) break
                    monitor.wait(remaining)
                }
                if (remote == null) bindRequested = false
                return remote ?: throw IllegalStateException("strategy_worker_bind_timeout")
            }
        }

        private fun markDisconnected() {
            synchronized(monitor) {
                remote = null
                monitor.notifyAll()
            }
        }

        private fun newConnection(): ServiceConnection = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
                synchronized(monitor) {
                    if (activeConnection !== this) return
                    remote = service?.let(::Messenger)
                    monitor.notifyAll()
                }
            }

            override fun onServiceDisconnected(name: ComponentName?) {
                disconnected(rebindExpected = true, "strategy_worker_disconnected")
            }

            override fun onBindingDied(name: ComponentName?) {
                disconnected(rebindExpected = false, "strategy_worker_binding_died")
                resetBinding(this)
            }

            override fun onNullBinding(name: ComponentName?) {
                disconnected(rebindExpected = false, "strategy_worker_null_binding")
                resetBinding(this)
            }

            private fun disconnected(rebindExpected: Boolean, reason: String) {
                val active = synchronized(monitor) {
                    if (activeConnection !== this) return@synchronized false
                    remote = null
                    bindRequested = rebindExpected
                    monitor.notifyAll()
                    true
                }
                if (active) failPending(this@WorkerClient, IllegalStateException(reason))
            }
        }

        private fun resetBinding(expected: ServiceConnection? = null) {
            val connection = synchronized(monitor) {
                if (expected != null && activeConnection !== expected) return@synchronized null
                val current = activeConnection
                val active = bindRequested
                remote = null
                bindRequested = false
                activeConnection = null
                monitor.notifyAll()
                current.takeIf { active }
            }
            if (connection != null) {
                try {
                    context.unbindService(connection)
                } catch (error: IllegalArgumentException) {
                    Log.d(TAG, "strategy_worker_already_unbound", error)
                }
            }
        }
    }
}
