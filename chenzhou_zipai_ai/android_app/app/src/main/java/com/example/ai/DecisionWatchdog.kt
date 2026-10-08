package com.example.ai

import java.util.concurrent.Executors
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

class DecisionWatchdog(
    private val timeoutMs: Long,
    private val onTimeout: (Long) -> Unit,
) : AutoCloseable {
    private val sequence = AtomicLong(0)
    private val activeDecision = AtomicLong(0)
    private val scheduler = Executors.newSingleThreadScheduledExecutor { runnable ->
        Thread(runnable, "aizipai-watchdog").apply { isDaemon = true }
    }

    class Ticket internal constructor(
        internal val id: Long,
        internal val future: ScheduledFuture<*>,
    )

    fun arm(runToken: Long): Ticket {
        val id = sequence.incrementAndGet()
        activeDecision.set(id)
        val future = scheduler.schedule(
            {
                if (activeDecision.compareAndSet(id, 0)) onTimeout(runToken)
            },
            timeoutMs,
            TimeUnit.MILLISECONDS,
        )
        return Ticket(id, future)
    }

    fun disarm(ticket: Ticket) {
        activeDecision.compareAndSet(ticket.id, 0)
        ticket.future.cancel(false)
    }

    fun cancel() {
        activeDecision.set(0)
    }

    override fun close() {
        cancel()
        scheduler.shutdownNow()
    }
}
