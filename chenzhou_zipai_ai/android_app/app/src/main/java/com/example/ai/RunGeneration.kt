package com.example.ai

import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

class RunGeneration {
    private val generation = AtomicLong(0)
    private val running = AtomicBoolean(false)

    fun start(): Long {
        if (!running.compareAndSet(false, true)) return generation.get()
        return generation.incrementAndGet()
    }

    fun stop(): Long {
        running.set(false)
        return generation.incrementAndGet()
    }

    fun isCurrent(token: Long): Boolean = running.get() && generation.get() == token

    fun isRunning(): Boolean = running.get()
}
