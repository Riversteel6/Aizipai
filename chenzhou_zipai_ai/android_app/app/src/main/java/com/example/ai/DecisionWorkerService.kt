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
import java.io.File
import java.util.concurrent.Executors
import org.json.JSONObject

class DecisionWorkerService : Service() {
    private companion object {
        const val TAG = "AizipaiDecisionWorker"
    }

    private val executor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "aizipai-decision-python").apply { isDaemon = true }
    }
    private lateinit var bridge: LocalPythonDecisionBridge
    private val messenger by lazy {
        Messenger(
            Handler(Looper.getMainLooper()) { message ->
                when (message.what) {
                    DecisionWorkerProtocol.EXECUTE -> {
                        execute(message)
                        true
                    }
                    DecisionWorkerProtocol.RESTART -> {
                        Log.e(TAG, "decision_worker_restart pid=${Process.myPid()}")
                        StrategyProcessPool.restartWorkers()
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
        StrategyProcessPool.initialize(applicationContext)
        if (!Python.isStarted()) Python.start(AndroidPlatform(this))
        bridge = LocalPythonDecisionBridge(this)
        Log.i(TAG, "decision_worker_started pid=${Process.myPid()}")
    }

    override fun onBind(intent: Intent?): IBinder = messenger.binder

    private fun execute(message: Message) {
        val replyTo = message.replyTo ?: return
        val requestId = message.data.getLong(DecisionWorkerProtocol.REQUEST_ID, -1L)
        val operation = message.data.getString(DecisionWorkerProtocol.OPERATION).orEmpty()
        val arguments = Bundle(message.data)
        Log.i(TAG, "decision_request_received id=$requestId operation=$operation")
        executor.execute {
            val startedAt = android.os.SystemClock.elapsedRealtime()
            Log.i(TAG, "decision_request_started id=$requestId operation=$operation")
            try {
                val payload = executeOperation(operation, arguments)
                Log.i(
                    TAG,
                    "decision_request_finished id=$requestId operation=$operation " +
                        "elapsed_ms=${android.os.SystemClock.elapsedRealtime() - startedAt}",
                )
                sendResult(replyTo, requestId, payload, null)
            } catch (error: Throwable) {
                Log.e(TAG, "decision_worker_call_failed id=$requestId operation=$operation", error)
                sendResult(replyTo, requestId, null, error.stackTraceToString())
            }
        }
    }

    private fun executeOperation(operation: String, arguments: Bundle): String = when {
        operation.startsWith("prewarm_profile_") -> Python.getInstance()
            .getModule("mobile_bridge")
            .callAttr(
                "prewarm_runtime_profile_json",
                filesDir.absolutePath,
                operation.removePrefix("prewarm_profile_"),
            )
            .toString()
        operation == "prewarm" -> bridge.prewarm().toString()
        operation == "diagnostic_production" -> Python.getInstance()
            .getModule("mobile_bridge")
            .callAttr(
                "current_frame_memory_production_diagnostics_json",
                File(cacheDir, "aizipai_current_opening.png").absolutePath,
                filesDir.absolutePath,
            )
            .toString()
        operation == "diagnostic_vision" -> Python.getInstance()
            .getModule("mobile_bridge")
            .callAttr(
                "current_frame_vision_diagnostics_json",
                File(cacheDir, "aizipai_current_opening.png").absolutePath,
            )
            .toString()
        operation == "warm_strategy_workers" -> Python.getInstance()
            .getModule("mobile_bridge")
            .callAttr("worker_pool_diagnostics_json")
            .toString()
        operation == "strategy_timing" -> Python.getInstance()
            .getModule("mobile_bridge")
            .callAttr("latest_strategy_timing_json")
            .toString()
        operation == "reset_transient" -> JSONObject()
            .put("value", bridge.resetTransientGuard(mode(arguments)))
            .toString()
        operation == "decide" -> bridge.decideRaw(frame(arguments), mode(arguments))
        operation == "probe" -> bridge.probe(frame(arguments)).let { result ->
            JSONObject()
                .put("actionable", result.actionable)
                .put("summary", result.summary)
                .put("elapsed_ms", result.elapsedMs)
                .toString()
        }
        operation == "decide_path" -> bridge.decideRaw(
                arguments.getString(DecisionWorkerProtocol.IMAGE_PATH).orEmpty(),
                mode(arguments),
                arguments.getInt(DecisionWorkerProtocol.WIDTH),
                arguments.getInt(DecisionWorkerProtocol.HEIGHT),
            )
        operation == "is_fresh" -> JSONObject()
            .put(
                "value",
                bridge.isFresh(
                    frame(arguments),
                    DecisionPlan("remote", "tap_sequence", emptyList(), signature(arguments)),
                ),
            )
            .toString()
        operation == "verify_before" -> bridge.verifyBeforeExecution(
            frame(arguments),
            DecisionPlan("remote", "tap_sequence", emptyList(), signature(arguments)),
        ).let { result ->
            val payload = JSONObject()
                .put("status", result.status.name.lowercase())
                .put("reason", result.reason)
                .put("elapsed_ms", result.elapsedMs)
            result.plan?.let { plan ->
                payload.put("plan", decisionPlanJson(plan))
            }
            payload.toString()
        }
        operation == "confirm_result" -> bridge.observeActionResult(
            frame(arguments),
            DecisionPlan("remote", "tap_sequence", emptyList(), signature(arguments)),
        ).let { result ->
            JSONObject()
                .put("status", result.status.name.lowercase())
                .put("reason", result.reason)
                .put("elapsed_ms", result.elapsedMs)
                .put("strong_transition", result.strongTransition)
                .toString()
        }
        operation == "prepare" -> bridge.prepareExecution(
            DecisionPlan("remote", "tap_sequence", emptyList(), signature(arguments)),
            mode(arguments),
        ).let { JSONObject().put("armed", it.armed).put("reason", it.reason).toString() }
        operation == "commit" -> JSONObject()
            .put(
                "value",
                bridge.commitExecuted(
                    DecisionPlan("remote", "tap_sequence", emptyList(), signature(arguments)),
                    mode(arguments),
                ),
            )
            .toString()
        operation == "abort" -> JSONObject()
            .put(
                "value",
                bridge.abortExecution(
                    DecisionPlan("remote", "tap_sequence", emptyList(), signature(arguments)),
                    mode(arguments),
                ),
            )
            .toString()
        else -> throw IllegalArgumentException("unsupported_decision_operation:$operation")
    }

    private fun frame(arguments: Bundle): LosslessFrame {
        val file = File(arguments.getString(DecisionWorkerProtocol.FRAME_PATH).orEmpty())
        val pixels = file.readBytes()
        val width = arguments.getInt(DecisionWorkerProtocol.WIDTH)
        val height = arguments.getInt(DecisionWorkerProtocol.HEIGHT)
        val rowBytes = arguments.getInt(DecisionWorkerProtocol.ROW_BYTES)
        require(pixels.size == Math.multiplyExact(rowBytes, height)) { "invalid_shared_frame_size" }
        return LosslessFrame(pixels, width, height, rowBytes)
    }

    private fun mode(arguments: Bundle): GameMode =
        if (arguments.getBoolean(DecisionWorkerProtocol.WILDCARD_ENABLED)) GameMode.WANG else GameMode.NO_WANG

    private fun signature(arguments: Bundle): String =
        arguments.getString(DecisionWorkerProtocol.SIGNATURE).orEmpty()

    private fun decisionPlanJson(plan: DecisionPlan): JSONObject = JSONObject()
        .put("action", plan.action)
        .put("ready", true)
        .put("execution_mode", plan.executionMode)
        .put("reason", "verified_current_frame")
        .put("validation", JSONObject().put("passed", true))
        .put(
            "clicks",
            org.json.JSONArray().apply {
                plan.steps.forEach { step ->
                    put(
                        JSONObject()
                            .put("target", step.target)
                            .put("x", step.x)
                            .put("y", step.y)
                            .put("delay_ms", step.delayMs)
                            .put("duration_ms", step.durationMs)
                            .also { click ->
                                step.fromX?.let { click.put("from_x", it) }
                                step.fromY?.let { click.put("from_y", it) }
                            },
                    )
                }
            },
        )
        .put("_mobile_signature", plan.signature)

    private fun sendResult(replyTo: Messenger, requestId: Long, payload: String?, error: String?) {
        val response = Message.obtain(null, DecisionWorkerProtocol.RESULT).apply {
            data = Bundle().apply {
                putLong(DecisionWorkerProtocol.REQUEST_ID, requestId)
                putBoolean(DecisionWorkerProtocol.OK, error == null)
                if (payload != null) putString(DecisionWorkerProtocol.PAYLOAD, payload)
                if (error != null) putString(DecisionWorkerProtocol.ERROR, error)
            }
        }
        runCatching { replyTo.send(response) }
            .onFailure { Log.w(TAG, "decision_result_delivery_failed id=$requestId", it) }
    }

    override fun onDestroy() {
        executor.shutdownNow()
        super.onDestroy()
    }
}
