package com.example.ai

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.content.Intent
import android.os.Bundle
import android.util.Log
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.chaquo.python.Python
import java.io.File
import java.io.FileOutputStream
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class MobilePythonSmokeTest {
    @Test
    fun decisionWorkerVisionStaysResponsiveAcrossFrames() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        assumeTrue(
            "current opening frame is not installed",
            File(context.cacheDir, "aizipai_current_opening.png").isFile,
        )
        DecisionProcessClient(context).use { client ->
            repeat(5) { index ->
                val result = JSONObject(
                    client.call("diagnostic_vision", timeoutMs = 8_000L),
                )
                Log.i("AizipaiBenchmark", "vision_${index + 1}=$result")
                assertEquals(result.toString(), 21, result.getInt("hand_count"))
            }
        }
    }

    @Test
    fun decisionWorkerSingleProductionWithoutPriorCall() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        assumeTrue(
            "current opening frame is not installed",
            File(context.cacheDir, "aizipai_current_opening.png").isFile,
        )
        DecisionProcessClient(context).use { client ->
            val result = JSONObject(
                client.call("diagnostic_production", timeoutMs = 15_000L),
            )
            Log.i("AizipaiBenchmark", "single_remote_production=$result")
            assertTrue(result.toString(), result.getDouble("elapsed_ms") < 5_000.0)
            assertTrue(result.toString(), result.getDouble("inner_elapsed_ms") < 5_000.0)
        }
    }

    @Test
    fun decisionWorkerTwoProductionCallsStayWarm() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        assumeTrue(
            "current opening frame is not installed",
            File(context.cacheDir, "aizipai_current_opening.png").isFile,
        )
        DecisionProcessClient(context).use { client ->
            repeat(2) { index ->
                val result = JSONObject(
                    client.call("diagnostic_production", timeoutMs = 15_000L),
                )
                Log.i("AizipaiBenchmark", "remote_production_${index + 1}=$result")
                assertTrue(result.toString(), result.getDouble("elapsed_ms") < 5_000.0)
                assertTrue(result.toString(), result.getDouble("inner_elapsed_ms") < 5_000.0)
            }
        }
    }

    @Test
    fun decisionWorkerSinglePrewarmProfileThenProduction() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        assumeTrue(
            "current opening frame is not installed",
            File(context.cacheDir, "aizipai_current_opening.png").isFile,
        )
        val profile = InstrumentationRegistry.getArguments()
            .getString("prewarm_profile", "rules")
        DecisionProcessClient(context).use { client ->
            val warm = JSONObject(
                client.call("prewarm_profile_$profile", timeoutMs = 10_000L),
            )
            val result = JSONObject(
                client.call("diagnostic_production", timeoutMs = 15_000L),
            )
            Log.i("AizipaiBenchmark", "single_prewarm=$warm production=$result")
            assertTrue(result.toString(), result.getDouble("elapsed_ms") < 5_000.0)
            assertTrue(result.toString(), result.getDouble("inner_elapsed_ms") < 5_000.0)
        }
    }

    @Test
    fun decisionWorkerProductionReturnsQuickly() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        assumeTrue(
            "current opening frame is not installed",
            File(context.cacheDir, "aizipai_current_opening.png").isFile,
        )
        DecisionProcessClient(context).use { client ->
            assertTrue(JSONObject(client.call("prewarm", timeoutMs = 30_000L)).getBoolean("ready"))
            val result = JSONObject(
                client.call("diagnostic_production", timeoutMs = 15_000L),
            )
            Log.i("AizipaiBenchmark", "decision_worker_production=$result")
            assertTrue(result.toString(), result.getDouble("elapsed_ms") < 5_000.0)
            assertTrue(result.toString(), result.getDouble("inner_elapsed_ms") < 5_000.0)
        }
    }

    @Test
    fun reproducedSingleDiscardEvaluationReturnsQuickly() {
        val index = InstrumentationRegistry.getArguments()
            .getString("probe_index", "1")
            .toInt()
        val result = JSONObject(
            Python.getInstance().getModule("mobile_bridge")
                .callAttr("single_discard_evaluation_diagnostics_json", index)
                .toString(),
        )
        Log.i("AizipaiBenchmark", "single_discard=$result")
        assertEquals(index, result.getInt("candidate_index"))
        assertTrue(result.toString(), result.getDouble("elapsed_ms") < 5_000.0)
    }

    @Test
    fun currentFrameExactInnerCandidateReturnsQuickly() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val source = File(context.cacheDir, "aizipai_current_opening.png")
        assumeTrue("current opening frame is not installed", source.isFile)
        val index = InstrumentationRegistry.getArguments()
            .getString("probe_index", "18")
            .toInt()
        val result = JSONObject(
            Python.getInstance().getModule("mobile_bridge")
                .callAttr(
                    "current_frame_inner_candidate_diagnostics_json",
                    source.absolutePath,
                    context.filesDir.absolutePath,
                    index,
                )
                .toString(),
        )
        Log.i("AizipaiBenchmark", "exact_inner_candidate=$result")
        assertEquals(index, result.getInt("candidate_index"))
        assertTrue(result.toString(), result.getDouble("elapsed_ms") < 5_000.0)
    }

    @Test
    fun reproducedInnerProductionCompletesWithoutOuterDecision() {
        val result = JSONObject(
            Python.getInstance().getModule("mobile_bridge")
                .callAttr("current_inner_production_diagnostics_json")
                .toString(),
        )
        Log.i("AizipaiBenchmark", "inner_only=$result")
        assertEquals("DISCARD", result.getString("selected_action"))
        assertTrue(result.toString(), result.getDouble("elapsed_ms") < 5_000.0)
    }

    @Test
    fun sequentialOuterLikeAndInnerProductionBothComplete() {
        val result = JSONObject(
            Python.getInstance().getModule("mobile_bridge")
                .callAttr("sequential_production_diagnostics_json")
                .toString(),
        )
        Log.i("AizipaiBenchmark", "sequential_production=$result")
        assertEquals("拾", result.getString("outer_label"))
        assertEquals("拾", result.getString("inner_label"))
        assertTrue(result.toString(), result.getDouble("outer_ms") < 5_000.0)
        assertTrue(result.toString(), result.getDouble("inner_ms") < 5_000.0)
    }

    @Test
    fun frozenPolicyConstructionDoesNotBlockInnerProduction() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        StrategyProcessPool.initialize(context)
        val result = JSONObject(
            Python.getInstance().getModule("mobile_bridge")
                .callAttr("policy_initialized_inner_diagnostics_json")
                .toString(),
        )
        Log.i("AizipaiBenchmark", "policy_then_inner=$result")
        assertEquals("拾", result.getString("selected_label"))
        assertTrue(result.toString(), result.getDouble("decision_ms") < 5_000.0)
    }

    @Test
    fun currentFrameTwoProductionEvaluationsCompleteWithoutFrozenSearch() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val source = File(context.cacheDir, "aizipai_current_opening.png")
        assumeTrue("current opening frame is not installed", source.isFile)
        val result = JSONObject(
            Python.getInstance().getModule("mobile_bridge")
                .callAttr("current_frame_production_pair_diagnostics_json", source.absolutePath)
                .toString(),
        )
        Log.i("AizipaiBenchmark", "current_frame_production_pair=$result")
        assertEquals(21, result.getInt("hand_count"))
        assertEquals("拾", result.getString("outer_label"))
        assertEquals("拾", result.getString("inner_label"))
        assertTrue(result.toString(), result.getDouble("outer_ms") < 5_000.0)
        assertTrue(result.toString(), result.getDouble("inner_ms") < 5_000.0)
    }

    @Test
    fun isolatedDecisionWorkerCanRestartWithoutKillingTheClientProcess() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val client = DecisionProcessClient(context)
        try {
            assertTrue(JSONObject(client.call("prewarm", timeoutMs = 30_000L)).getBoolean("ready"))
            client.restart("instrumented_test")
            Thread.sleep(500)
            assertTrue(JSONObject(client.call("prewarm", timeoutMs = 30_000L)).getBoolean("ready"))
        } finally {
            client.close()
        }
    }

    private fun optionalBenchmarkFrame(context: android.content.Context): File {
        val cached = File(context.cacheDir, "aizipai_lossless_benchmark.png")
        return if (cached.isFile) cached else File("/data/local/tmp/aizipai_lossless_benchmark.png")
    }

    private fun keepTargetForeground(context: android.content.Context) {
        context.startActivity(
            Intent(context, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
        )
        InstrumentationRegistry.getInstrumentation().waitForIdleSync()
    }

    @Test
    fun profileLosslessVisionOnOptionalHistoricalFrame() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        keepTargetForeground(context)
        assertTrue(PythonDecisionBridge(context).prewarm().getBoolean("ready"))
        val source = optionalBenchmarkFrame(context)
        assumeTrue("optional lossless benchmark frame is not installed", source.isFile)
        val bitmap = BitmapFactory.decodeFile(source.absolutePath)
        assertNotNull(bitmap)
        val frame = LosslessFrameFactory.fromBitmap(bitmap)
        bitmap.recycle()
        val profile = JSONObject(
            Python.getInstance().getModule("mobile_bridge")
                .callAttr(
                    "profile_rgba_vision_json",
                    frame.pixels,
                    frame.width,
                    frame.height,
                    frame.rowBytes,
                    context.cacheDir.absolutePath,
                )
                .toString(),
        )
        assertEquals(20, profile.getInt("hand_count"))
        assertTrue(profile.getJSONArray("buttons").toString().contains("pass"))
        Log.i("AizipaiBenchmark", "vision=$profile")
        File(context.cacheDir, "aizipai_vision_profile.json").writeText(profile.toString())
    }

    @Test
    fun profileProductizationVisionOnExternalFrame() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        keepTargetForeground(context)
        assertTrue(PythonDecisionBridge(context).prewarm().getBoolean("ready"))
        val source = File(
            InstrumentationRegistry.getArguments().getString(
                "frame_path",
                "/data/local/tmp/aizipai_productization_frame.png",
            ),
        )
        assumeTrue("productization benchmark frame is not installed", source.isFile)
        val bitmap = BitmapFactory.decodeFile(source.absolutePath)
        assertNotNull(bitmap)
        val frame = LosslessFrameFactory.fromBitmap(bitmap)
        bitmap.recycle()
        val profile = JSONObject(
            Python.getInstance().getModule("mobile_bridge")
                .callAttr(
                    "profile_rgba_vision_json",
                    frame.pixels,
                    frame.width,
                    frame.height,
                    frame.rowBytes,
                    context.cacheDir.absolutePath,
                )
                .toString(),
        )
        Log.i("AizipaiBenchmark", "productization_vision=$profile")
        assertEquals(21, profile.getInt("hand_count"))
    }

    @Test
    fun benchmarkLosslessV81OnOptionalHistoricalFrame() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        keepTargetForeground(context)
        val source = optionalBenchmarkFrame(context)
        assumeTrue("optional lossless benchmark frame is not installed", source.isFile)
        val bitmap = BitmapFactory.decodeFile(source.absolutePath)
        assertNotNull(bitmap)
        val copyStarted = System.nanoTime()
        val frame = LosslessFrameFactory.fromBitmap(bitmap)
        val copyMs = (System.nanoTime() - copyStarted) / 1_000_000
        bitmap.recycle()

        val bridge = PythonDecisionBridge(context)
        val coldStarted = System.nanoTime()
        val cold = bridge.decide(frame, GameMode.WANG)
        val coldMs = (System.nanoTime() - coldStarted) / 1_000_000
        assertNotNull(cold)
        assertEquals("pass", cold?.action)
        assertEquals("button:pass", cold?.plan?.steps?.single()?.target)

        val warmStarted = System.nanoTime()
        val warm = bridge.decide(frame, GameMode.WANG)
        val warmMs = (System.nanoTime() - warmStarted) / 1_000_000
        assertEquals(cold?.action, warm?.action)
        assertEquals(
            cold?.plan?.steps?.map { it.target },
            warm?.plan?.steps?.map { it.target },
        )

        val freshStarted = System.nanoTime()
        val freshness = JSONObject(
            Python.getInstance().getModule("mobile_bridge")
                .callAttr(
                    "validate_fresh_rgba_json",
                    frame.pixels,
                    frame.width,
                    frame.height,
                    frame.rowBytes,
                    cold!!.plan!!.signature,
                    context.filesDir.absolutePath,
                )
                .toString(),
        )
        val freshMs = (System.nanoTime() - freshStarted) / 1_000_000
        val result = JSONObject()
                .put("frame_copy_ms", copyMs)
                .put("cold_decision_ms", coldMs)
                .put("warm_decision_ms", warmMs)
                .put("freshness_ms", freshMs)
                .put("action", cold.action)
                .put("target", cold.plan!!.steps.single().target)
                .put("freshness", freshness)
                .put("process_pool", JSONObject(StrategyProcessPool.metricsJson()))
        Log.i("AizipaiBenchmark", "decision=$result")
        File(context.cacheDir, "aizipai_lossless_benchmark_result.json").writeText(result.toString())
        assertTrue(freshness.toString(), freshness.getBoolean("fresh"))
    }

    @Test
    fun prewarmMovesFirstHistoricalDecisionOntoWarmPath() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        keepTargetForeground(context)
        val source = optionalBenchmarkFrame(context)
        assumeTrue("optional lossless benchmark frame is not installed", source.isFile)
        val bitmap = BitmapFactory.decodeFile(source.absolutePath)
        assertNotNull(bitmap)
        val frame = LosslessFrameFactory.fromBitmap(bitmap)
        bitmap.recycle()
        val bridge = PythonDecisionBridge(context)

        val prewarmStarted = System.nanoTime()
        val prewarm = bridge.prewarm()
        val prewarmMs = (System.nanoTime() - prewarmStarted) / 1_000_000
        assertTrue(prewarm.toString(), prewarm.getBoolean("ready"))

        val decisionStarted = System.nanoTime()
        val decision = bridge.decide(frame, GameMode.WANG)
        val decisionMs = (System.nanoTime() - decisionStarted) / 1_000_000
        assertEquals("pass", decision?.action)
        assertEquals("button:pass", decision?.plan?.steps?.single()?.target)
        val steadySamples = mutableListOf<Long>()
        repeat(10) {
            val steadyStarted = System.nanoTime()
            val steady = bridge.decide(frame, GameMode.WANG)
            steadySamples += (System.nanoTime() - steadyStarted) / 1_000_000
            assertEquals(decision?.action, steady?.action)
            assertEquals(
                decision?.plan?.steps?.map { it.target },
                steady?.plan?.steps?.map { it.target },
            )
        }
        val sortedSamples = steadySamples.sorted()
        val p95Index = ((sortedSamples.size * 95 + 99) / 100 - 1)
            .coerceIn(0, sortedSamples.lastIndex)
        val steadyP95Ms = sortedSamples[p95Index]
        Log.i(
            "AizipaiBenchmark",
            "prewarm=${JSONObject().put("prewarm_ms", prewarmMs).put("first_decision_ms", decisionMs).put("steady_samples_ms", steadySamples).put("steady_p95_ms", steadyP95Ms).put("result", prewarm)}",
        )
        assertTrue("prewarmed initial full recognition took ${decisionMs}ms", decisionMs <= 4_000)
        assertTrue(
            "steady decision P95 took ${steadyP95Ms}ms samples=$steadySamples",
            steadyP95Ms <= 2_500,
        )
    }

    @Test
    fun benchmarkCachedOpeningDiscardOnTargetDevice() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        keepTargetForeground(context)
        val source = File(context.cacheDir, "aizipai_current_opening.png")
        assumeTrue("cached opening frame is not installed", source.isFile)
        val bitmap = BitmapFactory.decodeFile(source.absolutePath)
        assertNotNull(bitmap)
        val frame = LosslessFrameFactory.fromBitmap(bitmap)
        bitmap.recycle()
        val bridge = PythonDecisionBridge(context)

        val prewarmStarted = System.nanoTime()
        val prewarm = bridge.prewarm()
        val prewarmMs = (System.nanoTime() - prewarmStarted) / 1_000_000
        assertTrue(prewarm.toString(), prewarm.getBoolean("ready"))

        StrategyProcessPool.resetMetrics()
        val decisionStarted = System.nanoTime()
        val decision = bridge.decide(frame, GameMode.WANG)
        val decisionMs = (System.nanoTime() - decisionStarted) / 1_000_000
        assertEquals("discard", decision?.action)
        val python = Python.getInstance()
        val frozenModule = python.getModule("ai.frozen_two_player_strategy")
        val policy = frozenModule.callAttr("_frozen_policy")
        val jsonModule = python.getModule("json")
        val diagnostics = JSONObject(
            jsonModule.callAttr("dumps", policy.callAttr("diagnostics")).toString(),
        )
        val discardEvents = JSONArray(
            jsonModule.callAttr("dumps", policy.callAttr("discard_events")).toString(),
        )
        val result = JSONObject()
            .put("prewarm_ms", prewarmMs)
            .put("decision_ms", decisionMs)
            .put("action", decision?.action)
            .put("targets", decision?.plan?.steps?.map { it.target })
            .put("process_pool", JSONObject(StrategyProcessPool.metricsJson()))
            .put("strategy_diagnostics", diagnostics)
            .put(
                "last_discard_event",
                if (discardEvents.length() == 0) JSONObject.NULL
                else discardEvents.getJSONObject(discardEvents.length() - 1),
            )
        Log.i("AizipaiBenchmark", "opening_discard=$result")
        File(context.cacheDir, "aizipai_opening_discard_benchmark.json")
            .writeText(result.toString())
    }

    @Test
    fun benchmarkProductionDecisionOnExternalFrame() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        keepTargetForeground(context)
        val source = File(
            InstrumentationRegistry.getArguments().getString(
                "frame_path",
                "/data/local/tmp/aizipai_productization_frame.png",
            ),
        )
        assumeTrue("productization benchmark frame is not installed", source.isFile)
        val bitmap = BitmapFactory.decodeFile(source.absolutePath)
        assertNotNull(bitmap)
        val width = bitmap.width
        val height = bitmap.height
        bitmap.recycle()
        val sampleCount = InstrumentationRegistry.getArguments()
            .getString("sample_count", "3")
            .toInt()
            .coerceIn(1, 10)
        val samples = JSONArray()
        var failures = 0

        DecisionProcessClient(context).use { client ->
            repeat(sampleCount) { index ->
                // This benchmark measures the strategy, not the persisted answer cache.
                File(context.filesDir, "vision_memory_1v1-wang.json").delete()
                client.restart("productization_cold_sample_${index + 1}")
                Thread.sleep(750)
                val prewarmStarted = System.nanoTime()
                val sample = JSONObject().put("index", index + 1)
                try {
                    val prewarm = JSONObject(
                        client.call("prewarm", timeoutMs = 45_000L),
                    )
                    sample.put(
                        "prewarm_ms",
                        (System.nanoTime() - prewarmStarted) / 1_000_000,
                    )
                    sample.put("prewarm_ready", prewarm.getBoolean("ready"))
                    val arguments = Bundle().apply {
                        putString(DecisionWorkerProtocol.IMAGE_PATH, source.absolutePath)
                        putInt(DecisionWorkerProtocol.WIDTH, width)
                        putInt(DecisionWorkerProtocol.HEIGHT, height)
                        putBoolean(DecisionWorkerProtocol.WILDCARD_ENABLED, true)
                    }
                    val decisionStarted = System.nanoTime()
                    val raw = client.call(
                        "decide_path",
                        arguments,
                        timeoutMs = 25_000L,
                    )
                    val decisionMs = (System.nanoTime() - decisionStarted) / 1_000_000
                    val outcome = DecisionOutcome.parse(raw, width, height)
                    sample.put("decision_ms", decisionMs)
                    sample.put("action", outcome.action)
                    sample.put("reason", outcome.reason)
                    sample.put(
                        "targets",
                        JSONArray(outcome.plan?.steps?.map { it.target } ?: emptyList<String>()),
                    )
                    sample.put(
                        "strategy_timing",
                        JSONObject(client.call("strategy_timing", timeoutMs = 5_000L)),
                    )
                    sample.put("ok", true)
                } catch (error: Throwable) {
                    failures += 1
                    sample.put("ok", false)
                    sample.put("error", error.stackTraceToString())
                }
                samples.put(sample)
                if (sampleCount == 1) {
                    Log.i("AizipaiBenchmark", "productization_sample=$sample")
                }
                Log.i(
                    "AizipaiBenchmark",
                    "productization_sample_compact=" + JSONObject().apply {
                        put("index", sample.optInt("index", -1))
                        put("decision_ms", sample.optLong("decision_ms", -1L))
                        put("action", sample.optString("action", ""))
                        put("targets", sample.optJSONArray("targets") ?: JSONArray())
                        put("ok", sample.optBoolean("ok", false))
                        sample.optJSONObject("strategy_timing")?.let { strategy ->
                            put("strategy_ms", strategy.optDouble("decision_elapsed_ms", -1.0))
                            put("simulations", strategy.optInt("decision_simulations", -1))
                            put("production_label", strategy.optString("production_label", ""))
                            put("selected_label", strategy.optString("selected_label", ""))
                            put("refinement_ms", strategy.optJSONObject("refinement")?.optDouble("elapsed_ms", -1.0))
                            put("confirmation_ms", strategy.optJSONObject("confirmation")?.optDouble("elapsed_ms", -1.0))
                        }
                    },
                )
            }
        }

        val successfulMs = (0 until samples.length())
            .map { samples.getJSONObject(it) }
            .filter { it.optBoolean("ok", false) }
            .map { it.getLong("decision_ms") }
            .sorted()
        fun percentile(percent: Int): Long? {
            if (successfulMs.isEmpty()) return null
            val index = ((successfulMs.size * percent + 99) / 100 - 1)
                .coerceIn(0, successfulMs.lastIndex)
            return successfulMs[index]
        }
        val result = JSONObject()
            .put("sample_count", sampleCount)
            .put("failure_count", failures)
            .put("samples", samples)
            .put("p50_ms", percentile(50) ?: JSONObject.NULL)
            .put("p95_ms", percentile(95) ?: JSONObject.NULL)
            .put("max_ms", successfulMs.maxOrNull() ?: JSONObject.NULL)
        val compactSamples = JSONArray()
        for (index in 0 until samples.length()) {
            val sample = samples.getJSONObject(index)
            compactSamples.put(
                JSONObject()
                    .put("index", sample.optInt("index", -1))
                    .put("decision_ms", sample.optLong("decision_ms", -1L))
                    .put("action", sample.optString("action", ""))
                    .put("targets", sample.optJSONArray("targets") ?: JSONArray())
                    .put("ok", sample.optBoolean("ok", false)),
            )
        }
        Log.i(
            "AizipaiBenchmark",
            "productization_result_compact=" + JSONObject()
                .put("sample_count", sampleCount)
                .put("failure_count", failures)
                .put("samples", compactSamples)
                .put("p50_ms", percentile(50) ?: JSONObject.NULL)
                .put("p95_ms", percentile(95) ?: JSONObject.NULL)
                .put("max_ms", successfulMs.maxOrNull() ?: JSONObject.NULL),
        )
        Log.i("AizipaiBenchmark", "productization_result=$result")
        File(context.cacheDir, "aizipai_productization_benchmark.json")
            .writeText(result.toString())
        assertEquals(result.toString(), 0, failures)
    }

    @Test
    fun benchmarkSemanticChiTransactionOnRecordedThreeScreenChain() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        keepTargetForeground(context)
        // Recorded transaction benchmarks must not inherit the last real or
        // benchmark action from another run. The in-game duplicate-click guard
        // remains unchanged; only this isolated fixture state is reset.
        File(context.filesDir, "action_guard_1v1-wang.json").delete()
        File(context.filesDir, "vision_memory_1v1-wang.json").delete()
        val frames = (1..3).map { index ->
            File("/data/local/tmp/aizipai_chi_chain_$index.png")
        }
        frames.forEach { frame ->
            assumeTrue("CHI transaction frame is not installed: $frame", frame.isFile)
        }
        val samples = JSONArray()
        PythonDecisionBridge(context).use { bridge ->
            assertTrue(bridge.prewarm().getBoolean("ready"))
            assertTrue(bridge.resetTransientGuard(GameMode.WANG))
            var pendingPlan: DecisionPlan? = null
            for ((index, frame) in frames.withIndex()) {
                val bitmap = BitmapFactory.decodeFile(frame.absolutePath)
                assertNotNull(bitmap)
                val lossless = LosslessFrameFactory.fromBitmap(bitmap)
                bitmap.recycle()
                pendingPlan?.let { previous ->
                    val confirmation = bridge.observeActionResult(lossless, previous)
                    assertEquals(
                        "screen=${index + 1} previous=${previous.action} " +
                            "signature=${previous.signature} " +
                            "confirmation=$confirmation",
                        ActionResultStatus.CONFIRMED_SIGNAL,
                        confirmation.status,
                    )
                    assertTrue(
                        "screen=${index + 1} confirmation=$confirmation",
                        confirmation.strongTransition,
                    )
                    assertTrue(bridge.commitExecuted(previous, GameMode.WANG))
                }
                val started = System.nanoTime()
                val outcome = bridge.decide(lossless, GameMode.WANG)
                val elapsedMs = (System.nanoTime() - started) / 1_000_000
                val plan = outcome.plan
                assertNotNull(outcome.reason, plan)
                samples.put(
                    JSONObject()
                        .put("screen", index + 1)
                        .put("elapsed_ms", elapsedMs)
                        .put("action", plan?.action)
                        .put("surface_fast_path", plan?.surfaceFastPath)
                        .put("targets", JSONArray(plan?.steps?.map { it.target } ?: emptyList<String>())),
                )
                assertTrue(bridge.prepareExecution(plan!!, GameMode.WANG).armed)
                pendingPlan = plan
            }
        }

        val candidate = samples.getJSONObject(1)
        val compare = samples.getJSONObject(2)
        assertEquals(samples.toString(), "chi_option", candidate.getString("action"))
        assertEquals(samples.toString(), "compare_option", compare.getString("action"))
        assertTrue(samples.toString(), candidate.getBoolean("surface_fast_path"))
        assertTrue(samples.toString(), compare.getBoolean("surface_fast_path"))
        Log.i("AizipaiBenchmark", "semantic_chi_transaction=$samples")
        File(context.cacheDir, "aizipai_semantic_chi_transaction.json")
            .writeText(samples.toString())
    }

    @Test
    fun benchmarkTrustedLedgerChiInitialAgainstFullRecognition() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        keepTargetForeground(context)
        val guardFile = File(context.filesDir, "action_guard_1v1-wang.json")
        val ledgerFile = File(context.filesDir, "confirmed_hand_ledger_1v1-wang.json")
        guardFile.delete()
        ledgerFile.delete()
        File(context.filesDir, "vision_memory_1v1-wang.json").delete()
        val frame = File("/data/local/tmp/aizipai_chi_chain_1.png")
        assumeTrue("CHI initial frame is not installed: $frame", frame.isFile)
        PythonDecisionBridge(context).use { bridge ->
            assertTrue(bridge.prewarm().getBoolean("ready"))
            val bitmap = BitmapFactory.decodeFile(frame.absolutePath)
            assertNotNull(bitmap)
            val lossless = LosslessFrameFactory.fromBitmap(bitmap)
            bitmap.recycle()

            val fullStarted = System.nanoTime()
            val fullPlan = requireNotNull(bridge.decide(lossless, GameMode.WANG).plan)
            val fullMs = (System.nanoTime() - fullStarted) / 1_000_000
            assertEquals("chi", fullPlan.action)

            val signatureRoot = JSONObject(fullPlan.signature)
            val currentContext = signatureRoot.getJSONObject("context")
            val currentSignature = currentContext.getJSONArray("hand_signature")
            assertTrue(currentSignature.length() > 0)
            val targetLabel = currentSignature.getJSONArray(0).getString(0)
            val beforeSignature = JSONArray()
            val currentCounts = JSONObject()
            for (index in 0 until currentSignature.length()) {
                val item = currentSignature.getJSONArray(index)
                val label = item.getString(0)
                val count = item.getInt(1)
                currentCounts.put(label, count)
                beforeSignature.put(
                    JSONArray().put(label).put(count + if (label == targetLabel) 1 else 0),
                )
            }
            val ledgerContext = JSONObject(currentContext.toString())
                .put("hand_signature", beforeSignature)
                .put("planned_hand_target", JSONObject().put("label", targetLabel))
            val now = System.currentTimeMillis() / 1000.0
            guardFile.writeText(
                JSONObject()
                    .put("last_executed_signature", JSONArray().put("discard").put(JSONArray()))
                    .put("last_executed_at", now)
                    .put("last_context", ledgerContext)
                    .toString(),
            )
            ledgerFile.writeText(
                JSONObject()
                    .put("trusted", true)
                    .put("shadow_only", true)
                    .put("last_confirmed_action", "discard")
                    .put("updated_at", now)
                    .put("hand_counts", currentCounts)
                    .toString(),
            )

            val trustedSamples = JSONArray()
            repeat(5) {
                val started = System.nanoTime()
                val plan = requireNotNull(bridge.decide(lossless, GameMode.WANG).plan)
                val elapsedMs = (System.nanoTime() - started) / 1_000_000
                assertEquals(fullPlan.action, plan.action)
                assertEquals(
                    fullPlan.steps.map { it.target },
                    plan.steps.map { it.target },
                )
                trustedSamples.put(elapsedMs)
            }
            Log.i(
                "AizipaiBenchmark",
                "trusted_ledger_chi_initial=" + JSONObject()
                    .put("full_ms", fullMs)
                    .put("trusted_ms", trustedSamples)
                    .put("action", fullPlan.action)
                    .put("targets", JSONArray(fullPlan.steps.map { it.target }))
                    .toString(),
            )
        }
    }

    @Test
    fun benchmarkTrustedLedgerPassAgainstFullRecognition() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        keepTargetForeground(context)
        val guardFile = File(context.filesDir, "action_guard_1v1-wang.json")
        val ledgerFile = File(context.filesDir, "confirmed_hand_ledger_1v1-wang.json")
        guardFile.delete()
        ledgerFile.delete()
        val frame = File("/data/local/tmp/aizipai_pass_frame.png")
        assumeTrue("PASS frame is not installed: $frame", frame.isFile)
        PythonDecisionBridge(context).use { bridge ->
            assertTrue(bridge.prewarm().getBoolean("ready"))
            val bitmap = BitmapFactory.decodeFile(frame.absolutePath)
            assertNotNull(bitmap)
            val lossless = LosslessFrameFactory.fromBitmap(bitmap)
            bitmap.recycle()
            val fullStarted = System.nanoTime()
            val fullPlan = requireNotNull(bridge.decide(lossless, GameMode.WANG).plan)
            val fullMs = (System.nanoTime() - fullStarted) / 1_000_000
            assertEquals("pass", fullPlan.action)

            val contextPayload = JSONObject(fullPlan.signature).getJSONObject("context")
            val currentSignature = contextPayload.getJSONArray("hand_signature")
            val targetLabel = currentSignature.getJSONArray(0).getString(0)
            val beforeSignature = JSONArray()
            val currentCounts = JSONObject()
            for (index in 0 until currentSignature.length()) {
                val item = currentSignature.getJSONArray(index)
                val label = item.getString(0)
                val count = item.getInt(1)
                currentCounts.put(label, count)
                beforeSignature.put(
                    JSONArray().put(label).put(count + if (label == targetLabel) 1 else 0),
                )
            }
            val ledgerContext = JSONObject(contextPayload.toString())
                .put("hand_signature", beforeSignature)
                .put("planned_hand_target", JSONObject().put("label", targetLabel))
            val now = System.currentTimeMillis() / 1000.0
            guardFile.writeText(
                JSONObject()
                    .put("last_executed_signature", JSONArray().put("discard").put(JSONArray()))
                    .put("last_executed_at", now)
                    .put("last_context", ledgerContext)
                    .toString(),
            )
            ledgerFile.writeText(
                JSONObject()
                    .put("trusted", true)
                    .put("shadow_only", true)
                    .put("last_confirmed_action", "discard")
                    .put("updated_at", now)
                    .put("hand_counts", currentCounts)
                    .toString(),
            )

            val trustedSamples = JSONArray()
            repeat(5) {
                val started = System.nanoTime()
                val plan = requireNotNull(bridge.decide(lossless, GameMode.WANG).plan)
                trustedSamples.put((System.nanoTime() - started) / 1_000_000)
                assertEquals(fullPlan.action, plan.action)
                assertEquals(fullPlan.steps.map { it.target }, plan.steps.map { it.target })
            }
            Log.i(
                "AizipaiBenchmark",
                "trusted_ledger_pass=" + JSONObject()
                    .put("full_ms", fullMs)
                    .put("trusted_ms", trustedSamples)
                    .put("action", fullPlan.action)
                    .put("targets", JSONArray(fullPlan.steps.map { it.target }))
                    .toString(),
            )
        }
    }

    @Test
    fun androidBitmapArrivesAsExactLosslessBgrPixels() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val bitmap = Bitmap.createBitmap(3, 2, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(
            intArrayOf(
                Color.rgb(255, 0, 0),
                Color.rgb(0, 255, 0),
                Color.rgb(0, 0, 255),
                Color.rgb(11, 7, 3),
                Color.rgb(129, 128, 127),
                Color.rgb(252, 251, 250),
            ),
            0,
            3,
            0,
            0,
            3,
            2,
        )
        val frame = LosslessFrameFactory.fromBitmap(bitmap)
        bitmap.recycle()

        val probe = JSONObject(
            Python.getInstance().getModule("mobile_bridge")
                .callAttr(
                    "lossless_frame_probe_json",
                    frame.pixels,
                    frame.width,
                    frame.height,
                    frame.rowBytes,
                    context.cacheDir.absolutePath,
                )
                .toString(),
        )

        assertEquals("[2,3,3]", probe.getJSONArray("shape").toString())
        assertEquals(
            "[[0,0,255],[0,255,0],[255,0,0],[3,7,11],[127,128,129],[250,251,252]]",
            probe.getJSONArray("bgr").toString(),
        )
    }

    @Test
    fun androidGestureCommitUsesDesktopChiPendingContract() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val home = File(context.cacheDir, "execution_commit_home").apply {
            deleteRecursively()
            mkdirs()
        }
        val signature = JSONObject()
            .put(
                "plan",
                JSONArray()
                    .put("expand_chi_options")
                    .put(
                        JSONArray().put(
                            JSONArray()
                                .put("button:chi")
                                .put(1960)
                                .put(507),
                        ),
                    ),
            )
            .put(
                "context",
                JSONObject()
                    .put("flow_state", "play")
                    .put("hand_signature", JSONArray().put("三").put("三").put("叁")),
            )
            .toString()

        val module = Python.getInstance().getModule("mobile_bridge")
        val prepared = JSONObject(
            module
                .callAttr(
                    "prepare_external_action_json",
                    signature,
                    home.absolutePath,
                    true,
                )
                .toString(),
        )
        assertTrue(prepared.toString(), prepared.getBoolean("armed"))
        val duplicate = JSONObject(
            module
                .callAttr(
                    "prepare_external_action_json",
                    signature,
                    home.absolutePath,
                    true,
                )
                .toString(),
        )
        assertTrue(duplicate.toString(), !duplicate.getBoolean("armed"))
        assertEquals("duplicate_execution_same_context", duplicate.getString("reason"))

        val result = JSONObject(
            module
                .callAttr(
                    "commit_executed_action_json",
                    signature,
                    home.absolutePath,
                    true,
                )
                .toString(),
        )

        assertTrue(result.toString(), result.getBoolean("committed"))
        val guard = JSONObject(File(home, "action_guard_1v1-wang.json").readText())
        assertEquals("expand_chi_options", guard.getJSONArray("last_executed_signature").getString(0))
        assertEquals("chi", guard.getJSONObject("pending_response").getString("action"))
        assertEquals(1, guard.getJSONObject("pending_response").getInt("attempts"))
        assertTrue(!guard.has("inflight_action"))
    }

    @Test
    fun strategyTasksRunInCpuSizedIndependentProcesses() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        StrategyProcessPool.initialize(context)
        val expectedWorkers = StrategyProcessPool.workerCount()
        val diagnostics = JSONObject(
            Python.getInstance().getModule("mobile_bridge")
                .callAttr("worker_pool_diagnostics_json")
                .toString(),
        )
        assertEquals(4, expectedWorkers)
        assertEquals(expectedWorkers, diagnostics.getInt("worker_count"))
        assertEquals(expectedWorkers, diagnostics.getJSONArray("pids").length())
        assertTrue(diagnostics.getBoolean("all_ready"))
        val parentClock = diagnostics.getDouble("parent_perf_counter")
        val workerClocks = diagnostics.getJSONArray("worker_perf_counters")
        for (index in 0 until workerClocks.length()) {
            assertTrue(
                diagnostics.toString(),
                kotlin.math.abs(parentClock - workerClocks.getDouble(index)) < 5.0,
            )
        }
    }

    @Test
    fun workerDeadlinesUseComparableMonotonicClock() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        StrategyProcessPool.initialize(context)
        val diagnostics = JSONObject(
            Python.getInstance().getModule("mobile_bridge")
                .callAttr("worker_pool_diagnostics_json")
                .toString(),
        )
        val parentClock = diagnostics.getDouble("parent_perf_counter")
        val workerClocks = diagnostics.getJSONArray("worker_perf_counters")
        assertEquals(StrategyProcessPool.workerCount(), workerClocks.length())
        for (index in 0 until workerClocks.length()) {
            assertTrue(
                diagnostics.toString(),
                kotlin.math.abs(parentClock - workerClocks.getDouble(index)) < 5.0,
            )
        }
    }

    @Test
    fun primaryStrategyTaskSkipsQueuedBackgroundValidation() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        StrategyProcessPool.initialize(context)
        val result = JSONObject(
            Python.getInstance().getModule("mobile_bridge")
                .callAttr("worker_pool_priority_diagnostics_json")
                .toString(),
        )
        Log.i("AizipaiBenchmark", "priority=$result")
        assertEquals(
            result.toString(),
            StrategyProcessPool.workerCount() * 2,
            result.getInt("background_tasks"),
        )
        assertTrue(result.toString(), result.getDouble("primary_elapsed_ms") < 1_100.0)
    }

    @Test
    fun currentOpeningFrameReturnsBeforeDecisionTimeout() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val source = File(context.cacheDir, "aizipai_current_opening.png")
        assumeTrue("current opening frame is not installed", source.isFile)
        val bitmap = BitmapFactory.decodeFile(source.absolutePath)
        assertNotNull(bitmap)
        val frame = LosslessFrameFactory.fromBitmap(bitmap)
        bitmap.recycle()
        val scenario = ActivityScenario.launch(MainActivity::class.java)
        val bridge = PythonDecisionBridge(context)
        try {
            assertTrue(bridge.prewarm().getBoolean("ready"))
            StrategyProcessPool.resetMetrics()
            val started = System.nanoTime()
            val decision = bridge.decide(frame, GameMode.WANG)
            val elapsedMs = (System.nanoTime() - started) / 1_000_000
            val result = JSONObject()
                .put("elapsed_ms", elapsedMs)
                .put("disposition", decision.disposition.name)
                .put("action", decision.action)
                .put("target", decision.plan?.steps?.singleOrNull()?.target)
                .put("process_pool", JSONObject(StrategyProcessPool.metricsJson()))
            Log.i("AizipaiBenchmark", "current_opening=$result")
            File(context.cacheDir, "aizipai_current_opening_result.json")
                .writeText(result.toString())
            assertEquals(result.toString(), DecisionDisposition.EXECUTE, decision.disposition)
            assertEquals(result.toString(), "discard", decision.action)
            assertTrue(
                result.toString(),
                result.getJSONObject("process_pool").getLong("submitted") >= 8L,
            )
            assertTrue(result.toString(), elapsedMs < 25_000L)
        } finally {
            bridge.close()
            scenario.onActivity { it.finishAndRemoveTask() }
        }
    }

    @Test
    fun currentOpeningFrameIsStableAcrossWarmDecisions() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val source = File(context.cacheDir, "aizipai_current_opening.png")
        assumeTrue("current opening frame is not installed", source.isFile)
        val bitmap = BitmapFactory.decodeFile(source.absolutePath)
        assertNotNull(bitmap)
        val frame = LosslessFrameFactory.fromBitmap(bitmap)
        bitmap.recycle()
        val scenario = ActivityScenario.launch(MainActivity::class.java)
        val bridge = PythonDecisionBridge(context)
        try {
            assertTrue(bridge.prewarm().getBoolean("ready"))
            val elapsed = JSONArray()
            repeat(2) {
                val started = System.nanoTime()
                val decision = bridge.decide(frame, GameMode.WANG)
                val elapsedMs = (System.nanoTime() - started) / 1_000_000
                elapsed.put(elapsedMs)
                assertEquals(DecisionDisposition.EXECUTE, decision.disposition)
                assertEquals("discard", decision.action)
                assertEquals("hand:拾", decision.plan?.steps?.singleOrNull()?.target)
                assertTrue("warm decision took ${elapsedMs}ms", elapsedMs < 25_000L)
            }
            Log.i("AizipaiBenchmark", "current_opening_warm=$elapsed")
        } finally {
            bridge.close()
            scenario.onActivity { it.finishAndRemoveTask() }
        }
    }

    @Test
    fun currentOpeningMemoryPreparedProductionReturnsQuickly() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val source = File(context.cacheDir, "aizipai_current_opening.png")
        assumeTrue("current opening frame is not installed", source.isFile)
        StrategyProcessPool.initialize(context)
        val result = JSONObject(
            Python.getInstance().getModule("mobile_bridge")
                .callAttr(
                    "current_frame_memory_production_diagnostics_json",
                    source.absolutePath,
                    context.filesDir.absolutePath,
                )
                .toString(),
        )
        assertEquals(result.toString(), 21, result.getInt("hand_count"))
        assertTrue(result.toString(), result.getDouble("elapsed_ms") < 5_000.0)
        assertTrue(result.toString(), result.getDouble("inner_elapsed_ms") < 5_000.0)
    }

    @Test
    fun cancellingBusyWorkersRecoversWithoutAStaleResult() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        StrategyProcessPool.initialize(context)
        val expectedWorkers = StrategyProcessPool.workerCount()
        val diagnostics = JSONObject(
            Python.getInstance().getModule("mobile_bridge")
                .callAttr("worker_pool_cancel_recovery_json")
                .toString(),
        )
        assertEquals(expectedWorkers, diagnostics.getInt("cancelled"))
        assertEquals(expectedWorkers, diagnostics.getInt("recovered_worker_count"))
        assertTrue(diagnostics.getBoolean("all_ready"))
        assertTrue(diagnostics.toString(), diagnostics.getBoolean("same_worker_pids"))
        assertTrue(diagnostics.toString(), diagnostics.getDouble("recovery_ms") < 2_000.0)
    }

    @Test
    fun packagedPythonLoadsModelAndProcessesLandscapeFrame() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val diagnostics = JSONObject(
            Python.getInstance().getModule("mobile_bridge").callAttr("diagnostics_json").toString()
        )
        assertEquals("onnxruntime_android", diagnostics.getString("backend"))
        assertTrue(diagnostics.getString("opencv").isNotBlank())
        assertEquals(diagnostics.toString(), "王", diagnostics.getString("label"))
        assertEquals(0.22818, diagnostics.getDouble("confidence"), 0.002)
        assertEquals("五", diagnostics.getString("runner_up_label"))
        assertEquals(0.10140, diagnostics.getDouble("runner_up_confidence"), 0.002)
        assertEquals(1, diagnostics.getInt("session_count"))
        assertEquals("android_process_pool", diagnostics.getString("discard_executor"))
        assertEquals(StrategyProcessPool.workerCount(), diagnostics.getInt("strategy_workers"))

        val frame = File(context.cacheDir, "instrumented_blank_landscape.png")
        val bitmap = Bitmap.createBitmap(2344, 1080, Bitmap.Config.ARGB_8888)
        bitmap.eraseColor(Color.WHITE)
        FileOutputStream(frame).use { output ->
            assertTrue(bitmap.compress(Bitmap.CompressFormat.PNG, 100, output))
        }
        bitmap.recycle()

        val plan = PythonDecisionBridge(context).decide(
            frame.absolutePath,
            GameMode.NO_WANG,
            2344,
            1080,
        )
        assertNull(plan)
    }

    @Test
    fun onnxBridgeRejectsInvalidInputWithoutPoisoningSessionCache() {
        val before = AndroidOnnxBridge.cachedSessionCount()
        assertThrows(IllegalArgumentException::class.java) {
            AndroidOnnxBridge.run(
                "/missing/model.onnx",
                floatArrayOf(1.0f),
                longArrayOf(2),
            )
        }
        assertEquals(before, AndroidOnnxBridge.cachedSessionCount())

        assertThrows(IllegalArgumentException::class.java) {
            AndroidOnnxBridge.run(
                "/missing/model.onnx",
                floatArrayOf(1.0f),
                longArrayOf(1),
            )
        }
        assertEquals(before, AndroidOnnxBridge.cachedSessionCount())

        assertThrows(IllegalArgumentException::class.java) {
            AndroidOnnxBridge.runBytes(
                "/missing/model.onnx",
                byteArrayOf(0, 0, 0),
                longArrayOf(1),
            )
        }
        assertEquals(before, AndroidOnnxBridge.cachedSessionCount())
    }

    @Test
    fun benchmarkPackagedV81OnOptionalRealFrame() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val frame = File(context.cacheDir, "aizipai_benchmark.png")
        assumeTrue("optional benchmark frame is not installed", frame.isFile)
        StrategyProcessPool.initialize(context)
        val module = Python.getInstance().getModule("mobile_bridge")
        module.callAttr("worker_pool_diagnostics_json")
        StrategyProcessPool.resetMetrics()
        val benchmarkHome = File(context.cacheDir, "benchmark_home").apply {
            deleteRecursively()
            mkdirs()
        }

        val started = System.nanoTime()
        val raw = module.callAttr(
            "recommend_json",
            frame.absolutePath,
            true,
            benchmarkHome.absolutePath,
            2344,
            1080,
        ).toString()
        val plan = DecisionPlan.parse(
            raw,
            2344,
            1080,
        )
        val elapsedMs = (System.nanoTime() - started) / 1_000_000
        assertNotNull(plan)
        assertEquals("hand:玖", plan?.steps?.single()?.target)
        val freshnessStarted = System.nanoTime()
        val freshness = JSONObject(
            module.callAttr(
                "validate_fresh_plan_json",
                frame.absolutePath,
                plan?.signature,
            ).toString(),
        )
        val freshnessMs = (System.nanoTime() - freshnessStarted) / 1_000_000
        assertTrue(freshness.toString(), freshness.getBoolean("fresh"))
        val staleSignature = JSONObject(plan?.signature.orEmpty()).apply {
            getJSONObject("context").put("hand_signature", JSONArray())
        }.toString()
        val stale = JSONObject(
            module.callAttr(
                "validate_fresh_plan_json",
                frame.absolutePath,
                staleSignature,
            ).toString(),
        )
        assertTrue(stale.toString(), !stale.getBoolean("fresh"))
        assertTrue(
            stale.toString(),
            stale.getJSONArray("changed").toString().contains("hand_signature"),
        )
        File(context.cacheDir, "aizipai_benchmark_result.json").writeText(
            JSONObject()
                .put("elapsed_ms", elapsedMs)
                .put("freshness_ms", freshnessMs)
                .put("total_safe_path_ms", elapsedMs + freshnessMs)
                .put("action", plan?.action)
                .put("ready", plan != null)
                .put("signature", plan?.signature)
                .put(
                    "steps",
                    JSONArray().apply {
                        plan?.steps?.forEach { step ->
                            put(
                                JSONObject()
                                    .put("target", step.target)
                                    .put("from_x", step.fromX)
                                    .put("from_y", step.fromY)
                                    .put("x", step.x)
                                    .put("y", step.y),
                            )
                        }
                    },
                )
                .put("process_pool", JSONObject(StrategyProcessPool.metricsJson()))
                .toString(),
        )
    }

    @Test
    fun visibleHuUsesImmediatePriorityPathAndFreshConfirmation() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val frame = File(context.cacheDir, "aizipai_hu_benchmark.png")
        assumeTrue("optional HU benchmark frame is not installed", frame.isFile)
        val module = Python.getInstance().getModule("mobile_bridge")

        val started = System.nanoTime()
        val raw = module.callAttr("priority_action_json", frame.absolutePath).toString()
        val elapsedMs = (System.nanoTime() - started) / 1_000_000
        val plan = DecisionPlan.parse(raw, 2344, 1080)

        assertNotNull(raw, plan)
        assertEquals("hu", plan?.action)
        assertEquals("button:hu", plan?.steps?.single()?.target)
        assertEquals(1_985, plan?.steps?.single()?.x)
        assertEquals(510, plan?.steps?.single()?.y)
        val freshness = JSONObject(
            module.callAttr(
                "validate_fresh_plan_json",
                frame.absolutePath,
                plan?.signature,
            ).toString(),
        )
        assertTrue(freshness.toString(), freshness.getBoolean("fresh"))
        assertTrue("HU priority took ${elapsedMs}ms", elapsedMs < 1_500)
    }

}
