package com.example.ai;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.io.File;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Small, process-local ONNX Runtime bridge for Chaquopy Python. */
public final class AndroidOnnxBridge {
    private static final OrtEnvironment ENVIRONMENT = OrtEnvironment.getEnvironment();
    private static final Map<String, SessionHolder> SESSIONS = new ConcurrentHashMap<>();

    private AndroidOnnxBridge() {}

    public static float[] run(String modelPath, float[] input, long[] shape) {
        validateInput(modelPath, input, shape);
        return runBuffer(modelPath, FloatBuffer.wrap(input), shape);
    }

    public static float[] runBytes(String modelPath, byte[] input, long[] shape) {
        int elements = validateShape(modelPath, shape);
        if (input == null || input.length != Math.multiplyExact(elements, Float.BYTES)) {
            throw new IllegalArgumentException("input byte length does not match shape size");
        }
        FloatBuffer floats = ByteBuffer.wrap(input)
            .order(ByteOrder.LITTLE_ENDIAN)
            .asFloatBuffer();
        return runBuffer(modelPath, floats, shape);
    }

    private static float[] runBuffer(String modelPath, FloatBuffer input, long[] shape) {
        String canonicalPath;
        try {
            canonicalPath = new File(modelPath).getCanonicalPath();
        } catch (Exception exception) {
            throw new IllegalArgumentException("Invalid ONNX model path", exception);
        }

        SessionHolder holder = SESSIONS.computeIfAbsent(canonicalPath, AndroidOnnxBridge::openSession);
        synchronized (holder) {
            try (OnnxTensor tensor = OnnxTensor.createTensor(
                    ENVIRONMENT,
                    input,
                    shape
                ); OrtSession.Result result = holder.session.run(
                    Collections.singletonMap(holder.inputName, tensor)
                )) {
                if (result.size() == 0) {
                    throw new IllegalStateException("ONNX model returned no outputs");
                }
                OnnxValue output = result.get(0);
                return flattenFloats(output.getValue());
            } catch (OrtException exception) {
                throw new IllegalStateException("ONNX inference failed", exception);
            }
        }
    }

    public static int cachedSessionCount() {
        return SESSIONS.size();
    }

    static void clearSessionCacheForTests() {
        for (SessionHolder holder : SESSIONS.values()) {
            try {
                holder.session.close();
            } catch (OrtException ignored) {
                // Test cleanup must not hide the original assertion failure.
            }
        }
        SESSIONS.clear();
    }

    private static SessionHolder openSession(String modelPath) {
        File file = new File(modelPath);
        if (!file.isFile()) {
            throw new IllegalArgumentException("ONNX model does not exist: " + modelPath);
        }
        try {
            OrtSession session = ENVIRONMENT.createSession(modelPath, new OrtSession.SessionOptions());
            String inputName = session.getInputNames().iterator().next();
            return new SessionHolder(session, inputName);
        } catch (Exception exception) {
            throw new IllegalStateException("Unable to load ONNX model: " + modelPath, exception);
        }
    }

    private static void validateInput(String modelPath, float[] input, long[] shape) {
        int elements = validateShape(modelPath, shape);
        if (input == null) {
            throw new IllegalArgumentException("input is required");
        }
        if (elements != input.length) {
            throw new IllegalArgumentException(
                "input length " + input.length + " does not match shape size " + elements
            );
        }
    }

    private static int validateShape(String modelPath, long[] shape) {
        if (modelPath == null || modelPath.isBlank()) {
            throw new IllegalArgumentException("modelPath is required");
        }
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("shape is required");
        }
        long elements = 1;
        for (long dimension : shape) {
            if (dimension <= 0 || elements > Long.MAX_VALUE / dimension) {
                throw new IllegalArgumentException("shape must contain positive bounded dimensions");
            }
            elements *= dimension;
        }
        if (elements > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("shape is too large");
        }
        return (int) elements;
    }

    private static float[] flattenFloats(Object value) {
        int size = countFloats(value);
        float[] flattened = new float[size];
        fillFloats(value, flattened, new int[] {0});
        return flattened;
    }

    private static int countFloats(Object value) {
        if (value instanceof Number) {
            return 1;
        }
        if (value == null || !value.getClass().isArray()) {
            throw new IllegalStateException("ONNX output is not a numeric array");
        }
        int count = 0;
        for (int index = 0; index < Array.getLength(value); index++) {
            count += countFloats(Array.get(value, index));
        }
        return count;
    }

    private static void fillFloats(Object value, float[] output, int[] offset) {
        if (value instanceof Number) {
            output[offset[0]++] = ((Number) value).floatValue();
            return;
        }
        for (int index = 0; index < Array.getLength(value); index++) {
            fillFloats(Array.get(value, index), output, offset);
        }
    }

    private static final class SessionHolder {
        final OrtSession session;
        final String inputName;

        SessionHolder(OrtSession session, String inputName) {
            this.session = session;
            this.inputName = inputName;
        }
    }
}
