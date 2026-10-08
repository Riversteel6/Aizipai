package com.example.ai

internal object DecisionWorkerProtocol {
    const val EXECUTE = 1
    const val RESULT = 2
    const val RESTART = 3

    const val REQUEST_ID = "request_id"
    const val OPERATION = "operation"
    const val PAYLOAD = "payload"
    const val ERROR = "error"
    const val OK = "ok"

    const val FRAME_PATH = "frame_path"
    const val IMAGE_PATH = "image_path"
    const val WIDTH = "width"
    const val HEIGHT = "height"
    const val ROW_BYTES = "row_bytes"
    const val WILDCARD_ENABLED = "wildcard_enabled"
    const val SIGNATURE = "signature"
}
