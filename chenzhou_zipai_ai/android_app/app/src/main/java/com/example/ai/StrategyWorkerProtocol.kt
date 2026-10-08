package com.example.ai

internal object StrategyWorkerProtocol {
    const val EXECUTE = 1
    const val RESULT = 2
    const val CANCEL_ALL = 3
    const val RESTART_PROCESS = 4

    const val REQUEST_ID = "request_id"
    const val TASK_NAME = "task_name"
    const val PAYLOAD = "payload"
    const val OK = "ok"
    const val ERROR = "error"
}
