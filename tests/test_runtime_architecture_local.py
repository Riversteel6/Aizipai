"""Runtime architecture guardrails: no model/network calls inside local loop code."""

import re
from pathlib import Path

RUNTIME_ROOTS = [Path("chenzhou_zipai_ai"), Path("src/aizipai")]

FORBIDDEN_PATTERNS = {
    "openai": re.compile(r"\bopenai\b", re.IGNORECASE),
    "chatgpt": re.compile(r"\bchatgpt\b", re.IGNORECASE),
    "anthropic": re.compile(r"\banthropic\b", re.IGNORECASE),
    "deepseek": re.compile(r"\bdeepseek\b", re.IGNORECASE),
    "gemini": re.compile(r"\bgemini\b", re.IGNORECASE),
    "api_key": re.compile(r"\bapi_key\b", re.IGNORECASE),
    "requests": re.compile(r"\brequests\b", re.IGNORECASE),
    "httpx": re.compile(r"\bhttpx\b", re.IGNORECASE),
    "completion": re.compile(r"\bcompletion\b", re.IGNORECASE),
    "client\\.chat": re.compile(r"\bclient\\.chat\b", re.IGNORECASE),
    "client\\.responses": re.compile(r"\bclient\\.responses\b", re.IGNORECASE),
}


def _runtime_py_files() -> list[Path]:
    files: list[Path] = []
    for root in RUNTIME_ROOTS:
        for path in root.rglob("*.py"):
            if any(part.startswith(".") for part in path.parts):
                continue
            if "tests" in path.parts:
                continue
            if "build" in path.parts or "src/main/python" in path.as_posix():
                continue
            files.append(path)
    return files


def test_runtime_modules_do_not_contain_model_or_network_api_call_terms():
    hits: list[tuple[str, str]] = []
    for path in _runtime_py_files():
        text = path.read_text(encoding="utf-8")
        for name, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                hits.append((str(path), name))

    assert not hits, f"Detected forbidden runtime terms for model/network integration: {hits}"
