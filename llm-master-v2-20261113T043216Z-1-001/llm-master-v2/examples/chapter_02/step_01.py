"""2.1 接入模型：OpenAI 兼容请求封装与指数退避重试客户端"""
from dataclasses import dataclass
from typing import Callable, Dict, Set


RETRYABLE_STATUS: Set[int] = {429, 500, 502, 503, 504}
NON_RETRYABLE_STATUS: Set[int] = {400, 401, 403, 404}


@dataclass
class ModelResponse:
    status_code: int
    content: str
    finish_reason: str  # "stop" | "length" | "tool_calls" | "content_filter"
    attempts: int


def call_with_backoff(
    api_fn: Callable[[], Dict[str, object]],
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> ModelResponse:
    """按 HTTP 状态码分级重试，并记录最终 finish_reason 与尝试次数。"""
    delays = []
    for attempt in range(1, max_retries + 2):
        resp = api_fn()
        code = int(resp["status_code"])
        if code == 200:
            return ModelResponse(code, str(resp["content"]), str(resp["finish_reason"]), attempt)
        if code in NON_RETRYABLE_STATUS or attempt > max_retries:
            raise RuntimeError(f"请求终止: status={code}, attempts={attempt}, delays={delays}")
        if code in RETRYABLE_STATUS:
            delays.append(base_delay * (2 ** (attempt - 1)))
    raise RuntimeError("Unreachable")


if __name__ == "__main__":
    seq = iter([
        {"status_code": 429, "content": "", "finish_reason": ""},
        {"status_code": 503, "content": "", "finish_reason": ""},
        {"status_code": 200, "content": "OK", "finish_reason": "stop"},
    ])
    res = call_with_backoff(lambda: next(seq), max_retries=3)
    assert res.status_code == 200 and res.attempts == 3 and res.finish_reason == "stop"

    try:
        call_with_backoff(lambda: {"status_code": 400, "content": "", "finish_reason": ""}, max_retries=3)
        assert False, "400 不应重试"
    except RuntimeError as exc:
        assert "attempts=1" in str(exc)
    print("[2.1 PASS]", res)
