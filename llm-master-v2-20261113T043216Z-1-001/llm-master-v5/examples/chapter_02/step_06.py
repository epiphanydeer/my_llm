"""2.6 同步、异步与流式：SSE 流式块解析与完成状态校验"""
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class StreamResult:  # [无需死背-脚手架] 流式聚合结果类
    text: str
    finish_reason: Optional[str]
    chunk_count: int
    complete: bool


def consume_sse_stream(chunks: Iterable[dict]) -> StreamResult:
    """聚合流式 Delta，只有观测到有效 finish_reason 才视为完整交付。"""
    buf = []
    finish_reason = None
    count = 0
    for ch in chunks:  # [必背手撕-SSE增量块聚合] 逐块拼接 delta 并追踪最新 finish_reason
        count += 1
        delta = ch.get("delta", "")
        if delta:
            buf.append(delta)
        if ch.get("finish_reason"):
            finish_reason = ch["finish_reason"]
    return StreamResult(
        text="".join(buf),
        finish_reason=finish_reason,
        chunk_count=count,
        complete=(finish_reason in {"stop", "tool_calls"}),  # [必记设计决策-流式完整性门禁] 仅凭断流不算成功，必须观测到有效 finish_reason
    )


if __name__ == "__main__":  # [无需死背-脚手架] 正常流与断流对比测试
    ok_stream = [
        {"delta": "请检查", "finish_reason": None},
        {"delta": "安全阀。", "finish_reason": "stop"},
    ]
    broken_stream = [{"delta": "请检查", "finish_reason": None}]
    r1 = consume_sse_stream(ok_stream)
    r2 = consume_sse_stream(broken_stream)
    assert r1.complete is True and r1.text == "请检查安全阀。"
    assert r2.complete is False
    print("[2.6 PASS] complete:", r1.complete, "broken:", r2.complete)
