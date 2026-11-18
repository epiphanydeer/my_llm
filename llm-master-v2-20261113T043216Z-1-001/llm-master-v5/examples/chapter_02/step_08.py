"""2.8 Token、成本与延迟：含重试与降级的真实有效成本统计器"""
from dataclasses import dataclass
from typing import List


@dataclass
class AttemptTrace:  # [无需死背-脚手架] 单次尝试追踪记录
    latency_ms: float
    input_tokens: int
    output_tokens: int
    succeeded: bool


def summarize_effective_request(traces: List[AttemptTrace], in_rate_per_1k: float, out_rate_per_1k: float) -> dict:
    """将失败重试的延迟与 Token 成本一并计入单次用户请求的真实开销。"""
    total_latency = sum(t.latency_ms for t in traces)  # [必记设计决策-全链路延迟口径] 失败重试耗时必须累加进端到端延迟
    total_in = sum(t.input_tokens for t in traces)
    total_out = sum(t.output_tokens for t in traces)
    total_cost = (total_in / 1000) * in_rate_per_1k + (total_out / 1000) * out_rate_per_1k  # [必背手撕-含重试有效成本核算] 失败请求的 Token 同样计费
    final_ok = any(t.succeeded for t in traces)  # [必背手撕-最终成功状态判定]
    return {
        "attempts": len(traces),
        "final_ok": final_ok,
        "e2e_latency_ms": round(total_latency, 2),
        "effective_cost": round(total_cost, 6),
    }


if __name__ == "__main__":  # [无需死背-脚手架] 超时后重试成功的账单模拟
    traces = [
        AttemptTrace(1200, 1000, 50, False),  # 超时截断
        AttemptTrace(850, 1000, 200, True),   # 重试成功
    ]
    rep = summarize_effective_request(traces, in_rate_per_1k=0.002, out_rate_per_1k=0.006)
    assert rep["attempts"] == 2 and rep["e2e_latency_ms"] == 2050.0 and rep["final_ok"] is True
    print("[2.8 PASS]", rep)
