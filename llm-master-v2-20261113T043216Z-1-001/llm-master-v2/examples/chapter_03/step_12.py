"""3.12 Step 12: 在 Step 11 基础上新增逐请求端到端尾延迟（P50/P95/P99）与 Token 成本分析器"""
from dataclasses import dataclass
from decimal import Decimal
import math
from statistics import mean


@dataclass(frozen=True)
class RequestProfile:
    req_id: str
    retrieval_ms: float
    rerank_ms: float
    prefill_ms: float
    decode_ms: float
    in_tokens: int
    out_tokens: int


def nearest_rank_quantile(values: list[float], p: float) -> float:
    if not values or not (0.0 < p <= 1.0):
        raise ValueError("参数非法")
    s = sorted(values)
    return float(s[math.ceil(p * len(s)) - 1])


def analyze_perf_and_cost(
    profiles: list[RequestProfile],
    in_price_per_m: Decimal = Decimal("2.0"),
    out_price_per_m: Decimal = Decimal("6.0"),
) -> dict[str, object]:
    """逐请求先求和再求 P95，数学反证为何不能把各阶段 P95 直接相加。"""
    totals = [p.retrieval_ms + p.rerank_ms + p.prefill_ms + p.decode_ms for p in profiles]
    ttfts = [p.retrieval_ms + p.rerank_ms + p.prefill_ms for p in profiles]
    total_cost = sum(
        (Decimal(p.in_tokens) * in_price_per_m + Decimal(p.out_tokens) * out_price_per_m) / Decimal("1000000")
        for p in profiles
    )
    p50, p95, p99 = (nearest_rank_quantile(totals, q) for q in (0.50, 0.95, 0.99))
    assert p50 <= p95 <= p99
    return {"p50_ms": p50, "p95_ms": p95, "p99_ms": p99, "avg_ttft_ms": mean(ttfts), "total_cost": total_cost}


if __name__ == "__main__":
    reqs = [
        RequestProfile("R1", 100.0, 10.0, 10.0, 10.0, 1000, 200),
        RequestProfile("R2", 10.0, 100.0, 10.0, 10.0, 1000, 200),
        RequestProfile("R3", 10.0, 10.0, 10.0, 100.0, 1000, 200),
    ]
    res = analyze_perf_and_cost(reqs)
    # 若把各阶段 P95 直接相加会得出 300ms，但真实端到端 P95 只有 130ms
    assert math.isclose(res["p95_ms"], 130.0)
    assert res["total_cost"] == Decimal("0.0096")
    print("Step 12 self-test passed!")
