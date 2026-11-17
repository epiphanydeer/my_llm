"""10.1 模型选型：业务硬约束过滤与加权打分器"""
from dataclasses import dataclass
from typing import List


@dataclass
class ModelCandidate:
    name: str
    eval_score: float       # 0~100
    p95_latency_ms: float
    cost_per_1m_usd: float
    supports_on_prem: bool


def select_models(
    candidates: List[ModelCandidate],
    max_latency_ms: float,
    require_on_prem: bool,
    min_eval_score: float,
) -> List[str]:
    """先用硬约束筛掉不合规候选，再按质量/成本性价比排序。"""
    feasible = [
        m for m in candidates
        if m.p95_latency_ms <= max_latency_ms
        and m.eval_score >= min_eval_score
        and (not require_on_prem or m.supports_on_prem)
    ]
    feasible.sort(key=lambda m: (m.eval_score / max(m.cost_per_1m_usd, 0.01)), reverse=True)
    return [m.name for m in feasible]


if __name__ == "__main__":
    pool = [
        ModelCandidate("Cloud-Flagship", 92.0, 1800, 10.0, False),
        ModelCandidate("Qwen-32B-Int4", 87.5, 950, 1.2, True),
        ModelCandidate("Tiny-7B", 72.0, 300, 0.3, True),
    ]
    ranked = select_models(pool, max_latency_ms=1200, require_on_prem=True, min_eval_score=85.0)
    assert ranked == ["Qwen-32B-Int4"]
    print("[10.1 PASS]", ranked)
