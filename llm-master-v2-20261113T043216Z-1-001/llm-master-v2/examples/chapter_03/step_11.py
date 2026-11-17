"""3.11 Step 11: 在 Step 9 基础上新增面试项目实验指标与证据防虚构审计器"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MetricRecord:
    name: str
    baseline: Optional[float]
    optimized: Optional[float]
    evidence_id: Optional[str]


def audit_interview_claims(records: list[MetricRecord]) -> dict[str, object]:
    """区分有可回溯实验证据的指标与待测项，拦截无出处吹嘘。"""
    if not records:
        raise ValueError("实验指标记录不能为空")
    verified: list[str] = []
    pending: list[str] = []
    max_lift = 0.0
    for r in records:
        if r.baseline is not None and r.optimized is not None and r.evidence_id and r.evidence_id.strip():
            lift = (r.optimized - r.baseline) / r.baseline if r.baseline > 0 else 0.0
            max_lift = max(max_lift, lift)
            verified.append(f"{r.name}: {r.baseline:.2f} -> {r.optimized:.2f} (相对变化 {lift:+.1%}, 证据:{r.evidence_id})")
        else:
            pending.append(f"{r.name}: 尚未实测（面试时需如实说明边界与补测方案）")
    return {"verified": verified, "pending": pending, "max_relative_lift": max_lift}


if __name__ == "__main__":
    report = audit_interview_claims([
        MetricRecord("Recall@20", 0.68, 0.84, "EXP_2026_RRF"),
        MetricRecord("P95_Latency_50QPS", None, None, None),
    ])
    assert len(report["verified"]) == 1 and len(report["pending"]) == 1
    assert abs(report["max_relative_lift"] - (0.84 - 0.68) / 0.68) < 1e-6
    print("Step 11 self-test passed!")
