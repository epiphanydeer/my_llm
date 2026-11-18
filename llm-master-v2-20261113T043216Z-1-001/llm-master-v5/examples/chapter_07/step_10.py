"""7.10 Step 10: 多模态成本治理与分级级联路由 (Quality Cascade + 成功任务真实成本).

核心工程策略：
1. 先用低成本链路 (Low-Res / OCR) 初筛，仅把低置信度疑难样本升级到高分辨率裁剪复核。
2. 算账不看单次调用便宜了多少，而是核算“每个成功任务的真实总成本”（含重试与升级开销）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class CascadeSample:
    sample_id: str
    low_res_conf: float
    low_res_correct: bool
    high_res_correct: bool


def simulate_quality_cascade(
    samples: List[CascadeSample],
    conf_threshold: float = 0.75,
    low_cost_per_req: float = 0.002,
    high_cost_per_req: float = 0.015,
) -> Dict[str, float | int]:
    """对比全量高分辨率 vs 分级级联路由的准确率与单成功任务成本."""
    if not samples:
        raise ValueError("样本不能为空")
    if not 0.0 <= conf_threshold <= 1.0:
        raise ValueError("conf_threshold 必须在 [0, 1] 内")

    all_high_cost = len(samples) * high_cost_per_req  # [无需死背-脚手架]
    all_high_succ = sum(1 for s in samples if s.high_res_correct)

    cascade_cost = 0.0
    cascade_succ = 0
    escalated_count = 0

    for s in samples:
        cascade_cost += low_cost_per_req
        if s.low_res_conf >= conf_threshold:  # [必记设计决策-高置信度样本直接走低清链路省Token]
            if s.low_res_correct:
                cascade_succ += 1
        else:
            escalated_count += 1
            cascade_cost += high_cost_per_req  # [必记设计决策-低置信度样本升级高清复核并累加双倍账单]
            if s.high_res_correct:
                cascade_succ += 1

    cost_per_succ_all_high = all_high_cost / max(1, all_high_succ)
    cost_per_succ_cascade = cascade_cost / max(1, cascade_succ)  # [必背手撕-按单成功任务真实总成本核算ROI]

    return {
        "total_samples": len(samples),
        "escalated_count": escalated_count,
        "all_high_acc": round(all_high_succ / len(samples), 4),
        "cascade_acc": round(cascade_succ / len(samples), 4),
        "all_high_total_cost": round(all_high_cost, 4),
        "cascade_total_cost": round(cascade_cost, 4),
        "cost_per_success_saving_pct": round(
            (cost_per_succ_all_high - cost_per_succ_cascade) / cost_per_succ_all_high * 100, 2
        ),
    }


if __name__ == "__main__":  # [无需死背-脚手架]
    cases = [
        CascadeSample("1", 0.92, True, True),
        CascadeSample("2", 0.88, True, True),
        CascadeSample("3", 0.95, True, True),
        CascadeSample("4", 0.45, False, True),
    ]
    report = simulate_quality_cascade(cases, conf_threshold=0.75)
    assert report["escalated_count"] == 1
    assert report["cascade_acc"] == 1.0 and report["all_high_acc"] == 1.0
    assert report["cost_per_success_saving_pct"] > 50.0
    print(f"MM Step 10 self-test passed! report={report}")
