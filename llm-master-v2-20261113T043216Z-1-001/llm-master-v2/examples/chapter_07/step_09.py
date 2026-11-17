"""7.9 Step 9: 成对消融分析器 (Pairwise Ablation: Fixed vs Regressed + P95 尾延迟).

杜绝只报“总分涨了 2%”却掩盖大量原本做对样本退化的问题：
1. 强制校验 Baseline 与 Candidate 的 sample_id 集合及 gold 金标完全一致。
2. 逐样本统计 Fixed (错->对)、Regressed (对->错)、净提升与 P95 尾延迟变化。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass(frozen=True)
class RunSample:
    sample_id: str
    gold: str
    pred: str
    latency_ms: float


def compute_p95(latencies: List[float]) -> float:
    if not latencies:
        raise ValueError("延迟列表不能为空")
    return round(float(np.percentile(latencies, 95)), 2)


def pairwise_ablation(baseline: List[RunSample], candidate: List[RunSample]) -> Dict[str, object]:
    """执行严格样本对齐的成对消融对比."""
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("Baseline 与 Candidate 样本数必须一致且非空")

    base_map = {s.sample_id: s for s in baseline}
    cand_map = {s.sample_id: s for s in candidate}
    if set(base_map.keys()) != set(cand_map.keys()):
        raise ValueError("Baseline 与 Candidate 的 sample_id 集合不一致")

    fixed = 0
    regressed = 0
    both_right = 0
    both_wrong = 0

    for sid, b in base_map.items():
        c = cand_map[sid]
        if b.gold != c.gold:
            raise ValueError(f"GOLD_MISMATCH: 样本 {sid} 的金标被篡改 ({b.gold} != {c.gold})")
        b_ok = b.pred == b.gold
        c_ok = c.pred == c.gold
        if not b_ok and c_ok:
            fixed += 1
        elif b_ok and not c_ok:
            regressed += 1
        elif b_ok and c_ok:
            both_right += 1
        else:
            both_wrong += 1

    n = len(baseline)
    assert fixed + regressed + both_right + both_wrong == n
    base_acc = (both_right + regressed) / n
    cand_acc = (both_right + fixed) / n

    return {
        "n": n,
        "fixed": fixed,
        "regressed": regressed,
        "net_gain_count": fixed - regressed,
        "base_acc": round(base_acc, 4),
        "cand_acc": round(cand_acc, 4),
        "base_p95_ms": compute_p95([s.latency_ms for s in baseline]),
        "cand_p95_ms": compute_p95([s.latency_ms for s in candidate]),
    }


if __name__ == "__main__":
    base_runs = [
        RunSample("s1", "hazard", "safe", 120.0),
        RunSample("s2", "hazard", "safe", 130.0),
        RunSample("s3", "safe", "safe", 110.0),
        RunSample("s4", "safe", "safe", 140.0),
    ]
    cand_runs = [
        RunSample("s1", "hazard", "hazard", 210.0),
        RunSample("s2", "hazard", "hazard", 220.0),
        RunSample("s3", "safe", "hazard", 250.0),
        RunSample("s4", "safe", "safe", 200.0),
    ]
    res = pairwise_ablation(base_runs, cand_runs)
    assert res["fixed"] == 2 and res["regressed"] == 1 and res["net_gain_count"] == 1
    print(f"MM Step 9 self-test passed! res={res}")
