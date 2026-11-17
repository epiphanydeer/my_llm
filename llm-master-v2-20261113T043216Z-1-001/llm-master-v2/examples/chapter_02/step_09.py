"""2.9 模型版本与 Prompt 迭代：单变量回归评测矩阵"""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class EvalCase:
    case_id: str
    passed_v1: bool
    passed_v2: bool
    latency_v1_ms: float
    latency_v2_ms: float


def compare_prompt_versions(cases: List[EvalCase]) -> Dict[str, object]:
    """固定同一模型快照与测试集，输出通过率变化与退化（Regression）样本。"""
    n = len(cases)
    v1_rate = sum(c.passed_v1 for c in cases) / n
    v2_rate = sum(c.passed_v2 for c in cases) / n
    regressions = [c.case_id for c in cases if c.passed_v1 and not c.passed_v2]
    improvements = [c.case_id for c in cases if not c.passed_v1 and c.passed_v2]
    return {
        "v1_pass_rate": round(v1_rate, 3),
        "v2_pass_rate": round(v2_rate, 3),
        "regressions": regressions,
        "improvements": improvements,
        "ship_ready": (v2_rate >= v1_rate) and (len(regressions) == 0),
    }


if __name__ == "__main__":
    cases = [
        EvalCase("c1", True, True, 400, 380),
        EvalCase("c2", False, True, 500, 420),
        EvalCase("c3", True, True, 450, 410),
    ]
    res = compare_prompt_versions(cases)
    assert res["ship_ready"] is True and res["improvements"] == ["c2"]
    print("[2.9 PASS]", res)
