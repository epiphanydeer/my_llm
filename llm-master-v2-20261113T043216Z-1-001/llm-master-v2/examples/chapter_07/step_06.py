"""7.6 Step 6: 3x3 混淆矩阵、防刷分隐患召回率与困难切片正交评估.

核心口径：
1. 真实隐患 (gold="hazard") 被模型拒答 (pred="unknown") 时，绝不能剔除分母，必须计入未召回项！
2. 零除保护 (safe_div)：分母为 0 返回 None，绝不误报为 0.0。
3. 支持按场景切片 (day_clear / night_low_light / small_object) 独立统计。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set


VALID_LABELS = frozenset({"hazard", "safe", "unknown"})
LABEL_IDX = {"hazard": 0, "safe": 1, "unknown": 2}


@dataclass(frozen=True)
class EvalCase:
    sample_id: str
    gold: str
    pred: str
    group: str


def safe_div(num: int, den: int) -> Optional[float]:
    if den == 0:
        return None
    val = round(num / den, 4)
    assert 0.0 <= val <= 1.0
    return val


def compute_slice_metrics(cases: List[EvalCase]) -> Dict[str, object]:
    """计算单个集合上的 3x3 混淆矩阵与核心安全指标."""
    if not cases:
        raise ValueError("评测样本集不能为空")

    seen: Set[str] = set()
    matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for c in cases:
        if c.sample_id in seen:
            raise ValueError(f"重复样本 ID: {c.sample_id}")
        if c.gold not in VALID_LABELS or c.pred not in VALID_LABELS:
            raise ValueError(f"非法标签: gold={c.gold}, pred={c.pred}")
        seen.add(c.sample_id)
        matrix[LABEL_IDX[c.gold]][LABEL_IDX[c.pred]] += 1

    hit = matrix[0][0]
    fatal_miss = matrix[0][1]
    abstained_hazard = matrix[0][2]
    hazard_total = sum(matrix[0])
    assert hit + fatal_miss + abstained_hazard == hazard_total

    false_alarm = matrix[1][0]
    safe_total = sum(matrix[1])
    total_abstained = matrix[0][2] + matrix[1][2] + matrix[2][2]

    return {
        "n": len(cases),
        "strict_hazard_recall": safe_div(hit, hazard_total),
        "fatal_miss_rate": safe_div(fatal_miss, hazard_total),
        "false_alarm_rate": safe_div(false_alarm, safe_total),
        "abstention_rate": safe_div(total_abstained, len(cases)),
    }


def evaluate_with_slices(cases: List[EvalCase]) -> Dict[str, Dict[str, object]]:
    """同时输出全量 overall 指标与各业务切片指标."""
    grouped: Dict[str, List[EvalCase]] = defaultdict(list)
    for c in cases:
        grouped[c.group].append(c)
    report: Dict[str, Dict[str, object]] = {"overall": compute_slice_metrics(cases)}
    for grp, grp_cases in sorted(grouped.items()):
        report[f"slice:{grp}"] = compute_slice_metrics(grp_cases)
    return report


if __name__ == "__main__":
    sample_cases = [
        EvalCase("1", "hazard", "hazard", "day_clear"),
        EvalCase("2", "safe", "safe", "day_clear"),
        EvalCase("3", "hazard", "unknown", "night_low_light"),
        EvalCase("4", "hazard", "safe", "night_low_light"),
    ]
    rep = evaluate_with_slices(sample_cases)
    assert rep["overall"]["strict_hazard_recall"] == 0.3333
    assert rep["slice:night_low_light"]["false_alarm_rate"] is None
    print(f"MM Step 6 self-test passed! overall={rep['overall']}")
