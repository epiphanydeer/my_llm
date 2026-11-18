"""Step 7: Agent Trajectory Evaluator (轨迹评测、四分类归因与 Pass^k 稳定性指标).

在 Step 4~6 执行轨迹基础上新增 AgentTrajectoryEvaluator：
1. 四分类结果归因：SUCCESS / WRONG_TOOL / PARAM_ERROR / STEP_LIMIT。
2. 计算工具无效调用率、重复调用率与单任务平均步数。
3. 计算工业界常用的 Pass@k 与 Pass^k (连续 k 次全对的可靠性指标)。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List


class TrajectoryOutcome(str, Enum):
    SUCCESS = "SUCCESS"
    WRONG_TOOL = "WRONG_TOOL"
    PARAM_ERROR = "PARAM_ERROR"
    STEP_LIMIT = "STEP_LIMIT"


@dataclass
class TrajectoryRecord:
    """单次 Agent 运行的轨迹记录."""

    task_id: str
    expected_tools: List[str]
    actual_calls: List[Dict[str, Any]]  # [{"name": str, "args": dict, "valid": bool}]
    task_solved: bool
    hit_step_limit: bool = False


# [必记设计决策-四分类归因] 失败轨迹按因果优先级归因为：SUCCESS / STEP_LIMIT / PARAM_ERROR / WRONG_TOOL
def classify_outcome(record: TrajectoryRecord) -> TrajectoryOutcome:
    """对单条轨迹进行四分类故障归因."""
    if record.task_solved:
        return TrajectoryOutcome.SUCCESS
    if record.hit_step_limit:
        return TrajectoryOutcome.STEP_LIMIT

    actual_names = [c["name"] for c in record.actual_calls]
    if any(not c.get("valid", True) for c in record.actual_calls):
        return TrajectoryOutcome.PARAM_ERROR
    if set(record.expected_tools) - set(actual_names):
        return TrajectoryOutcome.WRONG_TOOL
    return TrajectoryOutcome.WRONG_TOOL


def evaluate_suite(records: List[TrajectoryRecord]) -> Dict[str, Any]:
    """聚合多条轨迹，输出成功率、无效工具率、重复率及 Pass^k 指标."""
    if not records:
        raise ValueError("评测轨迹列表不能为空")

    total_calls = 0
    invalid_calls = 0
    duplicate_calls = 0
    outcome_counts: Dict[str, int] = {o.value: 0 for o in TrajectoryOutcome}
    by_task: Dict[str, List[bool]] = {}

    for rec in records:
        outcome = classify_outcome(rec)
        outcome_counts[outcome.value] += 1
        by_task.setdefault(rec.task_id, []).append(rec.task_solved)

        seen_signatures: set[str] = set()
        for call in rec.actual_calls:
            total_calls += 1
            if not call.get("valid", True):
                invalid_calls += 1
            # [无需死背-脚手架] 轨迹内调用签名拼接用于统计重复调用率，无需死背字符串格式
            sig = f"{call['name']}::{sorted(call.get('args', {}).items())}"
            if sig in seen_signatures:
                duplicate_calls += 1
            seen_signatures.add(sig)

    total_runs = len(records)
    success_rate = outcome_counts[TrajectoryOutcome.SUCCESS.value] / total_runs
    # Pass@k: 同一任务 k 次尝试中至少成功 1 次的比例
    # [必背手撕-Pass@k] 同一任务 k 次尝试至少成功 1 次的比例（衡量能力上限）
    pass_at_k = sum(1 for runs in by_task.values() if any(runs)) / len(by_task)
    # Pass^k: 同一任务 k 次尝试全部成功的比例（生产可靠性核心指标）
    # [必背手撕-Pass^k] 同一任务 k 次尝试全部成功的比例（衡量生产稳定性，面试必考区分）
    pass_power_k = sum(1 for runs in by_task.values() if all(runs)) / len(by_task)

    return {
        "total_runs": total_runs,
        "success_rate": round(success_rate, 4),
        "pass_at_k": round(pass_at_k, 4),
        "pass_power_k": round(pass_power_k, 4),
        "invalid_tool_rate": round(invalid_calls / max(1, total_calls), 4),
        "duplicate_call_rate": round(duplicate_calls / max(1, total_calls), 4),
        "outcome_breakdown": outcome_counts,
    }


if __name__ == "__main__":
    sample_records = [
        TrajectoryRecord(
            task_id="T1",
            expected_tools=["query_ticket"],
            actual_calls=[{"name": "query_ticket", "args": {"ticket_id": "TK-1"}, "valid": True}],
            task_solved=True,
        ),
        TrajectoryRecord(
            task_id="T1",
            expected_tools=["query_ticket"],
            actual_calls=[{"name": "query_ticket", "args": {"id": "TK-1"}, "valid": False}],
            task_solved=False,
        ),
        TrajectoryRecord(
            task_id="T2",
            expected_tools=["check_node"],
            actual_calls=[{"name": "check_node", "args": {"node": "db-02"}, "valid": True}],
            task_solved=True,
        ),
        TrajectoryRecord(
            task_id="T2",
            expected_tools=["check_node"],
            actual_calls=[{"name": "check_node", "args": {"node": "db-02"}, "valid": True}],
            task_solved=True,
        ),
    ]
    report = evaluate_suite(sample_records)
    assert report["pass_at_k"] == 1.0, f"Pass@2 应为 1.0: {report}"
    assert report["pass_power_k"] == 0.5, f"Pass^2 应为 0.5: {report}"
    assert report["outcome_breakdown"]["PARAM_ERROR"] == 1
    assert report["invalid_tool_rate"] == 0.25
    print(f"Step 7 self-test passed! report={report}")
