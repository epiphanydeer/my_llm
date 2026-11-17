"""Step 10: Planner-Worker-Reviewer State Machine (多角色分工与只读审校断路器).

实现工业级多角色协作状态机：
1. Planner 拆解任务 -> Worker 执行（拥有写权限）-> Reviewer 独立审校（严格只读，禁止写权限）。
2. Reviewer 不通过时带结构化缺陷报告打回 Worker 返工。
3. 设置 max_reworks 硬熔断上限，防止无限互相推诿。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List


class Role(str, Enum):
    PLANNER = "planner"
    WORKER = "worker"
    REVIEWER = "reviewer"


@dataclass
class ReviewDecision:
    approved: bool
    feedback: str = ""


class MultiAgentOrchestrator:
    """Planner -> Worker -> Reviewer 三角色状态机."""

    def __init__(self, max_reworks: int = 2) -> None:
        if max_reworks < 0:
            raise ValueError("max_reworks 不能为负数")
        self.max_reworks = max_reworks
        self.audit_log: List[str] = []

    def run(
        self,
        goal: str,
        planner_fn: Callable[[str], List[str]],
        worker_fn: Callable[[List[str], str], Dict[str, str]],
        reviewer_fn: Callable[[Dict[str, str], SetPermissionGuard], ReviewDecision],
    ) -> Dict[str, object]:
        plan = planner_fn(goal)
        self.audit_log.append(f"PLANNER: generated {len(plan)} steps")

        feedback = ""
        rework_count = 0
        guard = SetPermissionGuard(allowed_mode="READ_ONLY")

        while True:
            artifact = worker_fn(plan, feedback)
            self.audit_log.append(f"WORKER(attempt={rework_count + 1}): produced artifact")

            decision = reviewer_fn(artifact, guard)
            self.audit_log.append(f"REVIEWER: approved={decision.approved}")

            if decision.approved:
                return {"status": "APPROVED", "reworks": rework_count, "artifact": artifact}

            rework_count += 1
            if rework_count > self.max_reworks:
                return {
                    "status": "ESCALATED_TO_HUMAN",
                    "reworks": rework_count - 1,
                    "last_feedback": decision.feedback,
                }
            feedback = decision.feedback


@dataclass
class SetPermissionGuard:
    """强制约束 Reviewer 只能使用只读工具."""

    allowed_mode: str = "READ_ONLY"

    def assert_tool_allowed(self, tool_mode: str) -> None:
        if self.allowed_mode == "READ_ONLY" and tool_mode != "READ":
            raise PermissionError(f"Reviewer 严禁调用写工具 (mode={tool_mode})")


if __name__ == "__main__":
    orch = MultiAgentOrchestrator(max_reworks=2)
    worker_calls = {"n": 0}

    def planner(goal: str) -> List[str]:
        return [f"定位 {goal} 根因", "生成修复建议"]

    def worker(plan: List[str], feedback: str) -> Dict[str, str]:
        worker_calls["n"] += 1
        if not feedback:
            return {"draft": "建议直接重启数据库"}
        return {"draft": "先切流再重启，附带回滚步骤", "addressed": feedback}

    def reviewer(artifact: Dict[str, str], guard: SetPermissionGuard) -> ReviewDecision:
        guard.assert_tool_allowed("READ")
        if "回滚" not in artifact["draft"]:
            return ReviewDecision(approved=False, feedback="缺少回滚步骤，请补充")
        return ReviewDecision(approved=True)

    res = orch.run("TK-101 连接池打满", planner, worker, reviewer)
    assert res["status"] == "APPROVED" and res["reworks"] == 1 and worker_calls["n"] == 2

    try:
        SetPermissionGuard("READ_ONLY").assert_tool_allowed("WRITE")
        assert False, "Reviewer 写操作必须被拒绝"
    except PermissionError:
        pass
    print("Step 10 self-test passed!")
