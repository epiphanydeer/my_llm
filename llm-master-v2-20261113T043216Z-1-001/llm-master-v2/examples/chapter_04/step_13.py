"""Step 13: Skill Readiness Auditor (能力前置条件图与就绪度审计).

将技能掌握度与系统能力模块建模为有向依赖图 (Prerequisite Graph)：
1. 校验技能依赖闭环与前置条件满足情况。
2. 输出可执行的学习/上线就绪度报告（已就绪、阻塞项、下一步推荐）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class SkillNode:
    skill_id: str
    title: str
    prerequisites: List[str]
    mastered: bool = False


class SkillReadinessAuditor:
    """校验前置依赖并计算技能/系统就绪度."""

    def __init__(self, skills: List[SkillNode]) -> None:
        self.skills: Dict[str, SkillNode] = {s.skill_id: s for s in skills}
        for s in skills:
            for pre in s.prerequisites:
                if pre not in self.skills:
                    raise ValueError(f"技能 {s.skill_id} 引用了未知前置项 {pre}")

    def audit(self) -> Dict[str, object]:
        mastered_set: Set[str] = {sid for sid, s in self.skills.items() if s.mastered}
        ready_to_learn: List[str] = []
        blocked: Dict[str, List[str]] = {}

        for sid, node in self.skills.items():
            if node.mastered:
                continue
            missing = [p for p in node.prerequisites if p not in mastered_set]
            if not missing:
                ready_to_learn.append(sid)
            else:
                blocked[sid] = missing

        readiness_score = round(len(mastered_set) / max(1, len(self.skills)), 4)
        return {
            "readiness_score": readiness_score,
            "mastered": sorted(mastered_set),
            "ready_next": sorted(ready_to_learn),
            "blocked": blocked,
        }


if __name__ == "__main__":
    nodes = [
        SkillNode("ota_loop", "O-T-A 闭环", [], mastered=True),
        SkillNode("tool_schema", "Tool Schema 与沙箱", ["ota_loop"], mastered=True),
        SkillNode("memory_window", "原子记忆窗口", ["ota_loop"], mastered=False),
        SkillNode("multi_agent", "多智能体交接", ["tool_schema", "memory_window"], mastered=False),
    ]
    report = SkillReadinessAuditor(nodes).audit()
    assert report["readiness_score"] == 0.5
    assert report["ready_next"] == ["memory_window"]
    assert report["blocked"] == {"multi_agent": ["memory_window"]}
    print(f"Step 13 self-test passed! report={report}")
