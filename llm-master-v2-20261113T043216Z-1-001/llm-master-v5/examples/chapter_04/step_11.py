"""Step 11: Handoff Projector & Token Budget Allocator (多角色交接白名单投影与全局预算池).

解决多智能体交接中的“Context 爆炸”与“无界并发超支”问题：
1. HandoffProjector：用白名单字段投影替代全量聊天历史透传，剔除中间推理废话与冗余日志。
2. TokenBudgetAllocator：线程安全/原子化的全局 Token 预算池，子 Agent 预扣配额、用完退还余量。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List


@dataclass
# [必记设计决策-白名单投影] 多角色交接严禁透传全量聊天历史，必须按 allowed_fields 白名单投影并截断超长字段
class HandoffProjector:
    """只保留下游角色必需的白名单字段，截断超长正文."""

    allowed_fields: List[str]
    max_field_chars: int = 300

    def project(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        """将上游完整状态投影为精简的交接载荷."""
        projected: Dict[str, Any] = {}
        for key in self.allowed_fields:
            if key not in raw_state:
                continue
            val = raw_state[key]
            # [必记设计决策-交接截断] 防止单字段超长堆栈或日志撑爆下游角色上下文窗口
            if isinstance(val, str) and len(val) > self.max_field_chars:
                val = val[: self.max_field_chars] + "...[TRUNCATED]"
            projected[key] = val
        return projected


@dataclass
class TokenBudgetAllocator:
    """多 Agent 全局 Token 预算池（支持预扣与余量回收）."""

    total_budget: int
    used_tokens: int = 0
    _reservations: Dict[str, int] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    # [必背手撕-原子预扣] 多子 Agent 并发时用锁(或 Redis Lua)先预扣配额，防各自超支突破总预算
    def reserve(self, agent_id: str, amount: int) -> bool:
        """原子预扣配额，余额不足时拒绝分配."""
        if amount <= 0:
            raise ValueError("预扣 Token 必须为正整数")
        with self._lock:
            reserved_sum = sum(self._reservations.values())
            available = self.total_budget - self.used_tokens - reserved_sum
            if amount > available:
                return False
            self._reservations[agent_id] = self._reservations.get(agent_id, 0) + amount
            return True

    # [必背手撕-结算退还] 执行完毕按 actual_used 结算并释放剩余预扣额度，确保账目守恒
    def settle(self, agent_id: str, actual_used: int) -> int:
        """结算实际消耗，退还未使用的预扣额度，返回当前剩余可用预算."""
        if actual_used < 0:
            raise ValueError("actual_used 不能为负")
        with self._lock:
            reserved = self._reservations.pop(agent_id, 0)
            assert actual_used <= reserved, f"{agent_id} 超额消耗: actual={actual_used} > reserved={reserved}"
            self.used_tokens += actual_used
            return self.remaining_available()

    def remaining_available(self) -> int:
        return self.total_budget - self.used_tokens - sum(self._reservations.values())


if __name__ == "__main__":
    raw_worker_state = {
        "ticket_id": "TK-101",
        "root_cause": "db-02 连接池耗尽",
        "raw_stack_trace": "Traceback..." * 100,
        "scratchpad": "我觉得可能是内存泄露，不对，再看看...",
    }
    projector = HandoffProjector(allowed_fields=["ticket_id", "root_cause"])
    handoff = projector.project(raw_worker_state)
    assert set(handoff.keys()) == {"ticket_id", "root_cause"}
    assert "raw_stack_trace" not in handoff and "scratchpad" not in handoff

    pool = TokenBudgetAllocator(total_budget=1000)
    assert pool.reserve("worker_1", 600) is True
    assert pool.reserve("worker_2", 500) is False, "超额并发预扣必须被拒绝"
    assert pool.reserve("worker_2", 400) is True
    rem = pool.settle("worker_1", actual_used=250)
    assert rem == 350 and pool.remaining_available() == 350
    print("Step 11 self-test passed!")
