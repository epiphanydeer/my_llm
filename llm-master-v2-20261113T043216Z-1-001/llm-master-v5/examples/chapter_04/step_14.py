"""Step 14: Dual-Mode Loop Breaker (双模式防死循环熔断器).

解决生产 Agent 最隐蔽的两类死循环：
1. 连续重复模式 (A-A-A)：连续 N 次调用完全相同的工具与规范化参数。
2. 乒乓震荡模式 (A-B-A-B)：在两个等价调用之间来回横跳（例如反复查 A 再查 B 再查 A 再查 B）。
采用规范化 JSON 序列化 (sort_keys=True) 生成指纹，避免键序不同导致漏判。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List


# [必记设计决策-规范化指纹] 必须用 json.dumps(..., sort_keys=True) 生成参数键序无关指纹，防 {"a":1,"b":2} 与 {"b":2,"a":1} 漏判
def canonical_signature(tool_name: str, args: Dict[str, Any]) -> str:
    """生成参数键序无关的规范化调用指纹."""
    payload = json.dumps({"tool": tool_name, "args": args}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class DualModeLoopBreaker:
    """支持 A-A-A 连续熔断与 A-B-A-B 震荡熔断."""

    consecutive_limit: int = 3
    oscillation_cycles: int = 2
    _history: List[str] = field(default_factory=list)

    def record_and_check(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """记录本次调用指纹，若触发熔断则返回中断原因."""
        sig = canonical_signature(tool_name, args)
        self._history.append(sig)

        # 1. 检测 A-A-A 连续重复
        # [必背手撕-AAA连续熔断] 检查尾部 N 次调用指纹集合长度是否为 1，拦截连续无效重试
        if len(self._history) >= self.consecutive_limit:
            tail = self._history[-self.consecutive_limit :]
            if len(set(tail)) == 1:
                return {
                    "tripped": True,
                    "mode": "CONSECUTIVE_REPEAT",
                    "hint": f"工具 {tool_name} 已连续 {self.consecutive_limit} 次使用完全相同参数调用，触发熔断。",
                }

        # 2. 检测 A-B-A-B 乒乓震荡
        window_len = self.oscillation_cycles * 2
        # [必背手撕-ABAB乒乓熔断] 检查尾部 2*cycles 窗口是否呈现 a!=b 且奇偶位交替震荡，防止两工具间来回横跳
        if len(self._history) >= window_len:
            w = self._history[-window_len:]
            a, b = w[0], w[1]
            if a != b and all(w[i] == (a if i % 2 == 0 else b) for i in range(window_len)):
                return {
                    "tripped": True,
                    "mode": "PING_PONG_OSCILLATION",
                    "hint": "检测到 A-B-A-B 往返震荡调用，禁止继续重试，请汇总已有信息或转交人工。",
                }

        # [无需死背-脚手架] 统一返回字典结构可按业务自定义，重点记住需附带纠偏 hint
        return {"tripped": False, "mode": None, "hint": None}


if __name__ == "__main__":
    b1 = DualModeLoopBreaker(consecutive_limit=3)
    assert not b1.record_and_check("query_ticket", {"a": 1, "b": 2})["tripped"]
    assert not b1.record_and_check("query_ticket", {"b": 2, "a": 1})["tripped"]
    res1 = b1.record_and_check("query_ticket", {"a": 1, "b": 2})
    assert res1["tripped"] and res1["mode"] == "CONSECUTIVE_REPEAT"

    b2 = DualModeLoopBreaker(consecutive_limit=3, oscillation_cycles=2)
    assert not b2.record_and_check("check_node", {"node": "db-01"})["tripped"]
    assert not b2.record_and_check("check_node", {"node": "db-02"})["tripped"]
    assert not b2.record_and_check("check_node", {"node": "db-01"})["tripped"]
    res2 = b2.record_and_check("check_node", {"node": "db-02"})
    assert res2["tripped"] and res2["mode"] == "PING_PONG_OSCILLATION"
    print("Step 14 self-test passed!")
