"""Step 5: Atomic Memory Window (短期上下文窗口管理与原子轮次淘汰).

在 Step 4 (ReActEngine) 的基础上新增 AtomicMemoryWindow：
1. 钉住 System Prompt（永不淘汰）。
2. 将一次完整的工具交互 (assistant[tool_calls] + 对应的 tool 消息) 绑定为不可分割的原子组 (Atomic Turn)。
3. 当总 Token 超出预算时，从最旧的非 System 原子组开始整组淘汰，避免只删半截导致 API 400 报错。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List


def estimate_tokens(msg: Dict[str, Any]) -> int:
    """粗略估算单条消息的 Token 数（中文按 1.5 字/token，英文按 4 字符/token 的保守上界）."""
    content = str(msg.get("content", ""))
    tool_calls = json.dumps(msg.get("tool_calls", []), ensure_ascii=False)
    total_chars = len(content) + len(tool_calls)
    return max(4, (total_chars // 2) + 4)


@dataclass
class AtomicMemoryWindow:
    """支持原子轮次分组与预算裁剪的短期记忆窗口."""

    max_tokens: int
    system_prompt: str
    _turns: List[List[Dict[str, Any]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_tokens <= 20:
            raise ValueError("max_tokens 必须大于 20 以容纳 System Prompt")

    def add_user_turn(self, content: str) -> None:
        """添加用户输入轮次."""
        if not content.strip():
            raise ValueError("用户输入不能为空")
        self._turns.append([{"role": "user", "content": content}])
        self._evict_if_needed()

    def add_assistant_text_turn(self, content: str) -> None:
        """添加纯文本助手回复轮次."""
        self._turns.append([{"role": "assistant", "content": content}])
        self._evict_if_needed()

    def add_atomic_tool_turn(
        self,
        assistant_msg: Dict[str, Any],
        tool_msgs: List[Dict[str, Any]],
    ) -> None:
        """将 assistant(tool_calls) 与所有对应的 tool 响应绑定为一个原子组."""
        tool_calls = assistant_msg.get("tool_calls", [])
        if not tool_calls:
            raise ValueError("assistant_msg 必须包含非空 tool_calls")

        expected_ids = {tc["id"] for tc in tool_calls}
        actual_ids = {tm.get("tool_call_id") for tm in tool_msgs}
        assert expected_ids == actual_ids, (
            f"原子组配对失败: expected={expected_ids}, actual={actual_ids}"
        )

        group = [assistant_msg] + list(tool_msgs)
        self._turns.append(group)
        self._evict_if_needed()

    def _system_msg(self) -> Dict[str, Any]:
        return {"role": "system", "content": self.system_prompt}

    def total_tokens(self) -> int:
        """计算当前窗口总 Token（含 System Prompt）."""
        total = estimate_tokens(self._system_msg())
        for turn in self._turns:
            total += sum(estimate_tokens(m) for m in turn)
        return total

    def _evict_if_needed(self) -> None:
        """按 FIFO 淘汰最旧的原子组，绝不拆散 tool_call 配对，且保留最新一轮."""
        while len(self._turns) > 1 and self.total_tokens() > self.max_tokens:
            self._turns.pop(0)

    def build_messages(self) -> List[Dict[str, Any]]:
        """输出可直接喂给 LLM API 的扁平消息列表，并校验 tool_call_id 完整性."""
        messages: List[Dict[str, Any]] = [self._system_msg()]
        for turn in self._turns:
            messages.extend(turn)

        # 关键不变量断言：窗口内不能出现悬空的 tool 消息或未闭合的 tool_calls
        pending_ids: set[str] = set()
        for msg in messages:
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    pending_ids.add(tc["id"])
            elif msg["role"] == "tool":
                tid = msg.get("tool_call_id")
                assert tid in pending_ids, f"检测到孤立的 tool 消息: {tid}"
                pending_ids.remove(tid)
        assert not pending_ids, f"检测到未闭合的 tool_calls: {pending_ids}"
        return messages


if __name__ == "__main__":
    window = AtomicMemoryWindow(
        max_tokens=160,
        system_prompt="你是企业工单排障专家，严禁编造工单状态。",
    )
    window.add_user_turn("帮我查一下工单 TK-101 为什么卡住了")
    window.add_atomic_tool_turn(
        assistant_msg={
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "name": "query_ticket", "arguments": {"ticket_id": "TK-101"}}],
        },
        tool_msgs=[
            {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "TIMEOUT", "node": "db-02"}'}
        ],
    )
    window.add_user_turn("那帮我再查一下 db-02 的 CPU 负载和连接数，看看是不是连接池打满了")
    window.add_atomic_tool_turn(
        assistant_msg={
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_2", "name": "check_node", "arguments": {"node": "db-02"}}],
        },
        tool_msgs=[
            {"role": "tool", "tool_call_id": "call_2", "content": '{"cpu": 0.96, "conn": "500/500"}'}
        ],
    )

    msgs = window.build_messages()
    assert msgs[0]["role"] == "system", "System Prompt 必须始终在首位"
    assert window.total_tokens() <= 160 or len(window._turns) == 1
    assert not any(m.get("tool_call_id") == "call_1" for m in msgs), "旧原子组应被整组淘汰"
    assert any(m.get("tool_call_id") == "call_2" for m in msgs), "最新原子组必须保留"
    print(f"Step 5 self-test passed! Remaining messages={len(msgs)}, tokens={window.total_tokens()}")
