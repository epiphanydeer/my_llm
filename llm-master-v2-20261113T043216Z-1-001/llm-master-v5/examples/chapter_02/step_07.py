"""2.7 上下文窗口：预算分配与完整轮次（不拆散 tool_calls）裁剪器"""
from typing import Dict, List


def count_msg_tokens(msg: Dict[str, str]) -> int:  # [无需死背-脚手架] 简化版消息 Token 估算函数
    return max(4, len(msg.get("content", "")) // 2 + 4)


def trim_history_atomic_turns(
    system_msg: Dict[str, str],
    turns: List[List[Dict[str, str]]],
    max_input_tokens: int,
) -> List[Dict[str, str]]:
    """保留 System，按原子轮次（user + assistant/tool）从最新往回装入窗口。"""
    budget = max_input_tokens - count_msg_tokens(system_msg)  # [必记设计决策-System提示词豁免裁剪] 优先扣除系统消息预算
    kept_turns: List[List[Dict[str, str]]] = []
    for turn in reversed(turns):  # [必背手撕-逆序原子轮次装填] 从最新一轮往回装，超预算立即 break
        turn_cost = sum(count_msg_tokens(m) for m in turn)
        if turn_cost <= budget:
            kept_turns.append(turn)
            budget -= turn_cost
        else:
            break
    result = [system_msg]
    for turn in reversed(kept_turns):
        result.extend(turn)  # [必记设计决策-完整轮次不拆散] 保证 assistant(tool_calls) 与 tool 结果成对保留
    return result


if __name__ == "__main__":  # [无需死背-脚手架] 多轮含工具链裁剪验证
    sys_m = {"role": "system", "content": "系统规则"}
    t1 = [{"role": "user", "content": "第1轮很长的历史消息" * 5}, {"role": "assistant", "content": "收到"}]
    t2 = [
        {"role": "user", "content": "查温度"},
        {"role": "assistant", "content": "call_1"},
        {"role": "tool", "content": "68C"},
        {"role": "assistant", "content": "正常"},
    ]
    trimmed = trim_history_atomic_turns(sys_m, [t1, t2], max_input_tokens=60)
    assert trimmed[0]["role"] == "system"
    assert len(trimmed) == 5  # 完整保留第2轮(含tool)，未拆散任何调用链
    print("[2.7 PASS] kept messages:", len(trimmed))
