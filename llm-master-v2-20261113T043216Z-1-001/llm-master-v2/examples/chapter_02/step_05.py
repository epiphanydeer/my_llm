"""2.5 Function Calling：参数校验、并行执行与 tool_call_id 严格配对闭环"""
import json
from typing import Any, Callable, Dict, List


def execute_tool_calls(
    assistant_msg: Dict[str, Any],
    registry: Dict[str, Callable[..., Dict[str, Any]]],
) -> List[Dict[str, str]]:
    """按 OpenAI Tool Calling 协议执行全部 tool_calls，并按 tool_call_id 逐一回传。"""
    tool_messages: List[Dict[str, str]] = []
    for tc in assistant_msg.get("tool_calls", []):
        call_id = tc["id"]
        fn_name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"]["arguments"])
            if fn_name not in registry:
                payload = {"error": f"未知工具: {fn_name}"}
            else:
                payload = registry[fn_name](**args)
        except Exception as exc:
            payload = {"error": f"参数或执行异常: {exc}"}

        tool_messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(payload, ensure_ascii=False),
        })
    return tool_messages


if __name__ == "__main__":
    tools = {"get_device_temp": lambda device_id: {"device_id": device_id, "temp_c": 68.5}}
    assistant_turn = {
        "role": "assistant",
        "tool_calls": [
            {"id": "call_01", "function": {"name": "get_device_temp", "arguments": '{"device_id": "crusher-A"}'}},
            {"id": "call_02", "function": {"name": "nonexistent_tool", "arguments": "{}"}},
        ],
    }
    outputs = execute_tool_calls(assistant_turn, tools)
    assert len(outputs) == 2
    assert outputs[0]["tool_call_id"] == "call_01" and "68.5" in outputs[0]["content"]
    assert outputs[1]["tool_call_id"] == "call_02" and "未知工具" in outputs[1]["content"]
    print("[2.5 PASS]", outputs)
