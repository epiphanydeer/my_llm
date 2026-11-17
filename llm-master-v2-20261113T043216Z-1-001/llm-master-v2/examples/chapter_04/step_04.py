"""4.4 Step 4: 在 Step 1~3 基础上升级标准 ReActEngine（强制 tool_call_id 配对 + 单步防重）"""
from step_03 import ToolSandboxRegistry, ToolSpec


class ReActEngine:
    """标准 ReAct 循环内核：严格维护 assistant(tool_call) 与 tool(tool_call_id) 配对消息栈。"""
    def __init__(self, registry: ToolSandboxRegistry, max_steps: int = 5) -> None:
        self.registry = registry
        self.max_steps = max_steps

    def run(self, goal: str, llm_policy_fn, role: str = "viewer") -> dict[str, object]:
        messages: list[dict[str, object]] = [{"role": "user", "content": goal}]
        seen_calls: set[str] = set()

        for step in range(1, self.max_steps + 1):
            decision = llm_policy_fn(messages, step)
            if decision["type"] == "FINAL":
                messages.append({"role": "assistant", "content": decision["answer"]})
                return {"status": "COMPLETED", "steps": step, "answer": decision["answer"], "messages": messages}

            call_id = str(decision["call_id"])
            tname = str(decision["tool_name"])
            targs = dict(decision["args"])
            fp = f"{tname}:{sorted(targs.items())}"
            if fp in seen_calls:
                return {"status": "STOPPED_DUPLICATE", "steps": step, "answer": "检测到重复无效调用，终止循环。", "messages": messages}
            seen_calls.add(fp)

            messages.append({"role": "assistant", "tool_calls": [{"id": call_id, "name": tname, "args": targs}]})
            res = self.registry.execute(tname, targs, role=role)
            # 强制使用相同 tool_call_id 回传 tool 结果，防止接口报 400
            messages.append({"role": "tool", "tool_call_id": call_id, "content": str(res)})

        return {"status": "STOPPED_MAX_STEPS", "steps": self.max_steps, "answer": "达到最大步数限制。", "messages": messages}


if __name__ == "__main__":
    reg = ToolSandboxRegistry()
    reg.register(ToolSpec("query_order", "READ", ("order_id",), lambda a: "paid_not_shipped"))

    def policy(msgs, step):
        if step == 1:
            return {"type": "TOOL", "call_id": "tc_1", "tool_name": "query_order", "args": {"order_id": "O100"}}
        return {"type": "FINAL", "answer": "订单已支付待发货。"}

    out = ReActEngine(reg, max_steps=4).run("查订单 O100", policy)
    assert out["status"] == "COMPLETED" and out["messages"][1]["tool_calls"][0]["id"] == out["messages"][2]["tool_call_id"]
    print("Agent Step 4 self-test passed!")
