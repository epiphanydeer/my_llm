"""4.1 Step 1: 最小 O-T-A (Observe-Think-Act) 有限状态机原型（纯标准库）"""
from dataclasses import dataclass, field


@dataclass
# [无需死背-脚手架] @dataclass 状态容器定义无需死背，重点记住 step / max_steps / observations / status 四字段
class AgentState:
    goal: str
    step: int = 0
    max_steps: int = 4
    observations: list[str] = field(default_factory=list)
    status: str = "RUNNING"  # RUNNING | DONE | STOPPED_MAX_STEPS


def run_ota_agent(goal: str, env_fn, max_steps: int = 4) -> AgentState:
    """最小 Agent 闭环：Observe -> Think -> Act，受 max_steps 硬上限约束。"""
    # [必记设计决策-入口边界] 入口强制校验空目标与 max_steps > 0，防止非法配置导致死循环
    if not goal.strip() or max_steps <= 0:
        raise ValueError("goal 不能为空且 max_steps 必须为正整数")
    state = AgentState(goal=goal, max_steps=max_steps)

    # [必背手撕-OTA闭环] 面试常考极简 Agent 循环：Check max_steps -> Think -> Act -> Observe -> 终态判断
    while state.status == "RUNNING":
        # [必记设计决策-步数硬上限] 任何 Agent Loop 必须把 max_steps 熔断写在每轮迭代最前面
        if state.step >= state.max_steps:
            state.status = "STOPPED_MAX_STEPS"
            break
        state.step += 1
        # Think：根据已有观察决定下一步动作
        if not state.observations:
            action = "check_metrics"
        elif "502_spike" in state.observations[-1]:
            action = "check_logs"
        else:
            state.status = "DONE"
            break
        # Act & Observe
        obs = env_fn(action)
        state.observations.append(f"{action}:{obs}")
        if "timeout_upstream" in obs:
            state.status = "DONE"

    assert state.step <= state.max_steps, "绝不允许突破最大步数"
    return state


if __name__ == "__main__":
    mock_env = lambda act: "502_spike" if act == "check_metrics" else "timeout_upstream"
    st = run_ota_agent("定位支付接口变慢原因", mock_env, max_steps=4)
    assert st.status == "DONE" and st.step == 2
    st_stop = run_ota_agent("无限探索", lambda act: "502_spike", max_steps=3)
    assert st_stop.status == "STOPPED_MAX_STEPS" and st_stop.step == 3
    print("Agent Step 1 self-test passed!")
