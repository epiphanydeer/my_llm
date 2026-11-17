"""4.2 Step 2: 在 Step 1 基础上新增 HybridRouter 混合路由分诊台（Workflow vs Agent）"""
from step_01 import run_ota_agent


def workflow_refund_check(order_id: str, is_shipped: bool, days_since_pay: int) -> dict[str, str]:
    """确定性规则分支 Workflow：规则明确写死在代码里，不让模型自由发挥。"""
    if days_since_pay > 7:
        return {"route": "WORKFLOW", "decision": "REJECT_EXPIRED", "reason": "超过7天退货期"}
    if not is_shipped:
        return {"route": "WORKFLOW", "decision": "AUTO_REFUND", "reason": "未发货自动退款"}
    return {"route": "WORKFLOW", "decision": "MANUAL_REVIEW", "reason": "已发货转人工审核"}


def hybrid_dispatch(task_type: str, payload: dict[str, object]) -> dict[str, object]:
    """混合分诊：流程确定走 Workflow，路径不确定才放行进 Step 1 的 Agent Loop。"""
    if task_type == "REFUND_CHECK":
        return workflow_refund_check(
            str(payload["order_id"]), bool(payload["is_shipped"]), int(payload["days"])
        )
    if task_type == "OPEN_FAULT_DIAGNOSIS":
        st = run_ota_agent(str(payload["goal"]), payload["env_fn"], max_steps=4)
        return {"route": "AGENT", "status": st.status, "steps": st.step, "evidence": st.observations}
    raise ValueError(f"未知任务类型: {task_type}")


if __name__ == "__main__":
    r_wf = hybrid_dispatch("REFUND_CHECK", {"order_id": "O1", "is_shipped": False, "days": 2})
    assert r_wf["route"] == "WORKFLOW" and r_wf["decision"] == "AUTO_REFUND"
    r_ag = hybrid_dispatch("OPEN_FAULT_DIAGNOSIS", {"goal": "查慢接口", "env_fn": lambda a: "ok"})
    assert r_ag["route"] == "AGENT" and r_ag["status"] == "DONE"
    print("Agent Step 2 self-test passed!")
