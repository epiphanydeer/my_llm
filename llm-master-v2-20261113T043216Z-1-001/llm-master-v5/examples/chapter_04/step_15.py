"""Step 15: Fact and Receipt Gate (终局事实核验与写操作回执门禁).

生产 Agent 上线前的最后一道防线，杜绝“口头谎报成功”与“无证据编造”：
1. 事实核验：输出中的关键结论必须能在工具观测证据 (observations) 中找到依据。
2. 写操作回执门禁：凡是声明“已退款/已重启/已关闭工单”的动作，必须在工具执行日志中存在对应的成功回执 (receipt_id)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
# [无需死背-脚手架] 回执与校验结果 dataclass 容器无需死背
class ToolReceipt:
    action_type: str
    receipt_id: str
    success: bool


@dataclass
class VerificationResult:
    passed: bool
    missing_evidence: List[str]
    missing_receipts: List[str]


class FactAndReceiptGate:
    """终局回答发布前的事实与回执门禁."""

    def verify(
        self,
        claimed_facts: List[str],
        claimed_write_actions: List[str],
        observations: List[str],
        receipts: List[ToolReceipt],
    ) -> VerificationResult:
        """核验每条关键事实与每项写操作声明是否均有真实工具证据支撑."""
        # [必记设计决策-事实证据核验] 最终输出宣称的关键数字/实体必须在工具 observations 中真实出现，防终局幻觉
        joined_obs = "\n".join(observations)
        missing_evidence = [fact for fact in claimed_facts if fact not in joined_obs]

        # [必背手撕-写操作回执门禁] 仅当工具执行返回 success=True 且带非空 receipt_id 时，才允许模型声称“已执行”
        valid_actions = {r.action_type for r in receipts if r.success and r.receipt_id.strip()}
        missing_receipts = [act for act in claimed_write_actions if act not in valid_actions]

        # [必记设计决策-双门禁缺一不可] 事实依据与写操作回执必须全部通过，未通过则退回重写或降级提示
        passed = not missing_evidence and not missing_receipts
        return VerificationResult(
            passed=passed,
            missing_evidence=missing_evidence,
            missing_receipts=missing_receipts,
        )


if __name__ == "__main__":
    gate = FactAndReceiptGate()

    # 1. 幻觉场景：模型口头声称“已重启 db-02”，但没有对应成功回执
    bad = gate.verify(
        claimed_facts=["db-02", "96%"],
        claimed_write_actions=["restart_service"],
        observations=['{"node": "db-02", "CPU": "96%"}'],
        receipts=[],
    )
    assert not bad.passed and bad.missing_receipts == ["restart_service"]

    # 2. 真实通过场景：既有观测事实，又有写操作回执
    good = gate.verify(
        claimed_facts=["db-02", "96%"],
        claimed_write_actions=["restart_service"],
        observations=['{"node": "db-02", "CPU": "96%"}'],
        receipts=[ToolReceipt(action_type="restart_service", receipt_id="rcpt-8891", success=True)],
    )
    assert good.passed and not good.missing_evidence and not good.missing_receipts
    print("Step 15 self-test passed!")
