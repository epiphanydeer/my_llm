"""5.1 Step 1: RAG 与 SFT/DPO 选型决策路由器."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RequirementProfile:
    knowledge_changes_weekly: bool
    requires_exact_citation: bool
    needs_strict_format_or_style: bool
    has_preference_pairs: bool
    labeled_samples: int


def route_rag_vs_sft(req: RequirementProfile) -> Dict[str, str]:
    """根据业务约束输出确定性的技术选型建议."""
    if req.labeled_samples < 0:
        raise ValueError("labeled_samples 不能为负数")

    if req.knowledge_changes_weekly or req.requires_exact_citation:
        if req.needs_strict_format_or_style and req.labeled_samples >= 500:
            return {"route": "RAG + SFT_LORA", "reason": "动态知识靠 RAG 检索，固定格式与话术靠 LoRA 固化"}
        return {"route": "RAG_ONLY", "reason": "知识高频变化且需溯源，微调不仅贵且无法实时更新"}

    if req.has_preference_pairs and req.labeled_samples >= 300:
        return {"route": "SFT_THEN_DPO", "reason": "先 SFT 学会基本能力，再用 DPO 对齐偏好与拒答边界"}

    if req.needs_strict_format_or_style and req.labeled_samples >= 200:
        return {"route": "SFT_LORA", "reason": "静态规则/格式/语气定制，LoRA 性价比最高"}

    return {"route": "PROMPT_ENGINEERING", "reason": "样本不足或需求简单，先用 Prompt + Few-shot 跑通基线"}


if __name__ == "__main__":
    assert route_rag_vs_sft(RequirementProfile(True, True, False, False, 1000))["route"] == "RAG_ONLY"
    assert route_rag_vs_sft(RequirementProfile(False, False, True, False, 800))["route"] == "SFT_LORA"
    assert route_rag_vs_sft(RequirementProfile(False, False, True, True, 500))["route"] == "SFT_THEN_DPO"
    print("Ch5 Step 1 self-test passed!")
