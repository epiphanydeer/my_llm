"""2.3 Few-shot、推理与反思：任务复杂度与推理策略路由器"""
from dataclasses import dataclass


@dataclass
class PromptStrategy:
    mode: str  # "zero_shot" | "few_shot" | "cot" | "self_refine"
    reason: str


def select_prompt_strategy(
    is_reasoning_model: bool,
    needs_strict_format: bool,
    multi_step_logic: bool,
    high_stakes_draft: bool,
) -> PromptStrategy:
    """根据模型类型与任务特征选择最小充分推理策略，避免对推理模型过度诱导 CoT。"""
    if is_reasoning_model:
        return PromptStrategy("zero_shot", "推理模型原生具备长链思考能力，直接给清晰目标与约束即可")
    if high_stakes_draft:
        return PromptStrategy("self_refine", "高风险输出需生成-校验-修订两阶段闭环")
    if multi_step_logic:
        return PromptStrategy("cot", "普通对话模型处理多步计算/逻辑推导需显式分步")
    if needs_strict_format:
        return PromptStrategy("few_shot", "通过2-3个正反样例锚定输出格式与边界口径")
    return PromptStrategy("zero_shot", "简单明确任务直接零样本调用以降低 Token 成本")


if __name__ == "__main__":
    s1 = select_prompt_strategy(True, False, True, False)
    s2 = select_prompt_strategy(False, False, True, False)
    s3 = select_prompt_strategy(False, True, False, False)
    assert s1.mode == "zero_shot" and s2.mode == "cot" and s3.mode == "few_shot"
    print("[2.3 PASS]", s1.mode, s2.mode, s3.mode)
