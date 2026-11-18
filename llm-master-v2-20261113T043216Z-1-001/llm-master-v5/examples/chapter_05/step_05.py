"""5.5 Step 5: 微调消融与收益归因追踪器 (Zero-shot vs Few-shot vs LoRA)."""

from __future__ import annotations

from typing import Dict, List


def compare_sft_ablation(
    gold: List[str],
    prompt_preds: List[str],
    lora_preds: List[str],
) -> Dict[str, object]:
    if not gold or not (len(gold) == len(prompt_preds) == len(lora_preds)):  # [无需死背-脚手架]
        raise ValueError("三组序列长度必须一致且非空")

    fixed = sum(1 for g, p, l in zip(gold, prompt_preds, lora_preds) if p != g and l == g)  # [必背手撕-成对消融Fixed错改对计数]
    regressed = sum(1 for g, p, l in zip(gold, prompt_preds, lora_preds) if p == g and l != g)  # [必背手撕-成对消融Regressed对改错计数]
    prompt_acc = sum(1 for g, p in zip(gold, prompt_preds) if p == g) / len(gold)
    lora_acc = sum(1 for g, l in zip(gold, lora_preds) if l == g) / len(gold)
    return {
        "prompt_acc": round(prompt_acc, 4),
        "lora_acc": round(lora_acc, 4),
        "fixed": fixed,
        "regressed": regressed,
        "net_gain": fixed - regressed,
        "worth_deploying": lora_acc > prompt_acc and (fixed - regressed) > 0,  # [必记设计决策-净收益为正且胜过Prompt基线才准上线]
    }


if __name__ == "__main__":  # [无需死背-脚手架]
    res = compare_sft_ablation(
        gold=["A", "B", "C", "D"],
        prompt_preds=["A", "X", "X", "D"],
        lora_preds=["A", "B", "C", "Y"],
    )
    assert res["fixed"] == 2 and res["regressed"] == 1 and res["worth_deploying"] is True
    print(f"Ch5 Step 5 self-test passed! res={res}")
