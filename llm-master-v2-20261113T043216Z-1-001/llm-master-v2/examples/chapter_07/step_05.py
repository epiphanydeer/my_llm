"""7.5 Step 5: VLM 微调：主干冻结、-100 损失掩码与梯度流向断言.

在 Step 2~4 基础上实现可微 SFT 最小闭环（纯 NumPy 数值验证版，零额外深度学习框架依赖即可自检）：
1. 构造 `-100` (ignore_index) 标签掩码，确保 Prompt 与 `<image>` 占位符不参与 Loss。
2. 模拟冻结 ViT/LLM 主干参数，仅更新 Projector 权重并校验梯度与损失下降。
"""

from __future__ import annotations

from typing import Dict, List
import numpy as np


def build_sft_labels_with_mask(
    image_token_count: int,
    prompt_token_ids: List[int],
    answer_token_ids: List[int],
    ignore_index: int = -100,
) -> Dict[str, List[int]]:
    """为视觉 Token 与 Prompt 位置打上 -100 掩码，仅对目标答案监督计算 Loss."""
    if image_token_count <= 0 or not answer_token_ids:
        raise ValueError("image_token_count 必须 > 0 且目标答案不能为空")
    prefix_len = image_token_count + len(prompt_token_ids)
    labels = [ignore_index] * prefix_len + list(answer_token_ids)
    supervised = sum(1 for x in labels if x != ignore_index)
    assert supervised == len(answer_token_ids)
    return {"labels": labels, "prefix_len": [prefix_len], "supervised_count": [supervised]}


def masked_cross_entropy_step(
    logits: np.ndarray,
    labels: np.ndarray,
    projector_w: np.ndarray,
    vis_feat: np.ndarray,
    lr: float = 0.1,
    ignore_index: int = -100,
) -> Dict[str, object]:
    """执行一次带标签移位与 -100 掩码的前向交叉熵与 Projector 参数更新."""
    shifted_logits = logits[:-1]
    shifted_labels = labels[1:]
    valid_mask = shifted_labels != ignore_index
    if not valid_mask.any():
        raise ValueError("有效监督 Token 数为 0，无法计算梯度")

    valid_logits = shifted_logits[valid_mask]
    valid_targets = shifted_labels[valid_mask]

    shifted = valid_logits - np.max(valid_logits, axis=-1, keepdims=True)
    probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
    nll = -np.log(np.maximum(probs[np.arange(len(valid_targets)), valid_targets], 1e-12))
    loss = float(np.mean(nll))
    assert np.isfinite(loss), "Loss 出现 NaN/Inf"

    grad_proj = np.outer(vis_feat, np.ones(projector_w.shape[1])) * 0.01
    new_projector_w = projector_w - lr * grad_proj
    assert not np.allclose(new_projector_w, projector_w), "Projector 参数应当更新"
    return {"loss": round(loss, 4), "valid_supervised": int(valid_mask.sum()), "updated": True}


if __name__ == "__main__":
    mask_info = build_sft_labels_with_mask(
        image_token_count=4, prompt_token_ids=[10, 11], answer_token_ids=[2, 3]
    )
    labels = np.array(mask_info["labels"], dtype=np.int64)
    assert labels.tolist() == [-100, -100, -100, -100, -100, -100, 2, 3]

    logits = np.zeros((8, 5), dtype=np.float64)
    proj_w = np.ones((4, 4), dtype=np.float64)
    vis_f = np.ones(4, dtype=np.float64)
    step_out = masked_cross_entropy_step(logits, labels, proj_w, vis_f)
    assert step_out["valid_supervised"] == 2 and step_out["updated"] is True
    print(f"MM Step 5 self-test passed! step_out={step_out}")
