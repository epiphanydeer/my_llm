"""5.2 Step 2: ChatML 多轮对话模板渲染与 SFT -100 损失掩码构造.

确保只有 Assistant 目标回复（含 <|im_end|> 结束符）参与交叉熵 Loss 计算，
System 与 User 提问全部打上 ignore_index = -100。
"""

from __future__ import annotations

from typing import Dict, List


def build_chatml_sft_sample(
    turns: List[Dict[str, str]],
    tokenize_fn,
    eos_id: int = 2,
    ignore_index: int = -100,
) -> Dict[str, List[int]]:
    """按轮次拼接 input_ids 与 labels，仅对 assistant 轮次保留真实 token_id."""
    if not turns:
        raise ValueError("对话轮次不能为空")

    input_ids: List[int] = []
    labels: List[int] = []
    assistant_tokens = 0

    for t in turns:
        role = t.get("role", "").strip()
        content = t.get("content", "")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"非法角色: {role}")

        toks = tokenize_fn(f"<{role}>\n{content}") + [eos_id]
        input_ids.extend(toks)
        if role == "assistant":
            labels.extend(toks)
            assistant_tokens += len(toks)
        else:
            labels.extend([ignore_index] * len(toks))

    assert len(input_ids) == len(labels)
    if assistant_tokens == 0:
        raise ValueError("样本中必须至少包含一轮有效的 assistant 回复")
    return {"input_ids": input_ids, "labels": labels, "supervised_tokens": [assistant_tokens]}


if __name__ == "__main__":
    simple_tok = lambda s: [ord(c) % 100 + 10 for c in s]
    sample = build_chatml_sft_sample(
        [
            {"role": "system", "content": "你是客服"},
            {"role": "user", "content": "怎么退款"},
            {"role": "assistant", "content": "7天内可退"},
        ],
        simple_tok,
    )
    assert sample["labels"][0] == -100 and sample["labels"][-1] == 2
    print("Ch5 Step 2 self-test passed!")
