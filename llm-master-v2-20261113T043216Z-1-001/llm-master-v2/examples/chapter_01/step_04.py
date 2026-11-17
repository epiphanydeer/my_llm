"""1.4 应用概念与生成采样：Temperature、Top-K、Top-P (Nucleus) 采样器"""
import numpy as np
from typing import Dict


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_v = np.exp(shifted)
    return exp_v / np.sum(exp_v)


def sample_next_token_probs(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> np.ndarray:
    """按标准推理引擎顺序执行：Temperature -> Top-K -> Top-P -> 归一化。"""
    if temperature <= 0.0:
        out = np.zeros_like(logits, dtype=np.float64)
        out[int(np.argmax(logits))] = 1.0
        return out

    scaled = logits.astype(np.float64) / temperature

    if top_k > 0 and top_k < len(scaled):
        kth_val = np.partition(scaled, -top_k)[-top_k]
        scaled = np.where(scaled < kth_val, -np.inf, scaled)

    probs = softmax(scaled)

    if 0.0 < top_p < 1.0:
        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]
        cum_probs = np.cumsum(sorted_probs)
        # 保留首个使累积概率 >= top_p 的 token，截断其后所有低概率长尾
        remove_mask = (cum_probs - sorted_probs) >= top_p
        filtered_sorted = np.where(remove_mask, 0.0, sorted_probs)
        probs = np.zeros_like(probs)
        probs[sorted_idx] = filtered_sorted
        probs = probs / np.sum(probs)

    return probs


if __name__ == "__main__":
    raw_logits = np.array([4.0, 3.0, 1.5, 0.2, -1.0])
    greedy = sample_next_token_probs(raw_logits, temperature=0.0)
    assert greedy[0] == 1.0 and np.sum(greedy[1:]) == 0.0
    topk_topp = sample_next_token_probs(raw_logits, temperature=0.8, top_k=3, top_p=0.9)
    assert np.isclose(np.sum(topk_topp), 1.0) and topk_topp[3] == 0.0 and topk_topp[4] == 0.0
    print("[1.4 PASS] greedy:", greedy.round(3), "top_k+top_p:", topk_topp.round(3))
