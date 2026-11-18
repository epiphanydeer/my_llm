"""1.4 应用概念与生成采样：Temperature、Top-K、Top-P (Nucleus) 采样器"""
import numpy as np
from typing import Dict


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)  # [必背手撕-数值稳定Softmax] 减去最大值防止 exp 溢出
    exp_v = np.exp(shifted)
    return exp_v / np.sum(exp_v)


def sample_next_token_probs(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> np.ndarray:
    """按标准推理引擎顺序执行：Temperature -> Top-K -> Top-P -> 归一化。"""
    if temperature <= 0.0:  # [必记设计决策-零温贪心退化] T<=0 时直接 argmax 取确定性输出
        out = np.zeros_like(logits, dtype=np.float64)
        out[int(np.argmax(logits))] = 1.0
        return out

    scaled = logits.astype(np.float64) / temperature  # [必背手撕-温度缩放] 先缩放 logits 再做截断

    if top_k > 0 and top_k < len(scaled):
        kth_val = np.partition(scaled, -top_k)[-top_k]  # [必记设计决策-采样顺序] 先 Top-K 屏蔽尾部，再转概率做 Top-P
        scaled = np.where(scaled < kth_val, -np.inf, scaled)

    probs = softmax(scaled)

    if 0.0 < top_p < 1.0:
        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]
        cum_probs = np.cumsum(sorted_probs)
        # 保留首个使累积概率 >= top_p 的 token，截断其后所有低概率长尾
        remove_mask = (cum_probs - sorted_probs) >= top_p  # [必背手撕-Nucleus截断边界] 保留首个使累积概率>=top_p的token
        filtered_sorted = np.where(remove_mask, 0.0, sorted_probs)
        probs = np.zeros_like(probs)
        probs[sorted_idx] = filtered_sorted
        probs = probs / np.sum(probs)

    return probs


if __name__ == "__main__":  # [无需死背-脚手架] 采样分布验证脚本
    raw_logits = np.array([4.0, 3.0, 1.5, 0.2, -1.0])
    greedy = sample_next_token_probs(raw_logits, temperature=0.0)
    assert greedy[0] == 1.0 and np.sum(greedy[1:]) == 0.0
    topk_topp = sample_next_token_probs(raw_logits, temperature=0.8, top_k=3, top_p=0.9)
    assert np.isclose(np.sum(topk_topp), 1.0) and topk_topp[3] == 0.0 and topk_topp[4] == 0.0
    print("[1.4 PASS] greedy:", greedy.round(3), "top_k+top_p:", topk_topp.round(3))
