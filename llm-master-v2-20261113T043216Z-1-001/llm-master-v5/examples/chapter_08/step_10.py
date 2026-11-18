"""8.10 Step 10: FlashAttention 核心思想——分块 Online Softmax 与全局 Softmax 数值等价验证."""

from __future__ import annotations

import numpy as np


def online_softmax_tiling(x: np.ndarray, block_size: int = 4) -> np.ndarray:
    """分块流式计算 Softmax，无需一次性把整行 (T_k,) 存进 SRAM."""
    m = -np.inf  # [必记设计决策-OnlineSoftmax状态] 仅维护局部最大值 m 与分母累加和 l，无需存整行到 SRAM
    l = 0.0
    for start in range(0, len(x), block_size):
        chunk = x[start : start + block_size]
        m_chunk = float(np.max(chunk))
        m_new = max(m, m_chunk)  # [必背手撕-分块最大值更新] 动态追踪全局最大 logit 保证指数不溢出
        l = l * np.exp(m - m_new) + float(np.sum(np.exp(chunk - m_new)))  # [必背手撕-重缩放累加] 用 exp(m - m_new) 修正历史分母并累加当前块
        m = m_new
    return np.exp(x - m) / l  # [必背手撕-最终归一化] 使用最终全局 m 与 l 输出等价 Softmax 概率


if __name__ == "__main__":  # [无需死背-脚手架] 数值等价性自测
    scores = np.random.default_rng(9).normal(0.0, 3.0, size=(20,))
    ref = np.exp(scores - np.max(scores)) / np.sum(np.exp(scores - np.max(scores)))
    tiled = online_softmax_tiling(scores, block_size=4)
    assert np.allclose(ref, tiled, atol=1e-12)
    print("Ch8 Step 10 self-test passed!")
