"""8.10 Step 10: FlashAttention 核心思想——分块 Online Softmax 与全局 Softmax 数值等价验证."""

from __future__ import annotations

import numpy as np


def online_softmax_tiling(x: np.ndarray, block_size: int = 4) -> np.ndarray:
    """分块流式计算 Softmax，无需一次性把整行 (T_k,) 存进 SRAM."""
    m = -np.inf
    l = 0.0
    for start in range(0, len(x), block_size):
        chunk = x[start : start + block_size]
        m_chunk = float(np.max(chunk))
        m_new = max(m, m_chunk)
        l = l * np.exp(m - m_new) + float(np.sum(np.exp(chunk - m_new)))
        m = m_new
    return np.exp(x - m) / l


if __name__ == "__main__":
    scores = np.random.default_rng(9).normal(0.0, 3.0, size=(20,))
    ref = np.exp(scores - np.max(scores)) / np.sum(np.exp(scores - np.max(scores)))
    tiled = online_softmax_tiling(scores, block_size=4)
    assert np.allclose(ref, tiled, atol=1e-12)
    print("Ch8 Step 10 self-test passed!")
