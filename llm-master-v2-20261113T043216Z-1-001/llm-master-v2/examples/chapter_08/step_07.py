"""8.7 Step 7: 手写 RoPE (旋转位置编码) 并验证内积只依赖相对距离 (m - n)."""

from __future__ import annotations

import numpy as np


def apply_rope(x: np.ndarray, pos: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """x: (T, D), D 为偶数 -> 对每对相邻维度 (2i, 2i+1) 施加角度 m * theta_i 旋转."""
    t, d = x.shape
    assert d % 2 == 0
    inv_freq = 1.0 / (base ** (np.arange(0, d, 2, dtype=np.float64) / d))
    angles = np.outer(pos, inv_freq)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x_even * cos_a - x_odd * sin_a
    out[:, 1::2] = x_even * sin_a + x_odd * cos_a
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    q, k = rng.normal(size=(1, 8)), rng.normal(size=(1, 8))
    dot_1 = float(np.sum(apply_rope(q, np.array([5])) * apply_rope(k, np.array([2]))))
    dot_2 = float(np.sum(apply_rope(q, np.array([15])) * apply_rope(k, np.array([12]))))
    assert np.isclose(dot_1, dot_2, atol=1e-10)
    print(f"Ch8 Step 7 self-test passed! dot_1={dot_1:.6f}, dot_2={dot_2:.6f}")
