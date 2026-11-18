"""8.7 Step 7: 手写 RoPE (旋转位置编码) 并验证内积只依赖相对距离 (m - n)."""

from __future__ import annotations

import numpy as np


def apply_rope(x: np.ndarray, pos: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """x: (T, D), D 为偶数 -> 对每对相邻维度 (2i, 2i+1) 施加角度 m * theta_i 旋转."""
    t, d = x.shape
    assert d % 2 == 0  # [无需死背-脚手架] 偶数维度边界检查
    inv_freq = 1.0 / (base ** (np.arange(0, d, 2, dtype=np.float64) / d))  # [必背手撕-RoPE逆频率] 几何级数衰减频率 theta_i = base^(-2i/d)
    angles = np.outer(pos, inv_freq)  # [必背手撕-外积角度] 位置 m 与频率外积得到旋转角矩阵 (T, d/2)
    cos_a, sin_a = np.cos(angles), np.sin(angles)  # [必记设计决策-绝对转相对] 通过正余弦旋转让 QK 内积只依赖相对距离 (m - n)

    x_even = x[:, 0::2]  # [必背手撕-奇偶配对] 拆分偶维与奇维构成二维复数旋转对
    x_odd = x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x_even * cos_a - x_odd * sin_a  # [必背手撕-RoPE旋转实部] x_2i*cos - x_{2i+1}*sin
    out[:, 1::2] = x_even * sin_a + x_odd * cos_a  # [必背手撕-RoPE旋转虚部] x_2i*sin + x_{2i+1}*cos
    return out


if __name__ == "__main__":  # [无需死背-脚手架] 相对位置不变性自测
    rng = np.random.default_rng(7)
    q, k = rng.normal(size=(1, 8)), rng.normal(size=(1, 8))
    dot_1 = float(np.sum(apply_rope(q, np.array([5])) * apply_rope(k, np.array([2]))))
    dot_2 = float(np.sum(apply_rope(q, np.array([15])) * apply_rope(k, np.array([12]))))
    assert np.isclose(dot_1, dot_2, atol=1e-10)
    print(f"Ch8 Step 7 self-test passed! dot_1={dot_1:.6f}, dot_2={dot_2:.6f}")
