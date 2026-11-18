"""9.3 Step 3: 手写 SwiGLU 门控前馈网络 (SiLU(x @ W_gate) * (x @ W_up)) @ W_down."""

from __future__ import annotations

import numpy as np


def silu(x: np.ndarray) -> np.ndarray:
    """Swish / SiLU 激活函数: x * sigmoid(x)，使用 clip 防止 exp(-x) 溢出."""
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))  # [必记设计决策-数值防溢出] clip 限制指数输入范围防止 exp(-x) 上溢
    return x * sig  # [必背手撕-SiLU激活] SiLU(x) = x * sigmoid(x)


class SwiGLUFeedForward:
    """现代大模型 (LLaMA/Qwen) 标配的 SwiGLU 门控前馈网络."""

    def __init__(self, d_model: int, d_ff: int, seed: int = 42) -> None:
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model 与 d_ff 必须为正整数")
        self.d_model = d_model
        self.d_ff = d_ff
        rng = np.random.default_rng(seed)
        scale = 0.02
        self.w_gate = rng.normal(0.0, scale, (d_model, d_ff))  # [必记设计决策-三矩阵门控] SwiGLU 含 gate/up/down 三个投影，d_ff 通常取 8/3 * d_model
        self.w_up = rng.normal(0.0, scale, (d_model, d_ff))
        self.w_down = rng.normal(0.0, scale, (d_ff, d_model))

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"输入必须为 (B, T, {self.d_model})")
        gated = silu(x @ self.w_gate) * (x @ self.w_up)  # [必背手撕-SwiGLU门控] 门控分支激活后与信息分支逐元素相乘
        out = gated @ self.w_down  # [必背手撕-降维投影] 映射回 (B, T, d_model)
        assert out.shape == x.shape
        return out


if __name__ == "__main__":  # [无需死背-脚手架] 自测入口
    ffn = SwiGLUFeedForward(d_model=16, d_ff=48)
    x = np.random.default_rng(1).normal(size=(2, 5, 16))
    out = ffn.forward(x)
    assert out.shape == (2, 5, 16) and np.isfinite(out).all()
    print("Ch9 Step 3 self-test passed!")
