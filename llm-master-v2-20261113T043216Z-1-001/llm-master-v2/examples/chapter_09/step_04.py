"""9.4 Step 4: 手写 LayerNorm 与 RMSNorm (严格把 eps 加在根号内防除零)."""

from __future__ import annotations

import numpy as np


class LayerNorm:
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        self.gamma = np.ones((d_model,), dtype=np.float64)
        self.beta = np.zeros((d_model,), dtype=np.float64)
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normed = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * normed + self.beta


class RMSNorm:
    """LLaMA/Qwen 标配：省去均值中心化与 beta 偏置，仅用均方根缩放."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        self.gamma = np.ones((d_model,), dtype=np.float64)
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return self.gamma * (x / rms)


if __name__ == "__main__":
    zeros = np.zeros((2, 4, 16))
    assert np.isfinite(LayerNorm(16).forward(zeros)).all()
    assert np.isfinite(RMSNorm(16).forward(zeros)).all()
    x = np.random.default_rng(3).normal(5.0, 2.0, size=(2, 4, 16))
    assert np.allclose(np.mean(LayerNorm(16).forward(x), axis=-1), 0.0, atol=1e-6)
    print("Ch9 Step 4 self-test passed!")
