"""5.3 Step 3: 从零手写 LoRA 低秩注入层 (W_new = W + (alpha/r) * B @ A) 与权重合并."""

from __future__ import annotations

import numpy as np


class LoRALinear:
    """纯 NumPy 手写 LoRA 线性层（冻结主干 + 低秩旁路 + 无损合并）."""

    def __init__(self, in_features: int, out_features: int, r: int = 4, lora_alpha: int = 8) -> None:
        if r <= 0 or r >= min(in_features, out_features):
            raise ValueError(f"rank={r} 必须为正整数且远小于矩阵维度 ({in_features}, {out_features})")
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = lora_alpha / r  # [必背手撕-LoRA核心缩放系数 alpha/r]

        rng = np.random.default_rng(42)
        self.weight = rng.normal(0.0, 0.05, size=(in_features, out_features)).astype(np.float64)
        self.lora_A = rng.normal(0.0, 0.02, size=(r, in_features)).astype(np.float64)  # [必背手撕-A矩阵高斯随机初始化]
        self.lora_B = np.zeros((out_features, r), dtype=np.float64)  # [必背手撕-B矩阵全零初始化保证初始ΔW=0]
        self.merged = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(f"输入形状必须为 (B, {self.in_features})，当前为 {x.shape}")
        base_out = x @ self.weight  # [必背手撕-主干冻结前向路径]
        if self.merged:  # [必记设计决策-合并态直通零额外推理延迟]
            return base_out
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T  # [必背手撕-低秩旁路分解先降维再升维]
        return base_out + self.scaling * lora_out  # [必背手撕-LoRA核心前向公式 Wx + (alpha/r)BAx]

    def merge_and_unload(self) -> None:
        """将 ΔW = (alpha/r) * (A.T @ B.T) 永久合并进冻结权重 W，消除推理旁路延迟."""
        if self.merged:
            return
        delta_w = self.scaling * (self.lora_A.T @ self.lora_B.T)  # [必背手撕-LoRA核心合并公式 ΔW=(alpha/r)BA]
        assert delta_w.shape == self.weight.shape
        self.weight = self.weight + delta_w  # [必背手撕-无损合并进主干权重]
        self.merged = True

    def param_savings_ratio(self) -> float:
        base_params = self.in_features * self.out_features
        lora_params = self.lora_A.size + self.lora_B.size
        return round(lora_params / base_params, 4)


if __name__ == "__main__":  # [无需死背-脚手架]
    layer = LoRALinear(in_features=64, out_features=64, r=4, lora_alpha=8)
    x = np.random.default_rng(0).normal(size=(3, 64))
    assert np.allclose(layer.forward(x), x @ layer.weight, atol=1e-12)

    layer.lora_B = np.random.default_rng(1).normal(0.0, 0.03, size=(64, 4))
    out_before_merge = layer.forward(x)
    layer.merge_and_unload()
    out_after_merge = layer.forward(x)
    assert np.allclose(out_before_merge, out_after_merge, atol=1e-10)
    print(f"Ch5 Step 3 self-test passed! param_ratio={layer.param_savings_ratio()}")
