"""9.2 Step 2: 手写多头注意力 (MultiHeadAttention: 两步拆头 -> 独立点积 -> 连续拼接)."""

from __future__ import annotations

from typing import Optional
import numpy as np
from step_01 import scaled_dot_product_attention


class MultiHeadAttention:
    """纯 NumPy 多头自注意力，严格校验 (B, T, C) <-> (B, H, T, d_k) 维度变换."""

    def __init__(self, d_model: int, num_heads: int, seed: int = 42) -> None:
        if d_model % num_heads != 0:  # [无需死背-脚手架] 整除校验
            raise ValueError(f"d_model={d_model} 必须能被 num_heads={num_heads} 整除")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # [必记设计决策-均分头维度] 总维度不变下拆分为 H 个独立子空间

        rng = np.random.default_rng(seed)
        scale = 0.02
        self.wq = rng.normal(0.0, scale, (d_model, d_model))  # [无需死背-脚手架] 权重矩阵初始化
        self.wk = rng.normal(0.0, scale, (d_model, d_model))
        self.wv = rng.normal(0.0, scale, (d_model, d_model))
        self.wo = rng.normal(0.0, scale, (d_model, d_model))

    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """(B, T, C) -> (B, T, H, d_k) -> (B, H, T, d_k)，绝不可直接一步 reshape."""
        b, t, c = x.shape
        return np.swapaxes(x.reshape(b, t, self.num_heads, self.d_k), 1, 2)  # [必背手撕-两步拆头] 先 reshape(B,T,H,d_k) 再 swapaxes(1,2)->(B,H,T,d_k)，严禁直接一步 reshape

    def combine_heads(self, x: np.ndarray) -> np.ndarray:
        """(B, H, T, d_k) -> (B, T, H, d_k) -> (B, T, C)."""
        b, h, t, dk = x.shape
        return np.ascontiguousarray(np.swapaxes(x, 1, 2)).reshape(b, t, h * dk)  # [必背手撕-连续拼接] swapaxes 回 (B,T,H,d_k) 后确保内存连续再合并为 (B,T,C)

    def forward(
        self,
        x: np.ndarray,
        causal: bool = True,
        key_padding_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"输入形状必须为 (B, T, {self.d_model})，当前为 {x.shape}")
        q = self.split_heads(x @ self.wq)  # [必背手撕-QKV投影拆头] 线性投影后拆分多头
        k = self.split_heads(x @ self.wk)
        v = self.split_heads(x @ self.wv)
        attn_out, _ = scaled_dot_product_attention(q, k, v, causal=causal, key_padding_mask=key_padding_mask)
        merged = self.combine_heads(attn_out)
        out = merged @ self.wo  # [必记设计决策-W_O跨头融合] 拼接后经 W_O 投影融合多头语义信息
        assert out.shape == x.shape
        return out


if __name__ == "__main__":  # [无需死背-脚手架] 自测入口
    mha = MultiHeadAttention(d_model=16, num_heads=4)
    x = np.random.default_rng(0).normal(size=(2, 5, 16))
    assert mha.forward(x, causal=True).shape == (2, 5, 16)
    print("Ch9 Step 2 self-test passed!")
