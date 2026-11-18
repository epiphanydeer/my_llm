"""9.5 Step 5: 手写标准 Pre-LN TransformerBlock (RMSNorm + MHA + SwiGLU 双重残差闭环)."""

from __future__ import annotations

from typing import Optional
import numpy as np
from step_02 import MultiHeadAttention
from step_03 import SwiGLUFeedForward
from step_04 import RMSNorm


class TransformerBlock:
    """Pre-LN 架构：x = x + MHA(RMSNorm(x)); x = x + FFN(RMSNorm(x))."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, seed: int = 42) -> None:
        self.norm1 = RMSNorm(d_model)  # [无需死背-脚手架] 子层组件实例化
        self.attn = MultiHeadAttention(d_model, num_heads, seed=seed)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model, d_ff, seed=seed + 1)

    def forward(
        self,
        x: np.ndarray,
        causal: bool = True,
        key_padding_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        h = x + self.attn.forward(self.norm1.forward(x), causal=causal, key_padding_mask=key_padding_mask)  # [必背手撕-Pre-LN注意力残差] 先 Norm 再进 Attention，最后与原输入 x 相加
        out = h + self.ffn.forward(self.norm2.forward(h))  # [必背手撕-Pre-LN前馈残差] 先 Norm 再进 SwiGLU，保留主干梯度高速公路
        assert out.shape == x.shape  # [必记设计决策-形状不变性] 任意 Block 输入输出同为 (B, T, C) 才能任意堆叠
        return out


if __name__ == "__main__":  # [无需死背-脚手架] 自测入口
    blk = TransformerBlock(d_model=16, num_heads=4, d_ff=48)
    x = np.random.default_rng(5).normal(size=(2, 6, 16))
    assert blk.forward(x).shape == (2, 6, 16)
    print("Ch9 Step 5 self-test passed!")
