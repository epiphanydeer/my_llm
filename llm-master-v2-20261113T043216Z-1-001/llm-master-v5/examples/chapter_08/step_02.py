"""8.2 Step 2: 根据架构类型 (Encoder / Decoder / Cross-Attention) 生成注意力掩码."""

from __future__ import annotations

import numpy as np


def build_attention_mask_by_arch(arch: str, t_q: int, t_k: int) -> np.ndarray:
    """True 表示允许关注, False 表示屏蔽."""
    if arch == "encoder":  # [必记设计决策-双向可见] Encoder 全连接双向注意力，无因果遮挡
        return np.ones((t_q, t_k), dtype=bool)
    if arch == "decoder":  # [必记设计决策-因果自回归] Decoder 要求 t_q == t_k 且禁止偷看未来 token
        if t_q != t_k:
            raise ValueError("Decoder 自注意力要求 t_q == t_k")
        return np.tril(np.ones((t_q, t_k), dtype=bool), k=0)  # [必背手撕-因果下三角掩码] 下三角 True 允许关注当前及历史
    if arch == "cross_attention":  # [必记设计决策-交叉注意力] Q 来自 Decoder(t_q)，K/V 来自 Encoder(t_k)，长度可不等
        return np.ones((t_q, t_k), dtype=bool)
    raise ValueError(f"未知架构: {arch}")


if __name__ == "__main__":  # [无需死背-脚手架] 自测入口与形状断言
    enc = build_attention_mask_by_arch("encoder", 4, 4)
    dec = build_attention_mask_by_arch("decoder", 4, 4)
    cross = build_attention_mask_by_arch("cross_attention", 3, 5)
    assert enc.all() and not dec[0, 1] and dec[3, 0] and cross.shape == (3, 5)
    print("Ch8 Step 2 self-test passed!")
