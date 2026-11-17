"""8.2 Step 2: 根据架构类型 (Encoder / Decoder / Cross-Attention) 生成注意力掩码."""

from __future__ import annotations

import numpy as np


def build_attention_mask_by_arch(arch: str, t_q: int, t_k: int) -> np.ndarray:
    """True 表示允许关注, False 表示屏蔽."""
    if arch == "encoder":
        return np.ones((t_q, t_k), dtype=bool)
    if arch == "decoder":
        if t_q != t_k:
            raise ValueError("Decoder 自注意力要求 t_q == t_k")
        return np.tril(np.ones((t_q, t_k), dtype=bool), k=0)
    if arch == "cross_attention":
        return np.ones((t_q, t_k), dtype=bool)
    raise ValueError(f"未知架构: {arch}")


if __name__ == "__main__":
    enc = build_attention_mask_by_arch("encoder", 4, 4)
    dec = build_attention_mask_by_arch("decoder", 4, 4)
    cross = build_attention_mask_by_arch("cross_attention", 3, 5)
    assert enc.all() and not dec[0, 1] and dec[3, 0] and cross.shape == (3, 5)
    print("Ch8 Step 2 self-test passed!")
