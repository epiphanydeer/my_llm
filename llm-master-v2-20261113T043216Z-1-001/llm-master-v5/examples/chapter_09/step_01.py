"""9.1 Step 1: 手写缩放点积注意力 (因果上三角掩码 + Key Padding 掩码 + 防 NaN 安全 Softmax)."""

from __future__ import annotations

import math
from typing import Optional
import numpy as np


def safe_softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定 Softmax：减去最大值防溢出，且若整行被掩码为 -inf 则置 0 防 NaN."""
    max_val = np.max(scores, axis=axis, keepdims=True)  # [必背手撕-安全Softmax] 减去最大值防 exp 溢出
    is_all_masked = np.isneginf(max_val)  # [必记设计决策-全遮挡防NaN] 整行被 Mask 时 -inf - (-inf) 会产生 NaN，必须显式兜底置 0
    shifted = np.where(is_all_masked, -1e9, scores - max_val)  # [必背手撕-防NaN平移] 全遮挡行替换安全负值避免产生 NaN
    exp_x = np.exp(shifted)
    exp_x = np.where(is_all_masked, 0.0, exp_x)
    denom = np.sum(exp_x, axis=axis, keepdims=True)
    return np.where(denom > 0.0, exp_x / np.maximum(denom, 1e-12), 0.0)


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    causal: bool = True,
    key_padding_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:  # [无需死背-脚手架] 输入张量维度校验
        raise ValueError(f"Q/K/V 维度不匹配: q={q.shape}, k={k.shape}, v={v.shape}")

    d_k = q.shape[-1]
    t_q, t_k = q.shape[-2], k.shape[-2]
    scores = (q @ np.swapaxes(k, -1, -2)) / math.sqrt(d_k)  # [必背手撕-缩放点积] (..., T_q, d_k) @ (..., d_k, T_k) / sqrt(d_k)

    if causal:
        causal_mask = np.triu(np.ones((t_q, t_k), dtype=bool), k=1)  # [必背手撕-因果上三角掩码] k=1 严格屏蔽未来 token
        scores = np.where(causal_mask, -np.inf, scores)  # [必记设计决策-先Mask后Softmax] 置为 -inf 使 Softmax 后权重严格归零

    if key_padding_mask is not None:
        pad_mask = ~key_padding_mask  # [必背手撕-Padding掩码对齐] 取反并扩维广播至 scores 形状
        while pad_mask.ndim < scores.ndim:
            pad_mask = np.expand_dims(pad_mask, axis=1)
        scores = np.where(pad_mask, -np.inf, scores)

    attn_weights = safe_softmax(scores, axis=-1)
    out = attn_weights @ v  # [必背手撕-价值加权聚合] (..., T_q, T_k) @ (..., T_k, d_v)
    assert out.shape == (*q.shape[:-1], v.shape[-1]) and np.isfinite(out).all()
    return out, attn_weights


if __name__ == "__main__":  # [无需死背-脚手架] 双重掩码自测入口
    rng = np.random.default_rng(42)
    q, k, v = rng.normal(size=(2, 4, 8)), rng.normal(size=(2, 4, 8)), rng.normal(size=(2, 4, 8))
    pad_mask = np.array([[True, True, True, True], [True, True, False, False]])
    out, weights = scaled_dot_product_attention(q, k, v, causal=True, key_padding_mask=pad_mask)
    assert np.allclose(weights[0, 0, 1:], 0.0) and np.allclose(weights[1, :, 2:], 0.0)
    print("Ch9 Step 1 self-test passed!")
