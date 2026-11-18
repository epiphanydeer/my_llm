"""8.9 Step 9: 单个 Pre-LN Block 的残差流形状与幅值稳定性追踪."""

from __future__ import annotations

import numpy as np


def rms_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)  # [必背手撕-RMSNorm核心] 仅用均方根缩放且 eps 在根号内防除零


def trace_pre_ln_block(x: np.ndarray) -> dict[str, object]:
    norm1 = rms_norm(x)  # [必记设计决策-Pre-LN位置] 归一化置于子层输入前，保留主干残差通路无阻
    h = x + norm1 * 0.1  # [必背手撕-注意力残差] 第一条残差分支汇合
    norm2 = rms_norm(h)
    out = h + norm2 * 0.1  # [必背手撕-FFN残差] 第二条残差分支汇合，保持 (B, T, C) 形状不变
    assert out.shape == x.shape
    return {"input_std": round(float(np.std(x)), 4), "output_std": round(float(np.std(out)), 4)}


if __name__ == "__main__":  # [无需死背-脚手架] 自测入口
    x = np.random.default_rng(3).normal(size=(2, 8, 32))
    res = trace_pre_ln_block(x)
    assert 0.8 < res["output_std"] < 1.3
    print(f"Ch8 Step 9 self-test passed! res={res}")
