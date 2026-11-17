"""8.9 Step 9: 单个 Pre-LN Block 的残差流形状与幅值稳定性追踪."""

from __future__ import annotations

import numpy as np


def rms_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def trace_pre_ln_block(x: np.ndarray) -> dict[str, object]:
    norm1 = rms_norm(x)
    h = x + norm1 * 0.1
    norm2 = rms_norm(h)
    out = h + norm2 * 0.1
    assert out.shape == x.shape
    return {"input_std": round(float(np.std(x)), 4), "output_std": round(float(np.std(out)), 4)}


if __name__ == "__main__":
    x = np.random.default_rng(3).normal(size=(2, 8, 32))
    res = trace_pre_ln_block(x)
    assert 0.8 < res["output_std"] < 1.3
    print(f"Ch8 Step 9 self-test passed! res={res}")
