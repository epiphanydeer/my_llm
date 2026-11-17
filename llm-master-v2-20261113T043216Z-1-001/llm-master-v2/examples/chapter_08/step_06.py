"""8.6 Step 6: GQA (Grouped-Query Attention) KV 头广播复用与显存压缩."""

from __future__ import annotations

import numpy as np


def repeat_kv(x: np.ndarray, n_rep: int) -> np.ndarray:
    """将 (B, H_kv, T, d_k) 沿头维度重复 n_rep 次对齐到 (B, H_q, T, d_k)."""
    if n_rep == 1:
        return x
    b, h_kv, t, dk = x.shape
    return np.repeat(x, repeats=n_rep, axis=1).reshape(b, h_kv * n_rep, t, dk)


if __name__ == "__main__":
    kv = np.random.default_rng(0).normal(size=(2, 4, 6, 16))
    expanded = repeat_kv(kv, n_rep=4)
    assert expanded.shape == (2, 16, 6, 16)
    assert np.allclose(expanded[:, 0], expanded[:, 1])
    print("Ch8 Step 6 self-test passed!")
