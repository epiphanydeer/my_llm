"""8.4 Step 4: 验证独立 W_Q / W_K 打破对称性，实现非对称词间检索."""

from __future__ import annotations

import numpy as np


def verify_qkv_asymmetry(x: np.ndarray) -> bool:
    rng = np.random.default_rng(1)
    d = x.shape[-1]
    wq = rng.normal(size=(d, d))
    wk = rng.normal(size=(d, d))
    sym_scores = x @ x.T
    asym_scores = (x @ wq) @ (x @ wk).T
    assert np.allclose(sym_scores, sym_scores.T)
    assert not np.allclose(asym_scores, asym_scores.T)
    return True


if __name__ == "__main__":
    x = np.random.default_rng(2).normal(size=(4, 8))
    assert verify_qkv_asymmetry(x) is True
    print("Ch8 Step 4 self-test passed!")
