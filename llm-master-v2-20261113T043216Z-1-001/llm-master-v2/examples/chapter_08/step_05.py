"""8.5 Step 5: 数值验证为什么点积注意力必须除以 sqrt(d_k) (方差从 d_k 归 1 防梯度消失)."""

from __future__ import annotations

import math
import numpy as np


def verify_scaling_variance(d_k: int = 256, n_samples: int = 2000) -> dict[str, float]:
    rng = np.random.default_rng(42)
    q = rng.normal(0.0, 1.0, size=(n_samples, d_k))
    k = rng.normal(0.0, 1.0, size=(n_samples, d_k))
    raw_dots = np.sum(q * k, axis=-1)
    scaled_dots = raw_dots / math.sqrt(d_k)
    return {
        "raw_var": round(float(np.var(raw_dots)), 2),
        "scaled_var": round(float(np.var(scaled_dots)), 2),
    }


if __name__ == "__main__":
    stats = verify_scaling_variance(256)
    assert stats["raw_var"] > 200.0 and 0.8 < stats["scaled_var"] < 1.2
    print(f"Ch8 Step 5 self-test passed! stats={stats}")
