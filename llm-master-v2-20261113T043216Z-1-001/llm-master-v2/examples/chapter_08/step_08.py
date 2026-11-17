"""8.8 Step 8: 残差分支方差缩放 (NANOGPT_SCALE_INIT: 1/sqrt(2*n_layer)) 稳定性验证."""

from __future__ import annotations

import math
import numpy as np


def simulate_residual_growth(num_layers: int = 24, d_model: int = 64, use_scale_init: bool = True) -> float:
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=(32, d_model))
    std = (0.02 / math.sqrt(2 * num_layers)) if use_scale_init else 0.02
    for _ in range(num_layers * 2):
        w = rng.normal(0.0, std * math.sqrt(d_model), size=(d_model, d_model))
        x = x + (x @ w) * 0.1
    return round(float(np.std(x)), 4)


if __name__ == "__main__":
    std_scaled = simulate_residual_growth(24, 64, use_scale_init=True)
    std_unscaled = simulate_residual_growth(24, 64, use_scale_init=False)
    assert std_scaled < std_unscaled
    print(f"Ch8 Step 8 self-test passed! scaled_std={std_scaled}, unscaled_std={std_unscaled}")
