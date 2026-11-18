"""8.8 Step 8: 残差分支方差缩放 (NANOGPT_SCALE_INIT: 1/sqrt(2*n_layer)) 稳定性验证."""

from __future__ import annotations

import math
import numpy as np


def simulate_residual_growth(num_layers: int = 24, d_model: int = 64, use_scale_init: bool = True) -> float:
    rng = np.random.default_rng(0)  # [无需死背-脚手架] 模拟输入初始化
    x = rng.normal(0.0, 1.0, size=(32, d_model))
    std = (0.02 / math.sqrt(2 * num_layers)) if use_scale_init else 0.02  # [必记设计决策-残差方差缩放] 每层有 Attn+FFN 两个残差分支，按 1/sqrt(2*L) 缩放防深层方差爆炸
    for _ in range(num_layers * 2):
        w = rng.normal(0.0, std * math.sqrt(d_model), size=(d_model, d_model))
        x = x + (x @ w) * 0.1  # [必背手撕-残差恒等通路] x + f(x) 保证梯度无损直通底层
    return round(float(np.std(x)), 4)


if __name__ == "__main__":  # [无需死背-脚手架] 对比实验自测
    std_scaled = simulate_residual_growth(24, 64, use_scale_init=True)
    std_unscaled = simulate_residual_growth(24, 64, use_scale_init=False)
    assert std_scaled < std_unscaled
    print(f"Ch8 Step 8 self-test passed! scaled_std={std_scaled}, unscaled_std={std_unscaled}")
