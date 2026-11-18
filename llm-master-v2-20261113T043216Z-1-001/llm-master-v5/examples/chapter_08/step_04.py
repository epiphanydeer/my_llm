"""8.4 Step 4: 验证独立 W_Q / W_K 打破对称性，实现非对称词间检索."""

from __future__ import annotations

import numpy as np


def verify_qkv_asymmetry(x: np.ndarray) -> bool:
    rng = np.random.default_rng(1)  # [无需死背-脚手架] 随机数与测试矩阵初始化
    d = x.shape[-1]
    wq = rng.normal(size=(d, d))
    wk = rng.normal(size=(d, d))
    sym_scores = x @ x.T  # [必记设计决策-对称缺陷] 直接 x @ x.T 必然对称，无法区分主语找谓语与谓语找主语
    asym_scores = (x @ wq) @ (x @ wk).T  # [必背手撕-非对称投影] 独立 W_Q / W_K 线性投影打破对称性，实现有向检索
    assert np.allclose(sym_scores, sym_scores.T)  # [无需死背-脚手架] 对称性对比断言
    assert not np.allclose(asym_scores, asym_scores.T)
    return True


if __name__ == "__main__":  # [无需死背-脚手架] 自测入口
    x = np.random.default_rng(2).normal(size=(4, 8))
    assert verify_qkv_asymmetry(x) is True
    print("Ch8 Step 4 self-test passed!")
