"""8.3 Step 3: 端到端张量形状追踪 ((B, T) -> (B, T, C) -> (B, T, V)) 与 Weight Tying."""

from __future__ import annotations

import numpy as np


def trace_transformer_shapes(batch: int, seq_len: int, d_model: int, vocab_size: int) -> dict[str, tuple[int, ...]]:
    rng = np.random.default_rng(0)  # [无需死背-脚手架] 随机数生成器与模拟输入
    tok_ids = rng.integers(0, vocab_size, size=(batch, seq_len))
    emb_table = rng.normal(size=(vocab_size, d_model))
    x = emb_table[tok_ids]  # [必背手撕-查表映射] (B, T) 整数索引查表映射为 (B, T, d_model)
    logits = x @ emb_table.T  # [必记设计决策-权重共享] Weight Tying 复用词表矩阵转置输出 (B, T, V)，省去海量参数
    assert x.shape == (batch, seq_len, d_model)  # [无需死背-脚手架] 形状断言校验
    assert logits.shape == (batch, seq_len, vocab_size)
    return {"tok_ids": tok_ids.shape, "hidden": x.shape, "logits": logits.shape}


if __name__ == "__main__":  # [无需死背-脚手架] 自测入口
    shapes = trace_transformer_shapes(2, 6, 16, 100)
    assert shapes["logits"] == (2, 6, 100)
    print(f"Ch8 Step 3 self-test passed! shapes={shapes}")
