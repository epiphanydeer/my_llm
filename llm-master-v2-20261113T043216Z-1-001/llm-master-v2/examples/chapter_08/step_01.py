"""8.1 Step 1: RNN 串行依赖 vs Transformer 并行全局交互路径与复杂度对比."""

from __future__ import annotations

from typing import Dict


def compare_rnn_vs_transformer(seq_len: int, d_model: int) -> Dict[str, object]:
    if seq_len <= 0 or d_model <= 0:
        raise ValueError("seq_len 与 d_model 必须为正整数")
    return {
        "rnn_max_path_len": seq_len,
        "transformer_max_path_len": 1,
        "rnn_sequential_ops": seq_len,
        "transformer_sequential_ops": 1,
        "self_attn_compute_flops_order": (seq_len**2) * d_model,
        "rnn_compute_flops_order": seq_len * (d_model**2),
    }


if __name__ == "__main__":
    res = compare_rnn_vs_transformer(seq_len=1024, d_model=512)
    assert res["transformer_max_path_len"] == 1 and res["rnn_sequential_ops"] == 1024
    print(f"Ch8 Step 1 self-test passed! res={res}")
