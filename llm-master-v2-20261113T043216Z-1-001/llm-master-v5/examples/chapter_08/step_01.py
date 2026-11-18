"""8.1 Step 1: RNN 串行依赖 vs Transformer 并行全局交互路径与复杂度对比."""

from __future__ import annotations

from typing import Dict


def compare_rnn_vs_transformer(seq_len: int, d_model: int) -> Dict[str, object]:
    if seq_len <= 0 or d_model <= 0:  # [无需死背-脚手架] 参数边界校验
        raise ValueError("seq_len 与 d_model 必须为正整数")
    return {
        "rnn_max_path_len": seq_len,  # [必记设计决策-路径长度] RNN 任意两词依赖路径为 O(T)，长程梯度易衰减
        "transformer_max_path_len": 1,  # [必背手撕-全局直连] Self-Attention 任意两词交互路径恒为 O(1)
        "rnn_sequential_ops": seq_len,  # [必记设计决策-串行瓶颈] RNN 必须按时间步串行 O(T)，无法跨步并行
        "transformer_sequential_ops": 1,  # [必背手撕-并行计算] Transformer 训练期所有位置 O(1) 并行
        "self_attn_compute_flops_order": (seq_len**2) * d_model,  # [必背手撕-复杂度] 注意力复杂度 O(T^2 * d)，长序列瓶颈所在
        "rnn_compute_flops_order": seq_len * (d_model**2),  # [必记设计决策-复杂度拐点] 当 T < d 时 Self-Attention 计算量反而低于 RNN
    }


if __name__ == "__main__":  # [无需死背-脚手架] 自测入口与断言
    res = compare_rnn_vs_transformer(seq_len=1024, d_model=512)
    assert res["transformer_max_path_len"] == 1 and res["rnn_sequential_ops"] == 1024
    print(f"Ch8 Step 1 self-test passed! res={res}")
