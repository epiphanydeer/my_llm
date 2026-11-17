# 第八章 Transformer 原理与面试深度拆解

本章聚焦概念闭卷复述 + 数学原理推导 + 形状/方差/复杂度验算，与第九章（纯手写 6 步拼装 TinyTransformerLM）形成完美互补。

| 小节 | 优先级 | 核心验算代码（Step 1~10） | 核心面试问题 |
| --- | --- | --- | --- |
| 8.1 为什么需要 Transformer | 【🔥 必会】 | compare_rnn_vs_transformer | 告别 RNN 串行瓶颈，训练并行 vs 推理串行 |
| 8.2 架构演进全景 | 【🔥 必会】 | build_attention_mask_by_arch | Encoder vs Decoder vs Encoder-Decoder |
| 8.3 端到端数据流 | 【🔥 必会】 | trace_transformer_shapes | (B,T)->(B,T,C)->(B,T,V) 与 Weight Tying |
| 8.4 QKV 三剑客 | 【🔥 必会】 | verify_qkv_asymmetry | 独立 $W_Q, W_K$ 打破对称性 |
| 8.5 Attention 缩放与 Mask | 【🔥 必会】 | verify_scaling_variance | 为什么除以 $\sqrt{d_k}$、-inf vs 0 |
| 8.6 多头与 GQA | 【🔥 必会】 | repeat_kv | 多子空间与 MHA -> MQA -> GQA 演进 |
| 8.7 位置编码与 RoPE | 【🔥 必会】 | apply_rope | RoPE 相对位置不变性与 NTK/YaRN 外推 |
| 8.8 残差、Norm 与 FFN | 【🔥 必会】 | simulate_residual_growth | LN vs RMSNorm、1/sqrt(2L) 缩放 |
| 8.9 Transformer Block | 【🔥 必会】 | trace_pre_ln_block | Pre-LN 双残差拓扑与 final_norm |
| 8.10 复杂度与 FlashAttention | 【🔥 必会】 | online_softmax_tiling | HBM vs SRAM、Online Softmax 与 7B 参数口算 |

