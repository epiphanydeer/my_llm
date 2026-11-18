# 第九章 手写 Transformer：从底层算子到训练生成

本章全程纯 NumPy + 标准库零黑盒从零手写，6 个模块逐层搭积木，到 9.6 节自然拼装成完整可运行的 TinyTransformerLM！

| 小节 | 优先级 | 核心手写积木（Step 1~6） | 核心手撕考点 |
| --- | --- | --- | --- |
| 9.1 手写缩放点积注意力 | 【🔥 必会】 | scaled_dot_product_attention | 因果 + Padding 双重掩码、防全 -inf NaN safe_softmax |
| 9.2 手写多头注意力 | 【🔥 必会】 | MultiHeadAttention | (B,T,C) -> (B,H,T,d_k) 两步规范拆头与连续拼接 |
| 9.3 手写 FFN 与 SwiGLU | 【🔥 必会】 | SwiGLUFeedForward | SiLU 防溢出、三矩阵门控与 $8/3 \cdot d$ 维度对齐 |
| 9.4 手写 LayerNorm 与 RMSNorm | 【🔥 必会】 | LayerNorm & RMSNorm | 严格把 eps 加在根号内防除零、混精度 FP32 归约 |
| 9.5 手写 TransformerBlock | 【🔥 必会】 | TransformerBlock | Pre-LN 双残差流闭环组装 |
| 9.6 TinyTransformerLM | 【🔥 必会】 | TinyTransformerLM | Embedding + Blocks + FinalNorm + Weight Tying + generate() |
