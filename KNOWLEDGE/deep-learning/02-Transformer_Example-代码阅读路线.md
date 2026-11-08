# Transformer_Example：从概念到 PyTorch 代码

源码副本：`external/Transformer_Example`（公开仓库 `lansinuote/Transformer_Example`）。

## 这不是 GPT，而是什么？

这是一个教学用的 **encoder–decoder Transformer**。它从输入序列 `x` 生成另一个序列 `y`：样例任务将字母转大写、数字变为 `9-数字`，再逆序。它包含 decoder 的自注意力和 decoder→encoder 的交叉注意力；标准 decoder-only GPT 没有 encoder，也没有这一交叉注意力。

## 先读运行路线，后读细节

```text
main.py
 ├─ 从 data.py 的 loader 得到 x:[8,50]、y:[8,51]
 ├─ model(x, y[:,:-1])
 ├─ CrossEntropyLoss 与 y[:,1:]
 └─ predict() 逐步生成

model.py
 ├─ PositionEmbedding：ID→[b,50,32] 并加位置
 ├─ Encoder：3 个 EncoderLayer
 ├─ Decoder：3 个 DecoderLayer
 └─ fc_out：[b,50,32]→[b,50,39]

util.py
 ├─ attention：QKᵀ/√8→mask→softmax→AV
 ├─ MultiHead：4 头、每头 8 维、残差/LN/输出投影
 ├─ PositionEmbedding：固定 sin/cos + Embedding
 └─ FullyConnectedOutput：32→64→32 的 FFN、残差/LN

mask.py
 ├─ mask_pad：禁止关注 PAD 的 key 列
 └─ mask_tril：禁止 decoder 看未来 token，也禁止看 PAD
```

## 固定超参数与形状字典

| 名称 | 值 / 形状 | 含义 |
|---|---|---|
| `b` | 训练时 8 | 一个 batch 里的样本数 |
| `T` | 50 | 固定序列长度 |
| `d_model` | 32 | 每个 token 的表示宽度 |
| `h` | 4 | 注意力头数 |
| `d_k=d_v` | 8 | 每头宽度，`32/4` |
| 词表大小 | 39 | 特殊符号、数字、小写/大写字母 |
| `Q/K/V` | `[b,4,50,8]` | 拆头后的张量 |
| 注意力分数 | `[b,4,50,50]` | 每个 query 对每个 key 的分数 |

## 阅读/练习协议

不要背代码，也不要一开始全量重写。

1. **跑通**：先运行，不改代码；记录一轮输入、目标、预测和 loss。
2. **地图**：从 `main.py` 顺着函数调用跳转；每个函数先写一句“输入→输出→职责”。
3. **预测形状**：读到 `reshape`、`permute`、`matmul`、`softmax` 前，先猜 shape 再验证。
4. **最小改动**：一次只改一处，例如打印 `attention` 的 shape，或将头数改为 2 并同步修改每头维度。
5. **复写小块**：只遮住 `attention()` 的 8–12 行，根据流程补写；不从头抄整个仓库。
6. **验证**：每次改动至少检查 shape、mask、softmax 行和、loss/预测其中一项。

## 分课顺序

1. `main.py`：训练目标、teacher forcing、`y[:,:-1]` 与 `y[:,1:]`；
2. `data.py`：数据、词表、Dataset/DataLoader；
3. `model.py`：模型总装配，先建立调用图；
4. `util.py: PositionEmbedding`：嵌入与 sin/cos；
5. `mask.py`：padding mask 与下三角因果 mask；
6. `util.py: attention()`：单头核心；
7. `util.py: MultiHead`：线性层、拆头、拼头、残差/LN；
8. `FullyConnectedOutput`、encoder/decoder 差异、`predict()`。

## 费曼问题

待开始 `main.py` 后填写。
