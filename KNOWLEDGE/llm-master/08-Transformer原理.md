# 第八章：Transformer 原理——从输入到一层 Block

> 本章优先级：🔥。仓库提供 9 篇原理文章；它们可作为 Build GPT 的中文解释。真正过关要能写出小模型、跟踪张量形状并解释训练结果。来源：[Transformer 目录](https://github.com/youngyangyang04/llm-master/tree/main/docs/llm/transformer)。

## 8.1 输入文本怎样变成张量【🔥】

**一句话：** 文本经过 tokenizer 变成 token ID；Embedding 表把每个 ID 映射到 C 维向量；模型还需要位置信息，然后进入多层 Block。

**例子：** 批量 B=2，每条序列 T=4 个 token，隐藏维 C=8。输入形状为 (2,4)，查 Embedding 后是 (2,4,8)。最后输出词表大小 V=1000 的 logits，形状为 (2,4,1000)；每个位置对应“下一个 token 各候选的分数”。

**费曼自述：** “模型先把词表编号变成向量；批量、时间长度和通道数分别占一个维度，最后把通道映射成词表分数。”

**概念闪卡：** Q：B、T、C 分别表示？A：批量、序列长度、隐藏维度。Q：logits 是概率吗？A：是未归一化分数，通常再经过 softmax 才是概率。

来源：[数据流](逐篇精读/transformer/transformer_data_flow.md) · [为什么 Transformer](逐篇精读/transformer/transformer_base_1.md)。

## 8.2 Encoder、Decoder、Encoder-Decoder【🔥】

**一句话：** Encoder 侧重对输入整体编码；自回归 Decoder 每次只依赖已有前文生成下一 token；Encoder-Decoder 用编码后的输入作为条件逐步生成。

**例子：** 读完整句子判断情绪可用 encoder 型结构；从“今天出差”续写后文要防止偷看未来 token，适合 decoder 的因果注意力；翻译是“先理解源句，再生成目标句”，常见于 encoder-decoder。现代模型有很多变体，不能仅凭名字断定内部细节。

**费曼自述：** “关键不是模型叫什么，而是每个位置能看哪些输入，以及输出是整体预测还是逐步生成。”

**概念闪卡：** Q：为什么 decoder 训练时需要因果 mask？A：防止当前位置看到未来答案。Q：encoder 注意力通常可看完整输入吗？A：典型双向 encoder 可以，但具体任务实现还要看 mask。

来源：[三种架构](逐篇精读/transformer/transformer_base_encoder_decoder.md)。

## 8.3 Q、K、V 与注意力计算【🔥】

**一句话：** 当前 token 的 Query 与各位置 Key 比较，得到注意力分数；分数经缩放、mask、softmax 后作为权重，对各位置 Value 加权求和。

~~~text
Q = XWq，K = XWk，V = XWv
score = QKᵀ / √dk
score 加上 mask（禁止看的位置设为极小）
weight = softmax(score)
output = weight × V
~~~

**最小手算：** 某位置对两个可见 token 的缩放后分数为 [0, 0]，softmax 得 [0.5, 0.5]；若对应 Value 是 [2,0] 和 [0,4]，输出就是 [1,2]。分数若变为 [2,0]，第一个 token 权重更大。mask 应在 softmax 前加入，不让被遮位置分到注意力权重。

**形状：** X=(B,T,C)；单头 Q、K、V 常为 (B,T,dₖ)；QKᵀ=(B,T,T)，表示每个位置对每个位置的分数。

**费曼自述：** “Q 像当前要找什么，K 像各位置的索引，V 是要取的信息。先比较相关性，再按权重汇总。”

**概念闪卡：** Q：为什么除以 √dₖ？A：控制点积分数随维度增大的尺度，避免 softmax 过度尖锐。Q：softmax 沿哪一维？A：通常沿可被关注的 key 位置维度。Q：因果 mask 应在什么时候加？A：softmax 之前。

来源：[Q/K/V 直觉](逐篇精读/transformer/qkv.md) · [一步步计算](逐篇精读/transformer/qkv_cal.md)。

## 8.4 多头注意力与维度变换【🔥】

**一句话：** 把隐藏维分给多个头，每个头做自己的 Q/K/V 注意力，再拼接并映射回隐藏维；多头增加可学习的表示路径，但不能保证每个头天然承担固定语法角色。

**例子：** B=2、T=4、C=8、头数 H=2，则每头 d=4。Q 可从 (2,4,8) 变成 (2,2,4,4)，维度分别是 (B,H,T,d)；注意力分数变成 (2,2,4,4)；两头输出合并回 (2,4,8)。容易错的是转置/reshape 后把 T 与 H 顺序弄反。

**费曼自述：** “多头不是简单把相同结果算两遍，而是有不同的可训练投影；每头在较小维度上处理，再合并。”

**概念闪卡：** Q：为什么 C 通常要能被 H 整除？A：要均分每头维度。Q：score 的最后两个维度是什么？A：查询位置 T 和被关注位置 T。

来源：[MHA 原理](逐篇精读/transformer/mha.md)。

## 8.5 位置编码与顺序【🔥】

**一句话：** 普通自注意力对输入顺序本身缺乏充分标识，模型需要位置信息来区分相同词以不同顺序出现的序列；实现有绝对、相对、旋转等多种形式。

**例子：** “狗追猫”与“猫追狗”词相同、顺序不同，任务答案可能完全相反。若每个 token 只有词向量而没有可用位置信号，模型难以稳定学出这种差别。仓库文章以正弦位置编码讲直觉；你在实际 GPT/VLM 中还会看到其他位置方案，不能把正弦编码当唯一标准。

**费曼自述：** “词向量说‘是什么’，位置表示补充‘在什么地方’，两者一起让模型处理顺序。”

**概念闪卡：** Q：位置编码为什么重要？A：词序影响语义与预测。Q：所有 Transformer 都使用正弦位置编码吗？A：不是。

来源：[位置编码](逐篇精读/transformer/pos_encode.md)。

## 8.6 FFN、残差与 LayerNorm【🔥】

**一句话：** 注意力在位置间交换信息；FFN 对每个位置的通道做非线性变换；残差保留输入通路；LayerNorm 稳定特征尺度。

~~~text
一种 Pre-Norm Block 示意：
X1 = X + Attention(LayerNorm(X))
X2 = X1 + FFN(LayerNorm(X1))
~~~

**例子：** X=(2,4,8)，注意力和 FFN 的最终输出也要是 (2,4,8)，才能与残差输入相加。FFN 中间可以临时扩到更大通道，再投影回 8。Pre-Norm 与 Post-Norm 把归一化放在不同位置，读实际代码时以其实现为准。

**费曼自述：** “注意力让 token 互相交流，FFN 在每个 token 内进一步加工；残差给原信息留路，归一化让训练更稳定。”

**概念闪卡：** Q：FFN 是否负责跨 token 通信？A：典型位置前馈层逐位置作用，跨位置交流主要来自注意力。Q：残差相加的形状要求？A：相同或可按设计兼容。

来源：[FFN/残差/LayerNorm](逐篇精读/transformer/ffn_ln.md) · [Block 结构](逐篇精读/transformer/transformer_structure.md)。

## 8.7 一层 Block 与整个 GPT【🔥】

**一句话：** 多层 Block 叠加仍保留 (B,T,C) 主形状，最后再映射到词表分数；训练时拿分数与真实下一 token 算损失，推理时按分数采样或选择新 token。

**例子：** 输入“安全帽应当”，模型预测下一 token；训练时真实续文提供目标，反向传播更新参数；生成时把新 token 追加进上下文继续预测。视觉大模型则还要把视觉表示放进输入序列或以其他结构接入。

**费曼自述：** “Block 是可重复的处理单元，不直接替我做完整训练；训练还需要数据、目标、损失、反传和验证。”

**概念闪卡：** Q：输出 logits 形状？A：常见自回归语言模型为 (B,T,V)。Q：跑通一次前向就说明模型学会了吗？A：不能，需训练/验证 loss 与任务测试。

来源：[Transformer Block](逐篇精读/transformer/transformer_structure.md)。

## 本章过关

闭卷画出 X→Q/K/V→注意力→FFN→logits；在 B=2、T=4、C=8、H=2 的例子上写对每一步形状；解释因果 mask 和验证 loss。卡住时回 [Build GPT 代码阅读路线](../deep-learning/02-Transformer_Example-代码阅读路线.md)。
