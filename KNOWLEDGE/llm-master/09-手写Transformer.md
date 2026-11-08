# 第九章：手写 Transformer——把原理变成代码

> 本章优先级：🔥 亲手实现与调试；仓库 6 篇代码文章可看解释，但“照抄示例能跑”不算掌握。尤其 [Tiny Transformer 文章](逐篇精读/transformer/tiny_transformer_code.md)主要演示 NumPy 前向，没有形成可训练、带因果 mask 的完整 GPT；你的主实践仍是 Build GPT。

## 9.1 手写单头 Attention【🔥】

**目标：** 给输入 X=(B,T,C)，独立写投影、点积、缩放、mask、softmax、加权求和，并用断言验证输出形状。

**例子：** B=1、T=3、C=4，第三个 token 可以看位置 1～3，第一个 token 只能看位置 1。若第一个位置的权重落到未来位置，因果 mask 就写错了。注意不同库布尔 mask 的 True/False 含义可能不同，要查所用 API。

**费曼自述：** “注意力代码先生成三个投影，求相关性并屏蔽不该看的位置，再把权重乘到 V 上。”

**概念闪卡：** Q：最容易写反的转置？A：K 的最后两个维度。Q：mask 后怎样验证？A：检查被遮位置 softmax 权重接近 0、输出形状正确。

来源：[单头 Attention 实现](逐篇精读/transformer/attention_code.md)。

## 9.2 多头拆分和拼接【🔥】

**目标：** 从 (B,T,C) 变到 (B,H,T,d)，每头算注意力，再转回 (B,T,C)。

**例子：** C=8、H=2、d=4；如果只 reshape 不调整轴顺序，把头维和时间维搞混，代码有时仍能运行但语义错。用小张量填可识别数字，打印每一步形状和元素位置，检查拆分/拼接是否互为逆操作。

**费曼自述：** “多头实现的关键是明确哪一维是头、哪一维是时间；算完后再按正确顺序合回。”

**概念闪卡：** Q：多头的维度条件？A：常见实现中 C 能被 H 整除。Q：张量形状对了就一定正确吗？A：不一定，轴顺序可能错而不报错。

来源：[MHA 实现](逐篇精读/transformer/mha_code.md)。

## 9.3 LayerNorm、残差和 FFN【🔥】

**目标：** 写出一层前馈网络，确认最后维度回到 C；把注意力和 FFN 分别接入残差。LayerNorm 可先使用框架模块，再通过小例子理解均值、方差和缩放。

**例子：** X 的形状为 (2,4,8)，FFN 中间到 32，最终回 8；若最终停在 32，就不能与输入做残差相加。LayerNorm 通常对每个位置的特征维归一化，不是对整批样本一起求一个均值。

**费曼自述：** “FFN 可以中间扩宽，但出口必须和残差输入对齐；Norm 是按约定的维度处理，不是随便求全局平均。”

**概念闪卡：** Q：FFN 一般包含什么？A：线性层、非线性激活、再线性映射。Q：LayerNorm 与 BatchNorm 最直观差别？A：归一化所依赖的维度/样本范围不同。

来源：[LayerNorm 与残差代码](逐篇精读/transformer/layernorm_residual_code.md) · [FFN 代码](逐篇精读/transformer/fnn_code.md)。

## 9.4 拼接 Block【🔥】

**目标：** 把注意力、Norm、FFN 和两次残差按选定 Pre-Norm/Post-Norm 顺序接起来，明确输入/输出都为 (B,T,C)。

**例子：** 不要为了“跑通”把因果 mask 省掉。训练小语言模型时，若前向允许看到未来 token，训练 loss 会虚低，生成时却没有未来信息，效果崩溃。可用“修改未来 token，观察当前位置输出是否变化”的实验检查因果性。

**费曼自述：** “Block 是两个处理子层加残差，能重复堆叠；自回归任务的注意力必须守住看不到未来的约束。”

**概念闪卡：** Q：怎样检测未来信息泄漏？A：保持前缀相同，仅改未来 token，比较较早位置输出。Q：Block 正常输出形状？A：仍为 (B,T,C)。

来源：[Block 实现](逐篇精读/transformer/transformer_block_code.md)。

## 9.5 从演示前向走到可训练 GPT【🔥】

**目标：** 自己补齐 tokenization、batch、训练/验证划分、损失、反传、生成和基线比较。

**例子：** Tiny Transformer 演示“给 token 产生一次输出”；可训练 GPT 还要对真实下一个 token 计算交叉熵、更新参数，在独立验证集看 loss，并逐 token 生成。与 bigram 基线比，才能知道更复杂的模型是否带来收益。训练代码也应记录 seed、配置和版本。

**费曼自述：** “一次 forward 只是模型能计算。学没学到，要训练、验证、生成，并与简单基线比较。”

**概念闪卡：** Q：演示 logits 与训练模型的差别？A：前者可来自随机权重，后者经损失反向传播更新。Q：为什么看验证集？A：检测对未见数据的表现与过拟合。

来源：[Tiny Transformer 示例](逐篇精读/transformer/tiny_transformer_code.md) · [Karpathy 课程与代码](https://github.com/karpathy/nn-zero-to-hero)。

## 本章过关

不看示例重写 Attention、MHA、Block 的关键路径；每一步能预报形状；跑通小 GPT 的训练与生成；比较 bigram 和 GPT 的验证 loss，并复盘至少一个自己遇到的形状或 mask 错误。
