# 第八章：Transformer 原理——张量怎样流过一层 Block

阅读方式：这是本章的完整学习文件，按知识链路合并原仓库文章。先读每节的解释与例子，再读技术细节；每节末尾闭卷做复述和闪卡。原文镜像仅供核对，不是必需的第二教材。

## 本章主线

从 token ID、Embedding 和位置表示开始，沿 Q/K/V→缩放点积→Mask→Softmax→多头→FFN→残差与 LayerNorm→logits 跟踪形状。每一步先说为什么需要，再写公式和维度。

本章动手证据： 用 B=2、T=4、C=8、H=2 手写每一步形状；对两个 token 手算一次 softmax 和加权 Value；解释因果 Mask 与梯度/训练的关系。

## 本章目录

- 8.1 Transformer：从静态词向量到上下文表示

- 8.2 Encoder、Decoder-only 与 Encoder-Decoder

- 8.3 Transformer 数据流：token、隐藏态与 logits

- 8.4 Q、K、V：查询、匹配与信息聚合

- 8.5 Attention 计算：点积、缩放、mask 和加权

- 8.6 多头注意力：形状拆分与合并

- 8.7 位置编码：正弦表、RoPE 与外推

- 8.8 残差、LayerNorm 与 FFN 的分工

- 8.9 Transformer Block：把子层串起来

- 8.10 Transformer 面试：形状、复杂度与缓存

## 与面试、代码的关系

面试串联： 讲注意力时先给 Q/K/V 的输入输出和张量形状，再写公式；解释 Mask 在 Softmax 前加入、为何缩放 √d、FFN 与残差的作用。用一个 2-token 手算例子检验理解。

需要亲手写的代码： 闭卷写 softmax(QKᵀ/√d + mask)V，逐步标出 (B,H,T,d)、(B,H,T,T) 和输出 (B,T,C)。

跨章节的需求分解、评测数据、指标与回归门禁见[系统设计与评测](横切-系统设计与评测.md)。

<a id="s1"></a>

## 8.1 Transformer：从静态词向量到上下文表示【🔥 必会】

〔补充自原仓库镜像〕〔本次新补，参考当前面试趋势〕

目标 + 例子。 同一个 token 的 embedding 查表结果相同，但经过注意力后，可因前文不同得到不同表示。本节用随机权重演示数据流，不解释词义，也不声称所有大模型都必须使用 Transformer。

### 概念与数据流

| 五要素 | 因果上下文化表示 |
| --- | --- |
| 输入 | token ID [B,T]，词表、位置容量与隐藏维 C。 |
| 做什么 | 查 token/位置 embedding，通过因果注意力聚合可见前文，再残差与归一化。 |
| 输出 | 每个位置的上下文表示 [B,T,C]。 |
| 哪里出错 | token 越界、位置超长、因果 mask 反向或将随机输出解释成语义。 |
| 参数影响 | 上下文长度、隐藏维、头数与注意力模式影响可见范围、计算和表达能力。 |

Transformer 使用注意力等组件处理序列，训练时可并行计算多个位置；全量自注意力的分数矩阵通常随序列长度平方增长。实际模型可能采用局部、稀疏或其他注意力，也存在非 Transformer 架构；工程选型需看任务和测量结果。

### 面试追问

【🔥 必会】为什么同一 token 能有不同表示？ 结论：注意力会聚合不同上下文。静态 token embedding 可以相同，经过位置表示和上下文交互后，隐藏态可不同。

【🔥 必会】decoder 的表示受未来 token 影响怎么办？ 结论：检查因果 mask 与输入位置。固定模型、只改未来 token，对比过去位置输出，验证允许范围。

【⭐ 理解】Transformer 为什么适合并行训练？ 结论：多个位置的注意力可批量计算。不等于生成阶段完全并行；自回归生成仍依赖之前生成的 token。

### 故障模式与排障

- 未来内容影响过去表示：检查 mask 三角方向和 API 布尔语义，用未来 token 扰动测试。

- token 或位置越界：核对 tokenizer、词表和最大上下文，输入前检查范围。

- 把随机表示差异解释成语义理解：检查权重来源和训练状态；用真实模型与独立任务评测验证能力。

### 工程 trick 与坑

- 下例只包含 embedding、单个注意力子层、残差和 LayerNorm，不是完整语言模型。

- nn.MultiheadAttention 本例布尔 mask 的 True 表示屏蔽；其他 API 可能相反。

- 本例无 padding、KV cache 和 tokenizer；完整手写 Block 与训练见第 9 章。

### Toy code：同 token 的上下文表示

运行前提：Python 3.10+、PyTorch；CPU 可运行，无模型下载。权重和 ID 均为随机/虚构。

```python
"""演示同一 token 在不同前文中可以得到不同上下文表示。"""
import torch
from torch import Tensor, nn


class ContextEncoder(nn.Module):
    """玩具因果自注意力层，返回静态词向量和上下文化结果。"""
    def __init__(self, vocab: int, channels: int, heads: int, max_len: int) -> None:
        super().__init__()
        if vocab < 2 or channels <= 0 or heads <= 0 or channels % heads or max_len <= 0:
            raise ValueError('词表、维度、头数或最大长度非法')
        self.vocab, self.max_len = vocab, max_len
        self.token = nn.Embedding(vocab, channels)
        self.position = nn.Embedding(max_len, channels)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        """输入 (B,T) token，输出两个 (B,T,C) 表示。"""
        if ids.ndim != 2 or ids.dtype != torch.long or ids.shape[0] == 0 or not 1 <= ids.shape[1] <= self.max_len:
            raise ValueError('输入必须是非空且长度受限的二维 long 张量')
        if int(ids.min()) < 0 or int(ids.max()) >= self.vocab:
            raise ValueError('token 超出词表')
        static = self.token(ids)
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = static + self.position(positions)[None, :, :]
        blocked = torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool, device=ids.device).triu(1)
        attended, _ = self.attn(x, x, x, attn_mask=blocked, need_weights=False)
        result = self.norm(x + attended)
        assert result.shape == static.shape
        return static, result


def main() -> None:
    """比较同 token 不同前文、改变未来 token 与非法输入。"""
    torch.manual_seed(71)
    model = ContextEncoder(vocab=16, channels=8, heads=2, max_len=8).eval()
    ids = torch.tensor([[1, 5, 7], [2, 5, 7]], dtype=torch.long)
    static, contextual = model(ids)
    assert torch.equal(static[0, 1], static[1, 1])
    assert not torch.allclose(contextual[0, 1], contextual[1, 1])
    changed = ids.clone(); changed[:, -1] = 8
    _, future_changed = model(changed)
    assert torch.allclose(contextual[:, :-1], future_changed[:, :-1], atol=1e-6)
    contextual.square().mean().backward()
    assert model.attn.in_proj_weight.grad is not None
    for bad in (torch.tensor([[16]]), torch.empty(1, 0, dtype=torch.long)):
        try:
            model(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('越界 token 或空序列必须拒绝')
    print('同 token 上下文化与因果边界自检通过；随机权重没有语义能力。')


if __name__ == '__main__':
    main()

```

逐段理解。 ContextEncoder 查表并加入位置，使用因果多头注意力和残差；自检比较相同 token 的静态 embedding 与不同前文的输出，并检查未来扰动与梯度。

### 这段代码怎么在面试里讲

- 不变量：ID 为 [B,T]，两种表示均为 [B,T,C]；同 token 查表值相同。

- 关键过程：加入位置信息后，注意力聚合允许看到的前文；不同前文可能改变输出。

- 边界与坑：随机权重仅验证计算关系，因果 mask、位置范围和真实任务表现需另行检查。

调库与手写怎么选。 本节用库层观察上下文化；第 9 章手写注意力和 Block 以理解底层，实际应用通常使用训练好的模型。

### 本节收尾：费曼自述与闪卡

费曼自述： token 查表得到静态向量，位置与注意力使表示能结合上下文；因果模型不能让过去位置看到未来输入。

- 问：同 token 的 embedding 一定相同吗？答：同一张固定表中的查表结果相同。

- 问：经过注意力后表示能不同吗？答：能，前文和位置可改变结果。

- 问：自回归生成能完全并行吗？答：不能，后续生成依赖前面生成的 token。

- 问：随机模型的表示差异证明理解语义吗？答：不证明。

### 原文核对 / 来源

[原仓库 Transformer 基础](原仓库Markdown/docs/llm/transformer/transformer_base_1.md)。

<a id="s2"></a>

## 8.2 Encoder、Decoder-only 与 Encoder-Decoder【🔥 必会】

〔补充自原仓库镜像〕〔本次新补，参考当前面试趋势〕

目标 + 例子。 对分类或文本向量任务可评估 encoder；开放式生成常用 decoder-only；翻译等输入到输出任务可评估 encoder-decoder。最终要在具体模型和数据上比较质量与成本。

### 概念与数据流

| 五要素 | 三类架构 |
| --- | --- |
| 输入 | encoder 接收源 token；decoder-only 接收上下文 token；encoder-decoder 接收源 token 和目标前缀。 |
| 做什么 | encoder 编码上下文；decoder-only 通常因果生成；encoder-decoder 在解码时还关注编码后的源序列。 |
| 输出 | encoder 常输出隐藏态；生成模型输出目标位置的词表 logits。 |
| 哪里出错 | 错用 mask、混淆源与目标长度、把随机模型输出当语义能力。 |
| 参数影响 | 序列长度、模型规模、目标任务和推理模式影响质量、延迟与资源。 |

典型 BERT encoder 在适当 mask 下编码上下文；典型 GPT 类 decoder-only 使用因果自注意力；T5 等 encoder-decoder 将源序列编码后供 decoder 跨注意力使用。这些是常见设计而非所有模型的绝对规则。Encoder 输出隐藏态并不天然等于适合检索的句向量，还需要合适的池化、训练与评测。

### 面试追问

【🔥 必会】三类架构怎样选？ 结论：先看任务与模型的实际能力。检索向量可评估专门训练的 encoder；生成可评估 decoder-only 或 encoder-decoder，并在同数据上测质量和成本。

【🔥 必会】encoder-decoder 输出长度为什么与输入不同？ 结论：输出位置对应目标序列。源编码长度和目标前缀长度可以不同；检查 decoder 输入、cross-attention mask 和形状。

【⭐ 理解】BERT 的任意隐藏态能直接当高质量 embedding 吗？ 结论：不能保证。池化方式、训练目标及领域数据影响向量质量，应在检索任务上验证。

### 故障模式与排障

- 输出长度不符合预期：分别检查源 token 和 decoder token 长度；按目标序列理解生成 logits。

- decoder 出现未来泄漏：检查因果 mask 和模型配置；改变未来输入，验证过去 logits。

- 检索向量效果差：核对模型是否针对 embedding 训练、池化和归一化；使用检索数据对比基线。

### 工程 trick 与坑

- 下例 encoder 返回隐藏态，另外两种返回 logits；它们的最后维度含义不同，不能直接当作能力比较。

- 所有模型随机初始化，无 tokenizer、padding 或真实任务评测。

- 具体模型的 attention mask、特殊 token 与输出类型以其官方文档和版本为准。

### Toy code：三种随机架构的形状

运行前提：Python 3.11、PyTorch、transformers==4.57.1；CPU 可运行，不下载权重。

```python
"""用随机小模型对照 encoder、decoder-only 与 encoder-decoder 形状。"""
import torch
from torch import Tensor
from transformers import (BertConfig, BertModel, GPT2Config, GPT2LMHeadModel,
                          T5Config, T5ForConditionalGeneration)


def validate(ids: Tensor, vocab: int, max_len: int) -> None:
    """拒绝空序列、越界 token 和超长输入。"""
    if ids.ndim != 2 or ids.dtype != torch.long or ids.shape[0] == 0:
        raise ValueError('输入必须是非空 (B,T) long 张量')
    if not 1 <= ids.shape[1] <= max_len or int(ids.min()) < 0 or int(ids.max()) >= vocab:
        raise ValueError('序列长度或 token ID 非法')


@torch.no_grad()
def run_models(source: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """构造随机小模型并返回三种架构的输出。"""
    vocab, hidden = 24, 16
    validate(source, vocab, 8); validate(target, vocab, 8)
    if source.shape[0] != target.shape[0]:
        raise ValueError('源和目标 batch 大小必须一致')
    encoder = BertModel(BertConfig(vocab_size=vocab, hidden_size=hidden, num_hidden_layers=1,
                                  num_attention_heads=2, intermediate_size=32)).eval()
    decoder = GPT2LMHeadModel(GPT2Config(vocab_size=vocab, n_positions=8, n_ctx=8,
                                         n_embd=hidden, n_layer=1, n_head=2)).eval()
    seq2seq = T5ForConditionalGeneration(T5Config(vocab_size=vocab, d_model=hidden, d_kv=8,
                                                  d_ff=32, num_layers=1, num_decoder_layers=1,
                                                  num_heads=2, decoder_start_token_id=0)).eval()
    encoded = encoder(input_ids=source).last_hidden_state
    decoder_logits = decoder(input_ids=source).logits
    seq2seq_logits = seq2seq(input_ids=source, decoder_input_ids=target).logits
    assert encoded.shape == (*source.shape, hidden)
    assert decoder_logits.shape == (*source.shape, vocab)
    assert seq2seq_logits.shape == (*target.shape, vocab)
    changed = source.clone(); changed[:, -1] = (changed[:, -1] + 1) % vocab
    changed_logits = decoder(input_ids=changed).logits
    assert torch.allclose(decoder_logits[:, :-1], changed_logits[:, :-1], atol=1e-6)
    return encoded, decoder_logits, seq2seq_logits


def main() -> None:
    """检查不同目标长度、因果性和非法输入。"""
    torch.manual_seed(67)
    source = torch.tensor([[1, 2, 3, 4], [1, 5, 6, 7]], dtype=torch.long)
    target = torch.tensor([[0, 8, 9], [0, 3, 4]], dtype=torch.long)
    a, b, c = run_models(source, target)
    assert a.shape[1] == b.shape[1] == 4 and c.shape[1] == 3
    for bad in (torch.tensor([[24]]), torch.empty(1, 0, dtype=torch.long)):
        try:
            validate(bad, 24, 8)
        except ValueError:
            pass
        else:
            raise AssertionError('越界或空序列必须拒绝')
    print('Encoder', tuple(a.shape), 'Decoder-only', tuple(b.shape), 'Encoder-decoder', tuple(c.shape))
    print('随机模型只用于形状验证，不评估能力。')


if __name__ == '__main__':
    main()

```

逐段理解。 先验证输入；分别创建随机 BERT、GPT-2、T5；断言隐藏态和 logits 的不同形状；最后修改未来 token 检查 GPT-2 因果性。

### 这段代码怎么在面试里讲

- 不变量：源和目标 batch 一致；encoder 输出 [B,Tsrc,C]，生成 logits 的最后一维是词表大小。

- 关键过程：encoder 编码源序列，decoder-only 对上下文输出 logits，encoder-decoder 根据源序列和目标前缀输出目标 logits。

- 边界与坑：这是随机小模型的形状实验；没有 tokenizer、padding 或质量评估，不应据此比较能力。

### 本节收尾：费曼自述与闪卡

费曼自述： Encoder 编码输入，decoder-only 通常按因果顺序生成，encoder-decoder 同时处理源序列和目标前缀；具体选型要通过任务评测。

- 问：encoder-decoder 的 logits 时间长度对应什么？答：decoder 目标前缀长度。

- 问：encoder 隐藏态天然是优质检索向量吗？答：不保证，要看训练、池化与评测。

- 问：GPT 类模型怎样限制未来可见？答：使用因果注意力，具体按模型实现确认。

- 问：本节随机模型能比较任务效果吗？答：不能，只验证接口与形状。

### 原文核对 / 来源

[原仓库三类架构对比](原仓库Markdown/docs/llm/transformer/transformer_base_encoder_decoder.md)。

<a id="s3"></a>

## 8.3 Transformer 数据流：token、隐藏态与 logits【🔥 必会】

〔补充自原仓库镜像〕〔本次新补，参考当前面试趋势〕

目标 + 例子。 将一批 token ID 输入因果语言模型，追踪 embedding、logits 和下一 token 的训练 loss；以随机 GPT-2 检查形状与因果边界。

### 概念与数据流

| 五要素 | decoder-only 数据流 |
| --- | --- |
| 输入 | token ID [B,T]、词表 V、上下文上限及可选 mask。 |
| 做什么 | 查 token/位置 embedding → 经因果 Transformer 层 → 词表投影；训练时比较下一个 token。 |
| 输出 | 隐藏态 [B,T,C]、logits [B,T,V]；有标签时可得到标量 loss。 |
| 哪里出错 | token 越界、位置超长、标签重复移位、padding 混入 loss、mask 错误。 |
| 参数影响 | 词表 V、上下文 T、隐藏维 C 和层数影响形状、计算及资源。 |

logits 是未归一化的词表分数；经过 softmax 才得到该位置的概率分布。训练时通常并行计算每个位置预测下一 token 的 loss；推理时读取最后有效位置的 logits 选择下一个 token，循环生成。真实 tokenizer、chat template 和 padding 的处理随模型变化，下例只使用虚构 ID。

### 面试追问

【🔥 必会】输入到 logits 的形状怎样变化？ 结论：[B,T]→[B,T,C]→[B,T,V]。embedding 将 ID 查表变成向量，模型处理上下文，输出层映射到词表分数。

【🔥 必会】训练 loss 正常但生成错位怎么查？ 结论：先核对 labels 的移位约定。检查模型是否内部移位、结束符和 padding 掩码；用短序列对照手算交叉熵。

【⭐ 理解】logits 最大就是概率吗？ 结论：不是，logits 是未归一化分数。经过 softmax 才是概率；argmax 在相同温度下位置相同，但采样还受温度及截断策略影响。

### 故障模式与排障

- 索引越界：检查 tokenizer 与模型词表版本，以及位置长度上限；验证输入 ID 范围。

- 训练目标错位：抽样打印 token 与标签，确认模型是否内部移位；不要重复右移。

- 未来信息泄漏：只改未来 token，比较早期 logits；检查因果 mask 和模型模式。

### 工程 trick 与坑

- 本例无 padding，且所有 token 均作为语言模型标签；指令微调时哪些位置计入 loss 由目标决定，参见第 5.2 节。

- GPT-2 的 labels 在本例所用接口内移位；不能把这一约定套到所有模型而不查文档。

- 随机小模型用于验证形状和计算，不证明语义或语言能力。

### Toy code：追踪 token 到 loss

运行前提：Python 3.11、PyTorch、transformers==4.57.1；CPU 可运行，无需下载模型。

```python
"""使用随机 GPT-2 追踪 token、embedding、logits 与因果 loss。"""
import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import GPT2Config, GPT2LMHeadModel


def validate(ids: Tensor, vocab: int, max_length: int) -> None:
    """只接收无 padding 的非空二维 long token。"""
    if ids.ndim != 2 or ids.dtype != torch.long or ids.shape[0] == 0:
        raise ValueError('token 必须是非空二维 long 张量')
    if not 2 <= ids.shape[1] <= max_length:
        raise ValueError('序列长度超出允许范围')
    if int(ids.min()) < 0 or int(ids.max()) >= vocab:
        raise ValueError('token 越出词表')


@torch.no_grad()
def trace(model: GPT2LMHeadModel, ids: Tensor) -> tuple[Tensor, Tensor]:
    """验证 embedding、logits 和手算的 next-token loss。"""
    validate(ids, model.config.vocab_size, model.config.n_positions)
    model.eval()
    embedding = model.transformer.wte(ids)
    result = model(input_ids=ids, labels=ids, use_cache=False)
    logits = result.logits
    assert embedding.shape == (*ids.shape, model.config.n_embd)
    assert logits.shape == (*ids.shape, model.config.vocab_size)
    if result.loss is None:
        raise RuntimeError('模型没有返回 loss')
    manually_shifted = F.cross_entropy(logits[:, :-1, :].reshape(-1, model.config.vocab_size),
                                      ids[:, 1:].reshape(-1))
    assert torch.isfinite(manually_shifted) and torch.allclose(result.loss, manually_shifted, atol=1e-6)
    return embedding, logits


def main() -> None:
    """检查形状、模型内部移位、因果性和非法 token。"""
    torch.manual_seed(61)
    config = GPT2Config(vocab_size=20, n_positions=8, n_ctx=8, n_embd=16, n_layer=1, n_head=2)
    model = GPT2LMHeadModel(config)
    ids = torch.tensor([[1, 4, 5, 2], [1, 7, 3, 2]], dtype=torch.long)
    embeddings, logits = trace(model, ids)
    changed = ids.clone(); changed[:, -1] = 9
    _, changed_logits = trace(model, changed)
    assert torch.allclose(logits[:, :-1], changed_logits[:, :-1], atol=1e-6)
    next_token = logits[:, -1, :].argmax(-1)
    assert next_token.shape == (2,) and torch.all((next_token >= 0) & (next_token < 20))
    for bad in (torch.tensor([[20, 1]]), torch.tensor([[1]])):
        try:
            validate(bad, 20, 8)
        except ValueError:
            pass
        else:
            raise AssertionError('越界 ID 或过短序列必须拒绝')
    print('token', tuple(ids.shape), 'embedding', tuple(embeddings.shape), 'logits', tuple(logits.shape))
    print('随机模型和虚构 token 不代表真实语言能力。')


if __name__ == '__main__':
    main()

```

逐段理解。 validate 检查 token 形状与范围；trace 获取 embedding 和 logits，并用手动移位交叉熵对照模型 loss；自检改变最后 token 验证因果性。

### 这段代码怎么在面试里讲

- 不变量：输入 [B,T]，embedding [B,T,C]，logits [B,T,V]，token 必须处于词表内。

- 关键过程：查表、经过模型输出 logits；将前 T-1 个位置与后 T-1 个 token 对齐计算 loss。

- 边界与坑：库可能已内部移位，padding 和最后有效位置需单独处理；随机模型不具备真实语言能力。

调库与手写怎么选。 本节用库观察完整数据流；从零实现注意力、Block 和训练循环见第 9 章，实际应用优先使用与 checkpoint 对齐的模型和 tokenizer。

### 本节收尾：费曼自述与闪卡

费曼自述： token ID 先变成带位置信息的向量，经因果 Transformer 得到隐藏态，再投影到词表；训练用当前位置预测下一个 token。

- 问：logits 的形状是什么？答：通常为 [B,T,V]。

- 问：logits 是概率吗？答：不是，softmax 后才是概率。

- 问：GPT-2 本例的 labels 要再次手工移位吗？答：不需要，模型内部已处理。

- 问：随机模型能证明业务能力吗？答：不能，只用于验证计算路径。

### 原文核对 / 来源

[原仓库 Transformer 数据流](原仓库Markdown/docs/llm/transformer/transformer_data_flow.md)；[GPT-2 labels 文档](https://huggingface.co/docs/transformers/model_doc/gpt2)。

<a id="s4"></a>

## 8.4 Q、K、V：查询、匹配与信息聚合【🔥 必会】

〔补充自原仓库镜像〕〔本次新补，参考当前面试趋势〕

目标 + 例子。 self-attention 的 Q/K/V 可以来自同一输入的不同投影；cross-attention 的 Q 可来自当前序列、K/V 来自另一序列。用不同长度的输入跟踪形状。

### 概念与数据流

| 五要素 | Q/K/V 投影与注意力 |
| --- | --- |
| 输入 | query [B,Tq,C]、context [B,Tk,C] 和三组可学习投影。 |
| 做什么 | query 投影成 Q，context 投影成 K/V；用 Q 与 K 匹配权重，再聚合 V。 |
| 输出 | Q [B,Tq,Dk]、K [B,Tk,Dk]、V [B,Tk,Dv] 和输出 [B,Tq,Dv]。 |
| 哪里出错 | Q/K 维度不一致、K/V 长度不一致、忽略 mask、混淆 self 与 cross 的来源。 |
| 参数影响 | 投影维度、可见性、序列长度和权重影响表达、计算与内存。 |

Q 表示当前位置用于匹配的查询，K 表示可匹配的键，V 是被加权聚合的值；这是帮助理解的类比，实际含义由训练学得。自注意力可由同一隐藏态产生三者，但投影权重通常不同；交叉注意力可让 Q 来自当前序列、K/V 来自外部上下文。本例为非因果演示；decoder 的自注意力还需要因果 mask。

### 面试追问

【🔥 必会】为什么 K 和 V 可以来自同一输入却不同？ 结论：它们经过不同可学习投影。K 参与匹配，V 被权重聚合；并非要求两个张量数值相同。

【🔥 必会】cross-attention 报维度错误怎么定位？ 结论：核对 Q/K 的特征维及 K/V 的序列长度。再检查 batch、mask 和投影；输出的时间长度由 query 决定。

【⭐ 理解】Q/K 维度必须等于 V 维度吗？ 结论：不必。点积要求 Q、K 的最后维匹配；V 的特征维可不同，输出随 V 的维度变化。

### 故障模式与排障

- 矩阵乘法维度不匹配：分别打印 Q/K/V 形状，核对 Q/K 最后一维和 K/V 的 context 长度。

- 结果与库实现不同：确认相同投影、缩放、mask 和 dropout，再用 SDPA 对照；本例无 mask。

- decoder 回答泄漏未来：检查是否使用因果 mask；本节非因果演示不能直接当作 decoder 自注意力。

### 工程 trick 与坑

- 本例只演示单头投影，不实现 padding、因果或输出投影；完整因果实现见 9.1—9.2。

- 使用同一输入产生 Q/K/V 不代表使用同一组权重；模型结构以具体实现为准。

- 注意力权重不是可靠的因果解释或事实证据，模型输出仍需独立评测。

### Toy code：Q/K/V 投影与 self/cross 形状

运行前提：Python 3.10+、PyTorch 2.x；CPU 可运行，随机数据不代表语义。

```python
"""用线性投影和库注意力对照 self / cross attention 的 QKV 数据流。"""
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ProjectedAttention(nn.Module):
    """Q 来自 query，K/V 来自 context；不处理 padding。"""
    def __init__(self, channels: int, head_dim: int, value_dim: int) -> None:
        super().__init__()
        if min(channels, head_dim, value_dim) <= 0:
            raise ValueError('维度必须为正')
        self.channels, self.head_dim, self.value_dim = channels, head_dim, value_dim
        self.q_proj = nn.Linear(channels, head_dim, bias=False)
        self.k_proj = nn.Linear(channels, head_dim, bias=False)
        self.v_proj = nn.Linear(channels, value_dim, bias=False)

    def forward(self, query: Tensor, context: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """输入 (B,Tq,C)/(B,Tk,C)，返回 Q、K、V 和非因果输出。"""
        if query.ndim != 3 or context.ndim != 3 or query.shape[0] == 0 or query.shape[0] != context.shape[0]:
            raise ValueError('输入必须是同批次非空三维张量')
        if query.shape[1] == 0 or context.shape[1] == 0 or query.shape[2] != self.channels or context.shape[2] != self.channels:
            raise ValueError('序列长度或通道维度非法')
        if any(t.device != self.q_proj.weight.device or t.dtype != self.q_proj.weight.dtype or not torch.isfinite(t).all() for t in (query, context)):
            raise ValueError('输入设备、dtype 或数值非法')
        q, k, v = self.q_proj(query), self.k_proj(context), self.v_proj(context)
        weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)
        out = weights @ v
        assert out.shape == (query.shape[0], query.shape[1], self.value_dim)
        return q, k, v, out


def main() -> None:
    """验证 self/cross 的长度、库等价和梯度。"""
    torch.manual_seed(59)
    model = ProjectedAttention(8, 4, 6)
    query = torch.randn(2, 3, 8, requires_grad=True)
    context = torch.randn(2, 5, 8, requires_grad=True)
    q, k, v, output = model(query, context)
    assert q.shape == (2, 3, 4) and k.shape == (2, 5, 4) and v.shape == (2, 5, 6)
    assert torch.allclose(output, F.scaled_dot_product_attention(q, k, v, dropout_p=0.0), atol=1e-6, rtol=1e-5)
    _, _, _, self_output = model(query, query)
    assert self_output.shape == (2, 3, 6)
    output.square().mean().backward()
    assert query.grad is not None and context.grad is not None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    try:
        model(query.detach(), torch.randn(2, 0, 8))
    except ValueError:
        pass
    else:
        raise AssertionError('空 context 必须拒绝')
    print('Q/K/V 投影、self/cross 形状与 SDPA 自检通过。')


if __name__ == '__main__':
    main()

```

逐段理解。 三个线性层分别生成 Q/K/V；手写权重聚合并与 SDPA 对照；同一输入演示 self、不同长度输入演示 cross；自检验证梯度和空 context。

### 这段代码怎么在面试里讲

- 不变量：Q/K 最后一维相同，K/V 的 context 长度相同；输出时间长度取 query。

- 关键过程：分别投影并计算 softmax(QKᵀ/√Dk)V，用相同张量对照库实现。

- 边界与坑：本例是非因果、无 padding 单头示例；decoder 使用时必须补正确 mask。

手写与调库怎么选。 手写投影和聚合用于理解来源与形状；实际应用优先模型内置注意力，并核对具体 mask 和缓存实现。

### 本节收尾：费曼自述与闪卡

费曼自述： Q 与 K 计算匹配，权重再聚合 V。自注意力通常来自同一输入的不同投影，交叉注意力可从另一序列提供 K/V。

- 问：cross-attention 输出长度由谁决定？答：query 的长度。

- 问：Q 和 K 的最后维度为何必须相同？答：点积需要对应维度匹配。

- 问：V 的最后维度必须与 Q 相同吗？答：不必，输出特征维由 V 决定。

- 问：本例能直接用于因果 decoder 吗？答：不能，还需正确的因果和 padding mask。

### 原文核对 / 来源

[原仓库 Q、K、V](原仓库Markdown/docs/llm/transformer/qkv.md)。

<a id="s5"></a>

## 8.5 Attention 计算：点积、缩放、mask 和加权【🔥 必会】

〔补充自原仓库镜像〕〔本次新补，参考当前面试趋势〕

目标 + 例子。 用两个 token 手算一次因果注意力：第一个 token 只能看自己，第二个可以看两个位置；再对照 PyTorch SDPA。完整带 padding 的实现见第 9.1 节。

### 概念与数据流

| 五要素 | 缩放点积注意力 |
| --- | --- |
| 输入 | Q、K、V，本例形状均为 [T,D]，以及因果可见性。 |
| 做什么 | scores=QKᵀ/√D，屏蔽未来位置，逐行 softmax，再与 V 相乘。 |
| 输出 | 注意力权重 [T,T] 和输出 [T,D]。 |
| 哪里出错 | 矩阵转置、mask 方向、softmax 轴、全遮蔽行或指数溢出。 |
| 参数影响 | D 决定缩放，mask 决定可见位置，序列长度 T 决定分数矩阵大小。 |

示例取 Q=K=[[1,0],[0,1]]，V=[[2,0],[0,4]]。第一行只能看到第一个位置，输出 [2,0]；第二行按分数 [0,1/√2] 经 softmax 加权两个 V。数值稳定 softmax 先减去每行最大有限分数；如果一整行都被屏蔽，应显式处理，不能直接对全 -inf 求 softmax。

### 面试追问

【🔥 必会】为什么 mask 要在 softmax 前处理？ 结论：让禁止位置得到零概率。对禁用位置加负无穷或使用等价实现后再归一化；还要处理整行不可见的情况。

【🔥 必会】softmax 输出 NaN 怎么排查？ 结论：先检查输入有限性和全遮蔽行。再检查数值精度、mask 和指数计算；使用稳定实现并明确空行策略。

【⭐ 理解】为什么要除以 √D？ 结论：控制点积随维度增大的典型尺度。在常见假设下避免 softmax 过早饱和，但它不保证任何输入都数值稳定。

### 故障模式与排障

- 第一个 token 看到未来：打印允许矩阵与权重，上三角应被屏蔽；用两 token 手算和扰动测试。

- 大分数导致 NaN：检查有限输入并使用减最大值的稳定 softmax；单独拒绝或处理全遮蔽行。

- 结果与库不一致：对齐 Q/K/V、缩放、mask、dropout、dtype 和库的布尔语义，再比较中间权重。

### 工程 trick 与坑

- 下例只处理没有 padding 的二维单头输入；批次、多头与 padding 的完整实现见 9.1—9.2。

- 浮点误差要求设置合理容差；手算等式不能替代设备与真实模型下的回归。

- 不同 API 的布尔 mask 方向可能相反；下例用 is_causal=True 对照，不手传 bool mask 给 SDPA。

### Toy code：两 token 手算与库对照

运行前提：Python 3.10+、PyTorch 2.x；CPU 可运行，无网络。

```python
"""手算两 token 的缩放因果注意力，并用 SDPA 交叉检查。"""
import math
import torch
from torch import Tensor
from torch.nn import functional as F


def stable_softmax(scores: Tensor, allowed: Tensor) -> Tensor:
    """按行计算稳定 softmax；拒绝没有任何可见项的行。"""
    if scores.ndim != 2 or scores.shape != allowed.shape or allowed.dtype != torch.bool or allowed.device != scores.device or not scores.is_floating_point():
        raise ValueError('scores 与布尔 allowed 必须是相同二维形状')
    if scores.numel() == 0 or not torch.isfinite(scores).all() or not allowed.any(dim=-1).all():
        raise ValueError('分数必须有限且每行至少有一个可见项')
    masked = scores.masked_fill(~allowed, float('-inf'))
    shifted = masked - masked.max(dim=-1, keepdim=True).values
    exp = shifted.exp()
    weights = exp / exp.sum(dim=-1, keepdim=True)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(scores.shape[0], dtype=scores.dtype, device=scores.device))
    return weights


def scaled_attention(q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """输入相同长度的 Q/K/V=(T,D)，只允许看到当前位置和过去。"""
    if q.ndim != 2 or q.shape != k.shape or q.shape != v.shape or q.shape[0] == 0 or q.shape[1] == 0:
        raise ValueError('Q/K/V 必须为同形状的非空二维张量')
    if not q.is_floating_point() or any(x.dtype != q.dtype or x.device != q.device or not torch.isfinite(x).all() for x in (q, k, v)):
        raise ValueError('Q/K/V 必须同设备同类型且有限')
    length, dim = q.shape
    allowed = torch.ones(length, length, dtype=torch.bool, device=q.device).tril()
    weights = stable_softmax(q @ k.T / math.sqrt(dim), allowed)
    output = weights @ v
    assert output.shape == q.shape
    return output, weights


def main() -> None:
    """检查手算、库对照、屏蔽位置和异常输入。"""
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    k = q.clone(); v = torch.tensor([[2.0, 0.0], [0.0, 4.0]], dtype=torch.float64)
    output, weights = scaled_attention(q, k, v)
    e = math.exp(1 / math.sqrt(2))
    expected = torch.tensor([[2.0, 0.0], [2 / (1 + e), 4 * e / (1 + e)]], dtype=torch.float64)
    library = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
    assert torch.allclose(output, expected, atol=1e-10)
    assert torch.allclose(output, library, atol=1e-10)
    assert weights[0, 1].item() == 0.0 and torch.isfinite(output).all()
    stable = stable_softmax(torch.tensor([[10000.0, 9999.0]]), torch.tensor([[True, True]]))
    assert torch.isfinite(stable).all() and stable[0, 0] > stable[0, 1]
    try:
        stable_softmax(torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))
    except ValueError:
        pass
    else:
        raise AssertionError('全遮蔽行必须拒绝')
    print('两 token 手算、数值稳定 softmax 与 SDPA 自检通过:', output.tolist())


if __name__ == '__main__':
    main()

```

逐段理解。 stable_softmax 先拒绝无效行并减去行最大值；scaled_attention 按因果可见性计算；main 对照手算公式和 SDPA，并验证大分数及全遮蔽错误。

### 这段代码怎么在面试里讲

- 不变量：Q/K/V 为相同的 [T,D]，每个 query 至少有一个可见 key，每行权重和为 1。

- 关键过程：点积、缩放、遮蔽、稳定 softmax，再与 V 相乘；第一位置禁止关注未来。

- 边界与坑：全遮蔽行必须单独处理，浮点比较用容差；本例不支持 padding 或不同 Q/V 维度。

手写与调库怎么选。 手算和手写用于检查因果关系与数值不变量；实际推理使用成熟注意力实现，并根据模型和设备核验 mask、精度和性能。

### 本节收尾：费曼自述与闪卡

费曼自述： Q 和 K 计算缩放分数，先屏蔽不可见位置，再对每行做稳定 softmax，最后加权 V；全遮蔽行不能直接归一化。

- 问：第一 token 的因果注意力能看哪里？答：只能看它自身及可见的更早位置。

- 问：为什么 softmax 前减去最大值？答：降低指数溢出风险，概率保持不变。

- 问：全遮蔽行怎么处理？答：按系统策略明确拒绝或定义安全输出，不能默认普通 softmax 可用。

- 问：权重矩阵是什么形状？答：本例是 [T,T]。

### 原文核对 / 来源

[原仓库 Attention 计算](原仓库Markdown/docs/llm/transformer/qkv_cal.md)；[PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。

<a id="s6"></a>

## 8.6 多头注意力：形状拆分与合并【🔥 必会】

〔补充自原仓库镜像〕〔本次新补，参考当前面试趋势〕

目标 + 例子。 对 B=2,T=4,C=8,H=2，把隐藏态拆成 [2,2,4,4]，按头做因果注意力后拼回 [2,4,8]。完整 QKV 投影和手写多头见第 9.2 节。

### 概念与数据流

| 五要素 | 多头形状变换 |
| --- | --- |
| 输入 | 投影后的张量 [B,T,C] 和标准 MHA 的头数 H。 |
| 做什么 | 将通道拆成 [H,C/H]，转置得到 [B,H,T,C/H]；各头运算后逆转置并合并。 |
| 输出 | 合并张量 [B,T,C]；完整 MHA 还需输出投影。 |
| 哪里出错 | C 不能整除 H、时间轴与头轴混淆、非连续张量错误 reshape。 |
| 参数影响 | 固定 C 时 H 越多，每头维度越小；实际性能和表示能力依模型及实现而定。 |

标准多头通常先由不同投影生成 Q/K/V，再分别拆头，头间输出拼接后通过输出投影。下例为聚焦形状，直接把同一输入当 Q/K/V，调用库注意力；它不是完整训练好的多头层。

### 面试追问

【🔥 必会】多头拆分的具体顺序是什么？ 结论：先拆通道，再交换头与时间轴。[B,T,C]→[B,T,H,D]→[B,H,T,D]，其中 D=C/H。

【🔥 必会】输出形状正确但内容不对怎么排查？ 结论：检查 transpose 和通道顺序。用可追踪的输入验证拆分后合并能否恢复原值，再核对投影权重和 mask。

【⭐ 理解】多头一定让模型更好吗？ 结论：不一定。头数改变每头维度和实现效率，是否有收益要结合模型、任务与资源实验。

### 故障模式与排障

- reshape 失败：检查 C%H 和实际张量形状；拒绝非法头数。

- 形状对但 token 顺序错：用拆分/合并可逆断言检查 transpose 顺序；必要时先转为连续内存。

- 未来信息泄漏：修改未来输入并检查过去输出；核对因果 mask 及 API 参数。

### 工程 trick 与坑

- 此例使用相同张量作为 Q/K/V，没有学习投影和输出投影；完整 MHA 见 9.2。

- 转置后张量可能非连续；本例调用 contiguous() 再合并。

- GQA/MQA 的 Q 和 KV 头数可能不同，不能直接套用标准 MHA 的形状假设。

### Toy code：拆头、库注意力与合并

运行前提：Python 3.10+、PyTorch 2.x；CPU 可运行，无 padding 和模型下载。

```python
"""练习多头拆分、因果注意力和合并，投影权重在第 9.2 节实现。"""
import torch
from torch import Tensor
from torch.nn import functional as F


def split_heads(x: Tensor, heads: int) -> Tensor:
    """将 (B,T,C) 重排为 (B,H,T,C/H)，不复制通道。"""
    if x.ndim != 3 or x.shape[0] == 0 or x.shape[1] == 0 or x.shape[2] == 0:
        raise ValueError('输入必须为非空 (B,T,C)')
    if isinstance(heads, bool) or not isinstance(heads, int) or heads <= 0 or x.shape[-1] % heads:
        raise ValueError('头数必须为正且整除 C')
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError('输入必须为有限浮点数')
    batch, length, channels = x.shape
    return x.reshape(batch, length, heads, channels // heads).transpose(1, 2)


def merge_heads(x: Tensor) -> Tensor:
    """将 (B,H,T,D) 合并为 (B,T,H*D)。"""
    if x.ndim != 4 or any(size == 0 for size in x.shape):
        raise ValueError('输入必须为非空 (B,H,T,D)')
    batch, heads, length, dim = x.shape
    result = x.transpose(1, 2).contiguous().reshape(batch, length, heads * dim)
    assert result.shape == (batch, length, heads * dim)
    return result


def main() -> None:
    """验证拆分可逆、因果注意力形状、未来位置不影响过去和梯度。"""
    torch.manual_seed(53)
    x = torch.randn(2, 4, 8, requires_grad=True)
    heads = split_heads(x, 2)
    assert heads.shape == (2, 2, 4, 4)
    assert torch.equal(merge_heads(heads), x)
    output = merge_heads(F.scaled_dot_product_attention(heads, heads, heads, is_causal=True, dropout_p=0.0))
    changed = x.detach().clone(); changed[:, -1] += 10
    other_heads = split_heads(changed, 2)
    other = merge_heads(F.scaled_dot_product_attention(other_heads, other_heads, other_heads,
                                                       is_causal=True, dropout_p=0.0))
    assert output.shape == x.shape and torch.allclose(output[:, :-1], other[:, :-1], atol=1e-6)
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for bad in (0, 3, True):
        try:
            split_heads(x.detach(), bad)
        except ValueError:
            pass
        else:
            raise AssertionError('错误头数必须拒绝')
    print('多头形状、合并、因果性和梯度自检通过。')


if __name__ == '__main__':
    main()

```

逐段理解。 split_heads 检查可整除并拆头；merge_heads 恢复通道；自检用 SDPA 做因果注意力，验证拆合可逆、未来位置不影响过去和梯度有限。

### 这段代码怎么在面试里讲

- 不变量：输入 [B,T,C]，拆后 [B,H,T,C/H]，合并后形状和通道顺序恢复。

- 关键过程：reshape 拆通道、transpose 换轴，逐头注意力后逆序操作。

- 边界与坑：非法头数要拒绝，非连续内存需正确处理；此例没有 QKV 和输出投影。

手写与调库怎么选。 手写拆合帮助理解形状；生产中用成熟多头层或优化注意力实现，完整手写与权重对照见 9.2。

### 本节收尾：费曼自述与闪卡

费曼自述： 多头将 C 分成 H 个子空间，转置后独立计算注意力，再恢复时间轴并拼接；完整模型还需要投影。

- 问：C=8,H=2 时每头几维？答：4 维。

- 问：拆头后形状是什么？答：[B,H,T,C/H]。

- 问：此示例是否实现完整 QKV 投影？答：没有，完整实现见 9.2。

- 问：transpose 后为什么注意连续性？答：内存布局可能不连续，直接 view 可能失败或不符合预期。

### 原文核对 / 来源

[原仓库多头注意力](原仓库Markdown/docs/llm/transformer/mha.md)。

<a id="s7"></a>

## 8.7 位置编码：正弦表、RoPE 与外推【🔥 必会】

〔补充自原仓库镜像〕〔本次新补，参考当前面试趋势〕

目标 + 例子。 纯自注意力本身没有显式的 token 顺序；模型需通过因果 mask、位置表示等结构利用顺序。手写正弦位置表与 RoPE，并和固定版本的 Transformers 实现对照。

### 概念与数据流

| 五要素 | 正弦位置表与 RoPE |
| --- | --- |
| 输入 | 位置索引 [T]、通道维 D；RoPE 还输入待旋转的 Q/K。 |
| 做什么 | 正弦表生成各位置不同频率的 sin/cos 表示；RoPE 按位置旋转 Q/K 的二维分量。 |
| 输出 | 位置表 [T,D]，或与 Q/K 同形状的旋转后张量。 |
| 哪里出错 | 增量解码的位置偏移错误、维度配对约定不一致、超长位置数值精度或外推退化。 |
| 参数影响 | 基频、旋转维度、上下文长度与扩展方法影响位置频率及实际长文本表现。 |

正弦绝对位置表常加到 embedding 上；RoPE 通常作用于注意力的 Q、K。对同一频率的旋转，Q/K 点积具有与相对位移相关的性质，但这不保证模型在训练长度之外仍有好的效果。长上下文外推需结合具体模型的位置缩放方法、训练范围和真实任务评测。因果 mask 也引入可见范围约束，不能简单说 decoder 完全不感知顺序。

### 面试追问

【🔥 必会】RoPE 为什么能表达相对位移？ 结论：两个位置旋转后的点积与角度差有关。相同频率下同时平移两个位置，角度差不变；实际模型还受其他结构与数值影响。

【🔥 必会】缓存生成的结果与全量前向不同怎么查？ 结论：检查新增 token 的位置偏移。对照前缀长度、position IDs、Q/K 旋转和 mask，再用短序列验证缓存与全量输出一致。

【⭐ 理解】RoPE 能自动保证无限长上下文吗？ 结论：不能。超出训练范围可能退化，具体扩展策略和质量须在目标模型与任务上验证。

### 故障模式与排障

- 增量解码重复使用位置 0：检查 cache 长度与新增 token 的 position ID；按模型接口传递正确偏移并与全量路径比较。

- 手写与库输出不一致：核对旋转维度配对、频率、基数、位置和 dtype；本例相邻维配对，Llama 版本使用前后半维配对，需先重排。

- 长文本精度或效果下降：核对训练上下文、位置缩放和数值精度；用真实长度分桶评测，不能只测短序列。

### 工程 trick 与坑

- RoPE 的具体维度排列、缩放和接口因模型、版本而异；示例仅比较固定配置下可等价转换的排列。

- 低精度和超大位置可能带来额外误差；下面的相对位移断言只是有限位置的数学自检。

- transformers==4.57.1 的模型内部辅助函数属于版本相关接口，升级后应重新核对。

### Toy code：正弦表和 RoPE 对照

运行前提：Python 3.11、PyTorch、transformers==4.57.1；CPU 可运行，不下载模型。示例只支持正偶数维、非负位置。

```python
"""手写正弦位置表与成对旋转位置编码，验证形状和相对位置性质。"""
import math
import torch
from torch import Tensor
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb


def angles(positions: Tensor, dim: int, base: float, dtype: torch.dtype) -> Tensor:
    """返回 (T,D/2) 相位；此示例要求偶数维和非负位置。"""
    if dim <= 0 or dim % 2 or not math.isfinite(base) or base <= 1:
        raise ValueError('维度必须为正偶数，base 必须大于 1')
    if positions.ndim != 1 or positions.numel() == 0 or positions.dtype != torch.long or (positions < 0).any():
        raise ValueError('位置必须是非空非负的一维 long 张量')
    exponents = torch.arange(0, dim, 2, device=positions.device, dtype=dtype) / dim
    frequencies = base ** (-exponents)
    return positions.to(dtype)[:, None] * frequencies[None, :]


def sinusoidal(positions: Tensor, dim: int, base: float = 10000.0) -> Tensor:
    """构造 (T,D) 的交错 sin/cos 位置表。"""
    phase = angles(positions, dim, base, torch.float64)
    table = torch.stack((phase.sin(), phase.cos()), dim=-1).reshape(-1, dim)
    assert table.shape == (positions.numel(), dim)
    return table


def rope(x: Tensor, positions: Tensor, base: float = 10000.0) -> Tensor:
    """按相邻维度旋转 x=(B,T,D)，不改变每对向量长度。"""
    if x.ndim != 3 or x.shape[0] == 0 or x.shape[1] != positions.numel() or not x.is_floating_point():
        raise ValueError('输入必须为浮点 (B,T,D)，且 T 匹配位置')
    if x.device != positions.device or not torch.isfinite(x).all():
        raise ValueError('输入与位置设备不同或输入非有限')
    phase = angles(positions, x.shape[-1], base, x.dtype)
    cos, sin = phase.cos().unsqueeze(0), phase.sin().unsqueeze(0)
    even, odd = x[..., 0::2], x[..., 1::2]
    result = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
    assert result.shape == x.shape
    return result


def main() -> None:
    """验证正弦表、旋转保范数、共同平移的点积和错误输入。"""
    torch.manual_seed(47)
    positions = torch.arange(4, dtype=torch.long)
    table = sinusoidal(positions, 8)
    assert table.shape == (4, 8) and torch.allclose(table[0, 0::2], torch.zeros(4, dtype=torch.float64))
    q, k = torch.randn(1, 1, 8, dtype=torch.float64), torch.randn(1, 1, 8, dtype=torch.float64)
    def dot(p: int, r: int) -> Tensor:
        return (rope(q, torch.tensor([p])) * rope(k, torch.tensor([r]))).sum()
    assert torch.allclose(dot(3, 8), dot(103, 108), atol=1e-10)
    assert torch.allclose(q.norm(), rope(q, torch.tensor([3])).norm(), atol=1e-10)
    # Llama 使用前后半维配对，本例用相邻维配对；先重排维度再对照。
    config = LlamaConfig(hidden_size=8, num_attention_heads=1, num_key_value_heads=1,
                         intermediate_size=16, num_hidden_layers=1, rope_theta=10000.0)
    samples = torch.randn(1, 4, 8)
    grouped = samples[..., [0, 2, 4, 6, 1, 3, 5, 7]].unsqueeze(1)
    cos, sin = LlamaRotaryEmbedding(config)(grouped, positions.unsqueeze(0))
    rotated, _ = apply_rotary_pos_emb(grouped, grouped, cos, sin)
    restored = rotated.squeeze(1)[..., [0, 4, 1, 5, 2, 6, 3, 7]]
    assert torch.allclose(rope(samples, positions), restored, atol=1e-6, rtol=1e-5)
    try:
        rope(torch.ones(1, 1, 7), torch.tensor([0]))
    except ValueError:
        pass
    else:
        raise AssertionError('奇数维示例必须拒绝')
    try:
        sinusoidal(torch.tensor([-1]), 8)
    except ValueError:
        pass
    else:
        raise AssertionError('负位置必须拒绝')
    print('正弦位置、RoPE 旋转与相对位移自检通过。')


if __name__ == '__main__':
    main()

```

逐段理解。 angles 生成不同频率的相位；sinusoidal 交错放置 sin/cos；rope 对相邻维做二维旋转；自检验证保范数、共同平移点积不变，并转换维度排列后与 Llama 辅助函数比较。

### 这段代码怎么在面试里讲

- 不变量：输入为 [B,T,D]，D 为正偶数；输出形状不变，二维旋转保持向量范数。

- 关键过程：按位置生成相位，对每对通道做旋转；Q/K 同时平移时，相位差不变。

- 边界与坑：不同实现的配对布局不同；生成位置偏移、长文本缩放及浮点精度都须按模型验证。

手写与调库怎么选。 手写用于理解二维旋转与相对位移；实际应用应使用目标模型内置位置实现，避免自行改变配对或缩放，代码中的库对照固定版本且不代表所有模型。

### 本节收尾：费曼自述与闪卡

费曼自述： 正弦表给每个位置一组不同频率的值；RoPE 根据位置旋转 Q/K，使点积与相对位移有关，但长文本表现仍需实际评测。

- 问：RoPE 通常旋转哪些张量？答：注意力中的 Q 和 K，具体实现按模型确认。

- 问：相邻配对与前后半维配对可以直接比较吗？答：不能，需按布局重排并对齐频率。

- 问：RoPE 保证无限上下文吗？答：不保证，训练范围外可能退化。

- 问：缓存解码的位置从哪里开始？答：按已有前缀和模型接口确定，不能每步都从零开始。

### 原文核对 / 来源

[原仓库正弦位置编码](原仓库Markdown/docs/llm/transformer/pos_encode.md)；[RoFormer / RoPE 论文](https://arxiv.org/abs/2104.09864)。

<a id="s8"></a>

## 8.8 残差、LayerNorm 与 FFN 的分工【🔥 必会】

〔补充自原仓库镜像〕〔本次新补，参考当前面试趋势〕

目标 + 例子。 注意力聚合上下文后，FFN 逐位置变换特征，残差保留输入旁路，LayerNorm 调整特征尺度；用实验验证形状和逐位置性。

### 概念与数据流

| 五要素 | Pre-norm FFN 残差子层 |
| --- | --- |
| 输入 | 隐藏态 [B,T,C]，以及归一化和 FFN 参数。 |
| 做什么 | 对每个位置归一化，经过 FFN，再与输入相加。 |
| 输出 | 与输入同形状的 [B,T,C]。 |
| 哪里出错 | 归一化轴错误、残差形状不一致、混淆 FFN 与跨 token 注意力。 |
| 参数影响 | eps、隐藏维、激活函数及归一化顺序影响数值、容量和计算。 |

Pre-norm 可写为 x + FFN(LN(x))，post-norm 可写为 LN(x + FFN(x))；二者不等价。标准 FFN 独立处理每个位置，但完整 Transformer 中注意力已经混合上下文。LayerNorm 的具体归一化维度由模型配置决定。

### 面试追问

【🔥 必会】残差与 FFN 分别做什么？ 结论：FFN 变换特征，残差保留旁路。残差也提供直接的梯度路径，但不保证任何网络都稳定训练。

【🔥 必会】归一化结果异常怎么定位？ 结论：核对归一化维度、eps 和参数。检查输入是否有限、是否为常数以及权重偏置；对照框架层的小样本结果。

【⭐ 理解】FFN 为什么不负责跨 token 交互？ 结论：标准 FFN 在每个位置独立变换。跨位置聚合由注意力等模块完成，具体以模型结构为准。

### 故障模式与排障

- 残差维度不匹配：打印输入和子层输出的 B/T/C，确认是否需要投影；避免依赖意外广播。

- 归一化出现非有限数：检查输入、eps 和计算精度；用常数输入和框架实现对照。

- 误以为 FFN 混合位置：只改变一个 token，比较其他位置的 FFN 输出；检查是否存在其他跨位置层。

### 工程 trick 与坑

- 示例的零均值断言只针对初始 weight=1,bias=0；训练后的缩放和平移可能改变均值与方差。

- 不同模型可能使用 RMSNorm、门控 FFN 或不同残差布局；不要套用单一公式。

- 本节使用框架层观察性质，手写实现与数值对照见第 9.3—9.5 节。

### Toy code：观察组件性质

运行前提：Python 3.10+、PyTorch；CPU 可运行。随机数据只用于不变量检查。

```python
"""观察 LayerNorm、逐位置 FFN 与残差的性质，不进行真实模型训练。"""
import torch
from torch import Tensor, nn


class Components(nn.Module):
    """以 pre-norm 方式将逐位置 FFN 放在残差分支中。"""
    def __init__(self, channels: int, hidden: int, eps: float = 1e-5) -> None:
        super().__init__()
        if channels <= 0 or hidden <= 0 or eps <= 0:
            raise ValueError('维度和 eps 必须为正')
        self.channels = channels
        self.norm = nn.LayerNorm(channels, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """返回归一化、FFN 输出与残差输出，形状均为 (B,T,C)。"""
        if x.ndim != 3 or x.shape[0] == 0 or x.shape[1] == 0 or x.shape[2] != self.channels:
            raise ValueError('输入必须为非空 (B,T,C)')
        if x.dtype != self.norm.weight.dtype or x.device != self.norm.weight.device or not torch.isfinite(x).all():
            raise ValueError('设备、dtype 或数值不合法')
        normed = self.norm(x)
        transformed = self.ffn(normed)
        result = x + transformed
        assert normed.shape == transformed.shape == result.shape == x.shape
        return normed, transformed, result


def main() -> None:
    """验证归一化维度、逐位置性、残差关系和梯度。"""
    torch.manual_seed(73)
    model = Components(8, 24)
    x = torch.randn(2, 4, 8, requires_grad=True)
    normed, transformed, result = model(x)
    assert torch.allclose(normed.mean(-1), torch.zeros(2, 4), atol=1e-6)
    assert torch.allclose(result, x + transformed)
    assert torch.allclose(model.norm(x + 10), model.norm(x), atol=2e-6)
    changed = x.detach().clone(); changed[:, -1] += 20
    _, other_ffn, _ = model(changed)
    assert torch.allclose(transformed[:, :-1], other_ffn[:, :-1], atol=1e-6)
    result.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None for p in model.parameters())
    for bad in (torch.empty(0, 4, 8), torch.randn(1, 4, 7)):
        try:
            model(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('空输入或错误维度必须拒绝')
    print('LayerNorm、FFN 逐位置性、残差与梯度自检通过。')


if __name__ == '__main__':
    main()

```

逐段理解。 Components 组合 LayerNorm、FFN 和残差；自检检查初始化时的均值、平移不变性、逐位置性以及梯度。

### 这段代码怎么在面试里讲

- 不变量：输入、归一化结果、FFN 结果和残差结果都为 [B,T,C]。

- 关键过程：先按最后一维归一化，再逐位置经过 FFN，最后与原输入相加。

- 边界与坑：零均值断言依赖初始参数；残差不保证收敛，不同模型的归一化和 FFN 可能不同。

### 本节收尾：费曼自述与闪卡

费曼自述： LayerNorm 按指定特征维处理数值，FFN 在每个位置变换特征，残差保留输入旁路；具体顺序由模型设计决定。

- 问：标准 FFN 混合不同 token 吗？答：不混合，它逐位置变换。

- 问：pre-norm 与 post-norm 相同吗？答：不相同，归一化和相加的顺序不同。

- 问：LayerNorm 最终输出总是零均值吗？答：不一定，可学习缩放和平移会改变结果。

- 问：残差保证训练一定稳定吗？答：不保证，还受模型和训练配置影响。

### 原文核对 / 来源

[原仓库 FFN 与 LayerNorm](原仓库Markdown/docs/llm/transformer/ffn_ln.md)。

<a id="s9"></a>

## 8.9 Transformer Block：把子层串起来【🔥 必会】

〔补充自原仓库镜像〕〔本次新补，参考当前面试趋势〕

目标 + 例子。 把注意力、FFN、残差和 LayerNorm 串成一个因果 pre-norm block；用框架层观察数据流，并通过未来扰动验证因果性。

### 概念与数据流

| 五要素 | 因果 pre-norm Block |
| --- | --- |
| 输入 | 隐藏态 [B,T,C] 和有效 token 标记 [B,T]。 |
| 做什么 | 归一化→因果多头注意力→残差，再归一化→FFN→残差。 |
| 输出 | [B,T,C]；本例额外将无效 query 位置清零。 |
| 哪里出错 | 默认库层并非因果、掩码方向错误、padding 泄漏、子层与残差维度不匹配。 |
| 参数影响 | 头数、FFN 宽度、dropout、归一化顺序和序列长度影响计算与行为。 |

注意力子层聚合允许位置的信息，FFN 对各位置独立变换特征。下面用 TransformerEncoderLayer 配合显式因果 mask 演示结构；这不是完整 decoder，也不包含位置编码、词表投影、交叉注意力或 KV cache。完整手写实现与权重对照见第 9.5 节。

### 面试追问

【🔥 必会】pre-norm Block 如何计算？ 结论：每个子层先归一化，再与旁路相加。先注意力子层和残差，再 FFN 子层和残差；具体结构仍要看模型实现。

【🔥 必会】库层运行正常却出现未来泄漏怎么查？ 结论：检查是否显式设置因果可见性。用相同输入只改变未来位置，比较过去输出，再核对库的 mask 语义。

【⭐ 理解】此处的 encoder 层就是完整 decoder 吗？ 结论：不是。它仅在显式因果 mask 等配置下可演示相似自注意力 block；完整模型还包含其他必要组件。

### 故障模式与排障

- 未来 token 影响过去：核对 src_mask、布尔方向及位置维，用扰动测试定位。

- padding 干扰有效结果：检查 key padding mask、序列真实长度和输出处理；无效位置的 loss 也需屏蔽。

- 数值或梯度异常：检查输入有限性、dtype、设备与维度；逐子层对照并检查梯度。

### 工程 trick 与坑

- TransformerEncoderLayer 默认不是自回归解码器；示例手动传入上三角禁止 mask。

- 示例只支持右侧 padding、固定配置和无 dropout；实际模型的 mask、norm 和缓存可能不同。

- 生产中优先使用与模型权重一致的实现；手写 block 和库参数对照见第 9.5 节。

### Toy code：框架 Block 的因果边界

运行前提：Python 3.10+、PyTorch 2.x；CPU 可运行，随机输入没有业务语义。

```python
"""观察配置为因果 pre-norm 的框架 Block，不等同完整 decoder。"""
import torch
from torch import Tensor, nn


class CausalBlock(nn.Module):
    """封装一个无 dropout 的库层并处理右侧 padding 输出。"""
    def __init__(self, channels: int, heads: int, hidden: int) -> None:
        super().__init__()
        if min(channels, heads, hidden) <= 0 or channels % heads:
            raise ValueError('维度必须为正且通道数能被头数整除')
        self.channels = channels
        self.layer = nn.TransformerEncoderLayer(d_model=channels, nhead=heads, dim_feedforward=hidden,
                                                dropout=0.0, activation='gelu', batch_first=True,
                                                norm_first=True)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        """输入 (B,T,C) 与右侧 padding 标记，输出同形状。"""
        if x.ndim != 3 or x.shape[0] == 0 or x.shape[1] == 0 or x.shape[-1] != self.channels:
            raise ValueError('输入必须为非空 (B,T,C)')
        if valid.shape != x.shape[:2] or valid.dtype != torch.bool or valid.device != x.device:
            raise ValueError('valid 必须为同设备 (B,T) bool')
        if not valid[:, 0].all() or ((~valid[:, :-1]) & valid[:, 1:]).any():
            raise ValueError('只支持非空和右侧 padding')
        weight = self.layer.self_attn.in_proj_weight
        if x.dtype != weight.dtype or x.device != weight.device or not torch.isfinite(x).all():
            raise ValueError('输入设备、dtype 或数值非法')
        blocked = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device).triu(1)
        output = self.layer(x, src_mask=blocked, src_key_padding_mask=~valid)
        result = output * valid.unsqueeze(-1)
        assert result.shape == x.shape
        return result


def main() -> None:
    """检查因果性、padding、梯度和非法形状。"""
    torch.manual_seed(79)
    block = CausalBlock(8, 2, 24).eval()
    x = torch.randn(2, 4, 8, requires_grad=True)
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    output = block(x, valid)
    changed = x.detach().clone(); changed[:, -1] += 50
    assert torch.allclose(output[:, :-1], block(changed, valid)[:, :-1], atol=1e-6)
    assert torch.count_nonzero(output[1, 2:]).item() == 0
    output.square().mean().backward()
    assert block.layer.self_attn.in_proj_weight.grad is not None
    assert x.grad is not None and torch.isfinite(x.grad).all()
    try:
        block(torch.randn(2, 4, 7), valid)
    except ValueError:
        pass
    else:
        raise AssertionError('通道错误必须拒绝')
    print('框架 Block 的形状、因果性、padding 和梯度自检通过。')


if __name__ == '__main__':
    main()

```

逐段理解。 CausalBlock 构造 pre-norm 库层并传入因果及 padding mask；自检验证输出形状、未来扰动、无效输出和梯度。

### 这段代码怎么在面试里讲

- 不变量：输入输出均 [B,T,C]，通道数可被头数整除，序列非空且右侧 padding。

- 关键过程：显式传因果 mask 和 key padding mask，运行注意力/FFN 两个残差子层并清零无效输出。

- 边界与坑：库 encoder 层只是结构对照，不能替代完整 decoder；还需位置、缓存和具体模型的 mask。

### 本节收尾：费曼自述与闪卡

费曼自述： Block 将注意力和 FFN 与归一化、残差组合，形状前后保持一致；因果性依赖正确的可见性约束。

- 问：此例 block 的输入输出形状？答：都是 [B,T,C]。

- 问：默认 encoder 层一定因果吗？答：不是，示例显式传入 mask。

- 问：padding query 如何处理？答：本例将对应输出清零，训练时还要处理 loss。

- 问：完整模型是否还需要位置表示？答：通常需要具体模型设计的位置机制。

### 原文核对 / 来源

[原仓库 Transformer Block](原仓库Markdown/docs/llm/transformer/transformer_structure.md)。

<a id="s10"></a>

## 8.10 Transformer 面试：形状、复杂度与缓存【🔥 必会】

〔补充自原仓库镜像〕〔本次新补，参考当前面试趋势〕

目标 + 例子。 面试时先给输入输出和形状，再解释因果 mask、位置、计算与缓存；用小张量核对理论字节数，避免把公式当实际显存监控。

### 概念与数据流

| 五要素 | 注意力与 KV cache 粗算 |
| --- | --- |
| 输入 | 层数 L、批次 B、查询头 Hq、KV 头 Hkv、序列长度 T、头维 D 和每元素字节数。 |
| 做什么 | 估算一层完整注意力分数 [B,Hq,T,T] 与各层 K/V [2,L,B,Hkv,T,D] 的理论大小。 |
| 输出 | 两个理论字节数；它们不是实际峰值显存。 |
| 哪里出错 | GQA 使用错误头数、混淆单层与所有层、遗漏 dtype、误以为优化实现一定物化分数矩阵。 |
| 参数影响 | T 加倍时完整分数张量元素数变四倍，KV 元素数变两倍；实际开销还取决于实现和调度。 |

全量注意力的逻辑分数矩阵规模与 T² 有关，但优化后的 attention 内核可能不完整物化该矩阵；不能直接把理论张量大小当作显存峰值。KV 粗算也不含模型权重、激活、对齐、碎片、共享和元数据；详细缓存边界见第 6.2 节。

### 面试追问

【🔥 必会】多头 attention 的分数形状？ 结论：标准自注意力为 [B,H,T,T]。Q/K 每头维度为 D，点积沿 D 收缩；cross-attention 的两个序列长度可不同。

【🔥 必会】长上下文推理 OOM 怎么定位？ 结论：区分权重、KV 和运行时开销。记录并发、上下文、KV 头数和 dtype；用实际显存监控和小规模实验验证粗算。

【⭐ 理解】GQA 的缓存量按哪个头数算？ 结论：按 KV 头数计算。查询头可能多于 KV 头；本例假设两者可整除，具体结构要核对模型。

### 故障模式与排障

- 估算远小于实际显存：检查是否漏掉模型权重、激活、对齐、碎片和额外缓冲；用框架或服务监控实测。

- GQA 缓存估得过大：核对查询头数与 KV 头数；缓存公式使用实际 KV 头数。

- 理论分数矩阵很大但实测显存较低：检查使用的 attention 后端；优化实现可能分块计算而不完整存储分数矩阵。

### 工程 trick 与坑

- 示例估算一层分数和所有层 KV，口径不同且不能直接相加为总峰值。

- 相同层形状、统一 dtype 和完整上下文是简化假设；滑窗、分层配置和分布式切分需要分别计算。

- 测真实服务时同时记录延迟、吞吐与失败率，参见第 6.4 节。

### Toy code：理论张量字节数

运行前提：Python 3.10+、PyTorch；CPU 上用小张量核对公式，不测 GPU 峰值显存。

```python
"""计算注意力分数和 KV cache 的理论张量字节数，不等于峰值显存。"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Dimensions:
    """统一层数、批次、查询头、KV 头、序列长度和头维度。"""
    layers: int
    batch: int
    query_heads: int
    kv_heads: int
    tokens: int
    head_dim: int
    bytes_per_element: int


def estimate(d: Dimensions) -> dict[str, int]:
    """估计单层完整注意力分数张量与全部层 K/V 张量的字节数。"""
    values = (d.layers, d.batch, d.query_heads, d.kv_heads, d.tokens, d.head_dim, d.bytes_per_element)
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values):
        raise ValueError('所有维度和字节数必须为正整数')
    if d.query_heads % d.kv_heads:
        raise ValueError('此示例的 GQA 要求查询头数是 KV 头数的整数倍')
    scores = d.batch * d.query_heads * d.tokens * d.tokens * d.bytes_per_element
    cache = 2 * d.layers * d.batch * d.kv_heads * d.tokens * d.head_dim * d.bytes_per_element
    return {'single_layer_attention_score_bytes': scores, 'all_layer_kv_bytes': cache}


def main() -> None:
    """用真实小张量核对字节数，并检查 GQA 与非法维度。"""
    dims = Dimensions(layers=2, batch=2, query_heads=4, kv_heads=2,
                      tokens=3, head_dim=4, bytes_per_element=4)
    result = estimate(dims)
    scores = torch.empty(dims.batch, dims.query_heads, dims.tokens, dims.tokens, dtype=torch.float32)
    kv = torch.empty(2, dims.layers, dims.batch, dims.kv_heads, dims.tokens, dims.head_dim, dtype=torch.float32)
    assert result['single_layer_attention_score_bytes'] == scores.numel() * scores.element_size()
    assert result['all_layer_kv_bytes'] == kv.numel() * kv.element_size()
    longer = estimate(Dimensions(2, 2, 4, 2, 6, 4, 4))
    assert longer['single_layer_attention_score_bytes'] == 4 * result['single_layer_attention_score_bytes']
    assert longer['all_layer_kv_bytes'] == 2 * result['all_layer_kv_bytes']
    for bad in (Dimensions(1, 1, 3, 2, 4, 4, 4), Dimensions(1, 1, 2, 2, 0, 4, 4)):
        try:
            estimate(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('错误头数或长度必须拒绝')
    print('理论张量字节数自检通过:', result)
    print('没有测量实际峰值显存；优化注意力实现可能不物化分数矩阵。')


if __name__ == '__main__':
    main()

```

逐段理解。 Dimensions 保存统一配置；estimate 分别计算一层分数矩阵与所有层 K/V 的字节数；自检用实际小张量核对，并验证长度变化趋势。

### 这段代码怎么在面试里讲

- 不变量：维度和每元素字节数为正；此例 GQA 要求查询头数为 KV 头数的整数倍。

- 关键过程：分数按 [B,Hq,T,T] 计数；缓存按 K/V、层、批次、KV 头、长度和头维计数。

- 边界与坑：两个结果是理论张量大小，实际优化实现、碎片和模型权重会改变峰值；需要实测。

### 本节收尾：费曼自述与闪卡

费曼自述： 注意力逻辑分数的形状随序列长度平方增长，KV 缓存随长度线性增长；实际显存受实现、权重和调度影响。

- 问：KV 粗算用查询头数还是 KV 头数？答：KV 头数。

- 问：T 加倍时完整分数张量元素数怎样变化？答：其他维度不变时变为四倍。

- 问：理论分数张量一定完整占用显存吗？答：不一定，优化内核可能不物化它。

- 问：单层分数与全层缓存相加就是峰值吗？答：不是，生命周期和其他开销也要考虑。

### 原文核对 / 来源

[原仓库 Transformer 面试题](原仓库Markdown/docs/interview/llm/transformer_interview.md)。

## 本章完成标准

用 B=2、T=4、C=8、H=2 手写每一步形状；对两个 token 手算一次 softmax 和加权 Value；解释因果 Mask 与梯度/训练的关系。

能闭卷解释本章 A/B 内容，并拿出代码、测试或实验记录后，才把本章标成“已学会”。

