# 第九章：手写 Transformer——从组件到训练与生成

阅读方式：这是本章的完整学习文件，按知识链路合并原仓库文章。先读每节的解释与例子，再读技术细节；每节末尾闭卷做复述和闪卡。原文镜像仅供核对，不是必需的第二教材。

## 本章主线

照着代码看懂不算会。本章的目标是从空白文件实现注意力、多头、FFN、LayerNorm 和 Block，能处理 batch/sequence/head 维度与 Mask，并用断言验证。

本章动手证据： 逐组件重写，不复制原文；逐组件测试形状和梯度，并对注意力测试因果 Mask；最后用小数据集训练 Tiny GPT，记录训练与验证 loss。

## 本章目录

- 9.1 手写缩放点积注意力：因果与 padding

- 9.2 手写多头注意力：拆头、mask 与拼接

- 9.3 手写 FFN：逐位置特征变换

- 9.4 手写 LayerNorm 与残差：维度、eps 和顺序

- 9.5 手写 Transformer Block：pre-norm、注意力与 FFN

- 9.6 Tiny Transformer：从零拼接、训练与生成

## 与面试、代码的关系

面试串联： 展示自己写过的模块与断言：多头 reshape/transpose、因果 Mask、残差形状、梯度流和训练 loss。面试中先说代码要满足的形状不变量，再写实现。

需要亲手写的代码： 能写 PyTorch 的 nn.Module、注意力 mask、reshape/transpose、FFN、LayerNorm/残差、训练循环和形状/梯度断言。

跨章节的需求分解、评测数据、指标与回归门禁见[系统设计与评测](横切-系统设计与评测.md)。

<a id="s1"></a>

## 9.1 手写缩放点积注意力：因果与 padding【🔥 必会】

〔补充自原仓库镜像〕〔本次新补，参考当前面试趋势〕

目标 + 例子。 给定 Q/K/V=[B,T,D]，自己计算缩放分数、因果 mask、softmax 与加权值；对右侧 padding 保证无效 key 不被关注，padding query 的输出为零，再与 PyTorch SDPA 对照。

### 概念与数据流

| 五要素 | 单头因果注意力 |
| --- | --- |
| 输入 | 已投影的 Q/K/V [B,T,D]，以及 valid[B,T]；示例限定三者相同维度、右侧 padding。 |
| 做什么 | 计算 QKᵀ/√D，屏蔽未来和无效 key，经 softmax 加权 V，再将无效 query 输出置零。 |
| 输出 | 上下文 [B,T,D] 和权重 [B,T,T]；示例返回的 padding query 权重不参与有效输出。 |
| 哪里出错 | mask 方向反、所有 key 被屏蔽、padding 泄漏、输入形状或数值非法。 |
| 参数影响 | 维度与缩放影响分数分布；mask 和序列长度决定可见范围与计算量。 |

QKᵀ 产生位置间分数，除以 √D 缓解维度增大时点积尺度上升，softmax 后对 V 加权。示例规定 valid=True 表示真实 token；手写 masked_fill 中 ~allowed=True 表示屏蔽。PyTorch SDPA 的布尔 attn_mask=True 表示允许参与，其他 API 的含义可能不同，应核对当前文档。

### 面试追问

【🔥 必会】怎么确认因果 mask 没反？ 结论：未来位置权重应为零。除检查矩阵上三角，也可修改未来 token 的 V 并确认前面输出不变；带 padding 时另验无效位置。

【🔥 必会】注意力出现 NaN 怎么排查？ 结论：先查是否有全遮蔽行或非有限输入。检查 mask、空序列、dtype 和分数；明确处理无有效 key 的 query，不对全 -inf 做普通 softmax。

【⭐ 理解】为什么要缩放 √D？ 结论：控制点积分数随维度增长的尺度。在常见独立同尺度假设下，点积方差随维度增加；缩放可降低 softmax 过早饱和的风险。

### 故障模式与排障

- 未来信息泄漏：检查 tril/triu 和布尔方向，观察上三角权重；构造未来值扰动测试。

- padding 改变有效输出：检查 key mask 和真实长度；将无效 key 排除，并按模型约定处理无效 query。

- 某些 batch 出现 NaN：检查空序列和全遮蔽行；本例拒绝空序列及非右侧 padding，其他布局需设计额外处理。

### 工程 trick 与坑

- 下例只支持右侧 padding、非空序列和相同 Q/K/V 维度；不是通用 cross-attention 实现。

- 返回的 padding query 权重仍是归一化分布，但最终输出已置零；训练时还需正确处理对应 loss。

- SDPA 中布尔 mask 的 True 与某些多头注意力 API 约定不同，不能照搬。

### Toy code：手写注意力与 SDPA 对照

运行前提：Python 3.10+、PyTorch 2.x；CPU 可运行。使用固定随机张量，无模型权重或网络。

```python
"""手写单头因果注意力，支持右侧 padding，并与 PyTorch SDPA 对照。"""
import math
import torch
from torch import Tensor
from torch.nn import functional as F


def attention(q: Tensor, k: Tensor, v: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """输入 Q/K/V=(B,T,D)、valid=(B,T)，输出上下文与注意力权重。"""
    if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError('Q/K/V 必须具有相同的三维形状')
    batch, length, dim = q.shape
    if batch == 0 or length == 0 or dim == 0 or not q.is_floating_point():
        raise ValueError('张量必须非空且为浮点数')
    if any(t.device != q.device or t.dtype != q.dtype or not torch.isfinite(t).all() for t in (q, k, v)):
        raise ValueError('Q/K/V 的设备、类型和数值必须一致且有限')
    if valid.shape != (batch, length) or valid.dtype != torch.bool or valid.device != q.device:
        raise ValueError('valid 必须为同设备的 (B,T) 布尔张量')
    if not valid[:, 0].all() or ((~valid[:, :-1]) & valid[:, 1:]).any():
        raise ValueError('每条序列非空，且示例只支持右侧 padding')
    causal = torch.ones(length, length, dtype=torch.bool, device=q.device).tril()
    allowed = causal.unsqueeze(0) & valid[:, None, :]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(dim)
    weights = torch.softmax(scores.masked_fill(~allowed, float('-inf')), dim=-1)
    result = (weights @ v) * valid.unsqueeze(-1)
    assert result.shape == (batch, length, dim) and weights.shape == (batch, length, length)
    return result, weights


def main() -> None:
    """验证因果性、padding、梯度及 SDPA 的 True=允许语义。"""
    torch.manual_seed(29)
    q = torch.randn(2, 4, 6, requires_grad=True)
    k = torch.randn(2, 4, 6, requires_grad=True)
    v = torch.randn(2, 4, 6, requires_grad=True)
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    actual, weights = attention(q, k, v, valid)
    allowed = torch.ones(4, 4, dtype=torch.bool).tril()[None, :, :] & valid[:, None, :]
    reference = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
    reference = reference * valid.unsqueeze(-1)
    assert torch.allclose(actual, reference, atol=1e-6, rtol=1e-5)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 4))
    assert torch.count_nonzero(weights.triu(1)).item() == 0
    assert torch.count_nonzero(actual[1, 2:]).item() == 0
    changed = v.detach().clone(); changed[1, 2:] += 100
    alternate, _ = attention(q, k, changed, valid)
    assert torch.allclose(actual[1, :2], alternate[1, :2])
    actual.square().sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
    for invalid in (torch.zeros_like(valid), torch.tensor([[True, False, True, False], [True, True, True, True]])):
        try:
            attention(q.detach(), k.detach(), v.detach(), invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('空序列或非右侧 padding 必须拒绝')
    print('手写注意力与 SDPA、因果性、padding 和梯度自检通过。')


if __name__ == '__main__':
    main()

```

逐段理解。 输入检查拒绝空序列和非右侧 padding；构造因果与 key 有效性组合 mask；手写 softmax 加权后对无效 query 清零；自检与 SDPA 比较，并验证因果、padding 和梯度。

### 这段代码怎么在面试里讲

- 不变量：Q/K/V 为 [B,T,D]，权重为 [B,T,T]，每条序列至少一个真实 token；无效 query 输出为零。

- 关键过程：先缩放 QK 分数，屏蔽未来与无效 key，再 softmax 加权 V；用同一允许 mask 对照 SDPA。

- 边界与坑：布尔含义因 API 而异，全遮蔽会产生 NaN；本例不支持左 padding 和不同 value 维度。

手写与调库怎么选。 手写帮助检查形状、掩码和梯度；真实模型优先使用经过优化的注意力实现，并在目标 PyTorch 版本与设备上验证数值及性能。

### 本节收尾：费曼自述与闪卡

费曼自述： Q 与 K 算分数，缩放后屏蔽不可见位置，softmax 得到权重，再加权 V；padding 和因果约束都需要明确。

- 问：SDPA 的布尔 True 表示什么？答：允许注意该位置；应核对所用版本文档。

- 问：全遮蔽一行为什么危险？答：普通 softmax 可能得到 NaN。

- 问：padding key 和 query 都要考虑吗？答：要；本例屏蔽 key 并将无效 query 输出清零。

- 问：注意力计算顺序是什么？答：点积、缩放、mask、softmax、加权 V。

### 原文核对 / 来源

[原仓库手撕 Attention](原仓库Markdown/docs/llm/transformer/attention_code.md)；[PyTorch SDPA 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。

<a id="s2"></a>

## 9.2 手写多头注意力：拆头、mask 与拼接【🔥 必会】

〔补充自原仓库镜像〕〔本次新补，参考当前面试趋势〕

目标 + 例子。 给定 X=[B,T,C]，通过线性层产生 Q/K/V，拆为 H 个头，分别做因果注意力后合并，并与相同权重的 PyTorch 多头层比较。

### 概念与数据流

| 五要素 | 多头自注意力 |
| --- | --- |
| 输入 | 隐藏态 [B,T,C]、头数 H 和有效 token mask [B,T]。 |
| 做什么 | 线性投影、拆头 [B,H,T,C/H]、按头做缩放因果注意力、拼回并输出投影。 |
| 输出 | 与输入同形状的 [B,T,C]；本例将 padding query 位置清零。 |
| 哪里出错 | C 不能整除 H、transpose 顺序错误、mask 语义弄反、无效 key 泄漏。 |
| 参数影响 | 头数改变每头维度；隐藏维和序列长度影响计算与内存，mask 决定可见位置。 |

多头不是把整段序列复制 H 次独立算完整维度；此例先将 C 拆成 H 个 C/H 维的头。每个头对自己的投影空间计算注意力，拼接后经输出投影混合。这里是标准 MHA；GQA/MQA 的 Q 与 KV 头数关系不同。

### 面试追问

【🔥 必会】拆头前后张量形状是什么？ 结论：[B,T,C]→[B,H,T,C/H]。计算注意力得到 [B,H,T,T]，加权 V 后转置拼回 [B,T,C]。

【🔥 必会】手写结果与库不同怎么查？ 结论：先对齐权重、mask 和模式。确认 QKV 的排列、投影偏置、batch_first、dropout、padding query 处理和数值容差。

【⭐ 理解】头数越多一定越好吗？ 结论：不一定。固定 C 时每头维度变小；性能和表达效果受模型结构与任务影响，应通过实验选择。

### 故障模式与排障

- reshape 报错或输出乱序：逐步打印 [B,T,H,D] 与 [B,H,T,D]，确认 C%H==0；转置后按需 contiguous 再合并。

- 库对照出现未来泄漏：检查 nn.MultiheadAttention 的布尔 mask；本例 True 为屏蔽，与 SDPA 布尔 mask 的约定不同。

- padding 影响有效位置：检查 key padding mask 并扰动无效输入；输出的无效 query 按任务约定清零或忽略。

### 工程 trick 与坑

- 对照时显式复制 QKV 和输出投影权重；两个随机初始化的模型不应直接比较数值。

- 本例只支持右侧 padding、非空序列和标准 MHA，没有 KV cache 或 cross-attention。

- nn.MultiheadAttention 布尔 mask 在本例中 True 表示禁止，不能与第 9.1 节 SDPA 的 True 表示允许混用。

### Toy code：手写 MHA 与库对照

运行前提：Python 3.10+、PyTorch 2.x；CPU 可运行，无下载。

```python
"""手写多头因果自注意力，与 torch.nn.MultiheadAttention 对照。"""
import math
import torch
from torch import Tensor, nn


class MultiHead(nn.Module):
    """输入输出 (B,T,C)，支持右侧 padding 和因果 mask。"""
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels <= 0 or heads <= 0 or channels % heads:
            raise ValueError('通道数和头数必须为正且整除')
        self.channels, self.heads = channels, heads
        self.head_dim = channels // heads
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        """按头拆分、屏蔽未来与无效 key、拼接并输出投影。"""
        if x.ndim != 3 or x.shape[-1] != self.channels or x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError('输入必须是非空 (B,T,C)')
        batch, length, channels = x.shape
        if valid.shape != (batch, length) or valid.dtype != torch.bool or valid.device != x.device:
            raise ValueError('valid 必须为同设备的 (B,T) bool')
        if not valid[:, 0].all() or ((~valid[:, :-1]) & valid[:, 1:]).any():
            raise ValueError('只支持非空序列与右侧 padding')
        if not torch.isfinite(x).all():
            raise ValueError('输入必须为有限数')
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(t: Tensor) -> Tensor:
            return t.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        allowed = causal[None, None] & valid[:, None, None, :]
        weights = torch.softmax(scores.masked_fill(~allowed, float('-inf')), dim=-1)
        merged = (weights @ v).transpose(1, 2).contiguous().reshape(batch, length, channels)
        result = self.proj(merged) * valid.unsqueeze(-1)
        assert result.shape == x.shape
        return result


def main() -> None:
    """复制权重到官方多头层，验证输出、因果、padding 和梯度。"""
    torch.manual_seed(31)
    model = MultiHead(8, 2).eval()
    reference = nn.MultiheadAttention(8, 2, batch_first=True, dropout=0.0).eval()
    with torch.no_grad():
        reference.in_proj_weight.copy_(model.qkv.weight)
        reference.in_proj_bias.copy_(model.qkv.bias)
        reference.out_proj.weight.copy_(model.proj.weight)
        reference.out_proj.bias.copy_(model.proj.bias)
    x = torch.randn(2, 4, 8, requires_grad=True)
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    actual = model(x, valid)
    blocked = torch.ones(4, 4, dtype=torch.bool).triu(1)
    expected, _ = reference(x, x, x, attn_mask=blocked, key_padding_mask=~valid, need_weights=False)
    expected = expected * valid.unsqueeze(-1)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    altered = x.detach().clone(); altered[:, -1] += 100
    assert torch.allclose(actual[:, :-1], model(altered, valid)[:, :-1], atol=1e-5)
    assert torch.count_nonzero(actual[1, 2:]).item() == 0
    actual.square().sum().backward()
    assert model.qkv.weight.grad is not None and torch.isfinite(model.qkv.weight.grad).all()
    for channels, heads in ((8, 3), (8, 0)):
        try:
            MultiHead(channels, heads)
        except ValueError:
            pass
        else:
            raise AssertionError('非法头数必须拒绝')
    print('手写多头与 PyTorch 对照、因果/padding/梯度自检通过。')


if __name__ == '__main__':
    main()

```

逐段理解。 qkv 一次生成三个投影；split 拆头并转置；组合因果与 key 有效性后 softmax；拼接和输出投影后清零无效 query；自检复制权重并验证结果、因果和梯度。

### 这段代码怎么在面试里讲

- 不变量：C%H==0，输入输出均 [B,T,C]，每头 [B,H,T,C/H]。

- 关键过程：投影→拆头→mask→逐头注意力→拼接→输出投影；将库权重对齐后做数值比较。

- 边界与坑：不同 API 的 bool mask 语义不同；空序列和左 padding 本例拒绝，真实模型还需处理缓存与其他注意力结构。

手写与调库怎么选。 手写用于掌握维度与 mask；实际项目优先成熟多头/优化注意力实现，并按设备和版本验证精度、掩码与性能。

### 本节收尾：费曼自述与闪卡

费曼自述： 多头把投影后的通道拆成多个头，分别计算注意力，再按原顺序拼接和投影；拆头和 mask 的维度必须准确。

- 问：每个头的维度是多少？答：标准 MHA 中 C/H，要求整除。

- 问：头输出怎样恢复 [B,T,C]？答：先转置回时间在前，再连续化并 reshape。

- 问：此例中 MHA 的布尔 True 表示什么？答：禁止注意；SDPA 的布尔语义不同。

- 问：为什么要复制库层权重再比较？答：不同随机参数的输出本来就不同。

### 原文核对 / 来源

[原仓库手撕 Multi-Head Attention](原仓库Markdown/docs/llm/transformer/mha_code.md)；[PyTorch MultiheadAttention 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)。

<a id="s3"></a>

## 9.3 手写 FFN：逐位置特征变换【🔥 必会】

〔补充自原仓库镜像〕〔本次新补，参考当前面试趋势〕

目标 + 例子。 给定隐藏态 [B,T,C]，对每个位置独立执行升维、非线性和降维；再把相同权重复制到 nn.Sequential，验证结果与梯度。

### 概念与数据流

| 五要素 | 两层 FFN |
| --- | --- |
| 输入 | 隐藏态 [B,T,C]；权重 W1[C,F]、W2[F,C] 和对应偏置。 |
| 做什么 | GELU(XW1+b1)W2+b2，共享权重但不混合不同时间位置。 |
| 输出 | [B,T,C]，可与残差相加。 |
| 哪里出错 | 权重方向错误、隐藏维不匹配、误认为 FFN 会跨 token 交互、padding loss 处理缺失。 |
| 参数影响 | 中间维度 F、激活函数和偏置影响参数量、计算量及表达能力。 |

注意力负责在可见 token 间聚合；此处 FFN 对每个 token 独立应用相同的特征变换。中间维常比 C 大，但倍率和激活因模型不同，不应把 4 倍和 GELU 当成所有模型的固定配置；部分模型使用门控 FFN。

### 面试追问

【🔥 必会】FFN 会让不同 token 互相交换信息吗？ 结论：标准逐位置 FFN 不会。它对每个位置使用相同变换；跨位置交互来自注意力等其他模块。

【🔥 必会】手写 FFN 与库输出不同怎么查？ 结论：检查权重方向、偏置与激活。nn.Linear.weight 形状是 [out,in]，本例手写权重按 [in,out] 存储，复制时需转置。

【⭐ 理解】FFN 中间维为什么常常更大？ 结论：扩展非线性特征空间。代价是更多参数与计算；具体倍率和结构应根据模型设计与目标负载选择。

### 故障模式与排障

- 矩阵相乘维度错误：核对 W1[C,F]、W2[F,C] 与输入最后一维；复制到库层时检查转置。

- 梯度不存在或非有限：检查参数是否参与前向、loss 是否有限与输入尺度；在小批次上打印梯度并对照库实现。

- padding 位置输出非零：偏置可能使其非零；按任务和整体模型约定处理 padding 与 loss mask，不能仅靠 FFN 自动忽略。

### 工程 trick 与坑

- 该代码采用 GELU 和两层矩阵，门控 FFN 的结构与参数量不同。

- 手写权重与 nn.Linear 的存储方向相反，数值对照必须复制并转置，而非分别随机初始化。

- FFN 的逐位置独立性不意味着整个 Transformer 对各位置独立；注意力已混合上下文。

### Toy code：手写 FFN 与 Sequential 对照

运行前提：Python 3.10+、PyTorch；CPU 可运行。示例使用随机数据，不代表训练效果。

```python
"""手写逐 token FFN，与权重一致的 PyTorch Sequential 对照。"""
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FFN(nn.Module):
    """两次仿射变换与 GELU，输入输出均为 (B,T,C)。"""
    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        if channels <= 0 or hidden <= 0:
            raise ValueError('维度必须为正')
        self.channels, self.hidden = channels, hidden
        self.w1 = nn.Parameter(torch.empty(channels, hidden))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.w2 = nn.Parameter(torch.empty(hidden, channels))
        self.b2 = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.w1.T, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2.T, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        """仅变换特征维度，不在时间位置之间混合。"""
        if x.ndim != 3 or x.shape[0] == 0 or x.shape[1] == 0 or x.shape[2] != self.channels:
            raise ValueError('输入必须是非空 (B,T,C)')
        if x.dtype != self.w1.dtype or x.device != self.w1.device or not torch.isfinite(x).all():
            raise ValueError('输入 dtype、设备或数值非法')
        output = F.gelu(x @ self.w1 + self.b1) @ self.w2 + self.b2
        assert output.shape == x.shape
        return output


def main() -> None:
    """对照库实现，验证逐位置独立性、梯度和维度检查。"""
    torch.manual_seed(37)
    model = FFN(8, 24)
    reference = nn.Sequential(nn.Linear(8, 24), nn.GELU(), nn.Linear(24, 8))
    with torch.no_grad():
        reference[0].weight.copy_(model.w1.T)
        reference[0].bias.copy_(model.b1)
        reference[2].weight.copy_(model.w2.T)
        reference[2].bias.copy_(model.b2)
    x = torch.randn(2, 4, 8, requires_grad=True)
    actual = model(x)
    assert torch.allclose(actual, reference(x), atol=1e-6, rtol=1e-5)
    altered = x.detach().clone(); altered[:, 3] += 50
    assert torch.allclose(actual[:, :3], model(altered)[:, :3])
    actual.square().mean().backward()
    assert model.w1.grad is not None and torch.isfinite(model.w1.grad).all()
    assert model.w2.grad is not None and torch.isfinite(model.w2.grad).all()
    try:
        model(torch.randn(2, 4, 7))
    except ValueError:
        pass
    else:
        raise AssertionError('错误的特征维度必须拒绝')
    print('手写 FFN、Sequential 对照、位置独立和梯度自检通过。')


if __name__ == '__main__':
    main()

```

逐段理解。 FFN 保存两组权重和偏置；前向只改变最后一维；将权重转置复制到 nn.Linear 后对照，并扰动一个 token 检查其他位置不变。

### 这段代码怎么在面试里讲

- 不变量：输入输出均为 [B,T,C]，中间态为 [B,T,F]，每个位置使用同一组参数。

- 关键过程：做线性升维、GELU 和线性降维，并用同权重库模型检查数值。

- 边界与坑：padding 输出未自动屏蔽；权重存储方向、非有限输入与不同激活都需核对。

手写与调库怎么选。 手写矩阵变换便于理解参数方向与梯度；实际模型使用框架层或模型自带 FFN，针对设备和版本做验证。

### 本节收尾：费曼自述与闪卡

费曼自述： FFN 对每个 token 的特征做同一组非线性变换，通常先扩展再映射回 C；它本身不让不同位置交换信息。

- 问：FFN 是否混合序列位置？答：标准逐位置 FFN 不混合。

- 问：两个权重形状是什么？答：本例为 [C,F] 和 [F,C]。

- 问：4 倍中间维是否通用？答：不是，模型可能采用其他倍率或门控结构。

- 问：padding 输出必然为零吗？答：不一定，偏置可能产生非零值。

### 原文核对 / 来源

[原仓库手撕 FFN](原仓库Markdown/docs/llm/transformer/fnn_code.md)。

<a id="s4"></a>

## 9.4 手写 LayerNorm 与残差：维度、eps 和顺序【🔥 必会】

〔补充自原仓库镜像〕〔本次新补，参考当前面试趋势〕

目标 + 例子。 对 [B,T,C] 的每个 token 在 C 维归一化，与 nn.LayerNorm 对照；再写出 pre-norm 和 post-norm 的不同残差顺序。

### 概念与数据流

| 五要素 | LayerNorm 与残差 |
| --- | --- |
| 输入 | 隐藏态 [B,T,C]、缩放与偏置 [C]、正数 eps，以及形状匹配的子层输出。 |
| 做什么 | 沿最后一维算均值和总体方差，归一化、缩放平移；残差把子层输入与输出相加。 |
| 输出 | 形状保持为 [B,T,C] 的张量。 |
| 哪里出错 | 按错误维度归一化、方差定义不一致、eps 无效、残差维度不匹配。 |
| 参数影响 | eps 影响低方差下的数值稳定，缩放和平移参数改变输出；pre/post 顺序改变数据流。 |

Pre-norm 常写作 x + sublayer(LN(x))；post-norm 常写作 LN(x + sublayer(x))。它们不是同一个函数，训练稳定性还受初始化、深度和其他设计影响，不能仅凭顺序保证模型一定收敛。

### 面试追问

【🔥 必会】LayerNorm 对哪个维度求统计量？ 结论：由 normalized_shape 决定。此例只沿每个 token 的最后 C 维计算；它不是沿 batch 求均值。

【🔥 必会】常数输入为什么不应出现 NaN？ 结论：分母里有正的 eps。常数输入的方差为零，归一化部分为零，输出等于可学习偏置；还需检查 dtype 和非有限输入。

【⭐ 理解】Pre-norm 和 post-norm 有什么区别？ 结论：归一化与残差相加的顺序不同。两种结构的数据和梯度路径不同；实际稳定性需结合完整模型和实验分析。

### 故障模式与排障

- 手写结果与库不一致：核对归一化维度、总体方差、eps、缩放与偏置；复制相同参数后比较。

- 常数输入出现非有限值：检查 eps 是否为正、输入是否有限以及计算 dtype；用常数样本做单元自检。

- 残差相加报形状错：核对 batch、序列和通道维；明确是否需要额外投影，不能依赖意外广播掩盖错误。

### 工程 trick 与坑

- torch.var 的方差修正参数会影响结果；本例显式求平方差均值，与 nn.LayerNorm 的总体方差口径对照。

- LayerNorm 后经过可学习缩放和平移，最终输出不一定严格零均值、单位方差。

- 一些模型使用 RMSNorm 等不同组件，不能把这里的公式当作所有模型实现。

### Toy code：LayerNorm 与 pre/post 残差

运行前提：Python 3.10+、PyTorch；CPU 可运行，随机数据只验证实现。

```python
"""手写最后一维 LayerNorm，并展示 pre/post norm 残差顺序。"""
import math
import torch
from torch import Tensor, nn


class ManualLayerNorm(nn.Module):
    """使用总体方差与可学习缩放、偏置的 LayerNorm。"""
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        if channels <= 0 or not math.isfinite(eps) or eps <= 0:
            raise ValueError('通道数和 eps 必须为正且有限')
        self.channels, self.eps = channels, eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        """对最后一维求均值和总体方差，输出形状与输入一致。"""
        if x.ndim != 3 or x.shape[0] == 0 or x.shape[1] == 0 or x.shape[-1] != self.channels:
            raise ValueError('输入必须是非空 (B,T,C)')
        if x.dtype != self.weight.dtype or x.device != self.weight.device or not torch.isfinite(x).all():
            raise ValueError('输入设备、dtype 或数值非法')
        mean = x.mean(dim=-1, keepdim=True)
        variance = (x - mean).square().mean(dim=-1, keepdim=True)
        result = (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias
        assert result.shape == x.shape
        return result


def main() -> None:
    """比较库 LayerNorm，检查常数输入、残差顺序和梯度。"""
    torch.manual_seed(41)
    manual = ManualLayerNorm(8)
    reference = nn.LayerNorm(8, eps=1e-5)
    with torch.no_grad():
        manual.weight.copy_(torch.randn(8))
        manual.bias.copy_(torch.randn(8))
        reference.weight.copy_(manual.weight)
        reference.bias.copy_(manual.bias)
    x = torch.randn(2, 4, 8, requires_grad=True)
    actual = manual(x)
    assert torch.allclose(actual, reference(x), atol=1e-6, rtol=1e-5)
    constant = manual(torch.ones(2, 3, 8))
    assert torch.isfinite(constant).all() and torch.allclose(constant[0, 0], manual.bias)
    layer = nn.Linear(8, 8)
    pre = x + layer(manual(x))
    post = manual(x + layer(x))
    assert pre.shape == post.shape == x.shape and not torch.allclose(pre, post)
    pre.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert manual.weight.grad is not None and layer.weight.grad is not None
    for invalid in (torch.empty(0, 3, 8), torch.randn(2, 3, 7)):
        try:
            manual(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('空输入或错误维度必须拒绝')
    print('手写 LayerNorm、库对照、常数输入和残差梯度自检通过。')


if __name__ == '__main__':
    main()

```

逐段理解。 ManualLayerNorm 计算最后一维均值与总体方差；复制参数到库层后比较；常数输入验证 eps，pre/post 表达式验证顺序差异并检查梯度。

### 这段代码怎么在面试里讲

- 不变量：输入输出为 [B,T,C]，统计量为 [B,T,1]，缩放和平移为 [C]。

- 关键过程：减均值、除以 sqrt(var+eps)、乘权重并加偏置，再按 pre/post 顺序连接残差。

- 边界与坑：方差口径、归一化维度、常数输入和 dtype 都需要对齐；有偏置后输出不保证零均值。

手写与调库怎么选。 手写用于理解统计维度和残差数据流；实际项目使用框架或模型提供的归一化层，并对齐精度与版本。

### 本节收尾：费曼自述与闪卡

费曼自述： LayerNorm 按指定特征维做归一化并学习缩放和平移；残差保留旁路，pre/post norm 的子层顺序不同。

- 问：此例 LayerNorm 沿哪个维度计算？答：每个 token 的最后 C 维。

- 问：常数输入的输出一定为零吗？答：归一化部分为零，最终输出等于偏置。

- 问：pre-norm 的顺序是什么？答：先归一化再经过子层，最后与输入相加。

- 问：LayerNorm 的最终输出总是单位方差吗？答：不保证，可学习缩放和平移会改变它。

### 原文核对 / 来源

[原仓库 LayerNorm 与残差](原仓库Markdown/docs/llm/transformer/layernorm_residual_code.md)；[PyTorch LayerNorm 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)。

<a id="s5"></a>

## 9.5 手写 Transformer Block：pre-norm、注意力与 FFN【🔥 必会】

〔补充自原仓库镜像〕〔本次新补，参考当前面试趋势〕

目标 + 例子。 将手写 LayerNorm、多头因果注意力、FFN 和两条残差组合为一个 block，再把参数复制到带因果 mask 的 PyTorch 层验证。

### 概念与数据流

| 五要素 | 单层 pre-norm 因果 Block |
| --- | --- |
| 输入 | 隐藏态 [B,T,C] 和有效位置 [B,T]。 |
| 做什么 | x1=x+MHA(LN1(x))，y=x1+FFN(LN2(x1))，最后按本例约定清零 padding 输出。 |
| 输出 | 与输入同形状的 [B,T,C]；实际堆叠还要遵循模型的 padding 约定。 |
| 哪里出错 | mask 方向、拆头维度、两次残差位置、归一化参数或权重复制不一致。 |
| 参数影响 | 头数、隐藏维、FFN 宽度、eps 和 dropout 等改变计算和行为；本例 dropout 为零。 |

这里只实现 decoder-only 中的一种因果自注意力 block，没有 encoder-decoder 的 cross-attention。对照代码借用 TransformerEncoderLayer 的相同 pre-norm 结构并显式传入因果 mask，只验证这一配置下的数值等价，不代表它是完整 decoder 模型。

### 面试追问

【🔥 必会】一个 pre-norm block 的计算顺序？ 结论：归一化→注意力→残差，再归一化→FFN→残差。中间隐藏态维度应保持一致；投影和 FFN 的内部维度可以变化。

【🔥 必会】block 输出与参考层不一致怎么定位？ 结论：逐子层对齐参数、mask 和数值模式。先比较 LayerNorm、注意力、第一条残差与 FFN；确保相同权重、激活、dropout 和 padding 处理。

【⭐ 理解】因果 mask 为什么只需要作用在注意力？ 结论：本例的 FFN 和 LayerNorm 逐位置计算。它们不读取其他时间位置；注意力负责限制可见 token，其他跨位置模块则需另行检查。

### 故障模式与排障

- 未来信息泄漏：改变未来 token，比较之前位置输出；核对注意力 mask 的布尔方向与广播。

- 对照结果差异大：检查 QKV 权重排列、归一化 eps、激活、残差顺序和 dropout；逐层比较定位首个分歧。

- padding 位置或梯度异常：检查有效 key 和无效 query 的处理；对非空右 padding 写独立测试，并确保 loss 不监督 padding。

### 工程 trick 与坑

- 这里使用库的 Linear 构造参数，但注意力、LayerNorm、残差和 FFN 流程手写；没有 KV cache、位置编码或完整训练循环，这些见 6.2 和 9.6。

- 库对照显式复制参数、设 norm_first=True 和 dropout=0，并添加因果 mask；默认的 encoder 层不天然等同因果 decoder。

- 此例只支持非空、右侧 padding；其他布局和多模态 mask 需要明确设计。

### Toy code：手写 Block 与库层对照

运行前提：Python 3.10+、PyTorch 2.x；CPU 可运行。示例随机初始化，不训练业务模型。

```python
"""手写 pre-norm decoder block，并与同权重 PyTorch 编码层对照。"""
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Norm(nn.Module):
    """在最后 C 维执行 LayerNorm。"""
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """按最后维度的总体方差归一化。"""
        mean = x.mean(dim=-1, keepdim=True)
        variance = (x - mean).square().mean(dim=-1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class Block(nn.Module):
    """因果多头注意力 + FFN，使用 pre-norm 和残差。"""
    def __init__(self, channels: int, heads: int, hidden: int) -> None:
        super().__init__()
        if channels <= 0 or heads <= 0 or channels % heads or hidden <= 0:
            raise ValueError('维度必须为正，通道数须能被头数整除')
        self.channels, self.heads, self.head_dim = channels, heads, channels // heads
        self.norm1, self.norm2 = Norm(channels), Norm(channels)
        self.qkv, self.out = nn.Linear(channels, 3 * channels), nn.Linear(channels, channels)
        self.fc1, self.fc2 = nn.Linear(channels, hidden), nn.Linear(hidden, channels)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        """传入非空 (B,T,C) 和右侧 padding 的有效位置。"""
        if x.ndim != 3 or x.shape[0] == 0 or x.shape[1] == 0 or x.shape[2] != self.channels:
            raise ValueError('输入必须为非空 (B,T,C)')
        batch, length, channels = x.shape
        if x.dtype != self.qkv.weight.dtype or x.device != self.qkv.weight.device:
            raise ValueError('输入 dtype 或设备与参数不一致')
        if valid.shape != (batch, length) or valid.dtype != torch.bool or valid.device != x.device:
            raise ValueError('valid 必须是同设备的 (B,T) 布尔值')
        if not valid[:, 0].all() or ((~valid[:, :-1]) & valid[:, 1:]).any() or not torch.isfinite(x).all():
            raise ValueError('只支持有限输入与非空、右侧 padding 序列')
        normalized = self.norm1(x)
        q, k, v = self.qkv(normalized).chunk(3, dim=-1)
        def split(t: Tensor) -> Tensor:
            return t.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        allowed = causal[None, None] & valid[:, None, None, :]
        weights = torch.softmax(scores.masked_fill(~allowed, float('-inf')), dim=-1)
        attended = (weights @ v).transpose(1, 2).contiguous().reshape(batch, length, channels)
        x = x + self.out(attended)
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        result = x * valid.unsqueeze(-1)
        assert result.shape == (batch, length, channels)
        return result


def main() -> None:
    """复制参数，与配置为因果 pre-norm 的库层比较。"""
    torch.manual_seed(43)
    model = Block(8, 2, 24).eval()
    reference = nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=24,
                                           dropout=0.0, activation='gelu', batch_first=True,
                                           norm_first=True, layer_norm_eps=1e-5).eval()
    with torch.no_grad():
        reference.self_attn.in_proj_weight.copy_(model.qkv.weight)
        reference.self_attn.in_proj_bias.copy_(model.qkv.bias)
        reference.self_attn.out_proj.load_state_dict(model.out.state_dict())
        reference.linear1.load_state_dict(model.fc1.state_dict())
        reference.linear2.load_state_dict(model.fc2.state_dict())
        reference.norm1.load_state_dict({'weight': model.norm1.weight, 'bias': model.norm1.bias})
        reference.norm2.load_state_dict({'weight': model.norm2.weight, 'bias': model.norm2.bias})
    x = torch.randn(2, 4, 8, requires_grad=True)
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    actual = model(x, valid)
    blocked = torch.ones(4, 4, dtype=torch.bool).triu(1)
    expected = reference(x, src_mask=blocked, src_key_padding_mask=~valid) * valid.unsqueeze(-1)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    changed = x.detach().clone(); changed[:, -1] += 100
    assert torch.allclose(actual[:, :-1], model(changed, valid)[:, :-1], atol=1e-5)
    assert torch.count_nonzero(actual[1, 2:]).item() == 0
    actual.square().sum().backward()
    assert model.qkv.weight.grad is not None and torch.isfinite(model.qkv.weight.grad).all()
    try:
        Block(8, 3, 24)
    except ValueError:
        pass
    else:
        raise AssertionError('通道不能整除头数时必须拒绝')
    print('手写 pre-norm Block 与库层、因果性和梯度自检通过。')


if __name__ == '__main__':
    main()

```

逐段理解。 Norm 手写均值和总体方差；Block 拆头、应用因果和 key mask、连接两条 pre-norm 残差；自检将相同权重复制到配置一致的库层，检查输出、因果性和梯度。

### 这段代码怎么在面试里讲

- 不变量：输入输出都是 [B,T,C]，C 可被头数整除；每个序列至少有一个有效 token。

- 关键过程：先 LN→注意力→残差，再 LN→FFN→残差；对照时显式复制全部参数并加因果 mask。

- 边界与坑：对照的 encoder 层只用于匹配这个子结构，不是完整 decoder；还需留意 padding、位置、dropout 和模型具体结构。

手写与调库怎么选。 手写一层用于掌握数据流和排障；实际模型通常用官方实现和优化后端，确保 mask、缓存与 checkpoint 的结构一致。

### 本节收尾：费曼自述与闪卡

费曼自述： Pre-norm block 先归一化再经注意力和残差，接着归一化经 FFN 和残差；同权重对照有助于找出 mask 或顺序错误。

- 问：两条残差分别绕过什么？答：注意力子层和 FFN 子层。

- 问：默认 encoder 层一定是因果的吗？答：不是，示例显式传入因果 mask。

- 问：为什么要禁用 dropout 做数值对照？答：减少随机性，使相同参数的输出可比。

- 问：此 block 是否包含位置编码和 KV cache？答：没有，需由完整模型和推理流程提供。

### 原文核对 / 来源

[原仓库手撕 Transformer Block](原仓库Markdown/docs/llm/transformer/transformer_block_code.md)；[PyTorch TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)。

<a id="s6"></a>

## 9.6 Tiny Transformer：从零拼接、训练与生成【🔥 必会】

〔补充自原仓库镜像〕〔本次新补，参考当前面试趋势〕

目标 + 例子。 手写一个只有一层的因果 decoder：token 与位置 embedding → pre-norm 多头注意力和 FFN → logits → 交叉熵训练 → 贪心生成；再用 Transformers 随机小模型跑相同玩具任务，对照接口和标签移位。

### 概念与数据流

| 五要素 | 微型因果语言模型 |
| --- | --- |
| 输入 | token ID [B,T]，训练时还需要右移的下一个 token 作为标签。 |
| 做什么 | token/位置 embedding → 多头因果注意力 → 残差、LayerNorm、FFN → 词表投影；训练时反向传播。 |
| 输出 | logits [B,T,V]，训练 loss；生成时逐步选择新 token。 |
| 哪里出错 | 因果 mask 反向、位置越界、头数不能整除、标签重复移位、训练/验证泄漏。 |
| 参数影响 | 层数、隐藏维、头数、上下文长度、学习率和训练步数影响容量、资源和收敛。 |

Pre-norm 示例先归一化再经过子层，然后残差相加；FFN 分别作用于每个位置的特征，注意力负责跨位置交互。训练目标是让位置 t 预测 t+1。下面的留出集仅按玩具序列起点划分，序列片段仍可能重合，不是可靠的独立泛化评测；训练 loss 下降也不代表真实语言能力。

### 面试追问

【🔥 必会】从输入 token 到 loss 的形状怎么走？ 结论：[B,T]→[B,T,C]→[B,T,V]→标量。多头内部拆为 [B,H,T,D]；计算交叉熵前将 logits 和下一 token 标签展平。

【🔥 必会】模型训练 loss 不降怎么排查？ 结论：先查数据、移位、梯度与优化器。验证 token 范围、因果 mask 和标签位置，检查 loss 有限、梯度非空及参数更新，再调整超参数。

【🔥 必会】怎么检查未来 token 是否泄漏？ 结论：只改未来输入，比较更早位置输出。保持模型在 eval 模式和其他输入不变，早期 logits 应在数值容差内一致；同时检查 mask 方向。

【⭐ 理解】为什么训练下降不能证明真实能力？ 结论：训练集可能被记住或过于简单。需要独立、贴近目标分布的评测集、基线与失败分析；本例只验证训练链路。

### 故障模式与排障

- 输出看到了未来：固定种子并修改末尾 token，比较前面位置 logits；检查上三角 mask 和库的布尔语义。

- loss 不下降或非有限：检查输入/标签错位、学习率、梯度和优化器；先在小样本上过拟合验证链路。

- 库训练和手写 loss 对不上：核对预测位置、标签、模型权重及 dropout；GPT-2 接口会在内部移位标签，不能额外移位两次。

### 工程 trick 与坑

- 此例用普通 nn.Linear、nn.Embedding 和自动微分，手写注意力、LayerNorm、Block 与训练循环；没有实现 tokenizer、padding、缓存、混合精度和分布式训练。

- 贪心生成每一步重新计算完整前缀，且不允许超过位置容量；生产系统通常使用 KV cache 和成熟推理引擎。

- 固定玩具序列可以验证代码，但留出起点不等于真实独立语料；不要在简历上将此例描述为预训练大模型。

### Toy code 一：手写 Tiny Transformer 与训练循环

运行前提：Python 3.10+、PyTorch；CPU 可运行，必要时设置 OMP_NUM_THREADS=1。虚构模运算序列只用于自检，不需模型下载。

```python
"""从零拼接单层 decoder 并训练虚构序列；不用于训练基座模型。"""
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """在最后一维做带可学习缩放和平移的层归一化。"""
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        if dim <= 0 or eps <= 0:
            raise ValueError('维度和 eps 必须为正')
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """输入输出均为 (..., C)。"""
        if x.ndim < 1 or x.shape[-1] != self.weight.numel():
            raise ValueError('LayerNorm 最后一维不匹配')
        mean = x.mean(dim=-1, keepdim=True)
        variance = (x - mean).square().mean(dim=-1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class Attention(nn.Module):
    """因果多头自注意力，输入输出均为 (B,T,C)。"""
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels <= 0 or heads <= 0 or channels % heads:
            raise ValueError('通道数必须能被头数整除')
        self.channels, self.heads, self.head_dim = channels, heads, channels // heads
        self.qkv = nn.Linear(channels, 3 * channels)
        self.out = nn.Linear(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        """拆分头、遮住未来位置、拼接后投影。"""
        if x.ndim != 3 or x.shape[-1] != self.channels or x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError('输入必须为非空 (B,T,C)')
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def heads(t: Tensor) -> Tensor:
            return t.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = heads(q), heads(k), heads(v)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        blocked = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        weights = torch.softmax(scores.masked_fill(blocked, float('-inf')), dim=-1)
        result = (weights @ v).transpose(1, 2).contiguous().reshape(batch, length, channels)
        assert result.shape == x.shape
        return self.out(result)


class Block(nn.Module):
    """Pre-norm 残差块；FFN 逐位置处理特征。"""
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.norm1, self.norm2 = LayerNorm(channels), LayerNorm(channels)
        self.attn = Attention(channels, heads)
        self.ff1, self.ff2 = nn.Linear(channels, 4 * channels), nn.Linear(4 * channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        """先 attention 残差，再 FFN 残差。"""
        x = x + self.attn(self.norm1(x))
        return x + self.ff2(F.gelu(self.ff1(self.norm2(x))))


class TinyLM(nn.Module):
    """随机初始化的微型 decoder，仅供学习和数值自检。"""
    def __init__(self, vocab: int, context: int, channels: int, heads: int) -> None:
        super().__init__()
        if vocab < 2 or context < 2:
            raise ValueError('词表和上下文长度至少为 2')
        self.vocab, self.context = vocab, context
        self.token = nn.Embedding(vocab, channels)
        self.position = nn.Embedding(context, channels)
        self.block = Block(channels, heads)
        self.norm = LayerNorm(channels)
        self.head = nn.Linear(channels, vocab, bias=False)

    def forward(self, ids: Tensor) -> Tensor:
        """接收 (B,T) long token，返回 (B,T,V) logits。"""
        if ids.ndim != 2 or ids.dtype != torch.long or ids.shape[0] == 0 or not 1 <= ids.shape[1] <= self.context:
            raise ValueError('token 批次或长度非法')
        if int(ids.min()) < 0 or int(ids.max()) >= self.vocab:
            raise ValueError('token 超出词表')
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.token(ids) + self.position(positions)[None, :, :]
        logits = self.head(self.norm(self.block(x)))
        assert logits.shape == (*ids.shape, self.vocab)
        return logits


@torch.no_grad()
def generate(model: TinyLM, ids: Tensor, new_tokens: int) -> Tensor:
    """贪心生成；超过位置容量时拒绝，不实现滑动上下文。"""
    if isinstance(new_tokens, bool) or not isinstance(new_tokens, int) or new_tokens < 0:
        raise ValueError('生成长度须为非负整数')
    if ids.ndim != 2 or ids.shape[1] + new_tokens > model.context:
        raise ValueError('生成后超过位置容量')
    model.eval()
    model(ids)  # 复用输入合法性检查，即便 new_tokens 为零
    result = ids
    for _ in range(new_tokens):
        next_id = model(result)[:, -1, :].argmax(dim=-1, keepdim=True)
        result = torch.cat((result, next_id), dim=1)
    return result


def make_data(starts: range, steps: range, length: int, vocab: int) -> Tensor:
    """构造可复现的模运算序列；这里只是玩具规律。"""
    rows = [[(start + step * i) % vocab for i in range(length)]
            for start in starts for step in steps]
    if not rows:
        raise ValueError('训练数据不能为空')
    return torch.tensor(rows, dtype=torch.long)


def loss_for(model: TinyLM, sequences: Tensor) -> Tensor:
    """用前 T 个 token 预测后 T 个 token。"""
    if sequences.ndim != 2 or sequences.shape[1] < 2:
        raise ValueError('序列必须是二维且至少两个 token')
    logits = model(sequences[:, :-1])
    return F.cross_entropy(logits.reshape(-1, model.vocab), sequences[:, 1:].reshape(-1))


def main() -> None:
    """验证因果性、梯度、训练 loss 与留出起点的验证 loss。"""
    torch.manual_seed(19)
    model = TinyLM(vocab=20, context=8, channels=24, heads=4)
    train = make_data(range(16), range(1, 4), 9, 20)
    valid = make_data(range(16, 20), range(1, 4), 9, 20)
    model.eval()
    with torch.no_grad():
        altered = train[:2, :-1].clone(); altered[:, -1] = (altered[:, -1] + 1) % 20
        assert torch.allclose(model(train[:2, :-1])[:, :-1], model(altered)[:, :-1], atol=1e-6)
        initial = float(loss_for(model, train))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    model.train()
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(model, train)
        if not torch.isfinite(loss):
            raise RuntimeError('训练 loss 非有限')
        loss.backward()
        assert model.block.attn.qkv.weight.grad is not None
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final, val = float(loss_for(model, train)), float(loss_for(model, valid))
    assert final < initial and math.isfinite(val)
    generated = generate(model, train[:2, :3], 2)
    assert generated.shape == (2, 5) and torch.equal(generated[:, :3], train[:2, :3])
    try:
        model(torch.tensor([[20]], dtype=torch.long))
    except ValueError:
        pass
    else:
        raise AssertionError('越界 token 必须拒绝')
    print(f'虚构序列训练 loss: {initial:.4f} -> {final:.4f}; 留出起点验证 loss: {val:.4f}')
    print('该玩具规律不代表真实语言能力，也不证明实际业务泛化。')


if __name__ == '__main__':
    main()

```

逐段理解。 LayerNorm 按最后一维归一化；Attention 投影 Q/K/V、拆头并施加因果 mask；Block 用 pre-norm 残差组合注意力和 FFN；TinyLM 加入可学习位置和词表投影；训练循环显式清梯度、前向、反向和更新，再验证因果性、loss 与生成长度。

这段代码怎么在面试里讲： ① ID 是 [B,T]，隐藏态是 [B,T,C]，每头是 [B,H,T,C/H]，logits 是 [B,T,V]；② 解释 mask、残差、标签错位与优化器；③ 说明无 padding、无 KV cache、上下文上限和玩具验证集的局限。

排障： loss 不降检查输入目标移位与梯度；未来泄漏检查 mask 和改变未来 token 的测试；越界错误检查词表、上下文与位置 embedding。

### Toy code 二：Transformers 的随机模型训练对照

运行前提：Python 3.11、PyTorch、transformers==4.57.1；CPU 可运行。随机 GPT-2 和虚构序列不需要下载权重或网络。

```python
"""用 Transformers 训练随机 GPT-2，验证库中的因果语言模型训练路径。"""
import math
import torch
from torch import Tensor
from transformers import GPT2Config, GPT2LMHeadModel


def make_data(starts: range, steps: range, length: int, vocab: int) -> Tensor:
    """构造虚构的模运算序列；不代表真实文本语料。"""
    rows = [[(start + step * i) % vocab for i in range(length)]
            for start in starts for step in steps]
    if not rows:
        raise ValueError('样本集不能为空')
    return torch.tensor(rows, dtype=torch.long)


def model_loss(model: GPT2LMHeadModel, ids: Tensor) -> Tensor:
    """GPT-2 在模型内部对 labels 执行因果移位。"""
    if ids.ndim != 2 or ids.dtype != torch.long or ids.shape[0] == 0 or not 2 <= ids.shape[1] <= model.config.n_positions:
        raise ValueError('token 批次或长度非法')
    if int(ids.min()) < 0 or int(ids.max()) >= model.config.vocab_size:
        raise ValueError('token 超出词表')
    result = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
    if result.loss is None or not torch.isfinite(result.loss):
        raise RuntimeError('loss 缺失或非有限')
    return result.loss


def main() -> None:
    """在同一玩具规律上训练并报告留出起点的 loss。"""
    torch.manual_seed(19)
    cfg = GPT2Config(vocab_size=20, n_positions=9, n_ctx=9, n_embd=24, n_layer=1, n_head=4,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    model = GPT2LMHeadModel(cfg)
    train = make_data(range(16), range(1, 4), 9, 20)
    valid = make_data(range(16, 20), range(1, 4), 9, 20)
    model.eval()
    with torch.no_grad():
        initial = float(model_loss(model, train))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    model.train()
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = model_loss(model, train)
        loss.backward()
        assert model.transformer.h[0].attn.c_attn.weight.grad is not None
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final, val = float(model_loss(model, train)), float(model_loss(model, valid))
    assert final < initial and math.isfinite(val)
    try:
        model_loss(model, torch.tensor([[20, 1]]))
    except ValueError:
        pass
    else:
        raise AssertionError('越界 token 必须拒绝')
    print(f'Transformers 随机模型训练 loss: {initial:.4f} -> {final:.4f}; 验证: {val:.4f}')
    print('玩具规律与随机模型不代表真实语言能力或任务泛化。')


if __name__ == '__main__':
    main()

```

逐段理解。 用配置创建随机模型，由 Transformers 实现注意力和 block；传入完整序列及同样的 labels，GPT-2 在内部移位；训练循环和留出数据评估仍由调用方管理。

这段代码怎么在面试里讲： ① 输入是无 padding 的 [B,T] token；② 库内部完成因果模型前向和标签移位，外层仍负责 loss、反传和更新；③ 不要再次移位，随机小模型和玩具数据不能证明实际任务性能。

排障： loss 无效先检查 token 范围和序列长度；模型参数无梯度检查 requires_grad 与反向传播；结果不可比时核对模型初始化、模板和标签位置。

手写与调库怎么选。 手写用于练习数据流和诊断；真实应用使用 Transformers 等成熟库加载经许可的模型、tokenizer 和训练工具，并在独立数据上评测。二者的模型结构和初始化可能不同，loss 数值不能直接当成优劣结论。

### 本节收尾：费曼自述与闪卡

费曼自述： Tiny decoder 将 token 与位置变成隐藏态，经因果注意力和逐位置 FFN 输出词表分数；训练预测下一 token，生成时逐步追加结果。

- 问：C 与头数有什么约束？答：此实现要求 C 能被头数整除。

- 问：如何验证因果 mask？答：只改未来 token，前面位置 logits 应保持不变。

- 问：GPT-2 的 labels 需要手工再右移吗？答：此接口内部移位，不应重复移位。

- 问：训练 loss 下降说明模型具备真实语言能力吗？答：不能，玩具数据只验证训练流程。

- 问：生成达到上下文上限怎么办？答：本例明确拒绝；真实系统按模型和缓存策略处理。

### 原文核对 / 来源

[手撕 Tiny Transformer](原仓库Markdown/docs/llm/transformer/tiny_transformer_code.md)；[GPT-2 接口与 labels](https://huggingface.co/docs/transformers/model_doc/gpt2)。

## 本章完成标准

逐组件重写，不复制原文；每个组件至少测形状、因果 Mask 和梯度；最后用小数据集训练 Tiny GPT，记录训练与验证 loss。

能闭卷解释本章 A/B 内容，并拿出代码、测试或实验记录后，才把本章标成“已学会”。

