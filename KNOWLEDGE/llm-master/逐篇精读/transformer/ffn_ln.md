# 残差连接、LayerNorm、FFN：Transformer里缺一不可的配角组件

> 重要程度：**🔥 必会** · [原仓库同版本文章](https://github.com/youngyangyang04/llm-master/blob/78b03ad475409ba1d6326e0a6e8d50b27025bb64/docs/llm/transformer/ffn_ln.md) · 原作者：程序员Carl（MIT License，见 [LICENSE](../LICENSE)）。
> 整理方法：保留技术解释、示例、代码和面试问答；压缩开篇叙事和推广语。下方“学习补充”由本笔记新增。

## 学习补充：先理解，再闭卷复述

### 8.6 FFN、残差与 LayerNorm【🔥】

**一句话：** 注意力在位置间交换信息；FFN 对每个位置的通道做非线性变换；残差保留输入通路；LayerNorm 稳定特征尺度。

~~~text
一种 Pre-Norm Block 示意：
X1 = X + Attention(LayerNorm(X))
X2 = X1 + FFN(LayerNorm(X1))
~~~

**例子：** X=(2,4,8)，注意力和 FFN 的最终输出也要是 (2,4,8)，才能与残差输入相加。FFN 中间可以临时扩到更大通道，再投影回 8。Pre-Norm 与 Post-Norm 把归一化放在不同位置，读实际代码时以其实现为准。

**费曼自述：** “注意力让 token 互相交流，FFN 在每个 token 内进一步加工；残差给原信息留路，归一化让训练更稳定。”

**概念闪卡：** Q：FFN 是否负责跨 token 通信？A：典型位置前馈层逐位置作用，跨位置交流主要来自注意力。Q：残差相加的形状要求？A：相同或可按设计兼容。

## 原文技术内容（整理版）

### 两者组合

在 Transformer 里，残差和 LayerNorm 是**捆绑使用**的，标准写法叫 **Add & Norm**：

$$
\text{输出} = \text{LayerNorm}(x + F(x))
$$

先残差相加，再 LayerNorm 归一化，每个子模块（Self-Attention 和 FFN）后面都跟一个这样的结构。

![Transformer前馈网络FFN结构示意图](https://file1.kamacoder.com/i/algo/a7c59854-ed2d-4e0b-8c20-5fb9c8134955.webp?v=20261005)


### FFN：Self-Attention 的"搭档"，不是装饰

Transformer 每一层的结构是：

```
Self-Attention → Add & Norm → FFN → Add & Norm
```

Self-Attention 负责让 Token 之间"互通信息"，FFN 负责什么？

很多初学者觉得 FFN 就是两层普通的线性变换，可有可无。但它其实在做一件 Self-Attention 做不了的事：**对每个 Token 的向量做非线性变换，提取更深层的特征。**

FFN 的结构很简单：

$$
\text{FFN}(x) = \text{ReLU}(x \cdot W_1 + b_1) \cdot W_2 + b_2
$$

两层线性变换，中间夹一个激活函数（ReLU 或 GELU）。

关键细节：**FFN 会先升维，再降维。**

以 d_model = 512 为例：

```
输入 x（512维）
    ↓ × W₁（512→2048）
中间层（2048维）← 升维到 4 倍
    ↓ 激活函数
    ↓ × W₂（2048→512）
输出（512维）
```


### 为什么要先升维再降维？

这里藏着 FFN 最重要的设计思路。

Self-Attention 做的是**线性组合**——把不同 Token 的 Value 向量加权求和，本质上是线性变换。**线性变换叠加再多层，表达能力还是有限的。**

FFN 中间插入了激活函数，带来了**非线性**。先升维，是为了在更高维的空间里让激活函数"发挥作用"——高维空间里，非线性能捕捉到更复杂的特征组合。再降维，把提取到的信息压缩回原来的维度，传给下一层。


还有一点值得注意：**FFN 是对每个 Token 独立计算的**，不同 Token 之间不共享中间结果。

这和 Self-Attention 正好相反——Self-Attention 让所有 Token 互相"看"，FFN 让每个 Token 自己"想"。

| | Self-Attention | FFN |
|---|---|---|
| **处理方式** | Token 之间互相交互 | 每个 Token 独立处理 |
| **变换类型** | 线性（加权求和） | 非线性（含激活函数） |
| **作用** | 聚合上下文信息 | 深化单个 Token 的表示 |

两者一个负责"横向"（Token 间）的信息交流，一个负责"纵向"（特征维度）的深度加工，缺一不可。


这三者各有分工，但都不可少：

- **残差连接**：给梯度留一条高速公路，让深层网络能够正常训练
- **LayerNorm**：每层计算完之后做数值体检，防止数值爆炸影响下一层
- **FFN**：补上 Self-Attention 缺少的非线性能力，让每个 Token 的表示更丰富

把这三个加上 Self-Attention，一个完整的 Transformer Encoder 层就搭好了。

---

