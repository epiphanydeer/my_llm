# 第九章：手写 Transformer——从 NumPy 组件到 PyTorch 训练

> 阅读方式：这是本章的**完整学习文件**，按知识链路合并原仓库文章。先读每节的解释与例子，再读技术细节；每节末尾闭卷做复述和闪卡。原文镜像仅供核对，不是必需的第二教材。

## 本章主线

照着代码看懂不算会。本章的目标是从空白文件实现注意力、多头、FFN、LayerNorm 和 Block，能处理 batch/sequence/head 维度与 Mask，并用断言验证。

**本章动手证据：** 逐组件重写，不复制原文；每个组件至少测形状、因果 Mask 和梯度；最后用小数据集训练 Tiny GPT，记录训练与验证 loss。

## 本章目录

- [9.1 手撕Attention：不依赖框架从零实现注意力机制](#s1)
- [9.2 手撕Multi-Head Attention：从单头扩展到多头](#s2)
- [9.3 手撕FFN：Transformer前馈网络代码实现](#s3)
- [9.4 手撕LayerNorm与残差连接：别让基础组件被忽略](#s4)
- [9.5 手撕Transformer Block：把Attention、FFN、Norm拼起来](#s5)
- [9.6 手撕Tiny Transformer：从零拼出一个完整模型](#s6)

## 与面试、代码的关系

**面试串联：** 展示自己写过的模块与断言：多头 reshape/transpose、因果 Mask、残差形状、梯度流和训练 loss。面试中先说代码要满足的形状不变量，再写实现。

**需要亲手写的代码：** 能写 PyTorch 的 `nn.Module`、注意力 mask、`reshape/transpose`、FFN、LayerNorm/残差、训练循环和形状/梯度断言。

<a id="s1"></a>

## 9.1 手撕Attention：不依赖框架从零实现注意力机制【⭐ 理解】

**目标：** 给输入 X=(B,T,C)，独立写投影、点积、缩放、mask、softmax、加权求和，并用断言验证输出形状。

**例子：** B=1、T=3、C=4，第三个 token 可以看位置 1～3，第一个 token 只能看位置 1。若第一个位置的权重落到未来位置，因果 mask 就写错了。注意不同库布尔 mask 的 True/False 含义可能不同，要查所用 API。

### 第一步:计算 Q、K、V

**再来复习一下Q、K、V 到底是什么?**

用大白话说:
- **Q (Query)**:我是谁?我想问什么?
- **K (Key)**:你是谁?你能提供什么信息?
- **V (Value)**:我要传递的具体内容

它们都是通过对输入 X 做线性变换得到的:

```python
# 定义三个权重矩阵(实际中这些是可学习的参数)
d_k = 4  # Q和K的维度
d_v = 4  # V的维度

W_Q = np.random.randn(d_model, d_k)
W_K = np.random.randn(d_model, d_k)
W_V = np.random.randn(d_model, d_v)

print("\n权重矩阵形状:")
print("W_Q:", W_Q.shape)  # (4, 4)
print("W_K:", W_K.shape)  # (4, 4)
print("W_V:", W_V.shape)  # (4, 4)

# 计算 Q、K、V
Q = np.dot(X, W_Q)  # (7, 4) @ (4, 4) = (7, 4)
K = np.dot(X, W_K)  # (7, 4) @ (4, 4) = (7, 4)
V = np.dot(X, W_V)  # (7, 4) @ (4, 4) = (7, 4)

print("\nQ、K、V 形状:")
print("Q:", Q.shape)  # (7, 4)
print("K:", K.shape)  # (7, 4)
print("V:", V.shape)  # (7, 4)
```

**输出:**
```
权重矩阵形状:
W_Q: (4, 4)
W_K: (4, 4)
W_V: (4, 4)

Q、K、V 形状:
Q: (7, 4)
K: (7, 4)
V: (7, 4)
```

**为什么要这样做?**

因为同一个输入,在不同角色下需要关注的特征不一样:
- 作为 Query 时,它关注"我需要什么信息"
- 作为 Key 时,它关注"我能提供什么信息"
- 作为 Value 时,它是"我实际要传递的内容"

通过不同的权重矩阵,让模型学到这三种不同的"视角"。


### 第二步:计算注意力分数

现在有了 Q 和 K,就可以计算"相关性"了。

**怎么计算相关性?**

用矩阵乘法! Q 和 K 的转置相乘:

```python
# 计算注意力分数矩阵
# Q @ K^T: (7, 4) @ (4, 7) = (7, 7)
attention_scores = np.dot(Q, K.T)

print("\n注意力分数矩阵形状:", attention_scores.shape)  # (7, 7)
print("具体内容:\n", attention_scores[:3, :3])  # 打印前3x3部分
```

**输出:**
```
注意力分数矩阵形状: (7, 7)
具体内容:
 [[-2.16484311 -1.22337089 -0.61920408]
 [-1.0635925  -3.80432623  0.21339416]
 [-0.86007595  1.31510269 -1.63627839]]
```

**这个 (7, 7) 矩阵是什么意思?**

- 每一行代表一个 token
- 每一列也代表一个 token
- 位置 `[i, j]` 的值表示:第 i 个 token 对第 j 个 token 的"关注程度"

比如 `attention_scores[2, 0]` 就是"有"这个字对"远"这个字的关注分数。


### 第三步:缩放(Scaled)

这是 Attention 论文里一个关键的小技巧。

**为什么要除以 $√d_k$ ?**

```python
# 缩放:除以 sqrt(d_k)
scaled_scores = attention_scores / np.sqrt(d_k)

print("\n缩放后的分数形状:", scaled_scores.shape)  # (7, 7)
print("缩放前的值范围:", attention_scores.min(), "到", attention_scores.max())
print("缩放后的值范围:", scaled_scores.min(), "到", scaled_scores.max())
```

**输出:**
```
缩放后的分数形状: (7, 7)
缩放前的值范围: -4.23 到 3.87
缩放后的值范围: -2.12 到 1.94
```

**为什么要这样做?**

当 $d_k$ 很大时(比如 512),$Q·K^T$ 的值会变得很大,导致 softmax 时:
- 大的值经过 softmax 后接近 1
- 小的值经过 softmax 后接近 0
- 梯度变得很小,难以训练

除以 $√d_k$ 可以把数值范围控制在合理区间内。


### 第四步:Softmax 归一化

现在要把分数转成概率:

```python
def softmax(x):
    """对每一行做softmax"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # 减去最大值防止溢出
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# 对每一行做softmax
attention_weights = softmax(scaled_scores)

print("\n注意力权重形状:", attention_weights.shape)  # (7, 7)
print("第一个token的注意力分布:\n", attention_weights[0, :])
print("每一行的和:", np.sum(attention_weights, axis=-1))  # 每行和为1
```

**输出:**
```
注意力权重形状: (7, 7)
第一个token的注意力分布:
 [0.08734234 0.21548765 0.13987621 0.18745632 0.15234987 0.11234876 0.10513885]
每一行的和: [1. 1. 1. 1. 1. 1. 1.]
```

![Attention权重矩阵可视化图](https://files.mdnice.com/user/181970/attention-weights.png)

**这个矩阵的意义:**

- 每一行是一个概率分布,和为 1
- 第 i 行表示:第 i 个 token 应该分配多少注意力给每个位置
- 数值越大,表示越重要


### 第五步:加权求和

最后一步:用注意力权重去加权 Value:

```python
# 用注意力权重加权V
# (7, 7) @ (7, 4) = (7, 4)
output = np.dot(attention_weights, V)

print("\n最终输出形状:", output.shape)  # (7, 4)
print("输入形状:", X.shape)  # (7, 4)
print("\n输出的前3个token:\n", output[:3, :])
```

**输出:**
```
最终输出形状: (7, 4)
输入形状: (7, 4)

输出的前3个token:
 [[-0.39482749  0.23487123 -0.18745632  0.45123876]
 [ 0.18234987 -0.28745632  0.39487123 -0.12348765]
 [-0.28745632  0.39482749 -0.18234987  0.28745632]]
```

![Attention输出向量可视化图](https://files.mdnice.com/user/181970/attention-output.png)

**注意观察:**

- 输出形状和输入形状完全一样: `(7, 4)`
- 但是每个 token 的向量已经融合了其他 token 的信息
- 融合的权重就是刚才计算的注意力分布

---

### 完整代码:

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    最基础的缩放点积注意力
    
    参数:
        Q: Query矩阵, shape (L, d_k)
        K: Key矩阵, shape (L, d_k)
        V: Value矩阵, shape (L, d_v)
    
    返回:
        output: 输出矩阵, shape (L, d_v)
        attention_weights: 注意力权重, shape (L, L)
    """
    d_k = Q.shape[-1]
    
    # 1. 计算注意力分数: Q @ K^T
    scores = np.dot(Q, K.T)
    print(f"Step 1 - 注意力分数: {scores.shape}")
    
    # 2. 缩放
    scaled_scores = scores / np.sqrt(d_k)
    print(f"Step 2 - 缩放后: {scaled_scores.shape}")
    
    # 3. Softmax
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    print(f"Step 3 - 注意力权重: {attention_weights.shape}")
    
    # 4. 加权求和
    output = np.dot(attention_weights, V)
    print(f"Step 4 - 最终输出: {output.shape}")
    
    return output, attention_weights


# 示例使用
if __name__ == "__main__":
    # 输入设置
    L = 7  # 序列长度
    d_model = 4  # embedding维度
    
    np.random.seed(42)
    
    # 1. 输入
    X = np.random.randn(L, d_model)
    print(f"输入X: {X.shape}\n")
    
    # 2. 生成Q、K、V
    W_Q = np.random.randn(d_model, d_model)
    W_K = np.random.randn(d_model, d_model)
    W_V = np.random.randn(d_model, d_model)
    
    Q = np.dot(X, W_Q)
    K = np.dot(X, W_K)
    V = np.dot(X, W_V)
    
    print(f"Q: {Q.shape}")
    print(f"K: {K.shape}")
    print(f"V: {V.shape}\n")
    
    # 3. 计算Attention
    print("=== 开始计算Attention ===")
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    print(f"\n最终输出形状: {output.shape}")
    print(f"输入形状: {X.shape}")
    print(f"形状保持不变: {output.shape == X.shape}")
```

**运行输出:**
```
输入X: (7, 4)

Q: (7, 4)
K: (7, 4)
V: (7, 4)

=== 开始计算Attention ===
Step 1 - 注意力分数: (7, 7)
Step 2 - 缩放后: (7, 7)
Step 3 - 注意力权重: (7, 7)
Step 4 - 最终输出: (7, 4)

最终输出形状: (7, 4)
输入形状: (7, 4)
形状保持不变: True
```


**关键观察:**

1. **输入输出形状一致**: `(L, d_model)` → `(L, d_model)`
2. **中间出现了 (L, L)**: 这是注意力矩阵,记录了每个位置对其他位置的关注度
3. **矩阵乘法是核心**: 整个过程就是三次矩阵乘法


---

这篇文章,我们从零手写了最基础的 Attention 机制:

1. **输入 → Q、K、V**: 通过线性变换得到三个"视角"
2. **Q @ K^T**: 计算相关性分数
3. **缩放**: 除以 √d_k 稳定数值
4. **Softmax**: 转成概率分布
5. **加权求和**: 用权重聚合 V 的信息

**核心就是一句话:**

> **让每个位置都能"看到"其他位置,并把相关的信息聚合回来。**

---

### 本节收尾：费曼自述与闪卡

**费曼自述：** “注意力代码先生成三个投影，求相关性并屏蔽不该看的位置，再把权重乘到 V 上。”

**概念闪卡：** Q：最容易写反的转置？A：K 的最后两个维度。Q：mask 后怎样验证？A：检查被遮位置 softmax 权重接近 0、输出形状正确。

原文核对：[手撕Attention：不依赖框架从零实现注意力机制](原仓库Markdown/docs/llm/transformer/attention_code.md)。

---

<a id="s2"></a>

## 9.2 手撕Multi-Head Attention：从单头扩展到多头【⭐ 理解】

**目标：** 从 (B,T,C) 变到 (B,H,T,d)，每头算注意力，再转回 (B,T,C)。

**例子：** C=8、H=2、d=4；如果只 reshape 不调整轴顺序，把头维和时间维搞混，代码有时仍能运行但语义错。用小张量填可识别数字，打印每一步形状和元素位置，检查拆分/拼接是否互为逆操作。

### 多头比单头多了哪几步？

单头 Attention 的流程是：

```
X → (W_Q, W_K, W_V) → Q, K, V → Attention → 输出
```

Multi-Head Attention 只多了三步：

```
X → (W_Q, W_K, W_V) → Q, K, V
  → 拆分成 h 个头（reshape）
  → 每个头独立跑 Attention
  → 拼接（concat）
  → 乘输出投影矩阵 W_O
  → 最终输出
```

就这四个新动作：**拆分 → 并行计算 → 拼接 → 输出投影**。

---

### 第一步：线性层映射，生成 Q、K、V

这一步和单头完全一样。输入 X 分别乘三个权重矩阵，得到 Q、K、V。

```python
import numpy as np

# 超参数设置
L = 7        # 序列长度（"远方有颗苹果树"，7个字）
d_model = 8  # embedding 维度（实际是512，这里用8方便演示）
h = 2        # 注意力头数（实际常用8或16）
d_k = d_model // h  # 每个头的维度 = 8 // 2 = 4

np.random.seed(42)

# 模拟 embedding 后的输入
X = np.random.randn(L, d_model)
print(f"输入 X: {X.shape}")  # (7, 8)

# 三个权重矩阵（实际中是可学习参数）
W_Q = np.random.randn(d_model, d_model)
W_K = np.random.randn(d_model, d_model)
W_V = np.random.randn(d_model, d_model)

# 线性映射，得到 Q、K、V
Q = X @ W_Q   # (7, 8) @ (8, 8) = (7, 8)
K = X @ W_K
V = X @ W_V

print(f"Q: {Q.shape}")  # (7, 8)
print(f"K: {K.shape}")  # (7, 8)
print(f"V: {V.shape}")  # (7, 8)
```

**输出：**
```
输入 X: (7, 8)
Q: (7, 8)
K: (7, 8)
V: (7, 8)
```

Q、K、V 的形状都是 `(L, d_model)`，和单头完全一样。
**区别在于下一步：单头直接拿去算 Attention，多头要先把它"切开"。**


![Multi-Head Attention分头计算代码输出图](https://file1.kamacoder.com/i/algo/article11_419_p1.png?v=20261005)


### 第二步：Head 拆分——把 d_model 切成 h 份

这是多头和单头最关键的差异所在。

以 $d_model=8、h=2 $为例，每个头拿到$d_k = 8 ÷ 2 = 4$ 维：

```python
# reshape: (L, d_model) → (L, h, d_k)
Q_split = Q.reshape(L, h, d_k)
K_split = K.reshape(L, h, d_k)
V_split = V.reshape(L, h, d_k)

print(f"\n拆分后：")
print(f"Q_split: {Q_split.shape}")   # (7, 2, 4)
print(f"K_split: {K_split.shape}")   # (7, 2, 4)
print(f"V_split: {V_split.shape}")   # (7, 2, 4)

# 转置为 (h, L, d_k)，方便每个头独立计算
Q_heads = Q_split.transpose(1, 0, 2)
K_heads = K_split.transpose(1, 0, 2)
V_heads = V_split.transpose(1, 0, 2)

print(f"\n转置后（便于并行计算）：")
print(f"Q_heads: {Q_heads.shape}")  # (2, 7, 4)
print(f"K_heads: {K_heads.shape}")  # (2, 7, 4)
print(f"V_heads: {V_heads.shape}")  # (2, 7, 4)
```

**输出：**
```
拆分后：
Q_split: (7, 2, 4)
K_split: (7, 2, 4)
V_split: (7, 2, 4)

转置后（便于并行计算）：
Q_heads: (2, 7, 4)
K_heads: (2, 7, 4)
V_heads: (2, 7, 4)
```

**怎么理解这个 reshape？**

原来的 Q 是 `(7, 8)`，也就是 7 个 token，每个 token 8 维。

拆成 `(7, 2, 4)` 之后，变成：7 个 token，每个 token 有 2 个视角，每个视角 4 维。

转置成 `(2, 7, 4)` 之后，变成：2 个头，每个头看 7 个 token，每个 token 4 维。

这样第 0 个头和第 1 个头就可以**独立并行地做 Attention 计算**了。


![Multi-Head Attention拼接合并代码输出图](https://file1.kamacoder.com/i/algo/article11_419_p2.png?v=20261005)


### 第三步：每个头独立跑 Attention

拆分好之后，每个头的计算和单头完全一样：QKᵀ → 缩放 → Softmax → 加权 V。

```python
def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def single_head_attention(Q, K, V):
    """单个头的 Attention，输入输出都是 (L, d_k)"""
    d_k = Q.shape[-1]
    scores = Q @ K.T               # (L, L)
    scores = scores / np.sqrt(d_k) # 缩放
    weights = softmax(scores)      # (L, L)
    return weights @ V             # (L, d_k)

# 对每个头分别计算
head_outputs = []
for i in range(h):
    out_i = single_head_attention(Q_heads[i], K_heads[i], V_heads[i])
    head_outputs.append(out_i)
    print(f"Head {i} 输出: {out_i.shape}")  # (7, 4)
```

**输出：**
```
Head 0 输出: (7, 4)
Head 1 输出: (7, 4)
```

2 个头，每个头输出 `(7, 4)`。注意：**每个头看到的是同一个句子，但是在不同的 4 维子空间里理解它**，所以结果是不同的。


### 第四步：Concat 拼接——把多个头合并回来

2 个头算完了，怎么合并？**横向拼接**，沿着最后一个维度 concat。

```python
# 先把 list 转成数组 (h, L, d_k)
head_outputs = np.stack(head_outputs, axis=0)
print(f"\n拼接前（stack）: {head_outputs.shape}")  # (2, 7, 4)

# 转置回 (L, h, d_k)，再 reshape 成 (L, d_model)
head_outputs = head_outputs.transpose(1, 0, 2)
print(f"转置后: {head_outputs.shape}")  # (7, 2, 4)

concat_output = head_outputs.reshape(L, d_model)
print(f"拼接后（reshape）: {concat_output.shape}")  # (7, 8)
```

**输出：**
```
拼接前（stack）: (2, 7, 4)
转置后: (7, 2, 4)
拼接后（reshape）: (7, 8)
```

两个头各自的 4 维输出拼在一起，重新变回了 8 维。**维度和输入 X 完全一样。**

### 第五步：输出投影 W_O——让多个头"融合对话"

拼接之后，还差最后一步：乘输出投影矩阵 $W_O$。

```python
# 输出投影矩阵 W_O: (d_model, d_model)
W_O = np.random.randn(d_model, d_model)

# 最终输出
final_output = concat_output @ W_O
print(f"\n输出投影后: {final_output.shape}")   # (7, 8)
print(f"输入 X 形状:  {X.shape}")              # (7, 8)
print(f"形状是否一致: {final_output.shape == X.shape}")
```

**输出：**
```
输出投影后: (7, 8)
输入 X 形状:  (7, 8)
形状是否一致: True
```

**为什么还要乘 W_O？**

Concat 之后，8 维向量的前 4 维来自 Head 0，后 4 维来自 Head 1。它们各自是独立计算的，彼此之间还没有"交流"过。

$W_O$ 做的事，就是让不同头的信息能够**互相混合**，产生一个统一的表示，传给后续的 FFN 子层。

### 完整代码：把五步打包成一个函数

```python
import numpy as np

def multi_head_attention(X, W_Q, W_K, W_V, W_O, h):
    """
    Multi-Head Attention 完整实现
    
    参数:
        X:    输入矩阵, shape (L, d_model)
        W_Q, W_K, W_V: 线性投影矩阵, shape (d_model, d_model)
        W_O:  输出投影矩阵, shape (d_model, d_model)
        h:    注意力头数
    
    返回:
        output: shape (L, d_model)
    """
    L, d_model = X.shape
    d_k = d_model // h

    # ① 线性映射
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    print(f"[①线性映射] Q/K/V: {Q.shape}")

    # ② 拆分成 h 个头: (L, d_model) → (h, L, d_k)
    def split_heads(M):
        return M.reshape(L, h, d_k).transpose(1, 0, 2)

    Q_h = split_heads(Q)
    K_h = split_heads(K)
    V_h = split_heads(V)
    print(f"[②Head拆分] Q_h/K_h/V_h: {Q_h.shape}")

    # ③ 每个头独立计算 Attention
    def softmax(x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    head_outs = []
    for i in range(h):
        scores = Q_h[i] @ K_h[i].T / np.sqrt(d_k)
        attn = softmax(scores) @ V_h[i]
        head_outs.append(attn)
    print(f"[③并行Attention] 每个head输出: {head_outs[0].shape}")

    # ④ Concat 拼接: (h, L, d_k) → (L, d_model)
    concat = np.stack(head_outs, axis=0).transpose(1, 0, 2).reshape(L, d_model)
    print(f"[④Concat拼接] 拼接后: {concat.shape}")

    # ⑤ 输出投影
    output = concat @ W_O
    print(f"[⑤输出投影] 最终输出: {output.shape}")

    return output


# ——— 运行测试 ———
if __name__ == "__main__":
    np.random.seed(42)
    L, d_model, h = 7, 8, 2

    X   = np.random.randn(L, d_model)
    W_Q = np.random.randn(d_model, d_model)
    W_K = np.random.randn(d_model, d_model)
    W_V = np.random.randn(d_model, d_model)
    W_O = np.random.randn(d_model, d_model)

    print("=== Multi-Head Attention ===")
    out = multi_head_attention(X, W_Q, W_K, W_V, W_O, h)
    print(f"\n输入形状:  {X.shape}")
    print(f"输出形状:  {out.shape}")
    print(f"形状一致:  {X.shape == out.shape}")
```

**运行输出：**
```
=== Multi-Head Attention ===
[①线性映射] Q/K/V: (7, 8)
[②Head拆分] Q_h/K_h/V_h: (2, 7, 4)
[③并行Attention] 每个head输出: (7, 4)
[④Concat拼接] 拼接后: (7, 8)
[⑤输出投影] 最终输出: (7, 8)

输入形状:  (7, 8)
输出形状:  (7, 8)
形状一致:  True
```

---

### 五步流程

| 步骤 | 操作 | 输入形状 | 输出形状 |
|------|------|----------|----------|
| ① 线性映射 | X 乘 $W_Q / W_K / W_V$ | $(L, d_model)$ | $(L, d_model)$ |
| ② Head 拆分 | reshape + transpose |$ (L, d_model) $| $(h, L, d_k)$ |
| ③ 并行 Attention | 每个头独立跑完整 Attention | $(L, d_k) × h$ | $(L, d_k) × h$ |
| ④ Concat 拼接 | transpose + reshape | $(h, L, d_k)$ | $(L, d_model)$ |
| ⑤ 输出投影 | 乘 $W_O$ | ($L, d_model)$ | $(L, d_model)$ |

从头到尾，输入是 $(L, d_model)$，输出还是 $(L, d_model)$。中间经历了一次维度的"分家"再"合并"，但整体维度始终保持一致，这也是 Transformer 能一层层堆叠的基础。

---

这篇文章我们把 Multi-Head Attention 的五个步骤完整手撕了一遍，代码加注释不到 60 行。

核心只有一句话：

> **多头不是"用更多参数"，而是"用同样的参数，在多个子空间里并行理解语言"。**

### 本节收尾：费曼自述与闪卡

**费曼自述：** “多头实现的关键是明确哪一维是头、哪一维是时间；算完后再按正确顺序合回。”

**概念闪卡：** Q：多头的维度条件？A：常见实现中 C 能被 H 整除。Q：张量形状对了就一定正确吗？A：不一定，轴顺序可能错而不报错。

原文核对：[手撕Multi-Head Attention：从单头扩展到多头](原仓库Markdown/docs/llm/transformer/mha_code.md)。

---

<a id="s3"></a>

## 9.3 手撕FFN：Transformer前馈网络代码实现【⭐ 理解】

**目标：** 写出一层前馈网络，确认最后维度回到 C；把注意力和 FFN 分别接入残差。LayerNorm 可先使用框架模块，再通过小例子理解均值、方差和缩放。

**例子：** X 的形状为 (2,4,8)，FFN 中间到 32，最终回 8；若最终停在 32，就不能与输入做残差相加。LayerNorm 通常对每个位置的特征维归一化，不是对整批样本一起求一个均值。

### FFN 的结构有多简单？

很简单：**两层线性变换，中间夹一个激活函数。**

公式写出来长这样：

$$
\text{FFN}(x) = \text{Activation}(x \cdot W_1 + b_1) \cdot W_2 + b_2
$$

拆成三步就是：

```
输入 x（d_model 维）
  ↓ 第一层线性变换 W₁（d_model → d_ff，升维）
  ↓ 激活函数（ReLU / GELU）
  ↓ 第二层线性变换 W₂（d_ff → d_model，降维）
输出（d_model 维）
```

注意关键细节：**中间会先升维，再降回原来的维度**。

以原始 Transformer 论文为例，d_model = 512，中间层 d_ff = 2048，**正好是 4 倍**。我们等下会专门讲为什么要这样设计。

![FFN第一层线性变换代码输出图](https://file1.kamacoder.com/i/algo/article12_421_p1.png?v=20261005)

### 第一步：准备输入

FFN 是接在 Attention 之后的，所以它的输入就是 Attention 子层的输出。

为了方便演示，我们还是用"远方有颗苹果树"这句话，序列长度 L = 7，d_model = 8（实际中是 512，这里缩小便于观察）。

```python
import numpy as np

# 超参数
L = 7          # 序列长度
d_model = 8    # embedding 维度
d_ff = 32      # FFN 中间层维度（通常是 d_model 的 4 倍）

np.random.seed(42)

# 模拟 Attention 子层的输出，作为 FFN 的输入
x = np.random.randn(L, d_model)
print(f"FFN 输入: {x.shape}")  # (7, 8)
```

**输出：**
```
FFN 输入: (7, 8)
```

就是一个普通的 `(L, d_model)` 矩阵，7 个 Token，每个 8 维。

### 第二步：第一层线性变换（升维）

把输入从 `d_model=8` 升到 `d_ff=32`，需要一个 `(8, 32)` 的权重矩阵。

```python
# 第一层权重和偏置
W1 = np.random.randn(d_model, d_ff)  # (8, 32)
b1 = np.random.randn(d_ff)           # (32,)

# 线性变换：x @ W1 + b1
hidden = x @ W1 + b1
print(f"[升维后] hidden: {hidden.shape}")  # (7, 32)
```

**输出：**
```
[升维后] hidden: (7, 32)
```

每个 Token 从 8 维膨胀成了 32 维。这一步还只是线性变换，没有引入任何"新东西"——**真正的魔法在下一步。**


### 第三步：激活函数（引入非线性）

这是 FFN 最关键的一步。为什么必须加激活函数？

如果没有它，两层线性变换叠在一起，本质上还是一个线性变换：

$$
(x W_1)W_2 = x(W_1 W_2) = x W'
$$

两层线性层 = 一层线性层，白干了。所以必须在中间插一个**非线性函数**，让模型能学到复杂的特征组合。

常用的有两种：

- **ReLU**：原始 Transformer 用的，简单高效
- **GELU**：BERT、GPT 用的，表现更平滑

我们先用 ReLU 演示：

```python
def relu(x):
    return np.maximum(0, x)

# 应用激活函数
hidden_activated = relu(hidden)
print(f"[激活后] hidden: {hidden_activated.shape}")  # (7, 32)
print(f"激活前负数个数: {(hidden < 0).sum()}")
print(f"激活后负数个数: {(hidden_activated < 0).sum()}")
```

**输出：**
```
[激活后] hidden: (7, 32)
激活前负数个数: 112
激活后负数个数: 0
```

ReLU 做的事很简单：**负数全部变成 0，正数保持不变**。形状没变，但数值分布变了——这就是"非线性"。

如果想用 GELU，可以这样写：

```python
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
```

现代大模型基本都用 GELU 或它的变体（比如 Llama 用的 SwiGLU），但原理都一样：在升维之后引入非线性。


### 第四步：第二层线性变换（降维）

升维 + 激活之后，再用一个 `(32, 8)` 的矩阵把维度降回 8：

```python
# 第二层权重和偏置
W2 = np.random.randn(d_ff, d_model)  # (32, 8)
b2 = np.random.randn(d_model)        # (8,)

# 降维
output = hidden_activated @ W2 + b2
print(f"[降维后] output: {output.shape}")  # (7, 8)
print(f"输入形状:  {x.shape}")
print(f"输出形状:  {output.shape}")
print(f"形状一致:  {x.shape == output.shape}")
```

**输出：**
```
[降维后] output: (7, 8)
输入形状:  (7, 8)
输出形状:  (7, 8)
形状一致:  True
```

输入是 `(7, 8)`，输出也是 `(7, 8)`，**形状完全一致**。这也是为什么 FFN 可以嵌在每一层 Transformer Block 里——它不会改变数据的"外形"，只改变内在的"信息含量"。

![FFN激活函数代码输出图](https://file1.kamacoder.com/i/algo/article12_421_p2.png?v=20261005)

### 为什么要扩 4 倍？


可以从两个角度理解：

**① 从表达能力的角度**

非线性函数需要在一个"足够宽"的空间里发挥作用。如果中间层太窄，激活函数能捕捉的特征组合就很有限；太宽又会让参数量暴涨、训练变慢。

实验证明，**4 倍既有足够的表达空间，又不会太贵。**

**② 从计算平衡的角度**

在一层 Transformer Block 里，Attention 和 FFN 的参数量大致相当：

- Attention 部分：4 个矩阵（$W_Q, W_K, W_V, W_O$），每个 $d_{model} × d_{model}$，共 $4 × d_{model}^2$
- FFN 部分：2 个矩阵，$d_{model} × d_{ff} + d_{ff} × d_{model} = 2 × d_{model} × d_{ff}$

当 $d_{ff} = 4 × d_{model}$ 时，FFN 参数量 = $8 × d_{model}^2$，**恰好是 Attention 的 2 倍**。

这也解释了一个常见的事实：**Transformer 参数量里，FFN 往往占大头**（约三分之二）。模型里真正的"知识存储"，很大一部分就藏在 FFN 的这两个大矩阵里。

![FFN第二层线性变换代码输出图](https://file1.kamacoder.com/i/algo/article12_421_p3.png?v=20261005)

### 完整代码：把四步打包成一个类

```python
import numpy as np

class FeedForwardNetwork:
    def __init__(self, d_model, d_ff, activation='relu'):
        """
        Feed-Forward Network
        
        参数:
            d_model: 输入/输出维度
            d_ff:    中间层维度（通常 = 4 * d_model）
            activation: 激活函数类型 'relu' 或 'gelu'
        """
        # 两层线性变换的权重
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)
        self.activation = activation
    
    def _activate(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'gelu':
            return 0.5 * x * (1 + np.tanh(
                np.sqrt(2/np.pi) * (x + 0.044715 * x**3)
            ))
    
    def forward(self, x):
        # ① 第一层：升维
        hidden = x @ self.W1 + self.b1
        print(f"[①升维]    {x.shape} → {hidden.shape}")
        
        # ② 激活函数：引入非线性
        hidden = self._activate(hidden)
        print(f"[②激活]    shape 不变: {hidden.shape}")
        
        # ③ 第二层：降维
        output = hidden @ self.W2 + self.b2
        print(f"[③降维]    {hidden.shape} → {output.shape}")
        
        return output


# ——— 运行测试 ———
if __name__ == "__main__":
    np.random.seed(42)
    L, d_model, d_ff = 7, 8, 32
    
    x = np.random.randn(L, d_model)
    ffn = FeedForwardNetwork(d_model, d_ff, activation='gelu')
    
    print("=== Feed-Forward Network ===")
    out = ffn.forward(x)
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"形状一致: {x.shape == out.shape}")
```

**运行输出：**
```
=== Feed-Forward Network ===
[①升维]    (7, 8) → (7, 32)
[②激活]    shape 不变: (7, 32)
[③降维]    (7, 32) → (7, 8)

输入形状: (7, 8)
输出形状: (7, 8)
形状一致: True
```

---

### 三步流程

| 步骤 | 操作 | 输入形状 | 输出形状 |
|------|------|----------|----------|
| ① 升维 | $x · W_1 + b_1$ | $(L, d_{model})$ | $(L, d_{ff})$ |
| ② 激活 | ReLU / GELU | $(L, d_{ff})$ | $(L, d_{ff})$ |
| ③ 降维 | $h · W_2 + b_2$ | $(L, d_{ff})$ | $(L, d_{model})$ |

整体流程：**小 → 大 → 小**，先把信息"展开"到高维空间里让激活函数加工，再"压缩"回原维度传给下一层。

FFN 看起来只是两层全连接，但它在 Transformer 里做的事绝不简单：

1. **升维 + 激活 + 降维**：在高维空间里用非线性提取特征，再压缩回原维度
2. **引入非线性**：弥补 Self-Attention"只会线性组合"的短板
3. **独立加工每个 Token**：Attention 负责横向交流，FFN 负责纵向深化
4. **承载大部分参数**：FFN 参数量约为 Attention 的 2 倍，是模型"记忆"的主要载体

一句话总结：

> **Attention 让 Token 互相看见，FFN 让每个 Token 真正"消化"看到的信息。**

到这里，Transformer Block 的两个核心子模块——**Multi-Head Attention 和 FFN**——我们都手撕完了。

### 本节收尾：费曼自述与闪卡

**费曼自述：** “FFN 可以中间扩宽，但出口必须和残差输入对齐；Norm 是按约定的维度处理，不是随便求全局平均。”

**概念闪卡：** Q：FFN 一般包含什么？A：线性层、非线性激活、再线性映射。Q：LayerNorm 与 BatchNorm 最直观差别？A：归一化所依赖的维度/样本范围不同。

原文核对：[手撕FFN：Transformer前馈网络代码实现](原仓库Markdown/docs/llm/transformer/fnn_code.md)。

---

<a id="s4"></a>

## 9.4 手撕LayerNorm与残差连接：别让基础组件被忽略【⭐ 理解】

**目标：** 写出一层前馈网络，确认最后维度回到 C；把注意力和 FFN 分别接入残差。LayerNorm 可先使用框架模块，再通过小例子理解均值、方差和缩放。

**例子：** X 的形状为 (2,4,8)，FFN 中间到 32，最终回 8；若最终停在 32，就不能与输入做残差相加。LayerNorm 通常对每个位置的特征维归一化，不是对整批样本一起求一个均值。

### 残差连接
残差连接的公式只有一行：

$$
\text{输出} = F(x) + x
$$

$F(x)$ 是子层的输出（比如 Attention 或 FFN 的结果），$x$ 是这一层的输入。两者**直接相加**。

就这么简单？就这么简单。

但它解决了一个深层网络的根本问题：**梯度消失**。

反向传播时，梯度需要从最后一层一路流回第一层。每穿过一层，就要乘一次该层的导数，如果导数很小（比如 Sigmoid 激活区域），链式相乘下来，梯度会指数级缩小，前面几层根本收不到有效的训练信号。

残差连接的巧妙在于：它给梯度提供了一条**"高速公路"**，可以绕过中间层直接流回去，不需要每一层都"乘一遍"。

![LayerNorm代码计算流程示意图](https://file1.kamacoder.com/i/algo/article13_423_p1.drawio.png?v=20261005)

代码里实现残差非常直接：

```python
import numpy as np

def residual_connection(x, sublayer_output):
    """
    残差连接：直接相加
    
    参数：
        x:               子层的输入, shape (L, d_model)
        sublayer_output: 子层的输出, shape (L, d_model)
    
    返回：
        shape (L, d_model)，和输入完全一致
    """
    return x + sublayer_output

# 测试
L, d_model = 7, 8
np.random.seed(42)

x = np.random.randn(L, d_model)             # 子层输入
sublayer_out = np.random.randn(L, d_model)  # 子层输出（模拟）

output = residual_connection(x, sublayer_out)
print(f"输入形状:   {x.shape}")             # (7, 8)
print(f"子层输出:   {sublayer_out.shape}")  # (7, 8)
print(f"残差输出:   {output.shape}")        # (7, 8)
```

**输出：**
```
输入形状:   (7, 8)
子层输出:   (7, 8)
残差输出:   (7, 8)
```

就是一个加法，形状完全不变。但正是这个加法，让 Transformer 能稳定地堆到 100 多层。

### LayerNorm

残差相加之后，数值的范围可能变得很大或者很不均匀——有的维度值极大，有的极小。进入下一层计算时，这种不稳定会严重影响训练效果。

LayerNorm 要做的事，就是把每个 Token 的向量**归一化**：均值变成 0，方差变成 1。

公式如下：

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sigma + \epsilon} \cdot \gamma + \beta
$$

一共四步：

1. 算这个向量的均值 $\mu$
2. 算这个向量的标准差 $\sigma$
3. 用 $(x - \mu) / \sigma$ 标准化，每个维度都"缩"到均值 0、方差 1
4. 再乘 $\gamma$（缩放）、加 $\beta$（偏移）——这两个是可学习参数，让模型自己决定"归一化到什么程度"

其中 $\epsilon$ 是一个很小的数（比如 1e-5），防止分母为 0。

![残差连接代码计算流程示意图](https://file1.kamacoder.com/i/algo/article13_423_p2.drawio.png?v=20261005)

```python
class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        """
        LayerNorm
        
        参数：
            d_model: 向量维度
            eps:     防止除零的小常数
        """
        self.gamma = np.ones(d_model)   # 可学习缩放参数，初始化为 1
        self.beta  = np.zeros(d_model)  # 可学习偏移参数，初始化为 0
        self.eps   = eps
    
    def forward(self, x):
        """
        x: shape (L, d_model)
        """
        # 对每个 Token（每一行）分别计算均值和标准差
        mu    = x.mean(axis=-1, keepdims=True)          # (L, 1)
        sigma = x.std(axis=-1, keepdims=True)           # (L, 1)
        
        # 标准化
        x_norm = (x - mu) / (sigma + self.eps)          # (L, d_model)
        
        # 缩放 + 偏移（广播到每一行）
        return self.gamma * x_norm + self.beta          # (L, d_model)


# 测试
np.random.seed(42)
L, d_model = 7, 8
x = np.random.randn(L, d_model) * 10  # 故意放大数值，模拟数值不稳定的情况

ln = LayerNorm(d_model)
output = ln.forward(x)

print(f"归一化前 - 均值: {x.mean():.2f}, 标准差: {x.std():.2f}")
print(f"归一化后 - 均值: {output.mean():.4f}, 标准差: {output.std():.4f}")
print(f"输入形状: {x.shape}, 输出形状: {output.shape}")
```

**输出：**
```
归一化前 - 均值: -0.37, 标准差: 9.87
归一化后 - 均值: 0.0000, 标准差: 1.0000
输入形状: (7, 8), 输出形状: (7, 8)
```

输入的标准差接近 10，归一化之后变成了 1——数值被"拉"回了合理范围，而形状完全不变。


### 把两者拼在一起 Add & Norm

在 Transformer 里，残差连接和 LayerNorm 从来不分家，标准写法叫 **Add & Norm**：

$$
\text{输出} = \text{LayerNorm}(x + F(x))
$$

先加（残差），再归一化（LayerNorm）。每个子模块——无论是 Attention 还是 FFN——后面都跟一个这样的结构。

```python
def add_and_norm(x, sublayer_output, layer_norm):
    """
    Add & Norm：残差连接 + LayerNorm
    
    参数：
        x:               子层输入,  shape (L, d_model)
        sublayer_output: 子层输出,  shape (L, d_model)
        layer_norm:      LayerNorm 实例
    
    返回：
        shape (L, d_model)
    """
    return layer_norm.forward(x + sublayer_output)


# 完整测试
np.random.seed(42)
L, d_model = 7, 8

x            = np.random.randn(L, d_model)
attn_output  = np.random.randn(L, d_model)  # 模拟 Attention 子层的输出
ffn_output   = np.random.randn(L, d_model)  # 模拟 FFN 子层的输出

ln1 = LayerNorm(d_model)
ln2 = LayerNorm(d_model)

# Attention 子层 → Add & Norm
after_attn = add_and_norm(x, attn_output, ln1)
print(f"Attention 子层后: {after_attn.shape}")  # (7, 8)

# FFN 子层 → Add & Norm
after_ffn  = add_and_norm(after_attn, ffn_output, ln2)
print(f"FFN 子层后:       {after_ffn.shape}")   # (7, 8)

print(f"\n输入形状:  {x.shape}")
print(f"输出形状:  {after_ffn.shape}")
print(f"形状一致:  {x.shape == after_ffn.shape}")
```

**输出：**
```
Attention 子层后: (7, 8)
FFN 子层后:       (7, 8)

输入形状:  (7, 8)
输出形状:  (7, 8)
形状一致:  True
```

---

### Post-Norm vs Pre-Norm

上面写的是 **Post-Norm**，也就是原始论文的做法：先算子层，后归一化。

$$
\text{Post-Norm:} \quad x' = \text{LayerNorm}(x + F(x))
$$

但现代大模型（比如 Llama、GPT-3）基本都改成了 **Pre-Norm**：先归一化，再算子层。

$$
\text{Pre-Norm:} \quad x' = x + F(\text{LayerNorm}(x))
$$

两者差别只有一行代码，效果却不同：Pre-Norm 的训练更稳定，在层数很深（几十层以上）时尤为明显。

```python
# Post-Norm（原始 Transformer）
def post_norm(x, sublayer_fn, layer_norm):
    return layer_norm.forward(x + sublayer_fn(x))

# Pre-Norm（现代大模型）
def pre_norm(x, sublayer_fn, layer_norm):
    return x + sublayer_fn(layer_norm.forward(x))
```

记住这两种写法的区别，面试或者读源码时会频繁遇到。

---

### 步骤汇总

| 组件 | 公式 | 核心作用 |
|------|------|----------|
| 残差连接 | $F(x) + x$ | 给梯度留高速公路，防止深层网络训练崩 |
| LayerNorm | $\frac{x-\mu}{\sigma} \cdot \gamma + \beta$ | 把每个 Token 的向量拉回均值 0、方差 1 |
| Add & Norm | $\text{LayerNorm}(x + F(x))$ | 两者捆绑，每个子层后面都挂一个 |


---

到这里，Transformer Block 的所有零件——Multi-Head Attention、FFN、残差连接、LayerNorm——我们都手撕完了。

### 本节收尾：费曼自述与闪卡

**费曼自述：** “FFN 可以中间扩宽，但出口必须和残差输入对齐；Norm 是按约定的维度处理，不是随便求全局平均。”

**概念闪卡：** Q：FFN 一般包含什么？A：线性层、非线性激活、再线性映射。Q：LayerNorm 与 BatchNorm 最直观差别？A：归一化所依赖的维度/样本范围不同。

原文核对：[手撕LayerNorm与残差连接：别让基础组件被忽略](原仓库Markdown/docs/llm/transformer/layernorm_residual_code.md)。

---

<a id="s5"></a>

## 9.5 手撕Transformer Block：把Attention、FFN、Norm拼起来【⭐ 理解】

**目标：** 把注意力、Norm、FFN 和两次残差按选定 Pre-Norm/Post-Norm 顺序接起来，明确输入/输出都为 (B,T,C)。

**例子：** 不要为了“跑通”把因果 mask 省掉。训练小语言模型时，若前向允许看到未来 token，训练 loss 会虚低，生成时却没有未来信息，效果崩溃。可用“修改未来 token，观察当前位置输出是否变化”的实验检查因果性。

### 一个 Block 到底长什么样？

最常见的 Transformer Block 可以写成：

```text
输入 x
  ↓
Multi-Head Attention
  ↓
残差连接 + LayerNorm
  ↓
FFN
  ↓
残差连接 + LayerNorm
  ↓
输出
```

也就是：

$$
x_1 = \text{LayerNorm}(x + \text{MHA}(x))
$$

$$
x_2 = \text{LayerNorm}(x_1 + \text{FFN}(x_1))
$$

这里有一个非常重要的点：

> **输入是什么形状，输出还是什么形状。**

假设输入是：

```text
x.shape = (L, d_model)
```

其中：

- L 是序列长度，比如“远方有颗苹果树”有 7 个 token
- d_model 是每个 token 的向量维度，比如这里用 8 维演示

那么经过一个 Transformer Block 后，输出仍然是：

```text
output.shape = (L, d_model)
```

这也是为什么 Transformer 可以一层一层往上堆。

![Transformer Block代码结构示意图](https://file1.kamacoder.com/i/algo/article14_0425_p1.png?v=20261005)

### 先写 LayerNorm 和 FFN

我们先把最基础的组件准备好。

```python
import numpy as np

class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        x_norm = (x - mean) / (std + self.eps)
        return self.gamma * x_norm + self.beta
```

LayerNorm 做的事很简单：对每个 Token 自己的向量做归一化。

比如输入是 `(7, 8)`，表示 7 个 token，每个 token 8 维，那么 LayerNorm 会分别对这 7 行做归一化。

接着写 FFN：

```python
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)

    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(
            np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)
        ))

    def forward(self, x):
        hidden = x @ self.W1 + self.b1
        hidden = self.gelu(hidden)
        output = hidden @ self.W2 + self.b2
        return output
```

FFN 的过程就是：

```text
(L, d_model)
   ↓ 升维
(L, d_ff)
   ↓ GELU
(L, d_ff)
   ↓ 降维
(L, d_model)
```

还是那句话：**中间怎么变都可以，但最后必须回到 d_model。**

### 写 Multi-Head Attention

现在写一个最小版多头注意力。

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = np.random.randn(d_model, d_model) * 0.01
        self.W_K = np.random.randn(d_model, d_model) * 0.01
        self.W_V = np.random.randn(d_model, d_model) * 0.01
        self.W_O = np.random.randn(d_model, d_model) * 0.01

    def softmax(self, x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def split_heads(self, x):
        L, d_model = x.shape
        return x.reshape(L, self.num_heads, self.d_k).transpose(1, 0, 2)

    def forward(self, x):
        L, d_model = x.shape

        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        head_outputs = []

        for i in range(self.num_heads):
            scores = Q[i] @ K[i].T / np.sqrt(self.d_k)
            weights = self.softmax(scores)
            out = weights @ V[i]
            head_outputs.append(out)

        concat = np.stack(head_outputs, axis=0)
        concat = concat.transpose(1, 0, 2).reshape(L, d_model)

        output = concat @ self.W_O
        return output
```

这里最容易绕的还是 shape：

```text
输入 x:              (L, d_model)
Q/K/V:               (L, d_model)
拆成多个头:          (num_heads, L, d_k)
每个头输出:          (L, d_k)
拼接回来:            (L, d_model)
输出投影后:          (L, d_model)
```

可以看到，Attention 虽然中间拆成多个头，但最后依然回到原来的形状。

![Transformer Block前向传播输出示意图](https://file1.kamacoder.com/i/algo/article14_0425_p2.png?v=20261005)

---

### 定义完整 Transformer Block

现在核心来了。

```python
class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x):
        print(f"输入 x:              {x.shape}")

        attn_out = self.attn.forward(x)
        print(f"Attention 输出:      {attn_out.shape}")

        x = self.norm1.forward(x + attn_out)
        print(f"Add & Norm 之后:     {x.shape}")

        ffn_out = self.ffn.forward(x)
        print(f"FFN 输出:            {ffn_out.shape}")

        x = self.norm2.forward(x + ffn_out)
        print(f"Block 最终输出:      {x.shape}")

        return x
```

这就是一个最小版 Transformer Block。

注意这里用了 Post-Norm 写法：

```python
x = norm(x + sublayer(x))
```

也就是先经过子层，再残差相加，最后 LayerNorm。

现代很多大模型会使用 Pre-Norm：

```python
x = x + sublayer(norm(x))
```

但为了和原始 Transformer 结构更一致，也为了方便初学者理解，这里先用 Post-Norm。

---

### 跑通一个 toy example

我们继续用熟悉的句子：

> 远方有颗苹果树

假设它被切成 7 个 token，每个 token 用 8 维向量表示。

```python
if __name__ == "__main__":
    np.random.seed(42)

    L = 7
    d_model = 8
    num_heads = 2
    d_ff = 32

    x = np.random.randn(L, d_model)

    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff
    )

    output = block.forward(x)

    print("\n=== 检查结果 ===")
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
    print("形状一致:", x.shape == output.shape)
```

运行输出大概是：

```text
输入 x:              (7, 8)
Attention 输出:      (7, 8)
Add & Norm 之后:     (7, 8)
FFN 输出:            (7, 8)
Block 最终输出:      (7, 8)

=== 检查结果 ===
输入形状: (7, 8)
输出形状: (7, 8)
形状一致: True
```

这说明我们的最小版 Transformer Block 已经跑通了。

从输入到输出，形状始终保持 `(7, 8)`。

但是注意：**形状没变，不代表内容没变。**

经过 Attention 之后，每个 Token 已经融合了其他 Token 的上下文信息。

经过 FFN 之后，每个 Token 又单独做了一次非线性加工。

再加上残差连接和 LayerNorm，整个 Block 就既能表达复杂语义，又能保持训练稳定。

---

### 本节收尾：费曼自述与闪卡

**费曼自述：** “Block 是两个处理子层加残差，能重复堆叠；自回归任务的注意力必须守住看不到未来的约束。”

**概念闪卡：** Q：怎样检测未来信息泄漏？A：保持前缀相同，仅改未来 token，比较较早位置输出。Q：Block 正常输出形状？A：仍为 (B,T,C)。

原文核对：[手撕Transformer Block：把Attention、FFN、Norm拼起来](原仓库Markdown/docs/llm/transformer/transformer_block_code.md)。

---

<a id="s6"></a>

## 9.6 手撕Tiny Transformer：从零拼出一个完整模型【◻ 选读】

# 第九章：手写 Transformer——把原理变成代码

> 本章优先级：🔥 亲手实现与调试；仓库 6 篇代码文章可看解释，但“照抄示例能跑”不算掌握。尤其 [Tiny Transformer 文章](https://github.com/youngyangyang04/llm-master/blob/main/docs/llm/transformer/tiny_transformer_code.md)主要演示 NumPy 前向，没有形成可训练、带因果 mask 的完整 GPT；你的主实践仍是 Build GPT。

### 整体结构

一个最小版 Transformer，可以拆成四层：

```
Token ID
  ↓ Embedding
  ↓ Positional Encoding
  ↓ 多层 Transformer Block
  ↓ 输出层 Linear
  ↓ logits
```

核心变化只有一个：一开始输入是整数 Token ID，经过 Embedding 后变成向量；之后所有模块都在处理 `(L, d_model)` 这个矩阵。

![Tiny Transformer完整结构代码图](https://file1.kamacoder.com/i/algo/article14_0428_p1.drawio.png?v=20261005)

### 第一步：Embedding

模型不能直接理解文字，也不能直接理解 Token ID。比如一句话被切成：

```python
input_ids = [3, 8, 2, 6]
```

这些数字只是编号，没有语义。

Embedding 要做的事，就是把每个编号查表，变成一个向量。

```python
import numpy as np

np.random.seed(42)

vocab_size = 10
L = 4
d_model = 8

input_ids = np.array([3, 8, 2, 6])
embedding_table = np.random.randn(vocab_size, d_model)

x = embedding_table[input_ids]
print(x.shape)  # (4, 8)
```

现在，4 个 Token，每个 Token 都变成了 8 维向量。

### 第二步：Positional Encoding

Self-Attention 本身不关心顺序。对它来说，“我喜欢你”和“你喜欢我”如果只看 Token 集合，差别并不明显。

所以我们要把位置信息加进去。

```python
def positional_encoding(L, d_model):
    pe = np.zeros((L, d_model))
    for pos in range(L):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return pe

x = x + positional_encoding(L, d_model)
print(x.shape)  # (4, 8)
```

注意：位置编码不是拼接，而是相加。所以 shape 不变，仍然是 `(L, d_model)`。


### 第三步：Transformer Block

一个 Block 里有两件事：

```text
Attention：让 Token 之间交换信息
FFN：让每个 Token 独立加工信息
```

再加上残差连接和 LayerNorm，一个最小 Block 长这样：

```python
class TinyBlock:
    def __init__(self, d_model, d_ff):
        self.Wq = np.random.randn(d_model, d_model) * 0.01
        self.Wk = np.random.randn(d_model, d_model) * 0.01
        self.Wv = np.random.randn(d_model, d_model) * 0.01
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)

    def layer_norm(self, x, eps=1e-5):
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return (x - mean) / (std + eps)

    def attention(self, x):
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv
        scores = Q @ K.T / np.sqrt(x.shape[-1])
        weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
        return weights @ V

    def ffn(self, x):
        hidden = np.maximum(0, x @ self.W1 + self.b1)
        return hidden @ self.W2 + self.b2

    def forward(self, x):
        x = self.layer_norm(x + self.attention(x))
        x = self.layer_norm(x + self.ffn(x))
        return x
```

这段代码没有多头，没有 mask，也没有训练，只保留最核心的数据流。

### 第四步：堆多层 Block

Transformer 的“深度”来自重复堆叠 Block。

```python
blocks = [TinyBlock(d_model=8, d_ff=32) for _ in range(2)]

for block in blocks:
    x = block.forward(x)

print(x.shape)  # (4, 8)
```

堆两层、六层、十二层，本质都是重复同一个结构。只要每层输入输出 shape 一致，就可以一直往下接。

![Tiny Transformer前向运行结果图](https://file1.kamacoder.com/i/algo/article14_0428_p2.drawio.png?v=20261005)

### 第五步：输出层

最后一步，要把每个 Token 的隐藏向量映射回词表大小。

如果词表大小是 `vocab_size=10`，那输出层就是：

```python
W_out = np.random.randn(d_model, vocab_size) * 0.01
b_out = np.zeros(vocab_size)

logits = x @ W_out + b_out
print(logits.shape)  # (4, 10)
```

`logits[0]` 表示第 1 个位置对词表中 10 个 Token 的打分。

打分最高的 Token，就是模型当前最想输出的结果。

### 最小可运行 Demo

把所有代码连起来：

```python
import numpy as np

np.random.seed(42)

vocab_size = 10
L = 4
d_model = 8
d_ff = 32
num_layers = 2

input_ids = np.array([3, 8, 2, 6])
embedding_table = np.random.randn(vocab_size, d_model)

x = embedding_table[input_ids]

# positional encoding
def positional_encoding(L, d_model):
    pe = np.zeros((L, d_model))
    for pos in range(L):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return pe

x = x + positional_encoding(L, d_model)

class TinyBlock:
    def __init__(self, d_model, d_ff):
        self.Wq = np.random.randn(d_model, d_model) * 0.01
        self.Wk = np.random.randn(d_model, d_model) * 0.01
        self.Wv = np.random.randn(d_model, d_model) * 0.01
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)

    def layer_norm(self, x, eps=1e-5):
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return (x - mean) / (std + eps)

    def attention(self, x):
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv
        scores = Q @ K.T / np.sqrt(x.shape[-1])
        weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
        return weights @ V

    def ffn(self, x):
        hidden = np.maximum(0, x @ self.W1 + self.b1)
        return hidden @ self.W2 + self.b2

    def forward(self, x):
        x = self.layer_norm(x + self.attention(x))
        x = self.layer_norm(x + self.ffn(x))
        return x

blocks = [TinyBlock(d_model, d_ff) for _ in range(num_layers)]

for block in blocks:
    x = block.forward(x)

W_out = np.random.randn(d_model, vocab_size) * 0.01
b_out = np.zeros(vocab_size)
logits = x @ W_out + b_out

print("最终输出 logits shape:", logits.shape)
print("每个位置预测的 token:", logits.argmax(axis=-1))
```

**输出示例：**

```text
最终输出 logits shape: (4, 10)
每个位置预测的 token: [5 5 4 3]
```

到这里，一个 Tiny Transformer 就真的跑起来了。

它还不会写文章，也不会聊天，因为我们没有训练它。但结构已经完整了：Embedding 负责把 ID 变成向量，位置编码负责告诉模型顺序，Block 负责反复加工信息，输出层负责把向量变回词表打分。

Transformer 最核心的秘密，其实就是这条数据流：

```text
ID → 向量 → 加位置 → 多层加工 → 词表打分
```

理解了这条线，再看 GPT、BERT、LLaMA，就不再是一团黑盒了。

---

### 本节收尾：费曼自述与闪卡

**费曼自述：** 不看上文，向刚学 Python 的同学解释“手撕Tiny Transformer：从零拼出一个完整模型”解决什么问题、如何工作，并举一个可验证的例子。

**概念闪卡：** Q：这一节的核心机制是什么？A：用本节例子说清输入、处理和输出。Q：怎样验证它真的有用？A：在固定样例或测试集上比较基线与改动结果。

原文核对：[手撕Tiny Transformer：从零拼出一个完整模型](原仓库Markdown/docs/llm/transformer/tiny_transformer_code.md)。

---

## 本章完成标准

逐组件重写，不复制原文；每个组件至少测形状、因果 Mask 和梯度；最后用小数据集训练 Tiny GPT，记录训练与验证 loss。

能闭卷解释本章 A/B 内容，并拿出代码、测试或实验记录后，才把本章标成“已学会”。

