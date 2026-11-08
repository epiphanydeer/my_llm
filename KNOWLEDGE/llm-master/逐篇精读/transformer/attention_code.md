# 手撕Attention：不依赖框架从零实现注意力机制

> 重要程度：**⭐ 理解** · [原仓库同版本文章](https://github.com/youngyangyang04/llm-master/blob/78b03ad475409ba1d6326e0a6e8d50b27025bb64/docs/llm/transformer/attention_code.md) · 原作者：程序员Carl（MIT License，见 [LICENSE](../LICENSE)）。
> 整理方法：保留技术解释、示例、代码和面试问答；压缩开篇叙事和推广语。下方“学习补充”由本笔记新增。

## 学习补充：先理解，再闭卷复述

### 9.1 手写单头 Attention【🔥】

**目标：** 给输入 X=(B,T,C)，独立写投影、点积、缩放、mask、softmax、加权求和，并用断言验证输出形状。

**例子：** B=1、T=3、C=4，第三个 token 可以看位置 1～3，第一个 token 只能看位置 1。若第一个位置的权重落到未来位置，因果 mask 就写错了。注意不同库布尔 mask 的 True/False 含义可能不同，要查所用 API。

**费曼自述：** “注意力代码先生成三个投影，求相关性并屏蔽不该看的位置，再把权重乘到 V 上。”

**概念闪卡：** Q：最容易写反的转置？A：K 的最后两个维度。Q：mask 后怎样验证？A：检查被遮位置 softmax 权重接近 0、输出形状正确。

## 原文技术内容（整理版）

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

