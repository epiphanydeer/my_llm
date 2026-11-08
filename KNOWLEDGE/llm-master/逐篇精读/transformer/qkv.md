# Attention机制详解：Q、K、V是什么，为什么它是Transformer的核心

> 重要程度：**🔥 必会** · [原仓库同版本文章](https://github.com/youngyangyang04/llm-master/blob/78b03ad475409ba1d6326e0a6e8d50b27025bb64/docs/llm/transformer/qkv.md) · 原作者：程序员Carl（MIT License，见 [LICENSE](../LICENSE)）。
> 整理方法：保留技术解释、示例、代码和面试问答；压缩开篇叙事和推广语。下方“学习补充”由本笔记新增。

## 学习补充：先理解，再闭卷复述

### 8.3 Q、K、V 与注意力计算【🔥】

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

## 原文技术内容（整理版）

### 注意力的直觉解释  
Attention本质上在做一件很像人类的事，大家先看这句话：  
> 小明打了小刚，因为他生气了

这里的“他”指的是谁？小明？小刚？放你看到这句话时，你不会随机回答，你会“先看一眼上下文”，再做出判断——小明。这就是Attention的直觉解释：  
> 从上下文中，挑出和当前词最相关的部分；也就是说，**每个Token都会关注其他Token，但关注的程度不一样**

### 为什么必须关注不同Token？  
如果没有Attention机制，会发生什么？  
1.每个词独立处理  
2.像RNN一样顺序传递信息  
这都会导致远距离的信息丢失，比如这句话：  
>我最近都在学习Transformer，前几天还去图书馆借了相关书籍，今天终于把Transformer学完了，它真的很强大  

“它”指的是谁？相关书籍？图书馆？Transformer？如果模型不能**“看整体”**，就很难捕捉到这样的语义信息  

### Query，Key，Value在Attention中扮演的角色  
大家一定也会常常听到Attention中的QKV矩阵，也是很多人搞不懂的地方：**QKV到底是做什么的？** 本篇文章依旧不涉及复杂公式，给大家形象介绍下它们的用途。  
Query（Q）：当前Token会问所有Token一个问题：我应该**“更注意”** 谁？  
比如这句话：
> *远方走来的是一位身穿白裙婀娜多姿面带微笑的少女*

这句话中，“身穿白裙”“婀娜多姿”“面带微笑”“少”等词对于“女”的语义影响是很大的，而“的”“是”等词就显得没有那么重要，于是Query矩阵就相当于“女”这个Token在发问：**我应该更关注谁？**

![Query Key Value角色关系示意图](https://file1.kamacoder.com/i/algo/a5d4c76f-01d3-4755-b481-cb8cfdc1a5ad.webp?v=20261005)


Key（K）：这个矩阵相当于在回答Query矩阵的问题，还是以上面这句话为例，“女”这个Token会问：我应该更关注谁？而Key矩阵则是回答：你应该更关注我。于是“身穿白裙”“婀娜多姿”“面带微笑”“少”则是回答“女”的query：**多关注我！**

![Attention根据QK相似度聚合Value示意图](https://file1.kamacoder.com/i/algo/42a2c5fd-ca0a-41ba-b101-7fe9679cd5b8.webp?v=20261005)


Value（V）：Value 是 Token 携带的**核心特征信息**。当 Query 发现某个 Key 很匹配时，它就会把对应的 Value 提取出来，融入到当前的语义表示中

