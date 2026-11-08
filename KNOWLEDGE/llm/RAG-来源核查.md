# RAG 学习资料来源核查

> 用途：为第三章学习笔记提供事实校准。核查时间：2026-11-08。这里区分「材料原话」「可验证的技术事实」与「应由项目实验决定的经验值」。

## 1. 材料范围

[卡码 RAG 专题目录](https://github.com/youngyangyang04/llm-master/blob/main/docs/topics/rag.md)只有 20 行，是学习顺序，不是单独的教程。其主干 10 篇与用户提供的页面一致：

1. [为什么需要 RAG](https://notes.kamacoder.com/llm/app/why_rag.html)
2. [完整链路](https://notes.kamacoder.com/llm/app/chain_of_rag.html)
3. [切片](https://notes.kamacoder.com/llm/app/how_to_chunking.html)
4. [Embedding](https://notes.kamacoder.com/llm/app/embedding.html)；用户附件提供了这篇的粘贴文本
5. [向量数据库](https://notes.kamacoder.com/llm/app/vector_database.html)
6. [问题排查](https://notes.kamacoder.com/llm/app/rag_problems.html)
7. [优化](https://notes.kamacoder.com/llm/app/rag_optimization.html)
8. [长文档与代码检索](https://notes.kamacoder.com/llm/app/long_document_code_retrieval.html)
9. [评估](https://notes.kamacoder.com/llm/app/rag_evaluation.html)
10. [Agentic RAG](https://notes.kamacoder.com/llm/intro/agentic_rag.html)

目录另列 4 篇面试深化，用户已给第 1 篇： [回答框架](https://notes.kamacoder.com/llm/app/rag_interview_framework.html)、[高频面试题](https://github.com/youngyangyang04/llm-master/blob/main/docs/interview/llm/rag_interview.md)、[落地难点](https://github.com/youngyangyang04/llm-master/blob/main/docs/interview/llm/rag_hardest_parts_interview.md)、[GraphRAG 与 LightRAG](https://github.com/youngyangyang04/llm-master/blob/main/docs/interview/llm/graphrag_interview.md)。因此总计 14 篇内容页，不能只按 11 篇主干/用户链接算。

## 2. 需要写进正式笔记的事实校准

| 主题 | 材料里容易误导初学者的说法 | 更准确的表述与依据 |
| --- | --- | --- |
| RAG 的作用 | RAG「解决幻觉」或有来源就可信 | RAG 给生成提供外部证据，可能改善知识密集问答的事实性和可更新性；检索错、引用错或模型推理错仍会产生错误。原始 [RAG 论文](https://arxiv.org/abs/2005.11401)报告相对基线的改进，不是无条件保证。 |
| Embedding 一致性 | 离线文档与在线查询「必须使用同一个模型」 | 必须进入**兼容且按模型设计对齐的向量空间**。单编码器常用同一模型；非对称检索可使用同一模型的 `encode_query` / `encode_document` 不同提示或路由，也可用配套训练的双编码器。更换其中一侧编码配置时通常需重新建索引或验证兼容性。见 [Sentence Transformers 语义检索文档](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)。 |
| 数据库选型 | 「MySQL 不能做向量检索」「百万级 pgvector 明显不如专用库」 | 传统 B-tree 不适合高维近邻搜索，但数据库扩展可以做。[pgvector](https://github.com/pgvector/pgvector)支持精确检索、HNSW 和 IVFFlat。其与专用库的性能/成本要在**相同数据、过滤条件、并发、召回目标和硬件**下比较，不可按数据量单独下结论。 |
| FAISS | 「FAISS 没有持久化」 | FAISS 是向量**索引库**而非完整数据库；它支持 `write_index` / `read_index` 序列化。它通常不直接提供完整数据库的元数据、权限、事务和服务运维能力。见 [FAISS Index I/O](https://github.com/facebookresearch/faiss/wiki/Index-IO,-cloning-and-hyper-parameter-tuning)。 |
| ANN 性能 | 「近似 O(log N)」「召回通常 95%+」「毫秒级」 | HNSW 原论文报告特定设置下的对数复杂度扩展趋势；生产延迟和 recall 由索引参数、数据分布、过滤、硬件等决定，不能承诺通用数字。IVF 访问多少候选列表也由 probes 决定。见 [HNSW 原论文](https://arxiv.org/abs/1603.09320)、[pgvector 索引参数说明](https://github.com/pgvector/pgvector)。 |
| 切片与重排 | 「256–512 token 最佳」「Top-20→Top-5 是标配」「一定要 Rerank」 | 这些是实验起点，不是定律。切片需按文档结构和问题粒度设定；重排先召回候选，再用 CrossEncoder 评估 Query/Chunk 对，收益与延迟要实测。见 [Sentence Transformers Retrieve & Re-Rank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)。 |
| 混合检索 | 向量检索永远优于关键词，或混合检索总优于单路 | 关键词适合名称、日期、产品码等精确词；语义检索适合词面不一致的问题。两路并行和 RRF 融合是可行方案，是否启用由评估集决定。见 [Azure AI Search 混合检索文档](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview)。 |
| GraphRAG / Agentic RAG | 普通 RAG 遇到多跳/全局问题「只能」换图或 Agent | 微软 [GraphRAG 论文](https://arxiv.org/abs/2404.16130)针对**全语料主题概括**展示了相对传统 RAG 基线的改进，不能推导出所有多跳或长文档任务都必须用图。查询拆分、迭代检索、结构化切片和层级摘要也是候选方案。Agentic RAG 中模型可在观察结果后决定下一次检索，类似 [ReAct](https://arxiv.org/abs/2210.03629) 的行动与观察循环；增加调用成本，先在固定流程基线上验证。 |

## 3. 评估指标的细节

[卡码评估文章](https://notes.kamacoder.com/llm/app/rag_evaluation.html)一处把 Recall@K 简写成「正确答案在 top-K 出现的比例」，随后又给出正确的分母「所有相关 chunk 总数」。前者若按每个问题只有“命中/未命中”计算，通常称 **Hit@K / Success@K**，与 Recall@K 不同。例：标准答案对应 3 个相关片段，前 5 个结果找到 2 个，则 Hit@5 = 1、Recall@5 = 2/3、Precision@5 = 2/5。先定义**什么算相关、标注单位是文档还是片段**，再算指标；切片方案变化后不能直接拿旧 chunk ID 当金标。

生成侧至少分开看：**正确性**（与可核实答案一致）、**忠实度**（每个事实是否有提供的上下文支持）、**引用正确性**（引文是否支持对应论断）。[Ragas 官方指标目录](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)分别列出 Context Precision/Recall、Response Relevancy、Faithfulness 等，不能把单一 LLM 打分当全部质量。

## 4. 建议的初学者实验顺序

1. 用 10–30 份公开文档和 20–50 个自己标注的问题，做最小链路：解析、保留标题/页码、切片、Embedding、检索、带引用回答。
2. 保存每题的**原文证据、检索前 5 名、最终答案、引用**。先算 Hit@5、Recall@5，再人工核对答案正确性和引用。
3. 失败按链路定位：解析是否丢内容 → 证据是否被切断 → 相关片段是否进入候选 → 排名是否靠前 → 模型是否正确使用证据。每次只改一个环节并复测。
4. 在基线上依次试结构切片、BM25/混合检索、Rerank；仅在具体问题未解决时再试查询改写、父子块、Agentic RAG 或 GraphRAG。记录质量、延迟与成本的变化。

面试文章里的「必考」「主流」「80% 工时」等属于作者经验描述，不能从这些资料推出目标企业 JD 的真实频率；求职优先级仍需另用实际 JD 样本统计。
