# 第三章 · RAG 检索增强系统（原理闭卷 + 渐进手撕 + 生产排障）

本章面向 2027 社招大模型/多模态应用算法工程师（Applied MLE）岗位，深度融合 llm-master 原教程严密逻辑与全部原图、补充高频面试盲区与扩展知识，并采用 Step 1 → Step 14 真正“搭积木式”渐进代码底座（每节仅在上一步基础上新增 1 个核心模块，全部附带类型标注、assert 断言、工业级调库对照与三段式面试话术）。

## 本章目录与代码演进路线图（共 14 节）

| 节号 | 标题 | 优先级 | 渐进式代码积木（Step 1 → Step 14） | 核心面试考点 |
| --- | --- | --- | --- | --- |
| 3.1 | [3.1-为什么需要RAG.md](3.1-%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81RAG.md) | 【🔥 必会】 | Step 1：Document + 权限/时效前置过滤 + 无证据拒答（~55行） | 1M 长窗口为何替代不了 RAG、RAG vs 微调、盲枚举防御 |
| 3.2 | [3.2-RAG完整链路.md](3.2-RAG%E5%AE%8C%E6%95%B4%E9%93%BE%E8%B7%AF.md) | 【🔥 必会】 | Step 2：封装 MiniRAGPipeline（离线入库 + Prompt 锚点 + 引用校验） | 离线/在线全链路、空间对齐铁律、虚假引用拦截 |
| 3.3 | [3.3-切片策略与来源偏移.md](3.3-%E5%88%87%E7%89%87%E7%AD%96%E7%95%A5%E4%B8%8E%E6%9D%A5%E6%BA%90%E5%81%8F%E7%A7%BB.md) | 【🔥 必会】 | Step 3：升级 SlidingWindowChunker（Overlap + 标点吸附 + 偏移恒等） | 四大切分策略选型、字符 vs Token、防 while 死循环 |
| 3.4 | [3.4-Embedding与相似度.md](3.4-Embedding%E4%B8%8E%E7%9B%B8%E4%BC%BC%E5%BA%A6.md) | 【🔥 必会】 | Step 4：新增 DenseVectorRetriever（L2 归一化 + 矩阵余弦 + 前缀） | 归一化下余弦/内积/L2 等价证明、防 NaN、BGE 非对称前缀 |
| 3.5 | [3.5-向量索引与数据库.md](3.5-%E5%90%91%E9%87%8F%E7%B4%A2%E5%BC%95%E4%B8%8E%E6%95%B0%E6%8D%AE%E5%BA%93.md) | 【🔥 必会】 | Step 5：新增 SimpleNavigableGraphIndex（Flat vs ANN 基准对照） | HNSW 三大超参、内存公式、FAISS vs Milvus/Qdrant/pgvector |
| 3.6 | [3.6-RAG分阶段排障.md](3.6-RAG%E5%88%86%E9%98%B6%E6%AE%B5%E6%8E%92%E9%9A%9C.md) | 【🔥 必会】 | Step 6：新增 PipelineTrace + 六层因果穿透排障状态机 | 五层排障 SOP、内在 vs 外在幻觉、Trace ID 全链路埋点 |
| 3.7 | [3.7-混合召回与重排.md](3.7-%E6%B7%B7%E5%90%88%E5%8F%AC%E5%9B%9E%E4%B8%8E%E9%87%8D%E6%8E%92.md) | 【🔥 必会】 | Step 7：新增 MiniBM25Retriever + 加权 RRF（k=60）+ 同文档去重 | 为何不直接相加、$k=60$ 数学原理、Bi vs Cross-Encoder |
| 3.8 | [3.8-长文档与代码检索.md](3.8-%E9%95%BF%E6%96%87%E6%A1%A3%E4%B8%8E%E4%BB%A3%E7%A0%81%E6%A3%80%E7%B4%A2.md) | 【🔥 必会】 | Step 8：新增 Python AST 切片（含装饰器降级）+ 首尾倒置父块扩展 | Parent-Child 顺序铁律、Claude Code 为何用 Grep+AST |
| 3.9 | [3.9-RAG评测.md](3.9-RAG%E8%AF%84%E6%B5%8B.md) | 【🔥 必会】 | Step 9：新增 evaluate_retrieval（五大指标）+ 防刷分忠实度 | Hit/Recall/MRR/NDCG 算例、RAGAS 四维度、误拒答漏洞 |
| 3.10 | [3.10-AgenticRAG.md](3.10-AgenticRAG.md) | 【⭐ 理解】 | Step 10：升级 agentic_rag_loop 动态路由 + 三层死循环熔断 | 四代演进、Multi-Query vs Multi-hop、CRAG、防 Context Drift |
| 3.11 | [3.11-RAG项目面试.md](3.11-RAG%E9%A1%B9%E7%9B%AE%E9%9D%A2%E8%AF%95.md) | 【⭐ 理解】 | Step 11：新增 audit_interview_claims（基线对比与防虚构审计） | 五步证据链框架、涉密脱敏话术、未测项坦诚答法 |
| 3.12 | [3.12-质量延迟与成本.md](3.12-%E8%B4%A8%E9%87%8F%E5%BB%B6%E8%BF%9F%E4%B8%8E%E6%88%90%E6%9C%AC.md) | 【🔥 必会】 | Step 12：新增 analyze_perf_and_cost（TTFT/TPOT + P95 反证） | 为何阶段 P95 不可相加、Fast-path 旁路、Prompt Cache |
| 3.13 | [3.13-文档更新与索引一致性.md](3.13-%E6%96%87%E6%A1%A3%E6%9B%B4%E6%96%B0%E4%B8%8E%E7%B4%A2%E5%BC%95%E4%B8%80%E8%87%B4%E6%80%A7.md) | 【🔥 必会】 | Step 13：新增 plan_incremental_sync（SHA-256 + 仅刷元数据 + 熔断） | 三大难点级联放大、CSV 解析、防爬虫抖动误删熔断 |
| 3.14 | [3.14-图增强检索.md](3.14-%E5%9B%BE%E5%A2%9E%E5%BC%BA%E6%A3%80%E7%B4%A2.md) | 【◻ 按需】 | Step 14：新增 LightGraphRetriever（带 chunk_id 溯源多跳 BFS） | 传统 RAG 天花板、GraphRAG Local/Global、LightRAG 选型 |
