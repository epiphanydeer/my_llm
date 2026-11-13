# 第三章 · RAG 检索增强（原仓库一字不删 + 你的仓库精华 + 零基础大白话 + 15 级渐进代码）

> **本章重构原则**：
> 1. **原仓库正文与配图 100% 完整保留**：每一节前半部分完整保留 `youngyangyang04/llm-master` 对应文章的全部段落、小标题与全部 38 张原始配图（包括第 4 节《Embedding 和 Rerank 有什么区别？》的完整文字与对比图，以及单独成篇的 `3.15-RAG落地最难的地方在哪.md`）。
> 2. **你的仓库 `llm-master` 14 节精华全数融合**：保留五要素表、高频排障题、工程 Trick 与 `3.12-质量延迟与成本.md`、`3.13-文档更新与索引一致性.md` 等全部扩展章节，并把所有生涩术语翻译成零基础一眼能懂的大白话。
> 3. **纯正 `.md` 格式上传 Google Drive**：全部以原生 `text/markdown` 格式存放在文件夹中，不再转成丑陋分页的 Google Doc，也没有任何 波浪号 删除线问题。

## 本章完整目录（15 节全覆盖）

| 节号 | 文件名 | 原仓库对应文章 + 你的仓库对应模块 | 渐进式手撕代码（Level 1 → 14） |
| --- | --- | --- | --- |
| 3.1 | [3.1-为什么需要RAG.md](3.1-为什么需要RAG.md) | `why_rag.md` 全文 + 3.1 外部知识与权限守卫 | Level 1：30 行最小关键词 RAG + 权限过滤与拒答 |
| 3.2 | [3.2-RAG完整链路.md](3.2-RAG完整链路.md) | `chain_of_rag.md` 全文 + 3.2 端到端链路与引用校验 | Level 2：端到端离线切片 → 在线召回 → 引用校验 |
| 3.3 | [3.3-RAG切片策略.md](3.3-RAG切片策略.md) | `how_to_chunking.md` 全文 + 3.3 标点吸附与偏移 | Level 3：滑动窗口 + 标点自然吸附 + 可逆偏移断言 |
| 3.4 | [3.4-Embedding与Rerank区别.md](3.4-Embedding与Rerank区别.md) | `embedding.md` 全文（含 Embedding vs Rerank 图解）+ 3.4 余弦几何等价 | Level 4：防除零 L2 归一化 + 余弦相似度 Top-K |
| 3.5 | [3.5-向量数据库与ANN索引.md](3.5-向量数据库与ANN索引.md) | `vector_database.md` 全文 + 3.5 Flat vs HNSW 对比 | Level 5：精确暴力搜索 vs 分层导航近似搜索对比 |
| 3.6 | [3.6-RAG常见问题排查.md](3.6-RAG常见问题排查.md) | `rag_problems.md` 全文 + 3.6 五层证据链排障漏斗 | Level 6：五层证据链断点自动诊断器 |
| 3.7 | [3.7-RAG优化：混合检索与Rerank.md](3.7-RAG优化：混合检索与Rerank.md) | `rag_optimization.md` 全文 + 3.3/3.7 手写 BM25 + 加权 RRF | Level 7：手写 BM25 + 加权 RRF 融合 + Rerank 精排 |
| 3.8 | [3.8-长文档与代码检索.md](3.8-长文档与代码检索.md) | `long_document_code_retrieval.md` 全文 + 3.8 AST 代码切分 | Level 8：带装饰器保留的 Python AST 语法树切分器 |
| 3.9 | [3.9-RAG评估体系.md](3.9-RAG评估体系.md) | `rag_evaluation.md` 全文 + 3.9 Hit/Recall/MRR/NDCG 手撕 | Level 9：手写 Hit@K / Recall@K / MRR / NDCG 与忠实度 |
| 3.10 | [3.10-AgenticRAG.md](3.10-AgenticRAG.md) | `agentic_rag.md` 全文 + 3.10 动态循环与死循环熔断 | Level 10：自适应改写循环 + 去重 + 最大步数硬熔断 |
| 3.11 | [3.11-RAG面试回答框架.md](3.11-RAG面试回答框架.md) | `rag_interview_framework.md` 全文 + 3.11 项目证据审计 | Level 11：面试指标真实性与相对提升计算审计器 |
| 3.12 | [3.12-质量延迟与成本.md](3.12-质量延迟与成本.md) | 你的仓库 `3.12` + 零基础看懂 TTFT/TPOT 与 P95 不可加 | Level 12：端到端 P50/P95/P99 分位数计算与反证 |
| 3.13 | [3.13-文档更新与索引一致性.md](3.13-文档更新与索引一致性.md) | 你的仓库 `3.13` + 零基础大白话讲透增量哈希与防误删锁 | Level 13：SHA-256 指纹增量同步 + 双保险防误删跳闸 |
| 3.14 | [3.14-GraphRAG与LightRAG.md](3.14-GraphRAG与LightRAG.md) | `graphrag_interview.md` 全文 + 3.14 带溯源多跳图检索 | Level 14：带 `chunk_id` 原文溯源的无环两跳图谱检索 |
| 3.15 | [3.15-RAG落地最难的地方与高频题汇总.md](3.15-RAG落地最难的地方与高频题汇总.md) | `rag_hardest_parts_interview.md` 全文 + `rag_interview.md` 全文 | 全章 12 大高频题 + 三大落地难点级联复盘总表 |

- **3.16-补充：RAG与微调选型对比与全章哈希对账单.md**：原仓库 `finetuning_vs_rag.md` 全文 + 3 张原图 + 量化选型决策器 + 第三章 16 篇原仓库与 14 篇个人仓库 100% 哈希对账单。
