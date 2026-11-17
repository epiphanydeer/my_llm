# 第六章 部署推理与压测优化：渐进式面试备考全集

本章围绕**「高并发大模型推理引擎与压测优化」**，从显存与 TCO 容量规划、纯手写 KV Cache 增量解码与 PagedAttention 块表，一路推进到 INT8/INT4 量化保真度审计与 TTFT/TPOT/Goodput 流式 SLA 压测。

| 小节 | 优先级 | 核心模块（Step 1~4） | 解决的上一步痛点 |
| --- | --- | --- | --- |
| 6.1 云 API 与自部署选型及显存 TCO 测算 | 【🔥 必会】 | Step 1: estimate_vram_and_concurrency | 不清楚单卡最大并发与自建 GPU 盈亏平衡点 |
| 6.2 手写 KV Cache 增量解码与 PagedAttention | 【🔥 必会】 | Step 2: IncrementalKVCacheAttention（纯手写） | 自回归重复重算 $O(T^2)$ 与连续预留碎片浪费 |
| 6.3 量化原理（INT8/INT4）与误差核验 | 【🔥 必会】 | Step 3: symmetric_quantize_int8 / audit_fidelity | 权重显存/访存带宽瓶颈与离群点量化崩盘 |
| 6.4 并发压测、TTFT/TPOT 与 P95 延迟统计 | 【🔥 必会】 | Step 4: evaluate_sla | 阶段 P95 误相加虚高与忽略长尾 Goodput |

