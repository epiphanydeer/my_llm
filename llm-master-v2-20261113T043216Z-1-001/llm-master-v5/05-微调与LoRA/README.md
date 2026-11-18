# 第五章 微调选型与 LoRA 低秩适配：渐进式面试备考全集

本章围绕**「垂类企业客服/合规微调全流程」**，从选型决策、ChatML -100 损失掩码、纯手写 LoRA 低秩注入与无损合并，一路叠加到防模板泄漏、消融归因与数据清洗脱敏。

| 小节 | 优先级 | 核心模块（Step 1~6） | 解决的上一步痛点 |
| --- | --- | --- | --- |
| 5.1 RAG 与 SFT/RLHF/DPO 选型决策 | 【🔥 必会】 | Step 1: route_rag_vs_sft | 盲目把高频更新知识硬训进权重 |
| 5.2 SFT 损失掩码与 ChatML 数据格式 | 【🔥 必会】 | Step 2: build_chatml_sft_sample | 输入全算 Loss 与漏掉 eos_id 导致推理停不下 |
| 5.3 手写 LoRA 低秩注入与 QLoRA | 【🔥 必会】 | Step 3: LoRALinear（纯手写） | 全参微调 70GB+ 显存爆炸与多业务部署冗余 |
| 5.4 微调项目防泄漏与训练排障 | 【🔥 必会】 | Step 4: SFTTrainingGuard | 同模板跨集泄漏、NaN 崩盘与过拟合背诵 |
| 5.5 微调消融与收益归因 | 【🔥 必会】 | Step 5: compare_sft_ablation | 无法证明 LoRA 相比 Prompt 的真实净收益 |
| 5.6 数据清洗、去重与合规脱敏 | 【⭐ 理解】 | Step 6: sanitize_sft_corpus | 重复脏数据复读与 PII 隐私泄漏进权重 |

