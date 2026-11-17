# 第四章 Agent 智能体系统：渐进式面试备考全集

本章围绕**「企业工单与故障诊断 Agent」**统一实战场景，从 Step 1 的 50 行极简 O-T-A 状态机起步，每节叠加一个核心工程模块，最终拼装成一套完整的工业级 Agent Harness（智能体工程脚手架）。

## 章节导航与代码演进路线图

| 小节 | 优先级 | 核心模块（Step 1~15） | 解决的上一步痛点 |
| --- | --- | --- | --- |
| 4.1 Agent 基本结构与 O-T-A 运行闭环 | 【🔥 必会】 | Step 1: run_ota_agent | 单轮问答无法多步取证与动态纠偏 |
| 4.2 Agent 与 Workflow 的边界选型 | 【🔥 必会】 | Step 2: HybridRouter | 过度 Agent 化导致确定性业务又慢又贵 |
| 4.3 Tool 设计与 Function Calling 安全沙箱 | 【🔥 必会】 | Step 3: ToolSandboxRegistry | 模型缺参、路径穿越越权与高危写操作失控 |
| 4.4 ReAct 推理循环与反思规划 | 【🔥 必会】 | Step 4: ReActEngine | 缺少 tool_call_id 协议配对与单次防重 |
| 4.5 记忆架构与上下文工程 | 【🔥 必会】 | Step 5: AtomicMemoryWindow | 上下文溢出、简单切片腰斩工具配对报 400 |
| 4.6 失败恢复、Schema 自愈与幂等控制 | 【🔥 必会】 | Step 6: ResilientExecutor | 503 抖动崩溃、Schema 报错空转与重复扣款 |
| 4.7 Agent 评测体系与 Pass^k 稳定性指标 | 【🔥 必会】 | Step 7: AgentTrajectoryEvaluator | 主观试跑漏掉随机性波动与过程冗余 |
| 4.8 MCP 协议与跨系统工具生态 | 【⭐ 理解】 | Step 8: MiniMCPServer | $M \times N$ 工具对接灾难与鉴权透传 |
| 4.9 Plan-and-Execute 与 DAG 并行调度 | 【⭐ 理解】 | Step 9: DAGTaskScheduler | 长链路串行太慢、环依赖与失败全盘重跑 |
| 4.10 多智能体角色分工与权限隔离 | 【🔥 必会】 | Step 10: MultiAgentOrchestrator | 单 Agent 自我原谅与无限返工踢皮球 |
| 4.11 多角色交接、上下文投影与全局预算治理 | 【🔥 必会】 | Step 11: HandoffProjector & TokenBudgetAllocator | 交接历史透传爆炸与并发子 Agent 超支 |
| 4.12 Browser / Web Agent 与 DOM 上下文工程 | 【⭐ 理解】 | Step 12: SafeWebExtractor | SSRF 内网探测、隐藏 DOM 注入与静默截断 |
| 4.13 Agent 能力图谱与上线就绪度审计 | 【◻ 按需】 | Step 13: SkillReadinessAuditor | 跳过基础防线盲目上线高阶多智能体 |
| 4.14 生产防线一：双模式防死循环熔断器 | 【🔥 必会】 | Step 14: DualModeLoopBreaker | A-A-A 连续空转与 A-B-A-B 乒乓震荡 |
| 4.15 生产防线二：终局事实核验与写操作回执门禁 | 【🔥 必会】 | Step 15: FactAndReceiptGate | 无证据编造数据与口头谎报写操作成功 |

