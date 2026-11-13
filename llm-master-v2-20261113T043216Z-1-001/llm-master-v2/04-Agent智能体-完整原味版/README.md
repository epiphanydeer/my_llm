# 第四章 · Agent 智能体（原仓库 25 篇一字不删 + 你的仓库 15 篇全量精华 + 零基础大白话 + 15 级搭积木手搓代码）

## 一、本章重构亮点与使用指南

1. **原仓库（`llm-master`）25 篇 Agent 全家桶 100% 零删减收录（含全部 142 张原图）**：
   - 不仅包含主干篇 14 篇（`agent_intro.md` 到 `agent_learning_roadmap.md`），还通过**哈希映射表**将散落在 `docs/interview/llm/` 下的 11 篇深化与架构专题（`agent_interview.md`、`agent_hallucination_control_interview.md`、`agent_drift_hallucination_interview.md`、`agent_hybrid_routing_interview.md`、`multi_agent_communication_interview.md`、`loop_engineering_interview.md`、`graph_engineering_interview.md`、`agent_skill_interview.md`、`agent_harness_observability_interview.md`、`multi_agent_harness_interview.md`、`production_agent_architecture_harness_org_talent.md`）精准匹配并 100% 全文合入对应小节！
2. **你的仓库（`llm-master-mine`）15 篇精华 100% 全量保留**：
   - 你整理的 `目标与例子`、`概念与五要素表`、`面试追问`、`故障模式与排障`、`工程 trick 与坑`、`进阶生产级 Toy Code`、`这段代码怎么在面试里讲（三段式）`、`费曼自述与闪卡` 一字不漏全部收录。
3. **双层渐进代码体系（先搭 40 行极简积木，再看工业级封装）**：
   - 专门针对“能手搓 Transformer 但直接看 150 行复杂工程类容易懵”的痛点，每一节第三部分先给出一份**40 行左右、只用基础字典和函数的“搭积木版手搓代码”**（从 4.1 最小闭环 -> 4.2 混合路由 -> 4.3 Schema 沙箱 -> 4.4 ReAct 循环 -> 4.5 原子记忆裁剪 -> 4.6 幂等退避重试……每节只长出 1 个模块），紧接着第四部分对照进阶工业级代码，全部通过 `python3` 自动化运行验证！
4. **零删除线 Bug、纯正原生 Markdown（`.md`）**：
   - 全章已清除所有波浪号 `到` 引起的删除线渲染错误，并以原生 `.md` 格式保存在云盘文件夹中。

## 二、第四章全量哈希对账表（`llm-master` 25 篇 + `llm-master-mine` 15 篇 100% 映射清单）

| 节次与文件名 | 对应原仓库（`llm-master`）源文件路径（共 25 篇） | 原图数 | 对应你的仓库（`llm-master-mine`）源文件 | 手搓积木演进（Level 1 到 15） |
| --- | --- | --- | --- | --- |
| `4.1-Agent基本结构.md` | `docs/llm/app/agent_intro.md` | 5 张 | `4.1-Agent基本结构.md` | Level 1：40行最小 O-T-A 闭环状态机 |
| `4.2-工作流与Agent.md` | `docs/llm/app/agent_vs_workflow.md` + `docs/interview/llm/agent_hybrid_routing_interview.md` | 11 张 | `4.2-工作流与Agent.md` | Level 2：看菜下碟混合路由分诊台 |
| `4.3-工具设计与FunctionCalling.md` | `docs/llm/app/function_calling.md` + `docs/llm/app/agent_tool_design.md` + `docs/interview/llm/agent_skill_interview.md` | 17 张 | `4.3-工具设计与FunctionCalling.md` | Level 3：Schema 契约校验与身份注入沙箱 |
| `4.4-ReAct与Loop工程.md` | `docs/llm/app/react_reflection_planning.md` + `docs/interview/llm/loop_engineering_interview.md` | 7 张 | `4.4-ReAct与Loop工程.md` | Level 4：`tool_call_id` 配对与防重 ReAct 循环 |
| `4.5-记忆与上下文管理.md` | `docs/llm/app/agent_memory.md` | 9 张 | `4.5-记忆与上下文管理.md` | Level 5：原子轮次打包与高低水位线记忆裁剪 |
| `4.6-失败模式与恢复.md` | `docs/llm/app/agent_failure_modes.md` | 8 张 | `4.6-失败模式与恢复.md` | Level 6：错误分级、幂等键防重与指数退避 |
| `4.7-Agent评测与可观测性.md` | `docs/llm/app/agent_evaluation.md` + `docs/interview/llm/agent_harness_observability_interview.md` | 17 张 | `4.7-Agent评测与可观测性.md` | Level 7：轨迹效率指标与 `Pass^k` 可靠性核算 |
| `4.8-MCP协议详解.md` | `docs/llm/app/mcp_protocol.md` | 9 张 | `4.8-MCP协议详解.md` | Level 8：MCP JSON-RPC 2.0 协议服务端分发器 |
| `4.9-DAG依赖与图编排.md` | `docs/llm/app/plan_execute_dag.md` + `docs/interview/llm/graph_engineering_interview.md` | 6 张 | `4.9-DAG依赖与图编排.md` | Level 9：Kahn 拓扑排序防死锁与并行批次调度 |
| `4.10-多角色分工与通信编排.md` | `docs/llm/app/multi_agent_roles.md` + `docs/interview/llm/multi_agent_communication_interview.md` | 9 张 | `4.10-多角色分工与通信编排.md` | Level 10：Worker-Reviewer 契约核验与返工熔断 |
| `4.11-多角色交接与Harness治理.md` | `docs/llm/app/multi_agent_context_governance.md` + `docs/interview/llm/multi_agent_harness_interview.md` | 12 张 | `4.11-多角色交接与Harness治理.md` | Level 11：白名单字段投影与隐私隔离交接器 |
| `4.12-网页内容提取与BrowserAgent.md` | `docs/llm/app/browser_agent_dom_context.md` | 3 张 | `4.12-网页内容提取与BrowserAgent.md` | Level 12：SSRF 拦截、DOM 脱水与防间接注入清洗 |
| `4.13-Agent学习路线与生产全景.md` | `docs/llm/app/agent_learning_roadmap.md` + `docs/interview/llm/production_agent_architecture_harness_org_talent.md` | 7 张 | `4.13-Agent学习路线与完成证据.md` | Level 13：技能拓扑树依赖推导与简历完备度审计 |
| `4.14-Agent大厂高频面试题汇总.md` | `docs/interview/llm/agent_interview.md` | 12 张 | `4.14-Agent面试：工具、停止条件与排障.md` | Level 14：A-A-A 连续重复与 A-B-A-B 振荡双模熔断 |
| `4.15-Agent幻觉控制与上下文漂移.md` | `docs/interview/llm/agent_hallucination_control_interview.md` + `docs/interview/llm/agent_drift_hallucination_interview.md` | 11 张 | `4.15-Agent幻觉控制：证据、动作与兜底.md` | Level 15：防漂移尾部锚点、事实核验与真实回执门控 |
