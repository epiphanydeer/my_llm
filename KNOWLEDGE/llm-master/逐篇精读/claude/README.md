# Claude学习专栏

> 重要程度：**◻ 按需查阅** · [原仓库同版本文章](https://github.com/youngyangyang04/llm-master/blob/78b03ad475409ba1d6326e0a6e8d50b27025bb64/docs/llm/claude/README.md) · 原作者：程序员Carl（MIT License，见 [LICENSE](../LICENSE)）。
> 整理方法：保留技术解释、示例、代码和面试问答；压缩开篇叙事和推广语。下方“学习补充”由本笔记新增。

## 学习补充：本专栏怎样读

**一句话：** 这里记录 Claude Code 的使用与 Agent 工程经验；先掌握上下文、工具调用和验证闭环，再按项目需要读 Skills、Hooks 与多 Agent。

**例子：** 让编码 Agent 修改一个函数时，给它任务目标、相关文件、测试命令和完成标准；跑完后查看 diff 与测试结果。这个闭环比只写“请帮我改好”更可复盘。

**费曼自述：** “编码 Agent 能读文件、改代码和跑命令；我需要约束目标与权限，并检查它留下的证据。”

**概念闪卡：** Q：什么是 Agent Loop？A：观察状态、决定动作、调用工具、检查结果，直到完成或停止。Q：何时学多 Agent？A：单 Agent 的任务边界和验证已清楚，确有可独立并行的工作时。

## 原文技术内容（整理版）

### 第一阶段：先把 Claude Code 用明白

先建立完整能力地图，避免一上来就在 Skills、Hooks、MCP 之间来回试错。

- [Claude Code高效使用指南：5件事把它调教成一个会自己干活的团队](./claude_code_efficient_guide.md)
- [Claude Code完整使用指南：CLAUDE.md、Skills、Subagents、MCP、Hooks、Plugins怎么用](./claude_code_toolkit_guide.md)
- [CLAUDE.md到底怎么写？项目记忆、团队规范和上下文管理一篇讲明白](./claude_md.md)

### 第二阶段：理解上下文和代码库

Claude Code 不是把整个仓库塞进上下文。它需要搜索、读取、压缩和缓存，也需要你给它清楚的项目入口。

- [Claude Code怎么读懂大代码库？Agent搜索、CLAUDE.md、Hooks、Skills、MCP和LSP一篇讲明白](./claude_code_large_codebase.md)
- [Claude Code为什么快？Prompt Cache、Plan Mode、MCP工具加载和上下文压缩一篇讲明白](./claude_prompt_cache.md)
- [为什么Agent时代大家都在做CLI？Claude Code、Codex与命令行的前世今生](./agent_cli.md)

### 第三阶段：从Prompt走向Agent Loop

当任务从“一次回答”变成“读取代码、执行修改、检查结果、继续修正”，重点就不再是某一句 Prompt，而是整个闭环能不能稳定运行。

- [Claude Code作者说“不写Prompt，写Loop”：AI编程从提示词到Agent闭环到底变了什么](./claude_code_loop.md)
- [深入理解Claude Code：从CLAUDE.md到Hooks、Skills、Subagents](./claude_code_extensions_evolution.md)
- [Claude Skills实战：Anthropic几百个Skill总结出的9大分类和实战经验](./claude_skills.md)
- [Claude Code动态工作流详解：让Claude自己现写一套harness](./dynamic_workflows.md)
- [Loop Engineering实战：14步路线图，从判断要不要做到上线后守住](./loop_engineering_guide.md)

### 第四阶段：团队与生产工程

最后再看规模化问题：怎么拆任务、怎么隔离执行环境、怎么验证迁移结果，以及怎样让 Agent 真正进入生产系统。

- [百万行代码两周迁完：Anthropic如何用Claude Code做大规模代码迁移](./ai_code_migration.md)
- [Claude Managed Agents详解：把Agent的“大脑”和“双手”拆开](./managed_agents.md)

### 第五阶段：看清AI编程真正改变了什么

工具会继续变，但人和Agent怎么分工、什么能力会变得更值钱，是更长期的问题。

- [40万次Claude Code会话揭示：真正拉开差距的不是Prompt](./claude_code_400k_sessions.md)

如果你刚开始用 Claude Code，从第一阶段按顺序看就够了。

如果你已经在项目里使用，可以直接从自己卡住的位置进入：规则总失效看 CLAUDE.md，仓库太大看大代码库，任务跑不稳看 Agent Loop，需要并行和隔离再看动态工作流与 Managed Agents。

