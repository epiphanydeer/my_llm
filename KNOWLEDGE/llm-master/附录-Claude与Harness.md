# 附录：Claude Code、AI 编程与 Agent harness

> 本附录对应仓库 [Claude 专栏首页](逐篇精读/claude/README.md)的 15 篇。当前优先级：◻；如果将来明确转向 Agent harness/AI 编程工具工程，再提升为 ⭐。其中具体产品命令、插件和套餐会变化，用时查看官方文档。

## A.1 项目说明与代码库读取【◻】

**一句话：** Agent 不会自动理解整个代码库；清晰的项目说明、按需搜索和能运行的测试，比把所有文件塞进上下文更有用。

**例子：** 在项目根目录说明“如何启动、测试、哪些目录存模型/数据、哪些文件不可改”，Agent 先搜索相关模块，再打开必要代码。长文档不必每轮全量重复。规则文件可以帮助工具，但不能替代真实测试。

**费曼自述：** “给 Agent 一张项目地图和可靠的验证命令，让它根据任务逐步找代码。”

**概念闪卡：** Q：项目说明文件能代替 README 吗？A：两者读者和用途可能不同；项目可保留给人看的 README 与给工具的工作规则。Q：大代码库为何先搜索？A：减少无关上下文并定位真正要改的模块。

来源：[CLAUDE.md](逐篇精读/claude/claude_md.md) · [大代码库](逐篇精读/claude/claude_code_large_codebase.md) · [高效使用](逐篇精读/claude/claude_code_efficient_guide.md)。

## A.2 Skills、Hooks、MCP 与扩展【◻】

**一句话：** Skill 封装任务方法，Hook 在事件触发时执行检查，MCP 连接外部工具/资源；它们解决不同问题，不能互相替代。

**例子：** “运行测试并解释失败”可以写成复用流程；每次提交前自动格式检查适合 Hook；读取外部文档系统可通过受控工具/MCP。是否需要这些机制取决于重复任务的频率和维护成本。

**费曼自述：** “流程说明、自动触发、外部连接是三种不同扩展点；先找到实际重复工作再加。”

**概念闪卡：** Q：Hook 与 Skill 的核心差别？A：前者按事件自动执行，后者是可选择使用的能力/流程封装。Q：MCP 直接保证数据权限吗？A：协议不替代服务端授权。

来源：[扩展演进](逐篇精读/claude/claude_code_extensions_evolution.md) · [工具指南](逐篇精读/claude/claude_code_toolkit_guide.md) · [Skills](逐篇精读/claude/claude_skills.md)。

## A.3 Loop 与 harness【◻】

**一句话：** 好的 Agent 不只靠一句 Prompt，而靠“尝试→观察结果→验证→修正→停止”的运行闭环；harness 是支撑这个循环的工具、权限、状态和验收环境。

**例子：** 让编程 Agent 修改一个接口：先定位代码，编辑，运行测试，看错误并修复，直到通过或达到停止预算。若没有测试和状态记录，Agent 可能反复改同一处。动态工作流和多 Agent 只是可选实现，不能自动提高质量。

**费曼自述：** “模型给出动作只是开始，验证反馈决定能否真正完成任务。harness 把这条反馈链变成可运行的系统。”

**概念闪卡：** Q：Loop 的停止条件有哪些？A：验收通过、失败不可恢复、预算/时间上限或需要人工输入。Q：为什么多 Agent 不一定更好？A：通信、权限和成本会增加。

来源：[Loop](逐篇精读/claude/claude_code_loop.md) · [Loop Engineering](逐篇精读/claude/loop_engineering_guide.md) · [动态工作流](逐篇精读/claude/dynamic_workflows.md) · [Managed Agents](逐篇精读/claude/managed_agents.md)。

## A.4 缓存、CLI 与案例文章【◻】

**一句话：** CLI 便于工具组合；Prompt 缓存只在符合服务规则的重复前缀上省计算；大型迁移案例提供任务拆分与验收思路，但数字与工具版本不宜背诵。

**例子：** 同一长项目规则被许多请求复用可能有缓存收益；每次变化的文件内容不能假定都命中缓存。百万行代码迁移案例可提炼“先做可判对错的测试，再扩大自动化范围”。

**费曼自述：** “从工具案例里抽取稳定的工程方法，不把某次统计或产品细节记成永久规律。”

**概念闪卡：** Q：缓存命中看什么？A：服务规则及可复用前缀。Q：自动迁移前为什么先做验收？A：没有判定标准，大规模执行会放大错误。

来源：[Prompt Cache](逐篇精读/claude/claude_prompt_cache.md) · [Agent CLI](逐篇精读/claude/agent_cli.md) · [代码迁移](逐篇精读/claude/ai_code_migration.md) · [会话研究](逐篇精读/claude/claude_code_400k_sessions.md)。

## 本附录过关

现在只需能说明“模型、工具、循环、验收”的关系。实际使用哪款 AI 编程工具时，再查对应最新文档；若将来投 harness 岗，再补系统实现和项目证据。

## 迁移页与事件页【⏭】

[intro 下的封号页](逐篇精读/intro/claude_code_china_ip_ban.md)、[大代码库页](逐篇精读/intro/claude_code_large_codebase.md)、[Loop 页](逐篇精读/intro/claude_code_loop.md)、[工具指南页](逐篇精读/intro/claude_code_toolkit_guide.md)、[CLAUDE.md 页](逐篇精读/intro/claude_md.md)、[缓存页](逐篇精读/intro/claude_prompt_cache.md)目前是“本文已迁移”的入口，正文在 claude 目录；不做重复笔记。[系统提示词泄露事件文章](逐篇精读/intro/fable5_system_prompt_leak.md)属于事件解读，只在研究该事件时回查。

**费曼自述：** “迁移页是路标，不能当独立课时；事件页提供线索，具体事实要核查一手材料。”

**概念闪卡：** Q：看到‘本文已迁移’应做什么？A：进入新正文页，不重复记两份。Q：事件文章需要背吗？A：不需要。
