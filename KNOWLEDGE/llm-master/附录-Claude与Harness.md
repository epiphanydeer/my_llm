# 附录：Claude Code 与 Agent Harness——把工具调用做成可验证的工作流

> 本附录按问题链路合并原仓库 Claude 专栏；这是 **◻ 按需深化**，当前学习主线仍是 Python、模型、RAG 和多模态项目。Claude 特定命令与产品细节会变，记住工具权限、上下文管理、执行循环、验证和可观察性。

## 本章主线

Harness 是围绕模型运行任务的控制层：输入任务→获取上下文→选择工具→执行→观察结果→验证→继续或停止。它还负责权限、失败恢复、日志和成本。模型权重决定一部分能力；Harness 决定能力如何在真实系统里受控地执行。面试时先解释任务闭环，再谈具体产品名。

~~~text
任务 → 计划/选择动作 → 工具执行 → 观察 → 验证
                    ↑             ↓
                    └── 未完成且未超预算 ──┘
~~~

**动手代码：** 用 Python 写一个最多 3 步的只读工具循环；工具调用前校验名字与参数，调用后记录输入/输出/错误，达到完成条件或预算时退出。把一次失败轨迹保存成可复盘的日志。

## 本章目录

- [附.1 agent_cli](#a1)
- [附.2 claude_code_efficient_guide](#a2)
- [附.3 claude_code_toolkit_guide](#a3)
- [附.4 claude_md](#a4)
- [附.5 claude_prompt_cache](#a5)
- [附.6 claude_code_large_codebase](#a6)
- [附.7 claude_code_loop](#a7)
- [附.8 claude_skills](#a8)
- [附.9 dynamic_workflows](#a9)
- [附.10 loop_engineering_guide](#a10)
- [附.11 managed_agents](#a11)
- [附.12 ai_code_migration](#a12)
- [附.13 claude_code_extensions_evolution](#a13)
- [附.14 claude_code_400k_sessions](#a14)

<a id="a1"></a>

## 附.1 为什么Agent时代大家都在做CLI？Claude Code、Codex与命令行的前世今生【◻】

**一句话：** CLI 便于工具组合；Prompt 缓存只在符合服务规则的重复前缀上省计算；大型迁移案例提供任务拆分与验收思路，但数字与工具版本不宜背诵。

**例子：** 同一长项目规则被许多请求复用可能有缓存收益；每次变化的文件内容不能假定都命中缓存。百万行代码迁移案例可提炼“先做可判对错的测试，再扩大自动化范围”。

### CLI 的前世今生：它从来不只是“黑框里敲字”

先把三个词掰清楚：Terminal 是承载输入输出的终端，Shell 是解释命令的程序，CLI 是用文本命令和软件交互的方式。Claude Code、Codex CLI 今天已经是带面板、状态和快捷键的 TUI，但它们真正依赖的底座，仍然是 Shell、文件系统和一整套命令行工具。

最早的计算机甚至谈不上命令行。人把程序打在卡片上，交给机器批处理，结果晚点再拿回来。**人和机器之间没有实时对话。**

后来分时系统和终端出现，人终于可以输入一行命令，马上看到一行结果。到了 Unix，Shell 又做了一次关键升级：它不只是启动程序的入口，还是一门可以组合程序的控制语言。

`stdin`、`stdout`、`stderr`、退出码、重定向、管道，这几个设计看起来朴素，却把一堆小工具接成了工作流。1978 年 Bourne 对 Unix Shell 的定义里，就已经同时强调了控制流、环境、重定向和进程管道。

GUI 兴起后，命令行没有消失。因为 GUI 解决的是**人的发现成本**：按钮摆在那里，不会写命令也能点。CLI 解决的是**系统的组合成本**：一个命令能进脚本、进 CI、跑远程机器，还能把输出交给下一个程序。

到了云原生时代，Git、Docker、Kubernetes、Terraform、各种云平台，几乎都把 CLI 当成一等公民。原因很简单：服务器没有必要配一块屏幕，自动化也不会拿鼠标点按钮。

Agent 时代，命令行又转了一圈回来。但这次敲命令的主角，开始从人变成模型。

![](https://file1.kamacoder.com/i/web/2026-07-16_10-48-27.jpg?v=20261005)

这张图回答的是：CLI 为什么在 GUI 时代没有死，到了 Agent 时代反而又站到前台。变化的不是命令行突然变好用了，而是“把人的意图翻译成精确语法”这份苦活，从人转移给了 Agent；CLI 原本最劝退人的门槛，恰好被大模型吃掉了。

### Agent + CLI，为什么是绝配？

很多文章会说“CLI 快、轻量、极客”。太浅了。**Agent 选择 CLI，不是因为终端酷，而是 CLI 天然提供了一套可执行、可观察、可组合的反馈系统。**

你让 Claude Code 修一个登录 Bug，它不是吐一段代码就完事，而是可以顺着仓库继续干：用搜索命令定位调用链，读取文件，修改代码，跑单测，看报错，再读 diff，直到证据说明任务完成。

这里最值钱的不是某一条命令，而是 CLI 把整个软件工具链压成了统一结构：

- **输入是文本或文件**：需求、路径、参数都能明确传入；
- **动作是命令**：Git、编译器、测试框架、数据库和云平台都能被调用；
- **反馈是证据**：标准输出、错误输出、退出码、测试报告和 diff 都能回到上下文；
- **组合靠管道和脚本**：一次成功的交互，很容易固化成可重复执行的流程；
- **边界由操作系统兜底**：工作目录、文件权限、网络策略、容器和沙箱能限制 Agent 的爆炸半径。

这就形成了一个完整闭环。

![](https://file1.kamacoder.com/i/web/2026-07-16_10-56-58.jpg?v=20261005)

这张图回答的是：Agent 为什么不能只有“大脑”。CLI 把文件系统和现成工具链接成它的“双手”，stdout、stderr、退出码和 diff 又变成“眼睛”；验证失败不是一句模糊的“不对”，而是下一轮推理能直接消费的证据。

更狠的一点是，CLI 还是**可递归的**。Agent 不只会调用 CLI，它自己也能变成 CLI 的一环：

```bash
cat build-error.txt | claude -p "定位根因并给出修复建议"
git diff | codex exec "检查这次修改是否引入并发问题"
```

交互工具因此变成了脚本组件，可以进 CI、定时任务和无人值守流程。网页聊天框很难自然做到这一点。

### GUI 输了吗？没有，只是它更适合人

GUI 最大的优势，是把功能摊开给人看。设计稿、数据大盘、复杂表格、可视化调试，这些场景让人直接看，比读几百行文本舒服得多。

但 Agent 操作 GUI，很多时候是在“猜像素”：按钮换个位置、弹窗多一层、页面加载慢一点，执行路径就变了。它当然可以靠 Computer Use 看屏幕、移动鼠标，但这种通道**延迟高、状态隐蔽、难并行、难精确复现**。

CLI 则把动作命名了。`git status` 不会因为窗口缩放就跑到右上角，退出码 `1` 也比截图里一行红字更容易判断。

所以真正的分工，不是 CLI 干掉 GUI，而是分成两层：**人用 App、IDE、Web 看计划、审 diff、做授权；Agent 在下面用 CLI、API、MCP 执行和取证。**

![](https://file1.kamacoder.com/i/web/2026-07-16_10-58-09.jpg?v=20261005)

这张图回答的是：IDE、App 和 CLI 到底谁替代谁。上层界面负责把复杂状态讲给人听，下层工具负责把确定动作交给 Agent；审批和证据在两层之间往返，而不是逼人永远盯着终端，也不是让 Agent 永远点像素。

### 但今天很多 CLI，并没有为 Agent 准备好

别因为 Agent 能跑命令，就觉得所有 CLI 都天然可靠。大量老工具默认操作者是人，会弹交互确认、输出彩色进度条、把错误混进 stdout，甚至失败了还返回 `0`。人能凭经验看懂，Agent 接进流水线就容易翻车。

真正 Agent 友好的 CLI，至少要做到几件事：

- 支持 `--json` 或稳定 Schema，不逼模型从花哨日志里猜字段；
- stdout 放结果，stderr 放诊断，退出码真实反映成功或失败；
- 提供 `--dry-run`、diff 和幂等操作，让 Agent 能先预演、失败后安全重试；
- 支持非交互模式、超时和取消，不要半夜卡在“是否继续？Y/n”；
- 权限最小化，凭证可按任务、目录、命令和时间收口；
- 输出可审计，明确谁执行了什么、改了哪里、依据是什么。

**未来 CLI 的竞争，不只是给人写得顺不顺手，还要看它能不能被 Agent 稳定调用。** 结构化输出会越来越重要，权限和沙箱也会从附加功能变成默认能力。

这也是为什么 Claude Code 和 Codex 都在权限模型上花重力气。Agent 越能干，误操作和提示注入的爆炸半径越大。没有边界的 CLI，不是自动化，是把生产事故也自动化。

### CLI 是终局吗？

我的判断很明确：**终端不是终局，CLI 也不会成为所有人的最终界面。**

当 Agent 从一次处理一个任务，变成同时管理十几个长期任务，纯终端很快就不够用了。你需要任务队列、并行状态、权限面板、diff 审核、通知和跨设备接力。OpenAI 后来做 Codex App，本身就在说明：终端适合启动和执行，但不一定适合人监督一支 Agent 队伍。

CLI 也不是所有工具的最佳协议。API 和 MCP 有明确 Schema，通常比解析自然语言日志更稳；Computer Use 则负责那些既没有 API、也没有 CLI 的最后一公里。

但 CLI 不会消失。它会更像 Agent 世界里的“窄腰层”：上面可以是 IDE、桌面 App、网页、手机，下面可以接 Git、测试、数据库、容器和云服务，中间都能通过命令和结构化结果完成交接。

![](https://file1.kamacoder.com/i/web/2026-07-16_10-58-57.jpg?v=20261005)

这张图回答的是：CLI 如果不是最终界面，为什么仍然会长期存在。未来不会由一个入口通吃，而是“人类监督层—Agent 编排层—工具执行层”三层分工；CLI 是执行层最成熟的一条通道，但会和 API、MCP、Computer Use 长期共存。

所以，真正的终局不是 CLI。

**终局是所有软件都长出一套 Agent 能调用、能验证、能审计、也能被安全关住的接口。**

CLI 只是最早准备好的那一个。

### 参考链接

- Anthropic 官方发布（Claude 3.7 Sonnet 与 Claude Code 研究预览）：https://www.anthropic.com/news/claude-3-7-sonnet
- Anthropic 官方产品页（Claude Code 跨工具链执行）：https://www.anthropic.com/product/claude-code
- Anthropic 工程博客（Claude Code 沙箱与权限边界）：https://www.anthropic.com/engineering/claude-code-sandboxing
- Claude Code 官方文档（CLI 与非交互模式）：https://code.claude.com/docs/en/cli-reference
- OpenAI 官方发布（Codex CLI 4 月上线、Codex Web 5 月上线）：https://openai.com/index/introducing-upgrades-to-codex/
- OpenAI 官方发布（Codex App 与多 Agent 监督界面）：https://openai.com/index/introducing-the-codex-app/
- OpenAI 官方仓库（Codex CLI）：https://github.com/openai/codex
- Bell Labs 论文（The UNIX Shell）：https://www.nokia.com/bell-labs/publications-and-media/publications/unix-time-sharing-system-the-unix-shell/

### 本节收尾：费曼自述与闪卡

**费曼自述：** “从工具案例里抽取稳定的工程方法，不把某次统计或产品细节记成永久规律。”

**概念闪卡：** Q：缓存命中看什么？A：服务规则及可复用前缀。Q：自动迁移前为什么先做验收？A：没有判定标准，大规模执行会放大错误。

原文核对：[为什么Agent时代大家都在做CLI？Claude Code、Codex与命令行的前世今生](原仓库Markdown/docs/llm/claude/agent_cli.md)。

---

<a id="a2"></a>

## 附.2 Claude Code 高效使用指南：别把它当聊天框，5 件事把它调教成一个会自己干活的团队【◻】

**一句话：** Agent 不会自动理解整个代码库；清晰的项目说明、按需搜索和能运行的测试，比把所有文件塞进上下文更有用。

**例子：** 在项目根目录说明“如何启动、测试、哪些目录存模型/数据、哪些文件不可改”，Agent 先搜索相关模块，再打开必要代码。长文档不必每轮全量重复。规则文件可以帮助工具，但不能替代真实测试。

### 第一件事：别让它每次都「从零认识你的项目」

新会话里的 Claude Code，是**失忆**的。它不知道你用什么技术栈、不知道你的代码规范、不知道这个项目哪几个坑不能踩。

你每次都口头交代一遍，就是在重复劳动。**正确的做法是把这些写进 `CLAUDE.md`**——放在项目根目录的一个 markdown 文件，Claude Code 每次开会话都会先读它。编码规范、架构决策、常用命令、踩过的坑，全写进去。

光这一步，就能把「它写出来的东西不符合我项目习惯」这类问题砍掉一大半。

更省心的是，Claude 干活时还会自己攒**自动内存**——构建命令、调试发现这些东西，它会跨会话记住，不用你写。怎么把 CLAUDE.md 写得既管用又不臃肿，[这一篇](原仓库Markdown/docs/llm/claude/claude_md.md)讲得很细，这里不重复。

**一句话：CLAUDE.md 是你给 Claude 的「项目说明书」，写一次，每个会话都受益。**

### 第二件事：把「每次都要解释一遍的流程」封成 Skill

CLAUDE.md 解决的是「always-on 的项目规则」，但有些活儿是**特定场景才用**的——比如「发版前怎么写 release notes」「我们团队的 PR 评审清单」「部署到 staging 的固定步骤」。

这些东西塞进 CLAUDE.md，会把它撑爆、还白占上下文。**它们该被封成 Skill。**

Skill 是**按需加载的专项能力**：平时不占上下文，Claude 判断这次任务用得上，才把它调起来，像 `/review-pr`、`/deploy-staging` 这样一键触发。而且 Skill 不是一个 markdown 文件，是一个**文件夹**——可以带脚本、带模板、带参考资料。

Anthropic 内部已经在用**几百个 Skill**，他们总结的实战经验我单独写过一篇 [Claude Skills 实战](原仓库Markdown/docs/llm/claude/claude_skills.md)，想认真用 Skill 的录友建议读一下。

**一句话：重复三次以上的流程，就该封成 Skill，让团队共享。**

### 第三件事：把「每次都得手动做的事」交给 Hooks 自动跑

有些事不是「让 Claude 决定做不做」，而是**每次都必须做**——比如每次改完文件自动格式化、提交前必须跑一遍 lint。

这种事靠你盯着、靠提醒 Claude，都不靠谱。**这是 Hooks 的活儿。**

Hooks 让你在 Claude Code 动手**之前或之后**自动执行 shell 命令。配一条「文件编辑后自动 format」、一条「commit 前先 lint」，从此这些动作脱离「人的注意力」，变成确定性发生的流程。

**Skill 和 Hooks 的区别记住这一句：Skill 是「需要时才调的能力」，Hooks 是「每次都确定触发的动作」。**

### 第四件事：大活儿别一个人扛——子代理和动态工作流

深度调研、安全审计、大范围重构这种活儿，让**一个** Claude 从头干到尾，上下文很快就撑满，还容易跑偏、偷懒、自夸。

Claude Code 的解法是**开一队 Claude 并行**：

- **子代理（sub-agents）**：主代理把任务拆成几块，分给多个子代理同时干，最后自己汇总合并。每个子代理有独立上下文，互不污染。
- **动态工作流（Dynamic Workflows）**：更进一步，Claude 针对你**当前这个任务**，现场写一套专属的编排（扇出汇总、对抗验证、锦标赛等模式），把活儿拆给一队 Claude 去跑。

这套「为什么要拆、怎么拆」的工程思路，我在 [Managed Agents](原仓库Markdown/docs/llm/claude/managed_agents.md) 和 [动态工作流详解](原仓库Markdown/docs/llm/claude/dynamic_workflows.md) 两篇里讲透了。

**一句话：上下文是稀缺资源，活儿大就拆开并行，别让一个 Claude 硬扛。**

### 第五件事：跳出聊天框——CLI 管道 + 定时任务

很多录友只在「交互式聊天」里用 Claude Code，其实它遵循 Unix 哲学，**是可以管道化、能塞进脚本和 CI 的**。

用 `claude -p`（print 模式）就能把它当成命令行里的一环：

```bash
# 盯日志，发现异常就 Slack 我
tail -200 app.log | claude -p "如果发现任何异常就 Slack 通知我"

# 批量审查改动文件的安全问题
git diff main --name-only | claude -p "审查这些改动文件有没有安全问题"
```

再往上一层是**定时任务**，让重复的活儿自动发生：早晨 PR 审查、夜里跑 CI 失败分析、每周依赖审计。

- 终端里临时轮询，用 [`/loop`](原仓库Markdown/docs/llm/claude/claude_code_loop.md)（这背后就是「写 Loop 不写 Prompt」那套思路）；
- 想要**电脑关机也照跑**的，用 **Routines**——它跑在 Anthropic 托管的基础设施上，还能被 API 调用或 GitHub 事件触发。

**一句话：能用一句话提问，也能塞进管道和定时任务无人值守地跑——后者才是效率的天花板。**

### 最后：换个地方接着干

Claude Code 不绑死在一个界面。终端、VS Code / JetBrains、桌面 App、网页，背后是**同一个引擎**——你的 CLAUDE.md、设置、MCP 服务器在哪儿都通用。

离开工位，用手机或浏览器**远程接管**正在跑的会话；网页或 iOS 上起一个长任务，回头用 `claude --teleport` 把它拽回终端继续。**工作跟着你走，不跟着设备走。**

说到底，高效用 Claude Code 就一条主线：**把"每次都要重复的东西"一层层固化下来**——项目规则固化进 CLAUDE.md，专项流程固化成 Skill，必做动作固化成 Hooks，大任务固化成并行编排。固化得越多，你要操的心越少。

### 参考链接

- Claude Code 官方文档（概述）：https://code.claude.com/docs/zh-CN/overview
- Claude Code 官方文档（常见工作流）：https://code.claude.com/docs/zh-CN/common-workflows

### 本节收尾：费曼自述与闪卡

**费曼自述：** “给 Agent 一张项目地图和可靠的验证命令，让它根据任务逐步找代码。”

**概念闪卡：** Q：项目说明文件能代替 README 吗？A：两者读者和用途可能不同；项目可保留给人看的 README 与给工具的工作规则。Q：大代码库为何先搜索？A：减少无关上下文并定位真正要改的模块。

原文核对：[Claude Code 高效使用指南：别把它当聊天框，5 件事把它调教成一个会自己干活的团队](原仓库Markdown/docs/llm/claude/claude_code_efficient_guide.md)。

---

<a id="a3"></a>

## 附.3 Claude Code完整使用指南：CLAUDE.md、Skills、Subagents、MCP、Hooks、Plugins怎么用【◻】

**一句话：** Skill 封装任务方法，Hook 在事件触发时执行检查，MCP 连接外部工具/资源；它们解决不同问题，不能互相替代。

**例子：** “运行测试并解释失败”可以写成复用流程；每次提交前自动格式检查适合 Hook；读取外部文档系统可通过受控工具/MCP。是否需要这些机制取决于重复任务的频率和维护成本。

### 一、为什么只会和 Claude Code 聊天，迟早会乱？

刚开始用 Claude Code，临时聊天完全够用。

“这个项目用 pnpm。”

“改完记得跑单测。”

“不要直接改数据库。”

“发版前按我们的清单检查。”

问题是，真实项目不会只聊三轮。

Claude 要读几十个文件、跑一堆命令、处理测试失败，还可能经历上下文压缩。**你临时说过的话，会和越来越多的新信息挤在同一个上下文里。**

团队协作更麻烦。

你告诉过 Claude 的规则，换个同事、换台电脑、换个项目，又要从头再讲一遍。

![临时聊天规则被新消息淹没](https://file1.kamacoder.com/i/web/20260729153229_claude_code_toolkit_02_background-compressed.jpg?v=20261005)

这张图讲的是：**聊天适合表达当前任务，不能承载整套工程制度。** 规则只存在聊天里，任务一长就容易被稀释；经验只存在个人会话里，团队就无法复用。

所以 Claude Code 的扩展能力，本质上都在回答下面六个问题：

| 问题 | 应该用什么 |
|---|---|
| 每次进入项目都应该知道什么？ | `CLAUDE.md` |
| 某类任务应该按什么方法做？ | Skills |
| 哪些工作需要独立上下文或独立角色？ | Subagents |
| 怎么访问仓库外的系统和数据？ | MCP |
| 哪些动作命中事件就必须执行？ | Hooks |
| 怎么把整套能力发给其他项目和队友？ | Plugins |

先记住一句：

**CLAUDE.md 管常驻规则，Skill 管按需方法，Subagent 管独立干活，MCP 管外部连接，Hook 管确定性动作，Plugin 管打包分发。**

下面逐个拆。

### 二、CLAUDE.md：先让 Claude 知道“这个项目怎么干”

#### 1. CLAUDE.md 是什么？

`CLAUDE.md` 是 Claude Code 会自动读取的项目说明文件。

它适合放**稳定、长期、几乎每次任务都用得上的信息**：

- 项目用什么技术栈；
- 构建、测试、格式化命令；
- 核心目录和模块边界；
- 编码规范和命名约定；
- 哪些文件不能改；
- 团队已经踩过、又很容易重踩的坑。

你可以把它理解成给新成员看的“上岗说明书”。

README 面向人，重点是项目是什么、怎么启动。

`CLAUDE.md` 面向 Agent，重点是**接到任务以后应该怎么行动**。

![CLAUDE.md项目说明书](https://file1.kamacoder.com/i/web/20260729153231_claude_code_toolkit_03_claude_md-compressed.jpg?v=20261005)

这张图讲的是：长期规则不该靠程序员一遍遍口头提醒。把命令、架构和禁区写成项目说明书，Claude 每次进项目就有稳定起点。

#### 2. CLAUDE.md 放在哪里？

最常用的四个位置是：

| 位置 | 作用范围 | 适合放什么 |
|---|---|---|
| `~/.claude/CLAUDE.md` | 你本机的所有项目 | 个人偏好、通用工作习惯 |
| `项目根目录/CLAUDE.md` | 当前项目，团队共享 | 技术栈、命令、全局规则 |
| `项目根目录/CLAUDE.local.md` | 当前项目，仅自己使用 | 本地地址、个人测试习惯 |
| `某个子目录/CLAUDE.md` | 进入该模块时按需加载 | 模块命令、局部架构和禁区 |

项目级文件应该提交到 Git。

`CLAUDE.local.md` 通常加入 `.gitignore`，不要把个人环境信息发给全团队。

大仓库不要把所有模块规则都塞进根文件。根目录管全局，前端、支付、数据等模块各放自己的 `CLAUDE.md`。

#### 3. CLAUDE.md 怎么写？

先从最小版本开始：

```md
### Project instructions

### Commands

- Install: `pnpm install`
- Test: `pnpm test`
- Build: `pnpm build`

### Architecture

- `apps/web/` is the frontend.
- `services/api/` is the API service.
- Shared types live in `packages/types/`.

### Rules

- Use pnpm, never npm.
- Add tests for every bug fix.
- Do not edit generated files under `dist/`.
- Never run database migrations without explicit approval.
```

判断一条内容该不该写进去，问三个问题：

1. 以后还会反复用到吗？
2. 它会改变 Claude 的行动吗？
3. 它是否值得每个会话都占上下文？

三个答案都是“是”，再写。

发布清单、安全审查步骤、几十页 API 文档，不要全塞进来。那是 Skill 的活。

#### 4. CLAUDE.md 不是安全边界

这一点很重要。

`CLAUDE.md` 是行为指导，不是强制执行器。

“不要执行危险命令”写在里面，能降低误操作概率，但不能提供绝对保证。

**真正必须阻止的动作，要交给权限设置或 `PreToolUse` Hook。**

如果 Claude 没按规则做，先运行 `/memory`，确认文件是否真的加载，再检查是不是规则太模糊、太长或者彼此冲突。

更完整的拆分方法，可以继续看 [CLAUDE.md 项目记忆与上下文管理](原仓库Markdown/docs/llm/claude/claude_md.md)。

### 三、Skills：把重复方法做成“按需工具箱”

#### 1. Skill 解决什么问题？

假设团队有一套发布流程：

1. 检查工作区；
2. 跑测试；
3. 生成变更说明；
4. 检查数据库变更；
5. 给出回滚方案。

这套流程很重要，但你写普通业务代码时不需要它。

如果塞进 `CLAUDE.md`，Claude 每个会话都要背着它。

**Skill 的价值，就是只在相关任务出现时加载专项知识和流程。**

它既可以由你用 `/skill-name` 主动触发，也可以由 Claude 根据 `description` 自动判断。

![Claude Code按需加载Skill](https://file1.kamacoder.com/i/web/20260729153233_claude_code_toolkit_04_skills-compressed.jpg?v=20261005)

这张图讲的是：Skill 不是把所有资料永久堆在桌面，而是面对发布、评审、排障等具体任务时，只拿当前需要的工具箱。

#### 2. 一个 Skill 长什么样？

项目级 Skill 放在：

```text
.claude/skills/release-check/
├── SKILL.md
├── checklist.md
└── scripts/
    └── verify.sh
```

其中 `SKILL.md` 是入口：

```md
---
name: release-check
description: Check whether the current changes are ready for release. Use for release preparation, pre-deployment review, or rollback planning.
disable-model-invocation: true
---

### Release check

1. Read `checklist.md`.
2. Inspect the current git diff.
3. Run `scripts/verify.sh`.
4. Report blockers, risks, and rollback steps.

Do not deploy. Only produce a release-readiness report.
```

这里有几个关键点：

- `description` 不是写给人看的广告，而是告诉 Claude **什么时候该加载它**；
- `disable-model-invocation: true` 表示只能由用户主动调用；
- 大段参考资料放到单独文件，`SKILL.md` 只做入口和导航；
- 脚本负责确定性检查，Claude 负责结合现场解释结果。

调用时输入：

```text
/release-check
```

个人通用 Skill 可以放在 `~/.claude/skills/`，项目共享 Skill 放在 `.claude/skills/`。

#### 3. 什么时候该做 Skill？

最实用的信号不是“这个知识很高级”，而是：

- 同一段说明已经复制了三次；
- `CLAUDE.md` 里某一节越来越像操作手册；
- 一个任务有固定输入、步骤和输出格式；
- 这套方法需要模板、示例或脚本；
- 多个角色都要复用同一套知识。

Skill 不是越多越好。

描述写得过宽，会在不相关任务里乱触发；内容太大，加载后一样会挤占上下文。

关于 Skill 的分类、渐进式披露和团队实践，可以继续看 [Claude Skills 实战](原仓库Markdown/docs/llm/claude/claude_skills.md)。

### 四、Subagents：把复杂工作交给独立角色

#### 1. 为什么有 Skill 还需要 Subagent？

Skill 给当前 Agent 加一套方法，但工作仍然发生在**当前上下文**。

安全审查、全仓探索、测试排查这类任务，会读很多文件、产生大量中间输出。

如果全部塞进主会话，主 Agent 的上下文很快变脏。

更麻烦的是，让写代码的 Agent 再审自己的代码，容易产生“我已经改好了，所以应该没问题”的自我认可。

Subagent 的价值是：**另开一个独立上下文，让一个专门角色完成任务，只把结论和证据带回来。**

![Claude Code Subagents独立分工](https://file1.kamacoder.com/i/web/20260729153234_claude_code_toolkit_05_subagents-compressed.jpg?v=20261005)

这张图讲的是：探索、测试和审查可以由独立角色分别完成。过程噪音留在各自上下文，主 Agent 只接收有证据的结果。

#### 2. 怎么创建一个 Subagent？

项目级 Subagent 放在 `.claude/agents/`：

```md
---
name: security-reviewer
description: Review code changes for authentication, authorization, injection, secret exposure, and unsafe data handling.
tools: Read, Grep, Glob, Bash
model: inherit
skills:
  - secure-coding
---

You are an independent security reviewer.

Review changes without modifying business code.

For every finding, report:

1. file and location;
2. trigger condition;
3. impact;
4. evidence;
5. recommended fix.

If no issue is found, list what you checked. Do not return only "looks good".
```

使用时可以直接说：

```text
让 security-reviewer 独立审查这次登录模块改动，只返回带证据的结论。
```

也可以运行 `/agents` 查看和管理。

#### 3. 什么任务适合拆 Subagent？

适合拆的任务通常有一个共同点：**过程很长，但主会话只需要结果。**

例如：

- 只读探索大代码库；
- 安全、性能、测试独立复核；
- 多个互不依赖模块的并行调查；
- 需要不同角色从相反角度挑错；
- 希望给角色限制工具、模型或权限。

不适合拆的情况也很明显：

- 一个文件的小修改；
- 强依赖主会话大量隐含信息；
- 多个子任务不断互相等待；
- 拆分成本比任务本身还高。

**Skill 是方法，Subagent 是拿着方法独立干活的角色。**

两者可以组合：在 Subagent 的 `skills` 字段里预加载安全规范或测试方法。

### 五、MCP：让 Claude Code 接上仓库外的世界

#### 1. MCP 到底是什么？

Claude Code 自带的文件和终端工具，擅长处理当前工作区。

但真实开发还要访问很多外部系统：

- GitHub 或其他代码托管平台；
- Jira、Linear 等工单系统；
- Sentry 和日志平台；
- 数据库、数据仓库；
- 内部文档、发布平台和业务后台；
- 浏览器或设计工具。

MCP，全称 Model Context Protocol，解决的是**AI 应用如何用统一方式发现和调用外部工具与数据**。

你不需要给每个 Agent 单独手写一套集成。

MCP Server 把能力暴露出来，Claude Code 作为 MCP Client 连接并调用。

![Claude Code通过MCP连接外部系统](https://file1.kamacoder.com/i/web/20260729153236_claude_code_toolkit_06_mcp-compressed.jpg?v=20261005)

这张图讲的是：MCP 像受控转接器，让 Claude Code 够到仓库外的代码平台、数据库和浏览器；权限钥匙仍然应该掌握在人和系统策略手里。

#### 2. 怎么给 Claude Code 添加 MCP Server？

添加远程 HTTP Server：

```bash
claude mcp add --transport http issue-tracker https://mcp.example.com/mcp
```

添加本地 stdio Server：

```bash
claude mcp add --transport stdio my-tools -- node ./tools/mcp-server.js
```

常用检查命令：

```bash
claude mcp list
claude mcp get issue-tracker
```

进入 Claude Code 后，用 `/mcp` 查看连接和认证状态。

团队共享的项目级 MCP 配置可以写进 `.mcp.json`。从仓库拉下来的配置需要经过信任和审批，不要因为文件进了 Git 就默认它安全。

#### 3. MCP 最容易踩什么坑？

**第一，权限给太大。**

查询数据库就用只读账号，不要上来就给生产库写权限。

代码评审只需要读 PR，就别顺手给删仓库权限。

**第二，工具太多。**

一口气接几十个 Server、暴露几百个工具，会增加选择成本，也会污染上下文。

先接最常用、返回结果最干净的能力。

**第三，把 MCP 当成工作方法。**

MCP 只负责“能连接、能调用”。

至于查询什么、怎么判断、输出什么格式，仍然应该交给 `CLAUDE.md`、Skill 或 Subagent。

所以最常见的组合是：

**MCP 提供手和眼，Skill 提供做事方法。**

想继续理解协议层，可以看 [MCP 协议详解](原仓库Markdown/docs/llm/app/mcp_protocol.md)。

### 六、Hooks：让关键动作不再依赖“记得”

#### 1. Hook 和提醒 Claude 有什么区别？

你在 `CLAUDE.md` 里写：

“每次改完文件都要格式化。”

这是提醒。

你配置 `PostToolUse` Hook，在编辑成功后自动运行格式化命令。

这是确定性动作。

Hooks 会在 Claude Code 生命周期的特定事件上触发，比如：

- 会话开始时注入环境信息；
- 工具执行前检查或拦截；
- 文件修改后自动格式化；
- 工具失败后记录诊断信息；
- Claude 等待确认时发桌面通知；
- 上下文压缩前保存关键状态。

![Claude Code Hooks自动门禁](https://file1.kamacoder.com/i/web/20260729153237_claude_code_toolkit_07_hooks-compressed.jpg?v=20261005)

这张图讲的是：格式化、测试和危险命令检查不该靠人追着提醒。Hook 像自动门禁，命中事件就执行，该通过的通过，该拦的拦住。

#### 2. 怎么配置一个 Hook？

Hooks 通常写在 `.claude/settings.json` 或用户级 `~/.claude/settings.json`。

下面这个例子在 Claude 使用 `Edit` 或 `Write` 后运行格式化命令：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "jq -r '.tool_input.file_path' | xargs npx prettier --write"
          }
        ]
      }
    ]
  }
}
```

配置后运行 `/hooks`，确认事件、匹配器和命令是否加载。

真实项目里，建议把复杂逻辑放进独立脚本：

```json
{
  "type": "command",
  "command": ".claude/hooks/check-edited-file.sh"
}
```

这样脚本能进 Git、能单独测试，也不用把一长串 shell 塞进 JSON。

#### 3. Hook 怎么用才不会变成新坑？

Hook 是自动执行的，所以要比普通 Prompt 更谨慎。

- 匹配范围尽量窄，不要什么事件都用 `.*`；
- 默认快速执行，重任务不要卡住每次编辑；
- 脚本要有明确退出码和错误信息；
- 不要在 Hook 里偷偷做发布、删除、推送等高风险动作；
- 先手动运行脚本，再接入 Hook；
- 需要复杂判断时，用 Skill 或独立审查 Agent，不要堆一坨 shell。

最适合 Hook 的，是“发生到这里就必须做”的动作。

最不适合 Hook 的，是需要阅读大量上下文、权衡多个方案的开放问题。

### 七、Plugins：把一套能力变成团队可安装的产品

#### 1. Plugin 不是第七种能力

很多录友把 Plugin 理解成“又一个工具”。

其实它更像**包装和分发格式**。

一个 Plugin 可以同时带上：

- Skills；
- Subagents；
- Hooks；
- MCP Server 配置；
- LSP 配置；
- 其他可复用组件。

你在一个项目的 `.claude/` 目录里做实验，适合快速迭代。

当这套配置已经稳定，需要跨项目、跨团队安装、升级和版本管理时，再把它做成 Plugin。

![Claude Code Plugin团队分发](https://file1.kamacoder.com/i/web/20260729153239_claude_code_toolkit_08_plugins-compressed.jpg?v=20261005)

这张图讲的是：Plugin 把零散的 Skill、Agent、Hook 和 MCP 配置装进同一个团队工具包。一份维护、多人复用，才不会每台电脑手工配一遍。

#### 2. 怎么安装 Plugin？

在 Claude Code 里输入：

```text
/plugin
```

可以打开插件面板，浏览、安装和管理插件。

安装官方市场中的插件：

```text
/plugin install github@claude-plugins-official
```

先看清插件提供什么组件、需要什么权限、会连接哪些外部系统。

安装插件不是给它无限授权。

#### 3. 一个自定义 Plugin 长什么样？

最小目录可以是：

```text
team-toolkit/
├── .claude-plugin/
│   └── plugin.json
├── skills/
│   └── release-check/
│       └── SKILL.md
├── agents/
│   └── security-reviewer.md
├── hooks/
│   └── hooks.json
└── .mcp.json
```

`plugin.json` 描述插件身份：

```json
{
  "name": "team-toolkit",
  "description": "Shared release and review workflows for our engineering team",
  "version": "1.0.0"
}
```

本地开发时，可以用 `--plugin-dir` 加载：

```bash
claude --plugin-dir ./team-toolkit
```

先在一两个项目里把组件跑稳，再发布到团队市场。

**Plugin 解决的是分发，不会自动修复里面写得很差的 Skill、过宽的 Hook 或权限过大的 MCP。**

### 八、六种能力怎么组合？看一个真实项目

假设录友正在给电商系统增加“退款审批”功能。

这个功能涉及业务规则、数据库、权限、测试、工单和发布。

正确做法不是把所有要求塞进一个超级 Prompt，而是按问题分层。

#### 1. CLAUDE.md：放每次都要知道的稳定事实

```md
- Money is stored in cents as integer values.
- Refund APIs must be idempotent.
- All approval actions require an audit log.
- Run `pnpm test:payments` after changing the payment module.
- Never run production migrations without explicit approval.
```

这些规则和某一次任务无关，任何支付改动都应该知道。

#### 2. Skill：保存退款评审方法

创建 `refund-review` Skill，里面放：

- 幂等检查清单；
- 金额精度规则；
- 审批状态机说明；
- 审计字段要求；
- 测试模板；
- 输出报告格式。

以后每次做退款功能，不用重新复制几十行要求。

#### 3. Subagent：独立做安全复核

让 `payment-security-reviewer` 只读检查：

- 是否能越权审批；
- 重放请求会不会重复退款；
- 日志是否泄露敏感信息；
- 状态转换是否能绕过；
- 有没有带证据的测试。

主 Agent 不参与它的推理，只接收结论，再决定怎么改。

#### 4. MCP：连接真实工程系统

通过 MCP：

- 读取退款工单；
- 查询错误平台中的历史异常；
- 查看代码托管平台上的 PR；
- 用只读账号查询测试环境数据。

连接负责拿到真实信息，判断规则仍然来自项目说明和 Skill。

#### 5. Hooks：自动执行硬检查

- 修改支付代码后自动跑格式化和静态扫描；
- 执行数据库命令前检查环境；
- 准备提交时确认关键测试是否通过；
- Claude 等待审批时给开发者发送通知。

不用指望 Agent 每次都“自觉记住”。

#### 6. Plugin：把成熟方案发给全团队

当退款 Skill、安全 Subagent、Hooks 和 MCP 配置都跑稳后，打包成 `payment-engineering` Plugin。

新同事安装以后，拿到的是同一套能力，不是某个人口口相传的经验。

这时候六层的关系就很清楚了：

```text
CLAUDE.md：告诉 Claude 这个项目长期怎么做
Skill：告诉 Claude 这类任务具体怎么做
Subagent：安排一个独立角色去做
MCP：给这个角色接上外部工具和数据
Hook：在关键节点自动检查和拦截
Plugin：把前面几层打包给团队
```

### 九、从零开始，应该按什么顺序配置？

不要第一天就安装几十个 Plugin、接十几个 MCP、开一队 Subagents。

配置越多，冲突、权限和上下文成本也越高。

建议按问题出现的顺序来：

#### 第一步：先把根 CLAUDE.md 写到能用

只写命令、架构、禁区和最常见的坑。

运行 `/memory` 确认它被加载。

#### 第二步：把重复出现的流程做成一个 Skill

从最常复制的发布、评审或排障流程开始。

运行 `/skills` 检查描述和作用域。

#### 第三步：给“必须发生”的动作加 Hook

先接格式化、lint、危险命令拦截这类确定动作。

运行 `/hooks` 检查配置，手动验证成功和失败路径。

#### 第四步：只为明确场景创建 Subagent

优先做安全审查或大范围只读探索。

运行 `/agents` 检查它能用哪些工具、模型和 Skills。

#### 第五步：按真实需求接 MCP

先接一个高频系统，给最小权限。

运行 `/mcp` 查看认证和连接状态。

#### 第六步：稳定以后再做 Plugin

先证明这套能力在项目里真的有用，再考虑跨项目分发。

插件化太早，只会把还没想清楚的配置更快地复制出去。

遇到配置不生效，还可以运行：

```text
/doctor
/status
/permissions
```

先确认“有没有加载、从哪里加载、最终权限是什么”，再怀疑模型。

### 十、最容易配错的六个地方

#### 1. 把 CLAUDE.md 写成百科全书

问题：每轮都占上下文，真正重要的规则反而不突出。

改法：稳定事实留下，专项流程移到 Skill，长资料按需加载。

#### 2. Skill 的 description 写得太空

问题：“帮助开发”这种描述什么都能匹配，也等于什么都没说。

改法：写清任务、触发语境和不适用范围。

#### 3. 为了“并行”滥用 Subagents

问题：拆任务、传上下文、汇总结果都要成本。

改法：只有任务能独立完成，或者过程噪音确实需要隔离时再拆。

#### 4. MCP 一上来就连生产写权限

问题：接入范围扩大，误操作半径也跟着扩大。

改法：只读优先、最小权限、敏感动作保留人工确认。

#### 5. Hook 做得又重又宽

问题：每次编辑都卡住，还可能对不相关文件误执行。

改法：缩小 matcher，把复杂逻辑放到可测试脚本，重任务改成按需 Skill。

#### 6. 把 Plugin 当“装得越多越强”

问题：组件可能互相冲突，工具列表膨胀，权限来源也变得难审计。

改法：只安装能解决明确问题的插件，定期用诊断命令检查来源。

### 十一、几个高频问题

#### CLAUDE.md 和自动 Memory 有什么区别？

`CLAUDE.md` 是团队明确维护的项目规则，可以进 Git、可以评审。

自动 Memory 是 Claude 在使用过程中积累的本机经验，适合保存调试发现和个人工作习惯。

团队制度写进 `CLAUDE.md`，不要等自动 Memory 自己猜。

#### Skill 和 Hook 到底怎么选？

需要结合上下文判断、执行一套方法，用 Skill。

命中事件就必须执行，而且结果应该稳定可验证，用 Hook。

例如“按团队标准审查 PR”是 Skill，“编辑后自动格式化”是 Hook。

#### Skill 和 Subagent 到底怎么选？

希望给当前上下文补知识，用 Skill。

希望把长过程隔离出去，只拿结果回来，用 Subagent。

如果一个安全角色需要固定审查方法，就让 Subagent 预加载对应 Skill。

#### MCP 和 Plugin 到底是什么关系？

MCP 是连接协议，解决“Claude 怎么调用外部系统”。

Plugin 是分发包，解决“怎么把 MCP 配置连同 Skills、Agents、Hooks 一起发给别人”。

#### 配齐六件套，就一定更好吗？

不一定。

这些能力不是等级勋章。

一个小项目可能只需要 30 行 `CLAUDE.md` 和一个格式化 Hook。只有重复流程出现了，才加 Skill；只有上下文需要隔离，才加 Subagent；只有仓库外能力确实要用，才接 MCP。

**最好的 Claude Code 配置，不是组件最多，而是每一层都在解决真实问题。**

### 参考链接

- Claude Code 官方文档（扩展能力总览）：https://code.claude.com/docs/en/features-overview
- Claude Code 官方文档（CLAUDE.md 与 Memory）：https://code.claude.com/docs/en/memory
- Claude Code 官方文档（Skills）：https://code.claude.com/docs/en/slash-commands
- Claude Code 官方文档（Subagents）：https://code.claude.com/docs/en/sub-agents
- Claude Code 官方文档（MCP）：https://code.claude.com/docs/en/mcp
- Claude Code 官方文档（Hooks）：https://code.claude.com/docs/en/hooks-guide
- Claude Code 官方文档（Plugins）：https://code.claude.com/docs/en/plugins

### 本节收尾：费曼自述与闪卡

**费曼自述：** “流程说明、自动触发、外部连接是三种不同扩展点；先找到实际重复工作再加。”

**概念闪卡：** Q：Hook 与 Skill 的核心差别？A：前者按事件自动执行，后者是可选择使用的能力/流程封装。Q：MCP 直接保证数据权限吗？A：协议不替代服务端授权。

原文核对：[Claude Code完整使用指南：CLAUDE.md、Skills、Subagents、MCP、Hooks、Plugins怎么用](原仓库Markdown/docs/llm/claude/claude_code_toolkit_guide.md)。

---

<a id="a4"></a>

## 附.4 CLAUDE.md到底怎么写？Claude Code项目记忆、团队规范和上下文管理一篇讲明白【◻】

**一句话：** Agent 不会自动理解整个代码库；清晰的项目说明、按需搜索和能运行的测试，比把所有文件塞进上下文更有用。

**例子：** 在项目根目录说明“如何启动、测试、哪些目录存模型/数据、哪些文件不可改”，Agent 先搜索相关模块，再打开必要代码。长文档不必每轮全量重复。规则文件可以帮助工具，但不能替代真实测试。

### 目录

1. [为什么你总是在重复教育 Claude Code？](#一为什么你总是在重复教育-claude-code)
2. [CLAUDE.md 到底是什么？](#二claudemd-到底是什么)
3. [它和 README、Prompt、Memory 有什么区别？](#三它和-readmepromptmemory-有什么区别)
4. [CLAUDE.md 应该放在哪里？](#四claudemd-应该放在哪里)
5. [什么内容值得写进去？](#五什么内容值得写进去)
6. [什么内容不要写进去？](#六什么内容不要写进去)
7. [怎么写才真的有效？](#七怎么写才真的有效)
8. [大项目怎么拆 CLAUDE.md？](#八大项目怎么拆-claudemd)
9. [CLAUDE.md 和上下文窗口有什么关系？](#九claudemd-和上下文窗口有什么关系)
10. [给你一个可以直接抄的模板](#十给你一个可以直接抄的模板)
11. [团队里怎么维护 CLAUDE.md？](#十一团队里怎么维护-claudemd)
12. [面试和工作中怎么讲这个能力？](#十二面试和工作中怎么讲这个能力)

### 一、为什么你总是在重复教育 Claude Code？

先说一个很常见的场景。

你打开一个老项目，让 Claude Code 帮你改一个接口。

你会先说一大堆：

- 这个项目是 VuePress 1.x，不要按 VuePress 2.x 写
- 命令要在 `docs/` 目录下执行
- 依赖版本别乱升
- 不要改部署脚本
- Markdown 图片必须有 ALT
- 改完之后跑 `npm run build`

这些话有用吗？

有用。

但问题是：**你每次开新任务都要说一遍。**

更麻烦的是，复杂任务跑久了以后，上下文越来越长，前面那些要求会被一堆文件内容、命令输出、错误日志淹没。

你以为 Claude Code "记得"这些规则，其实它只是当前上下文里还能看到这些规则。

一旦上下文压缩、裁剪、切换任务，它就可能忘。

所以项目规则不能只靠聊天输入。

它需要一个更稳定的入口。

这就是 `CLAUDE.md`。

**你可以把它理解成：写给 Claude Code 的项目交接文档。**

新人进组之前，你会告诉他：

- 项目怎么启动
- 哪些目录放什么
- 哪些地方不能动
- 常见坑在哪里
- 提 PR 前要跑什么检查

Claude Code 也是一样。

你不能指望它每次都自己摸索一遍。

摸索会浪费 token，也会制造风险。

### 二、CLAUDE.md 到底是什么？

`CLAUDE.md` 是一个普通的 Markdown 文件。

Claude Code 会自动读取它，把里面的内容作为项目上下文的一部分。

注意几个关键词：

**第一，它是普通 Markdown。**

不是配置文件，不是 JSON，不是 YAML，不需要写复杂语法。

你就按正常文档写：

```md
# 项目说明

- 使用 pnpm 管理依赖
- 开发命令：pnpm dev
- 构建命令：pnpm build
- 不要直接修改数据库 schema
```

**第二，它是给 Claude Code 看的。**

README 主要给人看，`CLAUDE.md` 主要给 AI 编程 Agent 看。

所以它不用写项目愿景、业务背景长文、团队文化。

它要写的是能直接影响 Claude Code 行动的信息。

**第三，它会进入上下文。**

这点很重要。

`CLAUDE.md` 不是在外面摆着好看的文档。

Claude Code 读到它之后，里面的规则会参与后续推理。

所以它能让 Claude Code 更稳定，但也会占上下文窗口。

这就决定了一个原则：

**CLAUDE.md 要短、准、可执行。**

不是越长越好。

<!-- drawio源文件: ./drawio/claude_md_01_positioning.drawio -->
![图1：临时聊天规则会被上下文稀释，CLAUDE.md 能把长期规则沉淀成稳定约束](https://file1.kamacoder.com/i/web/20260527112101_claude_md_01_positioning_compressed.png?v=20261005)

### 三、它和 README、Prompt、Memory 有什么区别？

很多录友会问：那我已经有 README 了，还要 `CLAUDE.md` 干嘛？

这俩不是一个东西。

README 的读者是人。

它通常会写：

- 项目介绍
- 功能说明
- 安装方式
- 使用文档
- 贡献指南

这些信息对人很友好，但对 Agent 不一定高效。

因为 Claude Code 真正需要的是行动约束：

- 这个任务应该从哪个目录入手
- 改代码前先看哪些文件
- 用哪个命令验证
- 哪些文件不要碰
- 遇到某类错误先检查哪里

Prompt 又不一样。

Prompt 是当前任务的临时要求。

比如你说："帮我修登录失败，但不要改后端接口。"

这句话就适合放当前对话里。

因为它只对这次任务生效。

而 `CLAUDE.md` 适合放跨任务都稳定的规则。

比如：

- 项目统一用 TypeScript
- 测试命令是什么
- API 层统一走 `src/api`
- UI 组件不要直接请求接口
- 禁止提交真实 token

Memory 也要分清。

Claude Code 的 memory 有多种层级，比如用户级、项目级、企业级。

`CLAUDE.md` 本身就是一种项目记忆。

但你个人的习惯，比如"我喜欢先看计划再改代码"，更适合放用户级记忆。

团队共享规则，才适合放项目里的 `CLAUDE.md`。

<!-- drawio源文件: ./drawio/claude_md_02_boundaries.drawio -->
![图2：不同类型的信息要分流到 README、Prompt、CLAUDE.md 和用户记忆](https://file1.kamacoder.com/i/web/20260527112102_claude_md_02_boundaries_compressed.png?v=20261005)

一句话区分：

**README 告诉人怎么理解项目，Prompt 告诉模型这次要做什么，CLAUDE.md 告诉 Claude Code 在这个项目里怎么干活。**

### 四、CLAUDE.md 应该放在哪里？

这块是最容易乱的地方。

Claude Code 的 memory 不是只有一个位置。

常见位置有这几类。

#### 1. 用户级：`~/.claude/CLAUDE.md`

这个文件属于你个人。

适合放你跨项目都稳定的偏好。

比如：

- 默认先解释风险再改代码
- 优先使用项目现有命令
- 修改代码后尽量运行相关测试
- 回答时用中文

这些不属于某个项目，而是你个人的工作习惯。

所以不要把它提交到仓库里。

#### 2. 项目级：`./CLAUDE.md` 或 `./.claude/CLAUDE.md`

这是最常用的位置。

适合放团队共享的项目规则。

比如：

- 项目架构
- 常用命令
- 代码风格
- 测试方式
- 禁止改动范围
- 常见问题排查

这个文件可以进 Git。

团队所有人用 Claude Code 时，都能获得同一套项目规则。

#### 3. 子目录级：模块自己的 `CLAUDE.md`

大仓库里，一个根目录 `CLAUDE.md` 往往不够。

比如：

```text
project/
  CLAUDE.md
  frontend/
    CLAUDE.md
  backend/
    CLAUDE.md
  docs/
    CLAUDE.md
```

前端、后端、文档的规则不一样。

强行塞进一个根文件，会越来越长，Claude Code 反而抓不住重点。

更好的方式是根文件写全局规则，子目录文件写模块规则。

官方文档里也提到，Claude Code 会从当前工作目录往上查找记忆；当前工作目录下面更深层的 `CLAUDE.md`，通常会在 Claude 读取对应子树文件时再纳入上下文。

这对大项目很有用。

**不要让前端任务背着后端所有细节跑，也不要让文档任务背着数据库规则跑。**

#### 4. 私人项目偏好：更推荐用 import

以前有些场景会用 `CLAUDE.local.md` 放个人项目偏好。

但现在更推荐在项目记忆里用 `@path` import 引入个人文件。

比如：

```md
# 个人偏好

- @~/.claude/my-project-preferences.md
```

这样个人规则不用进仓库，也更适合多 worktree 场景。

`@path` import 还可以引入项目里的其他文档。

比如：

```md
请参考 @README.md 了解项目概览。
测试规范见 @docs/testing.md。
```

但这里也要克制。

不要把一堆长文档全 import 进来。

能引用关键文档很好，滥用 import 就会把上下文变成垃圾桶。

<!-- drawio源文件: ./drawio/claude_md_03_scope.drawio -->
![图3：多级 CLAUDE.md 按任务路径加载，避免无关模块规则污染上下文](https://file1.kamacoder.com/i/web/20260527112104_claude_md_03_scope_compressed.png?v=20261005)

### 五、什么内容值得写进去？

判断一条信息该不该写进 `CLAUDE.md`，我有一个很简单的标准：

**它是否会反复影响 Claude Code 的行动？**

如果会，就值得写。

如果只对当前一次任务有效，就放 prompt。

如果只是给人看的背景，就放 README。

#### 1. 常用命令

这是最应该写的。

比如：

```md
### 常用命令

- 安装依赖：pnpm install
- 本地开发：pnpm dev
- 单元测试：pnpm test
- 构建检查：pnpm build
```

命令一定要写完整。

尤其是 monorepo 或 VuePress 这类项目，经常要求在特定目录执行。

比如：

```md
- 所有命令都在 `docs/` 目录下执行。
- 开发服务器：`cd docs && npm run dev`
- 构建：`cd docs && npm run build`
```

这比你每次提醒它强太多。

#### 2. 项目结构

写目录职责，不要贴完整文件树。

比如：

```md
### 目录结构

- `src/components/`：通用 UI 组件
- `src/pages/`：页面入口
- `src/api/`：接口封装，组件不要直接调用 fetch
- `src/store/`：全局状态
- `tests/`：单元测试和集成测试
```

这能帮助 Claude Code 少走弯路。

#### 3. 编码规范

写具体规则。

不要写：

```md
- 保持代码优雅
```

这句话没用。

要写：

```md
- 新组件使用函数组件
- 样式优先使用已有 token，不新增散乱颜色
- API 错误统一通过 `handleApiError()` 处理
- 不要在组件中直接写请求逻辑
```

**越具体，越有用。**

#### 4. 禁止事项

这类信息非常适合放 `CLAUDE.md`。

比如：

```md
### 禁止事项

- 不要修改数据库 schema，除非用户明确要求。
- 不要升级核心依赖版本。
- 不要提交 `.env`、token、密钥。
- 不要重写历史迁移文件。
- 不要删除用户已有改动。
```

AI 编程最怕的不是写得慢，而是乱动不该动的地方。

禁止事项写清楚，能减少很多事故。

#### 5. 验证流程

很多项目不是改完就完了。

要写清楚什么改动跑什么检查。

比如：

```md
### 验证要求

- 修改业务逻辑后，优先运行相关单测。
- 修改公共组件后，运行组件测试和构建。
- 修改依赖或配置后，运行完整构建。
- 如果测试无法运行，要在最终说明里写明原因。
```

这能让 Claude Code 更像一个靠谱同事。

#### 6. 常见坑

项目里有些坑，人踩过一次就知道。

AI 不知道。

所以要写。

比如：

```md
### 常见坑

- VuePress 1.x 插件必须使用 `@vuepress/back-to-top`，不要写成 VuePress 2.x 的插件名。
- dev 模式关闭了文件监听，改完文件需要手动刷新浏览器。
- 图片必须使用上传后的图床地址，不要引用本地临时文件。
```

这种信息放在聊天里很容易丢。

放在 `CLAUDE.md` 里，价值很高。

<!-- drawio源文件: ./drawio/claude_md_04_decision.drawio -->
![图4：一条信息能不能写进 CLAUDE.md，要经过反复性、行动性和可维护性判断](https://file1.kamacoder.com/i/web/20260527112106_claude_md_04_decision_compressed.png?v=20261005)

### 六、什么内容不要写进去？

`CLAUDE.md` 最大的问题不是没人写。

而是很多人越写越长，最后变成项目垃圾场。

以下内容不建议写进去。

#### 1. 完整接口文档

接口字段、返回值、错误码，如果很多，应该放专门文档。

`CLAUDE.md` 里最多写：

```md
- API 文档见 `docs/api.md`
- 修改接口调用前先检查 `src/api/`
```

不要把几十个接口全贴进去。

#### 2. 历史流水账

比如：

```md
- 2025-01-03 修了登录问题
- 2025-02-11 重构过首页
- 2025-03-07 和产品讨论过按钮颜色
```

这类信息对当前行动通常没帮助。

如果某个历史决策很重要，要写成规则：

```md
- 登录态统一由 `authStore` 管理，不要在页面组件里重复维护 token。
```

别写流水账，写结论。

#### 3. 空泛口号

比如：

```md
- 写高质量代码
- 注意性能
- 保持架构清晰
```

这类话看着正确，但不能指导行动。

改成可执行规则：

```md
- 列表页超过 100 条数据时使用分页，不要一次性渲染全量数据。
- 公共逻辑超过两个页面复用时，抽到 `src/hooks/`。
```

#### 4. 过期规则

过期规则比没有规则更危险。

比如项目已经从 npm 换成 pnpm，但 `CLAUDE.md` 里还写 npm。

Claude Code 会很听话地继续错。

所以 `CLAUDE.md` 不是写完就不管。

它要随着项目演进更新。

#### 5. 太细的临时任务

比如：

```md
- 今天先修用户 A 的导出 bug
```

这种就不该进项目记忆。

它属于当前对话。

任务结束后还留在 `CLAUDE.md` 里，只会污染后续任务。

### 七、怎么写才真的有效？

我建议录友记住三个词：

**短、准、硬。**

短，是不要写长篇大论。

准，是每条都和行动有关。

硬，是尽量写成明确约束，而不是温柔建议。

比如这句话：

```md
- 尽量注意测试。
```

太软。

改成：

```md
- 修改业务逻辑后，必须运行相关测试；如果无法运行，在最终回复说明原因。
```

这就好多了。

再比如：

```md
- 代码要符合项目风格。
```

太虚。

改成：

```md
- 新增 API 请求必须放在 `src/api/`，页面组件只能调用封装后的 API 方法。
```

这才是 Claude Code 能执行的规则。

我会把好 `CLAUDE.md` 的写法分成 6 类：

1. 项目是什么
2. 命令怎么跑
3. 目录怎么分
4. 代码怎么写
5. 什么不能做
6. 改完怎么验

其他内容能不放就不放。

**写 CLAUDE.md，不是为了显得你很懂 AI，而是为了让 AI 少猜。**

### 八、大项目怎么拆 CLAUDE.md？

小项目一个根 `CLAUDE.md` 就够了。

大项目一定要拆。

尤其是 monorepo。

比如一个项目有：

- Web 前端
- Node 后端
- 移动端
- 文档站
- 基础组件库

如果你把所有规则都写进根 `CLAUDE.md`，最后一定变成十几屏。

Claude Code 每次做一个很小的任务，都要背着一堆无关规则。

更好的结构是：

```text
repo/
  CLAUDE.md
  apps/web/CLAUDE.md
  apps/admin/CLAUDE.md
  packages/ui/CLAUDE.md
  docs/CLAUDE.md
```

根文件写全局规则：

- 包管理器
- Git 流程
- 安全要求
- 全局禁止事项
- 通用验证方式

模块文件写模块规则：

- 当前模块目录职责
- 本模块启动命令
- 本模块测试命令
- 本模块常见坑

这样 Claude Code 在处理具体模块时，拿到的是更相关的上下文。

这和我们前面讲 Agent 上下文管理是同一个逻辑：

**不是让模型看见更多，而是让它看见更有价值的信息。**

<!-- drawio源文件: ./drawio/claude_md_05_context_budget.drawio -->
![图5：大项目拆分 CLAUDE.md 的核心价值是减少无关规则占用上下文预算](https://file1.kamacoder.com/i/web/20260527112108_claude_md_05_context_budget_compressed.png?v=20261005)

### 九、CLAUDE.md 和上下文窗口有什么关系？

这里要讲一个很关键的点。

`CLAUDE.md` 虽然好用，但它不是免费空间。

它会进入 Claude Code 的上下文。

上下文窗口就像一张工作台。

你把项目规则放上去，模型就能看到。

但你放太多，工具输出、文件内容、用户要求就会被挤。

所以不要把 `CLAUDE.md` 当成无限记事本。

官方资料里也提到，Claude Code 会对 `CLAUDE.md` 使用 prompt caching。

这能降低重复读取带来的成本。

但录友要注意：**缓存降低的是计费压力，不代表它不占上下文空间，也不代表信息越多越好。**

一份 300 行的 `CLAUDE.md`，即使缓存了，模型每次也要在大量规则里找重点。

这就会带来两个问题：

第一，重要规则被淹没。

第二，无关规则干扰当前任务。

所以我更推荐：

- 根 `CLAUDE.md` 控制在几屏内
- 每条规则尽量一行说明
- 长文档用链接或 import 引用
- 低频规则放到对应子目录
- 定期删除过期内容

**CLAUDE.md 的目标不是大而全，而是让 Claude Code 进入项目后少踩坑。**

### 十、给你一个可以直接抄的模板

下面这个模板，录友可以直接放到项目根目录，然后按自己项目改。

```md
# 项目工作说明

### 项目概述

- 本项目是一个 XXX 应用，主要技术栈是 XXX。
- 主要代码在 `src/`，测试在 `tests/`。
- 优先遵循现有代码风格，不要引入新的架构风格。

### 常用命令

- 安装依赖：`pnpm install`
- 本地开发：`pnpm dev`
- 单元测试：`pnpm test`
- 构建检查：`pnpm build`

### 目录结构

- `src/components/`：通用组件
- `src/pages/`：页面入口
- `src/api/`：接口封装
- `src/hooks/`：可复用业务逻辑
- `tests/`：测试文件

### 编码规范

- 新增 API 请求必须放在 `src/api/`。
- 页面组件不要直接调用 `fetch`。
- 公共逻辑被两个以上模块复用时，抽到 `src/hooks/`。
- 修改已有功能时，优先保持现有接口兼容。

### 禁止事项

- 不要提交 `.env`、token、密钥。
- 不要升级核心依赖版本，除非用户明确要求。
- 不要修改数据库 schema，除非用户明确要求。
- 不要删除用户已有改动。

### 验证要求

- 修改业务逻辑后，运行相关测试。
- 修改公共组件后，运行构建检查。
- 如果测试无法运行，在最终回复里说明原因。

### 常见坑

- 本项目使用 pnpm，不要使用 npm 或 yarn。
- 修改配置文件后，需要重新启动开发服务器。
- 遇到鉴权问题，先检查 `src/api/auth.ts` 和 `src/store/auth.ts`。
```

这个模板不是让你照搬到所有项目。

它的价值是结构。

你真正要做的是把里面每一条改成自己项目里的真实规则。

**假的规范，比没有规范更坑。**

### 十一、团队里怎么维护 CLAUDE.md？

如果团队已经大量使用 AI 编程，我建议把 `CLAUDE.md` 当成工程资产维护。

不要让它变成某个人电脑里的私货。

团队可以这么做：

**第一，项目根 `CLAUDE.md` 进仓库。**

凡是团队共享规则，都应该进 Git。

比如构建命令、目录职责、代码规范、禁止事项。

**第二，个人偏好放用户级 memory。**

比如你喜欢中文回复、喜欢先看计划、喜欢最终说明简短。

这些不应该污染团队项目文件。

**第三，规则变更要像代码一样审查。**

尤其是禁止事项、测试命令、架构规则。

它们会直接影响 AI 的行动。

写错了，就是把错误流程自动化。

**第四，踩坑之后及时沉淀。**

如果 Claude Code 因为某个项目坑连续犯错，不要只在聊天里骂它。

把坑写进 `CLAUDE.md`。

比如：

```md
- 支付模块的金额单位是分，不是元。
```

这种一句话，可能能避免很多次错误。

**第五，定期删。**

`CLAUDE.md` 不是只加不删。

每隔一段时间看一眼：

- 有没有过期命令
- 有没有已经不存在的目录
- 有没有重复规则
- 有没有临时任务残留

越短越准，越有用。

### 十二、面试和工作中怎么讲这个能力？

很多人以为会用 Claude Code，就是会让它写代码。

其实更高级一点，是你能把 AI 编程变成稳定流程。

`CLAUDE.md` 就是一个很好的例子。

面试官如果问你："你怎么让 AI 编程工具更稳定？"

你可以这样答：

"我不会只靠临时 prompt 管项目规则。对于跨任务稳定的信息，比如项目架构、常用命令、测试方式、代码风格、禁止改动范围，我会沉淀到 `CLAUDE.md`。这样 Claude Code 每次进入项目时都能读取这些规则，减少重复解释，也减少上下文压缩后丢约束的问题。

但我不会把 `CLAUDE.md` 写成大杂烩。因为它会进入上下文，太长会稀释注意力。所以我会按全局规则和模块规则拆分，根目录写通用约束，子目录写模块细节；临时任务放 prompt，个人偏好放用户级 memory，团队规则放项目级 `CLAUDE.md`。核心目标不是让模型看到更多，而是让它看到更稳定、更有行动价值的信息。"

这个回答就不只是"我会用工具"。

它说明你理解 AI Coding 背后的工程化问题。

### 结尾

`CLAUDE.md` 不是魔法。

它不会让 Claude Code 瞬间变成懂你公司所有业务的老员工。

但它能解决一个非常实际的问题：

**别让 AI 每次进项目都从零猜。**

项目怎么启动，代码怎么写，哪里不能动，改完怎么验，这些规则越早沉淀，Claude Code 越像一个靠谱队友。

录友们不要把 `CLAUDE.md` 写成百科全书。

就写那些会反复影响行动的规则。

短一点，准一点，硬一点。

这就够了。

### 参考资料

- Claude Code Docs：Manage Claude's memory：https://docs.claude.com/en/docs/claude-code/memory)
- Claude Help Center：Give Claude context: CLAUDE.md and better prompts：https://support.claude.com/en/articles/14553240-give-claude-context-claude-md-and-better-prompts
- Claude Help Center：Claude Code cheatsheet：https://support.claude.com/en/articles/14553413-claude-code-cheatsheet

### 本节收尾：费曼自述与闪卡

**费曼自述：** “给 Agent 一张项目地图和可靠的验证命令，让它根据任务逐步找代码。”

**概念闪卡：** Q：项目说明文件能代替 README 吗？A：两者读者和用途可能不同；项目可保留给人看的 README 与给工具的工作规则。Q：大代码库为何先搜索？A：减少无关上下文并定位真正要改的模块。

原文核对：[CLAUDE.md到底怎么写？Claude Code项目记忆、团队规范和上下文管理一篇讲明白](原仓库Markdown/docs/llm/claude/claude_md.md)。

---

<a id="a5"></a>

## 附.5 Claude Code 为什么快？Prompt Cache、Plan Mode、MCP工具加载和上下文压缩一篇讲明白【◻】

**一句话：** CLI 便于工具组合；Prompt 缓存只在符合服务规则的重复前缀上省计算；大型迁移案例提供任务拆分与验收思路，但数字与工具版本不宜背诵。

**例子：** 同一长项目规则被许多请求复用可能有缓存收益；每次变化的文件内容不能假定都命中缓存。百万行代码迁移案例可提炼“先做可判对错的测试，再扩大自动化范围”。

### 一、先搞清楚：Prompt Cache 缓存的不是答案

很多录友一听缓存，第一反应是：

是不是用户问过的问题，下次直接把答案拿出来？

不是。

Prompt Cache 缓存的不是模型回答。

它缓存的是模型处理输入上下文时的中间计算。

你可以这么理解。

Claude Code 每一轮跟模型交互，并不是只把你刚输入的那句话发过去。

它要把一整套东西重新发给模型：

- 系统提示词
- 工具定义
- CLAUDE.md 里的项目规则
- 当前项目上下文
- 前面所有对话消息
- 工具调用结果
- 你这一轮的新输入

模型本身并不会在两次 API 请求之间“记住”上一次聊到哪了。

所以每一轮请求都要带上完整上下文。

如果没有缓存，那长会话越聊越贵，越聊越慢。

这也是为什么很多 Agent demo 一开始看起来很丝滑，但跑一会儿就开始变慢。

它不是突然变笨了。

是上下文越来越长，每次都在重算。

Prompt Cache 解决的就是这个问题：

**前面已经算过、下一轮又完全一样的部分，就不要再算一遍。**

但这里有一个关键限制。

它靠的是前缀匹配。

不是“这段内容大概一样”。

也不是“这个文件之前读过”。

而是从请求开头开始，连续一段内容必须一致，才能命中缓存。

![Prompt Cache 命中依赖请求前缀连续一致](https://file1.kamacoder.com/i/web/20260601095801_claude_prompt_cache_01_prefix_layers_compressed.png?v=20261005)

所以 Claude Code 的请求组织方式很讲究：

第一层，静态系统提示词和工具定义。

这部分最好全局稳定，不同项目都能复用。

第二层，CLAUDE.md 和项目上下文。

这部分在同一个项目里稳定。

第三层，会话上下文。

这部分在同一个会话里稳定。

第四层，对话消息和工具结果。

这部分会随着聊天增长。

第五层，本轮新增输入。

每一轮只让最后一点点内容变化。

这样前面大部分内容都能命中缓存。

**静态内容放前面，动态内容放后面。**

这是做 Agent 产品时最朴素、也最容易被忽略的缓存原则。

### 二、为什么“改一个小东西”也会让缓存断掉？

Prompt Cache 最反直觉的地方在这里。

很多人觉得：

我只是改了一下系统提示词里的时间。

我只是调整了一下工具顺序。

我只是把某个工具参数更新了一下。

应该没什么大问题吧？

有问题。

因为它会让前缀发生变化。

前缀一变，后面的缓存就接不上。

Anthropic 在原文里提到，他们也踩过类似坑：比如把详细时间戳放进静态系统提示词、工具定义顺序不是确定的、更新工具参数等。

这些变化看着都不大。

但从缓存角度看，都是前缀污染。

![Prompt Cache 前缀断点会导致后续缓存失效](https://file1.kamacoder.com/i/web/20260601095803_claude_prompt_cache_02_cache_breakers_compressed.png?v=20261005)

这就像你在一个大项目里改构建缓存 key。

你可能只改了一行配置。

但缓存系统看到的是：

不一样了。

那就重新构建。

对 Agent 也是一样。

如果系统提示词每轮都带一个实时日期，或者工具定义每轮顺序都不稳定，那缓存命中率就会很难看。

所以做 Agent 不能只问：

这段 Prompt 对模型有没有帮助？

还要问：

**这段内容是不是应该出现在稳定前缀里？**

如果它每轮都变，那就别放前面。

放到消息里。

### 三、动态信息别改系统提示词，放进消息里

Agent 运行过程中，确实会遇到很多动态信息。

比如：

- 当前时间变了
- 用户改了文件
- 权限状态变了
- 进入了某种工作模式
- 某个工具结果需要提醒模型

直觉上，你可能想更新系统提示词。

比如把“当前时间是 2026 年 5 月 28 日”写进系统提示词。

或者用户改了文件，就把“某文件已修改”更新到项目上下文里。

这在功能上没问题。

但在缓存上很差。

因为系统提示词和项目上下文都在前面。

你一改，后面全断。

Claude Code 的做法更聪明：

**动态信息尽量通过下一轮消息追加进去，而不是修改前面的稳定前缀。**

例如文件变化，可以在下一条用户消息或工具结果里追加一个系统提醒，告诉模型“这个文件已经变了，需要时重新读取”。

这样做的好处是：

前面的系统提示词、工具定义、CLAUDE.md 都不动。

只是末尾多了一条新消息。

缓存还能继续命中。

这也是为什么我们前面讲 CLAUDE.md 时反复强调：

CLAUDE.md 适合放长期稳定规则。

不要把一次性任务、临时状态、当前进度都塞进去。

**长期规则进 CLAUDE.md，临时变化进消息。**

这个分流，不只是为了上下文清爽。

也是为了缓存稳定。

![动态信息应该追加到消息末尾而不是修改稳定前缀](https://file1.kamacoder.com/i/web/20260601095805_claude_prompt_cache_03_dynamic_messages_compressed.png?v=20261005)

### 四、不要中途切模型，可能更贵

这点也很反直觉。

很多人会想：

复杂问题用强模型。

简单问题切便宜模型。

听起来很合理。

但在长会话里，不一定省钱。

因为 Prompt Cache 是按模型隔离的。

你在 Opus 里聊了 100k token，上下文已经缓存好了。

这时候你觉得下一个问题很简单，切到 Haiku。

从单次输出价格看，Haiku 便宜。

但 Haiku 没有 Opus 的缓存。

它要重新处理整段上下文。

这就可能出现一个很尴尬的情况：

**为了省一点输出成本，反而付出一大笔未缓存输入成本。**

所以 Anthropic 的建议是：不要在同一个长会话中随便换模型。

如果确实需要换模型，更好的方式是用 subagent。

主会话继续保持自己的模型和缓存。

子 Agent 用另一个模型处理局部任务。

然后把结果返回给主会话。

这其实又回到一个原则：

**不要破坏主会话的稳定前缀。**

主会话是主干。

子 Agent 是分支。

分支可以另起缓存。

但不要让主干为了一个小任务把整个缓存掀掉。

![长会话中途切模型会失去原模型缓存](https://file1.kamacoder.com/i/web/20260601095807_claude_prompt_cache_04_model_switch_compressed.png?v=20261005)

### 五、Plan Mode 为什么不能靠“删掉写入工具”实现？

Claude Code 的 Plan Mode，录友应该不陌生。

进入 Plan Mode 之后，Claude Code 会先读代码、分析问题、给计划，不直接改文件。

很多人如果自己设计这个功能，第一反应大概是：

计划模式只保留只读工具。

编辑工具、写文件工具、执行危险命令的工具都先移除。

听起来很安全。

但这会破坏缓存。

因为工具定义是前缀的一部分。

你一进入 Plan Mode，就把工具集合换了。

退出 Plan Mode，又换回来。

每次模式切换都在动前缀。

这就很亏。

![Plan Mode 的关键是保持工具定义稳定只追加模式状态](https://file1.kamacoder.com/i/web/20260601095808_claude_prompt_cache_05_plan_mode_compressed.png?v=20261005)

Claude Code 的设计更有意思。

它不是通过“换工具集合”来表达 Plan Mode。

而是始终保留完整工具集。

同时把 EnterPlanMode 和 ExitPlanMode 也设计成工具。

当用户进入 Plan Mode 时，系统在消息末尾追加说明：

当前处于计划模式，要探索代码库，不要编辑文件，计划完成后调用退出计划模式。

注意重点：

工具定义没变。

变化发生在对话末尾。

所以缓存前缀还能命中。

这就是工程产品和普通 demo 的差别。

普通 demo 可能只关心“功能能不能实现”。

Claude Code 这种产品还要关心：

**这个功能会不会让每一次模式切换都变贵、变慢？**

更妙的是，因为 EnterPlanMode 本身也是工具，模型在遇到复杂任务时，还可以自己判断进入计划模式。

功能更灵活。

缓存也没被破坏。

### 六、MCP 工具多了，不能每轮动态删工具

现在大家做 Agent，很容易接一堆 MCP。

GitHub 一个。

数据库一个。

浏览器一个。

文件系统一个。

内部平台再接几个。

工具越来越多。

然后就会遇到一个问题：

每次请求都把几十个工具的完整 schema 发给模型，太贵了。

那怎么办？

很多人的第一反应是：

本轮用不到的工具就删掉。

比如这轮只需要读文件，那数据库工具、浏览器工具都先不发。

听起来省 token。

但这又踩中了缓存雷区。

因为工具集合变了。

工具定义顺序变了。

缓存前缀就变了。

所以 Claude Code 的思路不是“动态删工具”，而是“延迟加载工具详情”。

![MCP 工具多时用稳定工具目录和按需加载 schema](https://file1.kamacoder.com/i/web/20260601095810_claude_prompt_cache_06_deferred_tools_compressed.png?v=20261005)

它可以先在稳定前缀里放轻量 stub。

也就是工具名、很少的元信息、以及 defer_loading 这样的标记。

完整工具 schema 先不塞进去。

当模型真的需要某个工具时，再通过 tool search 发现并加载完整 schema。

这样有两个好处：

第一，稳定前缀不会因为工具删来删去而变化。

第二，不常用工具的完整定义不会每轮都占 token。

所以 MCP 工具管理的核心不是：

我这一轮到底要不要给模型这个工具？

而是：

**哪些工具信息必须稳定存在，哪些工具详情可以按需展开？**

这个思路对我们自己做 Agent 也很重要。

不要一上来就把所有系统能力、所有接口文档、所有工具参数全塞进 Prompt。

先给工具目录。

需要时再展开详情。

### 七、上下文压缩不是另起一个总结请求

长会话跑久了，一定会遇到上下文窗口不够的问题。

Claude Code 里常见的处理方式就是 compact。

把前面的对话总结成一段摘要，然后用摘要替换原来的长历史，继续往下做。

这看起来很简单。

但从缓存角度看，这里面也有坑。

最容易想到的做法是：

单独发一个请求给模型。

系统提示词写“请总结下面这段对话”。

不带工具。

把完整历史塞进去。

让模型返回摘要。

这当然能跑。

但很贵。

因为这个总结请求的系统提示词、工具定义、上下文结构，和父会话完全不一样。

从第一个 token 开始就不匹配。

父会话前面攒下来的缓存，全用不上。

更糟的是：

越是需要 compact，说明对话越长。

对话越长，单独总结请求越贵。

这就是典型的“问题越严重，解决问题越贵”。

Claude Code 的做法是缓存友好分叉。

![上下文压缩应该复用父会话前缀再追加 compact 指令](https://file1.kamacoder.com/i/web/20260601095811_claude_prompt_cache_07_compaction_fork_compressed.png?v=20261005)

它在做压缩时，使用和父会话一样的系统提示词、用户上下文、系统上下文和工具定义。

先带上父会话的消息。

然后在末尾追加一条压缩指令。

从 API 视角看，这个请求和父会话上一次请求非常像。

前面大段前缀都一样。

只有末尾的压缩指令是新的。

这样就能复用父会话缓存。

当然，这也带来另一个工程要求：

系统必须预留压缩缓冲区。

不能等上下文窗口真的塞满了，再想着“现在总结一下吧”。

那时候已经没空间放压缩指令和摘要输出了。

所以成熟 Agent 产品会在上下文还没彻底满之前，就留出 compact buffer。

这点对我们自己做 Agent 也很关键。

不要把 compact 当成最后一刻的急救。

它应该是上下文生命周期管理的一部分。

### 八、缓存命中率要像服务可用性一样监控

Anthropic 在原文里有个细节很值得注意：

Claude Code 团队会监控 Prompt Cache 命中率。

命中率太低，会触发告警，甚至当成事故处理。

这个态度很说明问题。

很多团队做大模型应用，只监控这些指标：

- QPS
- 延迟
- 错误率
- token 消耗
- 单次请求成本

这些当然要看。

但做 Agent，还要看缓存。

尤其是长会话 Agent。

缓存命中率一掉，延迟和成本会一起变差。

用户体感就是：

怎么突然变慢了？

怎么同样的任务今天额度掉得特别快？

而研发侧如果只看请求成功率，可能完全看不出问题。

因为请求都成功了。

只是每次都在重算。

所以做 Agent 要多看两类 token：

cache creation input tokens：本轮写入缓存的输入 token。

cache read input tokens：本轮从缓存读取的输入 token。

![Prompt Cache 命中率需要纳入 Agent 监控闭环](https://file1.kamacoder.com/i/web/20260601095813_claude_prompt_cache_08_cache_metrics_compressed.png?v=20261005)

如果 read 很高，creation 很低，说明缓存用得好。

如果 creation 连续很高，说明你的前缀可能一直在变。

这时候不要先怀疑模型。

先查：

- 系统提示词是不是每轮都变
- 工具定义顺序是不是不稳定
- MCP 工具是不是动态增删
- 是否中途切了模型
- compact 是否另起了一个请求
- 是否把时间、文件变化、模式状态写进了前缀

很多 Agent 性能问题，不是模型问题。

是上下文组织问题。

### 九、这篇文章对普通开发者有什么用？

可能有录友会说：

这些都是 Claude Code 团队做产品的细节，我只是用 Claude Code 或者做一点大模型应用，有必要关心吗？

有必要。

因为这篇文章其实告诉我们一个大趋势：

**Agent 工程不是只会调 Prompt，也不是只会接几个工具。**

真正要做稳，要懂上下文结构、缓存策略、工具边界、模型切换、长会话生命周期。

如果你只是日常使用 Claude Code，可以记住这几条：

第一，别在长任务中频繁切模型。

切一次模型，下一轮可能会明显变慢，因为缓存要重建。

第二，CLAUDE.md 改完不一定立刻生效。

它属于会话开头加载的项目上下文，中途改了不一定会影响当前会话。要么重新开会话，要么在当前对话里明确补充。

第三，MCP 不要边做任务边频繁接入、断开。

MCP 工具定义会影响缓存，最好在一个任务开始前就把需要的 MCP 配好。

第四，compact 最好放在自然断点。

比如一个子任务完成后再压缩，而不是在关键编辑中间突然压缩。

第五，别把所有临时要求都塞进项目规则文件。

稳定规则写进 CLAUDE.md。

临时要求直接在当前对话里说。

如果你是做 Agent 产品的，那要再进一步：

- Prompt 结构按稳定性分层
- 工具定义顺序必须确定
- 工具集合尽量会话内稳定
- 动态状态用消息追加
- 多模型用 subagent 隔离
- 工具详情用延迟加载
- compact 走父会话缓存友好分叉
- 缓存命中率纳入监控

这不是优化清单。

这是架构清单。

### 十、最后说一句：缓存不是省钱小技巧，是 Agent 的地基

现在很多人讲 Agent，喜欢讲规划、记忆、工具调用、多 Agent 协作。

这些都重要。

但如果底层上下文每轮都在重算，工具集合每轮都在变，系统提示词每轮都不一样，模型动不动就切。

那这个 Agent 越跑越贵、越跑越慢，是迟早的事。

所以 Anthropic 这篇文章的价值，不在于告诉你“Prompt Cache 可以省钱”。

而在于告诉你：

**Agent 产品的功能设计，要服从缓存约束。**

Plan Mode 是这样。

MCP 工具加载是这样。

上下文压缩也是这样。

很多看起来是产品交互的问题，背后其实是缓存架构问题。

这也是 Claude Code 值得研究的地方。

它不是只把一个强模型塞进 CLI。

它是在模型之外，做了一整套能长期运行的 Agent harness。

而 Prompt Cache，就是这套 harness 里最容易被低估的一根主梁。

录友如果想做 AI 编程工具，或者想做真正可用的 Agent 系统，这篇原文建议认真读。

别只看它怎么省 token。

要看它怎么把缓存变成产品设计原则。

这才是重点。

### 本节收尾：费曼自述与闪卡

**费曼自述：** “从工具案例里抽取稳定的工程方法，不把某次统计或产品细节记成永久规律。”

**概念闪卡：** Q：缓存命中看什么？A：服务规则及可复用前缀。Q：自动迁移前为什么先做验收？A：没有判定标准，大规模执行会放大错误。

原文核对：[Claude Code 为什么快？Prompt Cache、Plan Mode、MCP工具加载和上下文压缩一篇讲明白](原仓库Markdown/docs/llm/claude/claude_prompt_cache.md)。

---

<a id="a6"></a>

## 附.6 Claude Code怎么读懂大代码库？Agent搜索、CLAUDE.md、Hooks、Skills、MCP和LSP一篇讲明白【◻】

**一句话：** Agent 不会自动理解整个代码库；清晰的项目说明、按需搜索和能运行的测试，比把所有文件塞进上下文更有用。

**例子：** 在项目根目录说明“如何启动、测试、哪些目录存模型/数据、哪些文件不可改”，Agent 先搜索相关模块，再打开必要代码。长文档不必每轮全量重复。规则文件可以帮助工具，但不能替代真实测试。

### 目录

1. [为什么大代码库不是小项目放大版？](#一为什么大代码库不是小项目放大版)
2. [Claude Code 不是靠索引背下整个仓库](#二claude-code-不是靠索引背下整个仓库)
3. [Agentic Search：像工程师一样边搜边读](#三agentic-search像工程师一样边搜边读)
4. [大代码库的关键不是看更多，而是找得准](#四大代码库的关键不是看更多而是找得准)
5. [Harness：模型之外真正影响体验的工程支架](#五harness模型之外真正影响体验的工程支架)
6. [CLAUDE.md：大代码库的第一层导航](#六claudemd大代码库的第一层导航)
7. [Hooks：把提醒模型变成自动执行](#七hooks把提醒模型变成自动执行)
8. [Skills：不要把所有专家知识塞进上下文](#八skills不要把所有专家知识塞进上下文)
9. [Plugins、MCP、LSP、Subagents 分别解决什么问题？](#九pluginsmcplspsubagents-分别解决什么问题)
10. [大代码库应该按什么顺序配置 Claude Code？](#十大代码库应该按什么顺序配置-claude-code)
11. [团队落地：为什么需要一个 Agent 管理人？](#十一团队落地为什么需要一个-agent-管理人)
12. [普通开发者怎么开始？](#十二普通开发者怎么开始)

### 一、为什么大代码库不是小项目放大版？

小项目里用 Claude Code，很简单。

你让它修一个 bug，它搜一下文件，读几个组件，改完跑测试。

路径短，文件少，命令也少。

但大代码库不是这样。

一个大代码库里，可能会有这些问题：

- 每个子目录都有不同的构建命令
- 前端、后端、脚本、部署工具混在一个仓库
- 同名函数在不同语言里出现几十次
- 老代码没有统一目录结构
- 微服务分散在多个仓库
- 构建日志一跑就是几千行
- 全量测试根本跑不动
- 某些目录是生成代码，根本不该读

这不是"文件变多了"这么简单。

这是信息密度、历史包袱、团队约定、工具链复杂度一起上来了。

所以大代码库里，Claude Code 最大的问题不是不会写代码。

而是它一开始不知道：

**从哪里看，读哪些文件，用什么命令验证，哪些地方不能碰。**

如果这些都靠 Claude 自己猜，它会浪费大量上下文。

更糟的是，它可能猜错方向。

所以大代码库的第一原则是：

**不要让 Agent 在仓库里盲搜。**

你要给它入口。

![图1：大代码库让 Agent 迷路的根因不是文件多，而是入口太多](https://file1.kamacoder.com/i/web/20260527192245_claude_large_codebase_01_complexity_compressed.png?v=20261005)

### 二、Claude Code 不是靠索引背下整个仓库

很多 AI 编程工具喜欢讲代码索引、Embedding、RAG。

这个思路很自然。

仓库太大，模型上下文放不下，那就先把代码切块、向量化、建索引，用户提问时再召回相关片段。

这套方案在很多场景有用。

但 Anthropic 这篇博客里提到一个关键问题：

**大规模活跃代码库里，索引很容易过期。**

几千个工程师每天都在提交代码。

函数今天改名，模块明天迁移，旧服务后天删掉。

如果索引管线跟不上，模型拿到的就是旧世界的地图。

它可能召回一个两周前已经改名的函数，也可能引用一个上个 sprint 已经删掉的模块。

更麻烦的是，召回结果本身不会主动告诉你："我过期了。"

Claude Code 的思路不一样。

它更像一个真实工程师：

- 在本地文件系统里看目录
- 用 grep 搜关键词
- 读当前最新文件
- 顺着引用关系继续找
- 根据测试和报错再调整方向

它不是先把整个仓库上传到某个中心索引里。

它是在开发者机器上的实时代码库里工作。

![图2：活跃代码库里索引容易和现场错位，Claude Code 更依赖当前工作区实时搜索](https://file1.kamacoder.com/i/web/20260527192246_claude_large_codebase_02_index_drift_compressed.png?v=20261005)

这就是 Anthropic 说的 agentic search。

不是"先把答案搜出来给模型"。

而是让 Agent 自己根据当前任务，一步一步决定下一步该搜什么、读什么、验证什么。

### 三、Agentic Search：像工程师一样边搜边读

Agentic Search 听起来很高级，其实你可以把它理解成：

**不是一次性检索答案，而是边行动边缩小范围。**

比如你让 Claude Code 修一个支付失败问题。

它不应该上来读整个支付系统。

更合理的路径是：

1. 先看报错关键词
2. 用 grep 找相关接口
3. 读调用链上的关键文件
4. 找到支付状态转换逻辑
5. 看测试怎么覆盖
6. 修改最小范围
7. 跑对应测试

![图3：Agentic Search 不是一次性搜索答案，而是根据新证据一步步缩小范围](https://file1.kamacoder.com/i/web/20260527192247_claude_large_codebase_03_agentic_search_compressed.png?v=20261005)

这和工程师查 bug 很像。

工程师也不会打开全仓库从第一行读到最后一行。

他会根据线索搜索。

所以 Claude Code 的优势不是"一次看完整个仓库"。

它的优势是：

**能用工具持续探索，并把探索结果转成下一步行动。**

但这里有一个前提。

Agentic Search 需要起点。

如果你只说："帮我优化这个十亿行代码库。"

那谁也救不了。

Claude Code 也会在上下文窗口耗尽之前迷路。

所以大代码库必须给 Claude Code 几类导航信息：

- 目录地图
- 根 CLAUDE.md
- 子目录 CLAUDE.md
- 局部测试命令
- 忽略规则
- LSP 符号导航

这些东西不是锦上添花。

它们是大代码库里让 Agent 找准方向的基础设施。

### 四、大代码库的关键不是看更多，而是找得准

很多录友一听大代码库，就觉得上下文窗口要越大越好。

窗口大当然有用。

但大代码库里，真正的问题不是"放不下全部代码"。

因为无论窗口多大，你都不应该把全部代码塞进去。

真正的问题是：

**Claude Code 能不能在读很少文件的情况下，找到真正相关的那几个文件。**

这就像你查线上 bug。

你不需要把所有日志都看一遍。

你需要知道哪个服务、哪个接口、哪个配置、哪个版本最可能相关。

所以大代码库要做的是上下文治理：

- 该加载的规则提前加载
- 不相关的目录不要读
- 生成代码和构建产物要排除
- 测试命令要按子目录收敛
- 同名函数要靠 LSP 区分
- 大范围探索交给 subagent

Anthropic 文章里提到一个很重要的点：

**太多上下文会降低性能，太少上下文会让 Claude 盲搜。**

这句话很适合放到团队规范里。

大代码库不是让 Claude Code 看更多。

而是让它每一步都看得更准。

![图4：上下文治理要避免太少导致盲搜，也要避免太多稀释注意力](https://file1.kamacoder.com/i/web/20260527192249_claude_large_codebase_04_context_governance_compressed.png?v=20261005)

### 五、Harness：模型之外真正影响体验的工程支架

很多人评价 Claude Code，第一反应是看模型。

用的是 Sonnet 还是 Opus？

跑 benchmark 排名怎么样？

这些当然重要。

但 Anthropic 这篇博客说得很直接：

**Claude Code 的效果，不只由模型决定，还由模型周围的 harness 决定。**

什么是 harness？

简单说，就是 Claude Code 周围这套工程支架：

- `CLAUDE.md`
- Hooks
- Skills
- Plugins
- MCP servers
- LSP
- Subagents

模型负责推理。

harness 负责把模型放到真实工程环境里。

没有 harness，Claude Code 就像一个很聪明的新同事，但没人告诉他项目规则，也不给他工具权限，还让他自己猜构建命令。

有 harness，Claude Code 才像一个接入了团队工程体系的 Agent。

![图5：同一个任务有没有 Harness，Claude Code 的执行路径会完全不同](https://file1.kamacoder.com/i/web/20260527192250_claude_large_codebase_05_harness_paths_compressed.png?v=20261005)

接下来我们一个个讲。

### 六、CLAUDE.md：大代码库的第一层导航

`CLAUDE.md` 永远应该先做。

原因很简单：

它是 Claude Code 每次进入项目时最稳定的上下文入口。

在大代码库里，根 `CLAUDE.md` 不应该写成百科全书。

它应该 lean and layered。

也就是：

**根文件写大图和关键坑，子目录文件写局部规则。**

根文件适合写：

- 仓库整体结构
- 顶层目录各自负责什么
- 全局禁区
- 包管理器
- 全局安全要求
- 重要文档入口

子目录文件适合写：

- 当前模块怎么启动
- 当前模块怎么测试
- 当前模块有什么本地约定
- 当前模块常见坑

不要把所有细节都塞进根文件。

根文件太长，每个任务都会背着一堆无关规则。

比如你只改前端登录页，却每次都加载后端数据库迁移规则、移动端构建说明、部署平台历史问题。

这就是上下文污染。

所以大代码库更需要分层 `CLAUDE.md`。

![图6：根 CLAUDE.md 管方向，子目录 CLAUDE.md 和 Skills 负责局部知识按需加载](https://file1.kamacoder.com/i/web/20260527192251_claude_large_codebase_06_context_routing_compressed.png?v=20261005)

这里和我们上一篇文章是连着的。

上一篇讲 `CLAUDE.md` 怎么写。

这篇要再强调一句：

**大代码库里，CLAUDE.md 不只是规则文档，它是 Agent 的导航系统。**

### 七、Hooks：把提醒模型变成自动执行

很多团队刚开始用 Claude Code，会喜欢在 `CLAUDE.md` 里写：

- 修改代码后记得格式化
- 提交前记得跑 lint
- 不要忘了更新文档
- 会话结束后总结这次踩过的坑

这些有用吗？

有用，但还不够。

因为只写在文档里，本质还是提醒模型。

而 Hooks 的价值是：

**把提醒模型，变成确定执行。**

比如格式化。

你当然可以写："修改代码后请运行 formatter。"

但更稳的方式是 hook 在合适的时机自动执行格式化。

再比如 lint。

你当然可以让 Claude 记得跑。

但 hook 自动跑 lint，比让模型记住更稳定。

Anthropic 文章里还提到一个更有意思的用法：

Hooks 不只是拦截错误，还可以让配置自我改进。

比如 stop hook 在会话结束时回看这次过程：

- Claude 今天是不是绕了远路？
- 是不是因为某个目录规则不清楚？
- 是不是某个测试命令应该写进子目录 CLAUDE.md？

如果是，就提出 `CLAUDE.md` 更新建议。

这样项目规则就不是一次性写完。

而是随着团队使用不断迭代。

这才是大团队落地 AI 编程需要的能力。

![图7：Hooks 把提醒模型变成自动执行，并把经验反哺到项目规则里](https://file1.kamacoder.com/i/web/20260527192252_claude_large_codebase_07_hooks_feedback_compressed.png?v=20261005)

### 八、Skills：不要把所有专家知识塞进上下文

大代码库里，会有很多专家知识。

比如：

- 安全审查怎么做
- 支付模块怎么部署
- 文档更新流程是什么
- 某个内部平台怎么查数据
- 某类日志怎么定位

如果你把这些全部写进 `CLAUDE.md`，它会爆炸。

而且很多任务根本用不到这些知识。

这时候就该用 Skills。

Skills 的关键是按需加载。

只有当任务需要安全审查时，才加载安全审查 skill。

只有当任务涉及文档处理时，才加载文档处理 skill。

只有当任务进入支付目录时，才加载支付部署 skill。

这就是 Anthropic 提到的 progressive disclosure。

我把它翻译得口语一点：

**别一上来把所有专家都请进会议室。谁用得上，再叫谁进来。**

当然，文章里不能真的这么写得太花。

落到工程上就是：

**稳定通用规则放 CLAUDE.md，专门任务流程放 Skills。**

这条边界很重要。

很多团队把 `CLAUDE.md` 写成巨型专家手册，最后 Claude 每次启动都要读一堆无关知识。

这不是增强。

这是拖慢。

### 九、Plugins、MCP、LSP、Subagents 分别解决什么问题？

Anthropic 文章里还讲了几个组件。

这些词很多录友第一次看会晕。

我们用一句话分别讲清楚。

**Plugins：把好用配置发给全团队。**

一个团队里，某个工程师配好了 hooks、skills、MCP。

如果只存在他电脑上，那就是个人经验。

如果打包成 plugin，新人第一天装上，就能拿到同样能力。

所以 Plugins 解决的是分发问题。

**MCP：让 Claude Code 连接内部工具。**

比如内部文档、工单系统、指标平台、发布平台、结构化搜索。

Claude Code 本地文件系统看不到这些。

MCP servers 就是把这些内部能力变成 Claude 可以调用的工具。

所以 MCP 解决的是外部工具接入问题。

**LSP：让 Claude Code 按符号找代码。**

大代码库里，单纯 grep 经常会出问题。

一个 `handle()`、`init()`、`process()`，可能全仓库几千个。

Grep 返回一堆字符串匹配，Claude 还要继续读文件判断哪个是真相关。

LSP 不一样。

它能像 IDE 一样做 go to definition、find references。

也就是按符号关系找，而不是按字符串找。

对于 C、C++、Java、C# 这种大工程，LSP 非常有价值。

**Subagents：把探索和编辑拆开。**

大代码库里，探索会产生大量噪音。

读目录、查引用、看历史、对比模块，这些都很吃上下文。

更稳的做法是让子 Agent 先做只读探索，最后把结论交给主 Agent。

主 Agent 再基于结论编辑代码。

这样主上下文不会被探索过程污染。

这和我们之前讲上下文管理是同一个逻辑：

**探索噪音隔离，关键结论回传。**

![图8：Plugins、MCP、LSP、Subagents 不是同一类东西，它们分别解决分发、接入、定位和隔离问题](https://file1.kamacoder.com/i/web/20260527192254_claude_large_codebase_08_component_boundaries_compressed.png?v=20261005)

### 十、大代码库应该按什么顺序配置 Claude Code？

这里我给一个落地顺序。

不是说所有团队都必须一模一样。

但大部分团队可以按这个顺序走。

**第一步，先写根 CLAUDE.md。**

不要追求完美。

先写：

- 仓库结构
- 常用命令
- 全局禁区
- 关键文档入口
- 最容易踩的坑

**第二步，补子目录 CLAUDE.md。**

哪个目录最常被 Claude Code 修改，就先给哪个目录补。

比如 `apps/web/CLAUDE.md`、`services/payment/CLAUDE.md`。

重点写本目录命令和规则。

**第三步，配置忽略和权限。**

生成代码、构建产物、第三方库、巨型快照文件，该排除就排除。

不要让 Claude Code 把时间浪费在垃圾上下文上。

**第四步，接 hooks。**

先接最确定的：

- format
- lint
- typecheck
- 单测
- 会话结束总结

**第五步，把重复专家流程做成 Skills。**

安全审查、文档更新、发布检查、事故复盘，这些都适合抽成 skill。

**第六步，再考虑 Plugins、MCP、LSP、Subagents。**

前面的基础没做好，直接上 MCP 和一堆复杂插件，容易越配越乱。

先让 Claude Code 在本地仓库里找得准、改得稳。

再接更多外部能力。

### 十一、团队落地：为什么需要一个 Agent 管理人？

技术配置只是第一步。

大团队里，Claude Code 真正难的是组织落地。

如果每个工程师都自己配一套：

- 自己写 CLAUDE.md
- 自己写 hooks
- 自己配 MCP
- 自己总结使用技巧

那很快就会变成各玩各的。

好经验留在个人电脑里。

踩过的坑没人沉淀。

新人进来还要重新摸索。

Anthropic 文章里提到一个角色：agent manager。

你可以理解成：

**负责维护 AI 编程工程体系的人。**

这个人不一定是专职。

小团队里，一个 DRI 就够。

但他要负责：

- 统一 CLAUDE.md 层级规范
- 管理 hooks 和权限策略
- 维护团队可用 Skills
- 选择和分发 Plugins
- 管理 MCP 接入边界
- 推动代码审查和安全流程
- 定期清理过期规则

大公司里，这个角色可能在 DevEx、开发效率、基础架构团队下面。

因为它本质上是开发者工具的一部分。

不是某个开发者的个人爱好。

![图9：团队落地 Claude Code 的关键是把个人经验沉淀成可复用的工程资产](https://file1.kamacoder.com/i/web/20260527192255_claude_large_codebase_09_team_asset_loop_compressed.png?v=20261005)

这里我觉得 Anthropic 说得很现实：

**自下而上的使用会带来热情，但没有集中维护，知识会停留在小圈子里。**

这句话非常适合国内团队。

很多公司现在都是几个工程师先偷偷用。

用得好的那个人效率很高。

但团队并没有真正吃到红利。

因为方法没有标准化，也没有沉淀成项目资产。

### 十二、普通开发者怎么开始？

如果你不是团队负责人，只是一个普通开发者，也能做很多事。

不要一上来就想着搭 MCP、写 plugin、搞一套平台。

先从最小动作开始。

**第一，给当前项目写一个根 CLAUDE.md。**

哪怕只有 20 行也行。

写清楚项目结构、常用命令、禁止事项。

**第二，把你反复提醒 Claude Code 的话沉淀进去。**

凡是你连续说过三次的规则，都应该考虑写进文件。

**第三，给常改目录补子目录 CLAUDE.md。**

比如你经常改 `docs/`，那就在 `docs/` 下面写文档规则。

比如你经常改 `services/payment/`，那就写支付模块规则。

**第四，排除明显无关文件。**

生成代码、构建产物、三方依赖，别让 Agent 老去读。

**第五，把重复流程整理成 skill 或脚本。**

比如每次发版前都要检查 5 件事。

不要每次口头提醒。

把它沉淀成可复用流程。

大代码库里，AI 编程不是"换个更强模型"就完了。

真正能拉开差距的是：

**你能不能把项目知识、工具链和团队规范，变成 Claude Code 可以稳定使用的工程支架。**

### 结尾

Claude Code 在大代码库里，不是靠把整个仓库背下来。

它更像一个会使用工具的新同事。

你给它清晰目录、局部规则、测试命令、符号导航、自动检查，它就能越做越稳。

你不给它这些，它就只能盲搜、乱猜、浪费上下文。

所以这篇文章最重要的一句话是：

**大代码库里的 Claude Code，不是模型单打独斗，而是模型 + 工程支架一起工作。**

会用 Claude Code，只是开始。

能把 Claude Code 接进团队工程体系，才是真正的 AI 编程落地。

加油。

### 参考资料

- Anthropic Blog：How Claude Code works in large codebases: Best practices and where to start：https://claude.com/blog/how-claude-code-works-in-large-codebases-best-practices-and-where-to-start
- Claude Code Docs：Manage Claude's memory：https://docs.claude.com/en/docs/claude-code/memory
- Claude Code Docs：Hooks：https://docs.claude.com/en/docs/claude-code/hooks
- Claude Code Docs：Skills：https://docs.claude.com/en/docs/claude-code/skills

### 本节收尾：费曼自述与闪卡

**费曼自述：** “给 Agent 一张项目地图和可靠的验证命令，让它根据任务逐步找代码。”

**概念闪卡：** Q：项目说明文件能代替 README 吗？A：两者读者和用途可能不同；项目可保留给人看的 README 与给工具的工作规则。Q：大代码库为何先搜索？A：减少无关上下文并定位真正要改的模块。

原文核对：[Claude Code怎么读懂大代码库？Agent搜索、CLAUDE.md、Hooks、Skills、MCP和LSP一篇讲明白](原仓库Markdown/docs/llm/claude/claude_code_large_codebase.md)。

---

<a id="a7"></a>

## 附.7 Claude Code作者说“不写Prompt，写Loop”：AI编程从提示词到Agent闭环到底变了什么【◻】

**一句话：** 好的 Agent 不只靠一句 Prompt，而靠“尝试→观察结果→验证→修正→停止”的运行闭环；harness 是支撑这个循环的工具、权限、状态和验收环境。

**例子：** 让编程 Agent 修改一个接口：先定位代码，编辑，运行测试，看错误并修复，直到通过或达到停止预算。若没有测试和状态记录，Agent 可能反复改同一处。动态工作流和多 Agent 只是可选实现，不能自动提高质量。

### 一、为什么只写 Prompt 不够了？

先别急着把 prompt 贬得一文不值。

Prompt 当然有用。

你问一个概念，让 AI 解释一下。

你贴一段代码，让 AI 帮你改写。

你让它生成一个 SQL、一个正则、一个接口文档。

这些场景里，prompt 就够了。

因为任务短、边界清楚、结果也容易人工判断。

但 AI 编程不是只做这种小任务。

真实项目里，你让 Claude Code 修一个 bug，往往不是一句话就能完成。

它要先读代码。

要定位调用链。

要判断哪个文件该改，哪个文件不能动。

要修改实现。

要跑测试。

测试失败了，还要回头看日志。

日志里可能暴露了另一个问题。

修完之后，还要确认没有引入新的副作用。

这时候你会发现，**prompt 只是任务的起点，不是任务的完成机制。**

你写一句：

“帮我修一下登录失败的问题。”

这句话只表达了目标。

但它没有说：

- 去哪里找入口
- 哪些文件不能动
- 怎么判断修好了
- 测试失败了怎么办
- 什么时候必须停下来问人
- 什么改动算越界

这些东西如果都靠模型自己猜，就会出现两个问题。

第一，模型会浪费大量上下文摸索。

第二，模型可能看起来很努力，最后却朝错方向越走越远。

很多录友说“AI 写代码不稳定”，根源不是它每一行代码都差。

而是你只给了它一个 prompt，却没有给它一个能稳定工作的闭环。

### 二、Loop 到底是什么？

Loop 不是玄学。

也不是让 AI 一直自动跑，跑到天荒地老。

一个最小的 Agent Loop，大概是这样：

**目标 → 计划 → 执行 → 观察 → 验证 → 反馈 → 下一步**

如果验证通过，就停止。

如果验证失败，就把失败信息带回去，重新计划。

如果触碰边界，就暂停，让人决策。

这才叫 loop。

![Agent Loop 的最小闭环结构图](https://file1.kamacoder.com/i/web/20260609100830_claude_code_loop_02_minimal_loop_compressed.png?v=20261005)

你看，这和普通 prompt 的差别很大。

普通 prompt 更像是：

“我说一句，你答一句。”

Loop 更像是：

“我给你目标、工具、规则和验收标准，你自己推进，但每一步都要留下证据。”

这也是为什么 Claude Code 这类工具和网页聊天框差别很大。

网页聊天框里，你主要是在和模型对话。

Claude Code 里，模型不是只回答你。

它会读文件、搜代码、改文件、跑命令、看报错、继续修。

这些动作串起来，才是 Agent Loop。

所以 Boris 说“不写 prompt，写 loop”，不是说一句提示词都不写了。

而是说：**不要把 AI 编程理解成“我怎么问得更好”，要理解成“我怎么设计一个可执行的闭环”。**

### 三、坏 Loop 比坏 Prompt 更危险

这里我要泼点冷水。

Loop 听起来高级，但坏 loop 很危险。

一个坏 prompt，最多就是回答不理想。

一个坏 loop，可能会一直跑、一直改、一直消耗 token，还可能把项目改乱。

很多人第一次用 Agent 工具，容易上来就说：

“你自己看着办，直到修好为止。”

这句话听起来很潇洒。

但对 Agent 来说，这就是一个危险指令。

因为“修好”没有被定义。

没有测试。

没有验收标准。

没有权限边界。

没有停止条件。

那它只能自己脑补。

脑补就会出事。

![坏 Loop 和好 Loop 的差异路径图](https://file1.kamacoder.com/i/web/20260609100832_claude_code_loop_03_bad_vs_good_loop_compressed.png?v=20261005)

好的 loop 至少要有四个东西：

**第一，有明确目标。**

不是“优化一下性能”，而是“让接口 P95 从 800ms 降到 300ms 以内，不能改变返回结构”。

**第二，有可执行工具。**

不能只让它想。

要能读代码、跑测试、看日志、查文档、调用脚本。

**第三，有验证反馈。**

测试结果、lint 结果、构建结果、接口返回、日志差异，这些都要能回到上下文里。

**第四，有停止条件。**

什么时候算完成？

什么时候算失败？

什么时候必须停下来问人？

这四个东西缺一个，loop 就会变成“AI 自己猜自己对不对”。

这种用法，不是先进。

是危险。

### 四、Prompt 工程和 Loop 工程不是一回事

很多录友之前学 prompt engineering，重点是怎么把一句话写清楚。

比如：

- 给角色
- 给背景
- 给输出格式
- 给示例
- 给约束

这些还要不要？

当然要。

但在 Agent 场景里，这些只是第一层。

更重要的是 loop engineering。

它关心的是：

- 任务怎么拆
- 信息怎么进上下文
- 工具怎么调用
- 失败怎么反馈
- 规则怎么长期保存
- 验收怎么自动化
- 权限怎么限制
- 成本怎么控制

你会发现，这已经不是“文案能力”了。

这更像工程能力。

![Prompt 工程和 Loop 工程的能力边界对比图](https://file1.kamacoder.com/i/web/20260609100834_claude_code_loop_04_prompt_vs_loop_engineering_compressed.png?v=20261005)

举个例子。

你让 Claude Code 修改一个支付回调 bug。

只写 prompt 的人会说：

“帮我修一下支付回调偶尔失败的问题，注意代码质量。”

这太虚了。

会写 loop 的人会这样给任务：

“先从支付回调入口开始读，不要修改数据库迁移文件。定位失败原因后，给出改动计划。修改完成后跑支付模块单测，如果测试失败，优先根据错误日志修复；如果涉及幂等逻辑或金额计算，先停下来说明风险，不要直接改核心规则。”

你看，这里不是提示词更华丽。

而是它把 loop 的关键部件放进去了：

- 搜索入口
- 禁止修改范围
- 计划阶段
- 测试反馈
- 错误恢复
- 高风险暂停点

这就是差别。

**Prompt 工程解决“模型怎么回答”。**

**Loop 工程解决“任务怎么被可靠完成”。**

### 五、Claude Code 为什么天然适合 Loop？

Claude Code 的价值，不是它把 Claude 模型搬进了终端。

如果只是终端里聊天，那没什么稀奇。

它真正厉害的地方，是把 AI 编程常见动作做成了可循环执行的系统。

一个真实的 Claude Code 工作流大概是这样：

你给一个目标。

它先读项目规则，比如 `CLAUDE.md`。

然后用搜索工具找入口。

读相关文件。

形成修改计划。

改代码。

跑命令。

看输出。

如果报错，就把错误日志作为新证据，继续修。

如果测试通过，就收束改动，给你结果。

![Claude Code 从读代码到验证结果的 Agent Loop 工作流图](https://file1.kamacoder.com/i/web/20260609100836_claude_code_loop_05_claude_code_workflow_compressed.png?v=20261005)

这套东西看起来像“AI 会自己干活”。

但拆开看，里面都是工程问题：

搜索准不准。

上下文有没有污染。

工具返回值够不够清楚。

测试命令能不能跑。

错误信息有没有被正确利用。

权限边界有没有卡住。

所以我一直说，Claude Code 不是一个“更聪明的聊天框”。

它是一个 Agent 执行环境。

你写 prompt，是在给模型一句话。

你写 loop，是在给 Agent 一套工作方式。

### 六、没有验证的 AI 编程，都是表面完成

AI 编程最容易骗过人的地方是什么？

不是代码写得差。

而是它很会“看起来完成了”。

它会给你一个很完整的解释。

会列出改了哪些文件。

会告诉你问题已经修复。

甚至还能把理由说得很顺。

但如果没有测试、构建、日志、人工验收这些反馈，它说“完成了”没有意义。

**AI 的完成感，不等于工程上的完成。**

![没有验证反馈的 AI 编程为什么容易表面完成图](https://file1.kamacoder.com/i/web/20260609100838_claude_code_loop_06_validation_failure_compressed.png?v=20261005)

真实项目里，我们判断一个改动能不能交付，不是看它解释得多漂亮。

而是看：

- 单测有没有过
- 构建有没有过
- 关键路径有没有回归
- 日志有没有异常
- 用户行为有没有变化
- 边界条件有没有覆盖

这些东西，就是 loop 的“眼睛”。

没有它们，Agent 只能靠语言自信。

语言自信是最不值钱的。

很多录友做 AI 项目，喜欢把重点放在 prompt 模板上。

我建议你换个角度：

**先问自己，这个任务有没有反馈信号。**

没有反馈信号，就不要轻易让 Agent 自动循环。

它不知道什么叫对。

不知道什么叫错。

那就只能越跑越玄学。

### 七、开发者不是被替代了，而是位置变了

看到“不写 prompt，写 loop”，有些人会紧张：

那是不是以后开发者不用写代码，只写规则？

没这么简单。

更准确地说，开发者的工作位置在变化。

以前你亲手写每一行代码。

后来你开始用 AI 生成代码，但你还要一直盯着它、提醒它、复制粘贴错误。

再往后，你要把这些反复提醒，沉淀成规则、测试、脚本、文档和权限边界。

也就是把“人盯人”变成“系统约束”。

![开发者从盯着 AI 到设计规则闭环的角色迁移图](https://file1.kamacoder.com/i/web/20260609100839_claude_code_loop_07_human_role_shift_compressed.png?v=20261005)

比如以前你会反复提醒：

“别改这个文件。”

现在你应该把它写进项目规则，或者用权限拦住。

以前你会反复说：

“改完跑测试。”

现在你应该让测试命令成为默认验证步骤。

以前你会在聊天里解释项目结构。

现在你应该把关键入口写进 `CLAUDE.md`。

以前你靠肉眼判断它有没有改对。

现在你要补测试、补脚本、补验收样例。

这不是开发者价值下降。

恰恰相反。

**越是 Agent 能干活，越需要懂工程的人来定义什么叫干对。**

不会写代码的人，很难写出好的 loop。

因为他不知道哪里危险，不知道哪里要验证，不知道什么改动会引发连锁问题。

所以别把 loop 理解成“AI 替我做所有事”。

它更像是：

**开发者把自己的工程判断，外化成 Agent 可以执行的闭环。**

### 八、未来开发者到底要写什么？

如果说过去开发者主要写代码。

现在开发者开始写 prompt。

那再往后，开发者会越来越多地写这些东西：

- 任务目标
- 项目规则
- 测试用例
- 验收标准
- 工具接口
- 权限边界
- 错误恢复策略
- 监控和评估指标

这些东西合在一起，就是 loop 的骨架。

![AI Coding 时代人、Agent、模型、工具的新分工图](https://file1.kamacoder.com/i/web/20260609100841_claude_code_loop_08_new_division_compressed.png?v=20261005)

模型负责生成。

Agent 负责执行。

工具负责连接真实环境。

测试和日志负责反馈。

人负责目标、边界和最终判断。

这个分工如果看懂了，你就不会再纠结“提示词该怎么写得更神”。

你会开始关心：

**我的项目有没有足够清楚的规则？**

**我的任务有没有可验证的完成标准？**

**我的测试能不能给 Agent 稳定反馈？**

**我的工具返回值能不能让模型看懂？**

**我的权限边界会不会让 Agent 乱动生产资源？**

这些问题，才是 AI 编程真正开始工程化之后的核心问题。

### 九、普通开发者怎么开始写 Loop？

如果你现在还只是偶尔让 AI 写点代码，不需要一下子搞很复杂。

从几个小动作开始就行。

**第一，别只写目标，要写验收。**

不要只说“修复登录问题”。

加一句：

“修完后跑登录模块测试，并说明失败用例是否恢复。”

**第二，别只让它改，要让它先定位。**

不要上来就说“直接改”。

先让它：

“先找登录入口、认证中间件和错误日志位置，列出可能原因，再改。”

**第三，给它边界。**

比如：

“不要修改数据库 schema。”

“不要改公开 API 返回结构。”

“不要调整鉴权策略，除非先说明风险。”

**第四，把反复说的话写进项目规则。**

如果你每次都提醒同一件事，就别再靠聊天提醒了。

写进 `CLAUDE.md`。

这就是从 prompt 走向 loop 的第一步。

**第五，优先补反馈信号。**

没有测试的项目，用 Agent 会很累。

不是不能用。

但你会一直靠人工判断。

一旦补了测试、lint、构建脚本，Agent 才有东西可以观察，loop 才跑得起来。

### 十、最后

Boris 这句话真正值得关注的地方，不是“不写 prompt”这四个字。

而是它提醒我们：

**AI 编程正在从对话技巧，进入工程闭环。**

Prompt 是入口。

Loop 是系统。

只会写 prompt，你最多让模型回答得更像你想要的样子。

会写 loop，你才能让 Agent 在真实项目里持续推进、自动验证、出错纠偏、边界内行动。

这就是差距。

以后 AI 编程里最值钱的能力，不是会不会喊 AI 干活。

而是你能不能把一个模糊任务，拆成一个可执行、可验证、可停止的闭环。

这才是工程能力。

### 本节收尾：费曼自述与闪卡

**费曼自述：** “模型给出动作只是开始，验证反馈决定能否真正完成任务。harness 把这条反馈链变成可运行的系统。”

**概念闪卡：** Q：Loop 的停止条件有哪些？A：验收通过、失败不可恢复、预算/时间上限或需要人工输入。Q：为什么多 Agent 不一定更好？A：通信、权限和成本会增加。

原文核对：[Claude Code作者说“不写Prompt，写Loop”：AI编程从提示词到Agent闭环到底变了什么](原仓库Markdown/docs/llm/claude/claude_code_loop.md)。

---

<a id="a8"></a>

## 附.8 Claude Skills实战：Anthropic用了几百个Skill，总结出9大分类和写好Skill的全部经验【◻】

**一句话：** Skill 封装任务方法，Hook 在事件触发时执行检查，MCP 连接外部工具/资源；它们解决不同问题，不能互相替代。

**例子：** “运行测试并解释失败”可以写成复用流程；每次提交前自动格式检查适合 Hook；读取外部文档系统可通过受控工具/MCP。是否需要这些机制取决于重复任务的频率和维护成本。

### 先纠正一个误解：Skill 不是一个 markdown 文件

很多录友一听 Skill，第一反应是「不就是写个 `.md` 文件告诉 Claude 怎么做嘛」。

**这个理解是错的，而且会让你写出来的 Skill 特别弱。**

官方原话：Skill 是「指令、脚本和资源的文件夹」（folders of instructions, scripts, and resources）。注意，是**文件夹**，不是文件。

![Skill文件夹解剖](https://file1.kamacoder.com/i/web/20260616181017_skill_anatomy.png?v=20261005)

一个像样的 Skill 长这样：

- `SKILL.md`：主指令 + 一段描述。这段描述很关键，它决定了 Claude **什么时候**会触发这个 Skill。
- `references/`：参考文档。Claude 不会一上来全读，而是用到哪份才读哪份。
- `scripts/`：可执行脚本。能跑代码就别让 Claude 现写，直接调。
- `assets/`：模板、示例数据这类资源。
- `config.json`：配置信息，缺了 Claude 还能反问你来补。

为什么这个区别重要？因为一旦你把 Skill 当成「一个文档」，你就会拼命往一个 md 里塞东西，最后塞成一坨谁也读不动的长文。而把它当成「一个文件夹」，你才会去想：哪些是脚本、哪些是按需读的参考、哪些是模板。这才是 Skill 真正的威力。

### Anthropic 把几百个 Skill，归成了 9 类

Anthropic 内部把所有 Skill 盘了一遍，发现它们能干净地落进 **9 个分类**。而且他们有个很重要的观察：

**最好的 Skill 只干一类事；什么都想干的 Skill，反而会把 Claude 搞糊涂。**

这 9 类长这样：

![Skill九大分类](https://file1.kamacoder.com/i/web/20260616181018_skill_nine_categories.png?v=20261005)

挨个说一下，对照着看你自己手头哪些活能做成 Skill：

1. **库 / API 参考**：讲清楚某个库、CLI、SDK 怎么用，重点是把边界情况写明白。比如内部计费库的各种坑、内部平台 CLI 的子命令示例。
2. **产品验证**：写完代码怎么验证它真的跑通了、真的能用。常配合 Playwright、tmux 这类工具。原文专门点名——**这一类在内部对产出质量的提升是最可量化、最明显的。**
3. **数据获取与分析**：接上数据和监控，比如漏斗查询、留存对比、Grafana / Datadog 的字段映射。
4. **业务流程与团队自动化**：把重复的协作流程压成一条命令。比如自动聚合大家的日报、按固定 schema 建工单。
5. **代码脚手架 / 模板**：按框架生成样板代码，注解、配置都预先填好。
6. **代码质量与评审**：强制代码规范、辅助 review。一个很妙的玩法是「对抗式评审」——另起一个全新视角的子 Agent 来挑刺。
7. **CI/CD 与部署**：拉代码、推代码、发布。比如盯着 PR、CI 挂了自动重试，灰度放量、出问题自动回滚。
8. **Runbook 排障手册**：从一个症状出发（一条 Slack 报错、一个报警），用多个工具一步步查，最后产出一份结构化的排查报告。
9. **基础设施运维**：带护栏的运维操作，比如清理孤儿资源、依赖管理。

这 9 类不用背，关键是那个原则：**一个 Skill 对应一类事**。你要是发现自己在写一个「既能查数据又能发布还能排障」的超级 Skill，停一下，拆开。

### 怎么写好一个 Skill？这几条最硬

分类讲完，重头戏是怎么把一个 Skill 写好。原文给了一长串原则，我挑出对录友们最有用的几条。

#### 别说废话，只写能把 Claude「推出惯性」的东西

Claude 本身就懂很多编程知识。所以如果你的 Skill 主要是传递知识，**别去写它本来就会的东西**。

把笔墨花在那些能**把 Claude 推出默认思路**的信息上。原文举的例子是前端设计 Skill：与其告诉 Claude「怎么写 CSS」（它会），不如告诉它「我们团队的设计偏好是什么」（它不知道，而且默认风格往往不是你要的）。

一句话：**写 Claude 不知道的，别写它已经知道的。**

#### Gotchas（坑）区，是整个 Skill 信号最强的部分

原文有个判断我特别认同：**一个 Skill 里信息密度最高、最值钱的部分，是「坑」。**

什么叫坑？就是那种「不告诉你，你一定会踩」的东西。比如：

- 同一个字段，在 A 系统叫 `user_id`，在 B 系统叫 `uid`；
- 某张表是 append-only 的，你以为能 update，其实只能插入。

这种知识在文档里通常找不到，全靠人踩出来。把它写进 Skill 的 Gotchas 区，价值远大于你复述十遍官方文档。

#### 渐进式披露：SKILL.md 当索引，用到才读

这条是 Skill 设计的精髓，也是它和「一个长 md」最本质的区别。

![渐进式披露](https://file1.kamacoder.com/i/web/20260616181019_progressive_disclosure.png?v=20261005)

笨办法是：把所有参考文档、所有脚本源码、所有模板、每个边界情况的长篇说明，**一股脑全塞进上下文**。结果还没开始干活，上下文窗口就满了，Claude 反而抓不住重点。

聪明办法叫**渐进式披露（progressive disclosure）**：`SKILL.md` 只放一个轻量的概要和「指针」——告诉 Claude「要做计费相关的事，去看 `references/billing.md`」。Claude 真用到了，才去读那个文件。

这其实就是**上下文工程**。文件夹结构本身就是你给 Claude 安排的上下文加载顺序。关于 Claude Code 怎么管理上下文、怎么读大代码库，我们在 [Claude Code 作者说「不写 Prompt，写 Loop」](原仓库Markdown/docs/llm/claude/claude_code_loop.md) 里聊过同一套思路，可以串起来看。

#### 描述（description）是写给模型看的，不是写给人看的

`SKILL.md` 里那段描述，决定了 Claude **什么时候触发这个 Skill**。

所以它不是给人看的简介，是给模型看的「触发条件」。原文的建议很直接：**把触发关键词直接塞进去**。比如一个盯 PR 的 Skill，描述里就该出现「babysit」这种用户真会说的词，而不是写一句文绉绉的「本技能用于持续监控合并请求的状态」。

录友们写的时候换个视角：用户会用什么话来唤起这个能力？把那些词写进描述。

#### 让 Claude 记住自己干过什么

这条很容易被忽略，但很实用：**Skill 可以存数据，让 Claude 读自己的历史。**

做法是在 Skill 里维护一个 append-only 的日志或 JSON 文件，Claude 每次跑的时候读一眼自己上次的记录，据此调整这次的行为。官方还提了一个 `${CLAUDE_PLUGIN_DATA}` 环境变量，用来做持久化存储。

相当于给 Skill 加了一层「记忆」。Agent 的记忆怎么分短期、长期，我们在 [Agent 的记忆](原仓库Markdown/docs/llm/app/agent_memory.md) 里专门拆过，Skill 这个用法本质上就是在做长期记忆。

#### 存脚本、给现成代码，让 Claude 把算力花在「决策」上

还有一条工程味很重的建议：**能给现成脚本和函数，就别让 Claude 现写。**

原文说得很到位——这样 Claude 就能「把它的回合花在组合上，决定下一步做什么，而不是重新拼样板代码」。

翻译成人话：Claude 的每一步推理都是有成本的。你把重复的样板活做成脚本，它就能省下脑子去想真正需要判断的事。这跟我们一直强调的 [工具设计决定 Agent 上限](原仓库Markdown/docs/llm/app/agent_tool_design.md) 是一个道理。

### 别想着一上来就写个完美 Skill

最后这条，是我觉得整篇博客最让人安心的一句话。

Anthropic 内部那些最好用的 Skill，**几乎都不是设计出来的，是养出来的。**

原文：大多数 Skill 一开始就是**几行字 + 一个坑**，后来变好，是因为大家在 Claude 不断撞到新边界情况的时候，一条一条往里加。

![Skill成长路径](https://file1.kamacoder.com/i/web/20260616181021_skill_evolution.png?v=20261005)

所以路径很清楚：

- 起步：写最小版，几行字 + 一个你已知的坑；
- 撞坑：Claude 在真实使用里遇到新情况、出了错；
- 补一条：把这个坑写进 Gotchas；
- 重复上面两步，循环往复，它就慢慢长成了一个靠谱、高频复用的 Skill。

别一上来就憋大招，憋出来的往往是个又长又没人用的文档。**先写最小版，让它在真实使用里长大。**

### 写在最后

Skill 这套东西，官方自己都说「最佳实践还在演进中」。但方向已经很明确了：它不是「一个 md 文件」，而是一个**带脚本、带资源、带记忆、按需加载**的文件夹；它最好**只干一类事**；它最值钱的部分是**坑**；它最该有的姿势是**先小再长大**。

录友们与其纠结理论，不如挑一个你每天都在重复的活——验证某个功能、查某张表、走某个发布流程——写成一个最小的 Skill，明天就开始用。

### 参考链接

- Anthropic 官方博客（How we use skills）：https://claude.com/blog/lessons-from-building-claude-code-how-we-use-skills

### 本节收尾：费曼自述与闪卡

**费曼自述：** “流程说明、自动触发、外部连接是三种不同扩展点；先找到实际重复工作再加。”

**概念闪卡：** Q：Hook 与 Skill 的核心差别？A：前者按事件自动执行，后者是可选择使用的能力/流程封装。Q：MCP 直接保证数据权限吗？A：协议不替代服务端授权。

原文核对：[Claude Skills实战：Anthropic用了几百个Skill，总结出9大分类和写好Skill的全部经验](原仓库Markdown/docs/llm/claude/claude_skills.md)。

---

<a id="a9"></a>

## 附.9 Claude Code 动态工作流详解：让 Claude 自己现写一套 harness，把一个任务拆给一队 Claude 去干【◻】

**一句话：** 好的 Agent 不只靠一句 Prompt，而靠“尝试→观察结果→验证→修正→停止”的运行闭环；harness 是支撑这个循环的工具、权限、状态和验收环境。

**例子：** 让编程 Agent 修改一个接口：先定位代码，编辑，运行测试，看错误并修复，直到通过或达到停止预算。若没有测试和状态记录，Agent 可能反复改同一处。动态工作流和多 Agent 只是可选实现，不能自动提高质量。

### 先搞清楚：默认那套 harness，是为「写代码」定制的

harness 是什么？就是控制 Claude **读什么、什么时候动手、产出怎么验证**的那层编排。Claude Code 自带的 harness 打磨得很好，但有个前提——它是冲着**写代码**这件事调出来的。

可你实际让 Claude Code 干的活，远不止写代码：深度调研、安全审计、给一队 Agent 分工、代码评审……这些任务的「最佳跑法」和写代码不一样。过去要把它们做到极致，你得**自己在 Claude Code 之上手搓一套 harness**。

动态工作流的意思就一句话：**这套针对任务的 harness，不用你手搓了，Claude 自己现场写。**

<!-- drawio源文件: ./drawio/dynamic_workflows_01_harness.drawio -->

![固定 harness 想通吃所有任务，对比动态工作流让 Claude 按任务现写专属 harness](https://file1.kamacoder.com/i/web/20260617184001_dw01.png?v=20261005)

这张图回答的是：默认 harness 是一套尺寸想套住所有任务，要做到极致只能你手搓；动态工作流则是 Claude 分析完任务，给调研、安全、评审各现写一套量身的外壳。

### 关键一步：工作流是「真的代码」，不是提示词

很多录友一听「工作流」，第一反应是「不就是写一段精心调过的提示词嘛」。**不是。**

**动态工作流是 Claude 现场生成、然后真的拿去执行的 JavaScript 文件。** 它里头有几个基本积木：

- `agent(prompt, opts)`：开一个子 Agent，给它一个干净独立的上下文窗口，还能用 JSON Schema 约束它输出的结构；
- `parallel(...)`：一道「栅栏」，几个任务全跑完才放行往下走；
- `pipeline(items, ...)`：让一批东西流水线式地过几道工序，A 项可以先走，B 项慢一点也不挡道。

Claude 拿这几块积木，针对你的任务搭出一套编排：要不要开子 Agent、开几个、哪步用强模型哪步用便宜模型、要不要用 git worktree 把不同方案隔开、哪些并行哪些串行——**这些都不是你预先规定死的，是 Claude 分析完任务自己定的。**

### 为什么非得这么折腾？单个 context 扛不动

因为长任务里，一个 Claude 单挑全部时，老三样毛病会准时冒出来（这几个我们在 [Agent 的失败模式](原仓库Markdown/docs/llm/app/agent_failure_modes.md) 里专门拆过）：

<!-- drawio源文件: ./drawio/dynamic_workflows_02_isolation.drawio -->

![单个上下文窗口扛全部任务会偷懒自夸跑偏，拆成多个隔离子Agent后各个击破](https://file1.kamacoder.com/i/web/20260617184002_dw02.png?v=20261005)

- **偷懒（agentic laziness）**：审 50 个文件审到 35 个就说「干完了」——不是模型坏了，是上下文塞满了，它把「差不多」当成了「做完」；
- **自夸（self-preferential bias）**：让 Claude 验自己的活，它会偏向维护自己之前的结论，跟「让学生给自己的卷子打分」一个道理；
- **跑偏（goal drift）**：对话拉长，反复压缩摘要，把边界要求、约束条件一点点丢掉，最后解的已经不是原来那个问题了。

动态工作流的解法是**结构性**的——与其让一个 Claude 扛全部，不如开一堆互相隔离、各自上下文干净、只盯一件事的子 Agent：

- 隔离 → 没有哪个 Agent 握着整个任务，**偷懒无处藏**；
- 独立的验证 Agent → 验活的和干活的不是同一个，**自夸自然消失**；
- 每个窗口都短 → **跑偏被摁住**。

### 六种最常复用的编排模式

博客把实战里反复出现的编排归成了六类，照着对你自己手头的活：

| 模式 | 怎么跑 | 适合 |
|------|--------|------|
| 分类即处理 | 先分流，再把每一类路由给专门的 Agent | 工单 triage、产出归档 |
| 扇出+汇总 | 拆成多份并行各跑一个 Agent，最后栅栏处汇总 | 审计、跨模块独立评审、深度调研 |
| 对抗验证 | 一个 Agent 提出，另一个专挑刺，两边不共享上下文 | 根因排查、结论复核 |
| 生成+筛选 | 先广撒网生成一堆候选，再按标准筛掉弱的 | 取名、方案探索、测试用例 |
| 锦标赛 | 多个 Agent/模型同题竞赛，裁判两两比较选优 | 模型路由、方案择优 |
| 跑到收工为止 | 反复「动手—检查—修复」，直到满足明确的停止条件 | 开放式排查、清扫式发现 |

其中**对抗验证**最该记住：验活的和干活的彻底分家、互不通气，一个结论只有在「反驳失败」之后才算数。这跟 [Claude Skills 实战](原仓库Markdown/docs/llm/claude/claude_skills.md) 里讲的「另起一个全新视角的子 Agent 来挑刺」是同一招，只不过这里把它做成了可复用的编排。

### 怎么开、怎么存、怎么分享

- **触发**：直接说「给这个任务做一套工作流」，或者把 effort 调到 `ultracode`，让 Claude 自己判断什么时候该上工作流；
- **盯进度**：`/workflows` 打开一个运行面板，每个阶段、每个子 Agent、每次工具调用和 token 消耗都看得到，还能暂停、跳过卡住的、重试单个 Agent；
- **可恢复**：每次运行有 ID，断了能接着跑，跑过的阶段直接走缓存、只补没干完的——这点和 Managed Agents 的 session 可恢复是一个味道；
- **存下来复用**：在面板里按 `s` 存到 `~/.claude/workflows`。一次漂亮的根因排查，就能沉淀成一套「根因排查工作流」；一次漂亮的迁移，就能沉淀成「迁移工作流」。

分享也简单：把这个 JS 文件丢进 [Skill](原仓库Markdown/docs/llm/claude/claude_skills.md) 文件夹发给队友。但记住它是**模板不是死脚本**——会根据具体任务自适应，而不是一行不差地重放。

### 但别滥用：它很烧 token

最后泼盆冷水。**动态工作流比普通单 Agent 会话烧多得多的 token**，因为它要协调一整队 Agent。

一个诚实的判断标准：**这个任务，是不是真的需要比「一个上下文窗口」更多的算力？**

- 值得上：50 个文件的安全审计、上千行的排序打分、高不确定要反复探索的调研、高风险要独立验证的改动、高复用值得沉淀成模板的流程；
- 别上：两行的 bug、改一个文件——那纯属杀鸡用牛刀，钱白烧。

它和 [Agent vs Workflow](原仓库Markdown/docs/llm/app/agent_vs_workflow.md) 那篇的判断一脉相承：先想清楚任务到底需不需要这么重的编排，再决定上不上。它也是 Agent Teams 的「按需版」——角色固定就用 Teams，角色得临场拆解就用动态工作流。目前这功能还是 research preview，跑在 Claude Opus 4.8 上。

### 写在最后

从 [不写 Prompt，写 Loop](原仓库Markdown/docs/llm/claude/claude_code_loop.md)，到 Managed Agents 把 harness 托管，再到今天 Claude 自己现写 harness——这条线越来越清楚：**编排这件事，正在从「人来设计架构」挪到「模型自己决定」。** 你只负责说清楚什么算成功、信任边界在哪，剩下那套外壳，Claude 自己搭。

录友们哪怕暂时不开这个功能，「把一个大任务拆给一队互相隔离的 Claude、再汇总」这个思路，也值得记进脑子里。

### 参考链接

- Anthropic 官方博客（A harness for every task: dynamic workflows in Claude Code）：https://claude.com/blog/a-harness-for-every-task-dynamic-workflows-in-claude-code

### 本节收尾：费曼自述与闪卡

**费曼自述：** “模型给出动作只是开始，验证反馈决定能否真正完成任务。harness 把这条反馈链变成可运行的系统。”

**概念闪卡：** Q：Loop 的停止条件有哪些？A：验收通过、失败不可恢复、预算/时间上限或需要人工输入。Q：为什么多 Agent 不一定更好？A：通信、权限和成本会增加。

原文核对：[Claude Code 动态工作流详解：让 Claude 自己现写一套 harness，把一个任务拆给一队 Claude 去干](原仓库Markdown/docs/llm/claude/dynamic_workflows.md)。

---

<a id="a10"></a>

## 附.10 Loop Engineering 实战：14 步路线图，从判断要不要做到上线后守住【◻】

**一句话：** 好的 Agent 不只靠一句 Prompt，而靠“尝试→观察结果→验证→修正→停止”的运行闭环；harness 是支撑这个循环的工具、权限、状态和验收环境。

**例子：** 让编程 Agent 修改一个接口：先定位代码，编辑，运行测试，看错误并修复，直到通过或达到停止预算。若没有测试和状态记录，Agent 可能反复改同一处。动态工作流和多 Agent 只是可选实现，不能自动提高质量。

### Loop 不是免费的，先问四个问题

Loop 很诱人，但它不是免费午餐。**它烧 token、要花时间搭、出了问题你还得去 debug 一个你没亲眼看它跑的系统。**

动手之前，先把这四个问题想清楚：

<!-- drawio源文件: ./drawio/loop_engineering_guide_01_decision.drawio -->

![判断是否需要 Loop](https://file1.kamacoder.com/i/web/20260708113429_loop_decision.jpg?v=20261005)

**任务是重复的吗？** Loop 的搭建成本要靠多次运行摊回来。一次性的活儿，写个好 prompt 更快。

**有自动检查机制吗？** 测试、类型检查、linter，随便哪个都行。**没有自动检查，你就得自己逐行读 diff**，loop 就并没有帮你节省时间。

**Token 预算够吗？** Loop 会反复读上下文、重试、试探，不管有没有产出都在烧 token。

**Agent 能跑自己写的代码吗？** 需要有日志、能复现、看得到哪里崩。

还有个附加题，**比上面四个都重要**：**你打算 review 它产出的代码吗？** 不打算，就别建 loop。

**谁适合上手：** 有强测试套件的团队，干 CI 失败分类、依赖升级、lint-and-fix 这类任务。

**谁不适合：** 消费级套餐的个人开发者（token 预算不够烧）、测试覆盖不够的代码库（没闸门）、瓶颈在 review 而不在打字速度的团队。

### Loop 的五个核心构件

@0xCodez 把 loop 拆成五个构件，好在每个都能单独用、单独试。

<!-- drawio源文件: ./drawio/loop_engineering_guide_02_components.drawio -->

![Loop 的五个核心构件](https://file1.kamacoder.com/i/web/20260708113435_loop_components.jpg?v=20261005)

**Automations——loop 的心跳。** 按节奏触发，跑完一轮，停下。**关键是停止条件要写死，别让它无限跑。**

**Worktrees——并行不打架。** 多个 Agent 同时干活，最怕改同一个文件。Git worktree 给每个 Agent 一份独立工作区，互不干扰。但 worktree 有 200-500ms 的 setup 开销，只在真需要并行修改文件时才用。

**Skills——把背景写下来。** 项目用什么框架、有什么约定、踩过什么坑，写成一个 skill 存着，Agent 每轮直接读。这块我们在 [Claude Skills 实战](原仓库Markdown/docs/llm/claude/claude_skills.md) 里讲得很透了。

**Connectors——连上真实工具链。** 通过 MCP 接上 GitHub、Linear/Jira、Slack、Sentry，loop 才算真正接入你的工作流。关于 MCP 协议本身，可以看 [MCP 协议详解](原仓库Markdown/docs/llm/app/mcp_protocol.md)。

**Sub-agents——写的和验的分开。** 写代码的模型给自己打分太宽容。换一个带不同指令的第二个 Agent 来验收，能抓到第一个自我说服过去的问题。

### 14 步完整路线图

<!-- drawio源文件: ./drawio/loop_engineering_guide_03_roadmap.drawio -->

![14 步完整路线图](https://file1.kamacoder.com/i/web/20260708113442_loop_roadmap.jpg?v=20261005)

#### 第一段：先想清楚要不要做（5 步）

1. **确认这活是重复的** - 一次性任务用好 Prompt 更划算
2. **确认有自动检查机制** - 测试、类型检查、linter，至少一个
3. **确认 token 预算够** - loop 不产出也照样烧钱
4. **确认 Agent 能自己验证** - 有日志、能复现、看得到哪崩了
5. **确认你真打算 review** - 不打算，就别建

#### 第二段：搭一个最小能跑的 loop（8 步）

6. **先让一次手动运行稳定下来** - 顺序别跳，先手动跑一遍确认每一步都通
7. **把项目背景沉淀成一个 Skill** - 省得每轮从零解释
8. **加一个状态文件** - 记下做完了什么、下一步干啥，这是 loop 的记忆
9. **设一道硬闸门** - 测试/构建过不了就自动拒，这是你能放心走开的唯一保障
10. **配一个 Automation** - 按节奏触发，用停止条件控制什么时候停
11. **多 Agent 并行就上 Worktree** - 别让它们改同一个文件打架
12. **接上 Connectors** - 让 loop 能开 PR、更新 ticket、发 Slack
13. **拆出 Sub-agents** - 写代码的和验收的分开，验收的那个要专挑刺

#### 第三段：上线之后守住（1 步，但最难）

14. **盯住每个被接受的改动成本，定期复审权限、读 diff、别让 loop 碰架构**

**盯一个指标：每个被接受的改动的成本。** 如果接受率低于 50%，这 loop 就在亏本。

**定期复审权限。** 今天加一个写权限，明天再加一个，每 30 天复审一次，砍掉不需要的。

**读 diff。** 别因为「loop 产出的」就不读了。抽查闸门，确保它真的在按你的规则工作。

**别让 loop 碰架构。** loop 能干的是重复劳动、机械校验、低风险的小改动。架构决策、重构、核心逻辑，必须人来做。

### 上线后的三种翻车方式

<!-- drawio源文件: ./drawio/loop_engineering_guide_04_pitfalls.drawio -->

![Loop 的三种翻车方式](https://file1.kamacoder.com/i/web/20260708113449_loop_pitfalls.jpg?v=20261005)

**假装干完了。** 工程师 Geoffrey Huntley 管这叫 Ralph Wiggum 循环：Agent 提前发「完成」信号，活干一半就退。原因只有一个：**没有硬闸门。** 解法：回到第 9 步，设一道硬闸门。

**理解债务。** loop 越快交付你没写过的代码，**「仓库里有什么」和「你理解什么」的差距就越大。** 有一天，你得 debug 一个团队里没人读过的系统。解法：读 diff，抽查。

**认知投降。** 你慢慢不再自己判断，loop 返回啥就收啥。**这是最隐蔽、也最致命的一种翻车。** 解法：守住第 14 步，盯指标、读 diff、不让 loop 碰架构。

### 安全红线

**无人值守的 loop，就是无人值守的攻击面。**

- **生成代码未审就上线** - 闸门里得加 SAST、依赖审计、密钥扫描
- **Skill 是注入入口** - 社区 17022 个 skill 里有 520 个会泄露凭证，自动安装前先读源码
- **凭证泄露进日志** - 生产 loop 关掉 verbose 日志
- **权限蔓延** - 今天加一个写权限，明天再加一个，每 30 天复审一次

**无人值守不等于无人负责。** loop 出了安全问题，背锅的还是你。

### 写在最后

两年来，与编码 Agent 协作的杠杆一直在提示词上。而现在，**工作流成了真正的护城河。**

loop engineering 不是银弹，它有成本、有门槛、有风险。但对于那些重复、能机器校验、有强测试套件的任务，它能把「十二个小时的活，干成二十分钟」。

录友们与其纠结理论，不如挑一个你每天都在重复、有测试覆盖、能自动验证的活——CI 失败分类、依赖升级、lint-and-fix——按这 14 步走一遍。

### 参考链接

- Anthropic 工程文档（Managed Agents 与 Dynamic Workflows）：https://claude.com/blog
- Addy Osmani 关于 loop 工程的长文：https://addyosmani.com/blog/

### 本节收尾：费曼自述与闪卡

**费曼自述：** “模型给出动作只是开始，验证反馈决定能否真正完成任务。harness 把这条反馈链变成可运行的系统。”

**概念闪卡：** Q：Loop 的停止条件有哪些？A：验收通过、失败不可恢复、预算/时间上限或需要人工输入。Q：为什么多 Agent 不一定更好？A：通信、权限和成本会增加。

原文核对：[Loop Engineering 实战：14 步路线图，从判断要不要做到上线后守住](原仓库Markdown/docs/llm/claude/loop_engineering_guide.md)。

---

<a id="a11"></a>

## 附.11 Claude Managed Agents详解：把Agent的「大脑」和「双手」拆开，从原型到上线只要几天【◻】

**一句话：** 好的 Agent 不只靠一句 Prompt，而靠“尝试→观察结果→验证→修正→停止”的运行闭环；harness 是支撑这个循环的工具、权限、状态和验收环境。

**例子：** 让编程 Agent 修改一个接口：先定位代码，编辑，运行测试，看错误并修复，直到通过或达到停止预算。若没有测试和状态记录，Agent 可能反复改同一处。动态工作流和多 Agent 只是可选实现，不能自动提高质量。

### 先看一条演进线：Agent 的「跑法」是怎么一步步变的

要理解 Managed Agents 解决了什么，得先看清楚我们是从哪一步走过来的。

![Agent架构演进](https://file1.kamacoder.com/i/web/20260616203446_managed_agents_evolution.png?v=20261005)

**2023 年，简单 API 时代。** 最早的 Claude API 故意做得很朴素：tokens in, tokens out（喂进去 token，吐出来 token）。模型只负责思考，至于怎么让它调工具、跑循环、连文件系统——那套「外壳」（harness）全得开发者自己搭。

**2025 年，Claude Code 和 Agent SDK。** Claude Code 自带了一套打磨过的内部 harness，后来通过 Claude Agent SDK 开放出来，开发者不用自己写 Agent 循环了。这一步我们在 [Claude Code 作者说「不写 Prompt，写 Loop」](原仓库Markdown/docs/llm/claude/claude_code_loop.md) 里聊过——Agent 的核心就是那个「想—做—看结果—再想」的闭环。

**现在，Managed Agents。** 不只是给你 harness，而是把 **harness + 整套生产基础设施**一起托管起来。你专注写 Agent 的逻辑和领域知识，运行、伸缩、安全这些脏活累活，交给平台。

### 核心架构：把「大脑」和「双手」拆开

这是整篇博客最值钱的一个设计，我反复琢磨了几遍。

先说老架构的问题。过去 Agent 的 harness（负责推理、调度的「大脑」）**和它操作的 sandbox 文件系统（执行代码的「双手」）经常挤在同一个容器里**。听起来没毛病，但一上规模，三个问题全冒出来：

- **启动慢**：每跑一个 Agent 都要冷启动一个容器，时间全耗在这上面；
- **凭证暴露**：你的 API key、数据库密码，和 Agent 现场生成的代码待在同一个容器里——等于把钥匙和陌生人锁一个屋；
- **挂了就全丢**：容器一崩，跑到一半的进度全没了，没法恢复。

Managed Agents 的解法很干脆：**把大脑和双手拆开，放到不同的地方，中间用一个 session 连接。**

![大脑和双手解耦](https://file1.kamacoder.com/i/web/20260616203448_managed_agents_decouple.png?v=20261005)

这个 session 是关键——它是**一条 append-only 的日志，记录每一次模型调用、每一次工具调用、每一个返回结果**。大脑在一边推理，双手在另一边执行，两边通过这条只追加、不修改的事件流对话。这条 append-only 日志，跟我们在 [Claude Skills 实战](原仓库Markdown/docs/llm/claude/claude_skills.md) 里讲的「让 Claude 读自己的历史」是同一个思路，只不过这里把它抬到了整个运行架构的层面。

拆开之后，整套东西由**三种资源**组成，理解了这三个，你就理解了 Managed Agents：

- **Agents（智能体）**：配置。用哪个模型、什么提示词、能调哪些工具、有什么护栏。
- **Environments（环境）**：执行上下文。sandbox 容器、预装好的依赖包。
- **Sessions（会话）**：一次具体的运行。完整的事件历史存在服务端，可观察、可恢复。

「配置 / 环境 / 运行」三层分开，这就是把大脑和双手解耦之后，自然长出来的结构。

### 拆开之后，到底赚到了什么

解耦不是为了好看，是为了解决前面那三宗罪，外加几个生产环境上的真实收益。

**第一，凭证安全。** 凭证不再跟生成的代码同处一室，而是单独放进一个 **vault（保险库）**。存进去之前先做信封加密（envelope encryption），取出来还得带一个签了名的请求 token。Agent 现场写的代码再怎么折腾，也碰不到你的密钥。

**第二，延迟大降。** 拆掉了「每次都冷启动容器」这块开销，**首 token 时间（time-to-first-token）中位数（p50）降了大约 60%，最慢的那批（p95）降了 90% 以上。** 这个数字很能说明问题——以前用户等的，很多根本不是模型在想，是容器在启动。

![首token时间改前改后对比](https://file1.kamacoder.com/i/web/20260616203449_managed_agents_latency.png?v=20261005)

**第三，会话可持久、可恢复。** session 是一条持续的事件流，你能**实时看到 Agent 干活时的每一步**，也能在之后任意时刻**把某个 session 接着跑下去**。容器崩了不再等于从头再来。

**第四，部署灵活。** 既可以用 Anthropic 托管的云容器，图省心；也可以把 sandbox 放进你自己的 VPC 里自托管，满足合规和数据不出域的要求。

### 一个容易被忽略、但很关键的 lesson

博客里有一段我特别想拎出来讲：**模型在进化，harness 必须跟着一起进化。**

原文举了个具体例子：在 Claude Sonnet 4.5 上，Agent 快用到上下文末尾时会**「赶工」**——感觉自己快没空间了，就急着草草收尾。他们的修法是给 harness 加了**上下文重置（context reset）**。结果呢？到了下一代模型，这个毛病没了，那段优化也就不需要了。

这件事戳中了一个现实：**如果你自己维护 harness，你就得永远追着模型的脾气改。** 这一代模型有这个怪癖你补一段，下一代变了你再删一段。而 Managed Agents 的意义就在这——这些跟着模型版本来回横跳的优化，平台替你扛了。

### 那么，谁该用 Managed Agents

博客给了一个很清醒的判断：**对绝大多数团队来说，维护一套 harness 是「不产生差异化」的纯开销。**

你的产品凭什么赢？是你的领域知识、你对上下文的组织能力，不是你那套 Agent 循环写得比别人精巧。基础设施这层，能交出去就交出去，把精力放在真正拉开差距的地方。关于「什么时候真需要一个 Agent、什么时候不需要」，可以回看 [Agent vs Workflow](原仓库Markdown/docs/llm/app/agent_vs_workflow.md)，先想清楚再决定上不上这套东西。

几个客户数据也挺有说服力：

- **Notion**：用自定义 Agent 把「十二个小时的活，干成了二十分钟」；
- **Sentry**：调试 + 自动生成 PR 的 Agent，**单个工程师几周就搞定**，过去得几个月；
- **Rakuten、Asana、Atlassian**：跨部门把 Agent 铺开，大约一周就能上。

它还带了一堆生产级特性：Memory 和 Dreaming（定时回顾 session、从中提炼模式、整理记忆）、Outcomes（Agent 按评分标准给自己的活打分）、多 Agent 编排、权限策略、webhook，以及用 MCP tunnel 连你私网里的服务器。关于 MCP 这个协议本身，可以看 [MCP 协议详解](原仓库Markdown/docs/llm/app/mcp_protocol.md)。

### 写在最后

从 tokens in/out，到自带 harness，再到今天把整个 harness 和基础设施托管起来——Anthropic 这条线其实一直在干同一件事：**把开发者从「伺候 Agent 怎么跑」里解放出来，让你只管「Agent 该干什么」。**

而这一切的技术支点，就是那个朴素又关键的决定：把大脑和双手拆开。录友们自己做 Agent 项目时，哪怕暂时用不上 Managed Agents，这个「推理和执行解耦、用一条 append-only 日志连接」的思路，也值得抄进自己的架构里。

### 参考链接

- Anthropic 官方博客（building with Claude Managed Agents）：https://claude.com/blog/building-with-claude-managed-agents

### 本节收尾：费曼自述与闪卡

**费曼自述：** “模型给出动作只是开始，验证反馈决定能否真正完成任务。harness 把这条反馈链变成可运行的系统。”

**概念闪卡：** Q：Loop 的停止条件有哪些？A：验收通过、失败不可恢复、预算/时间上限或需要人工输入。Q：为什么多 Agent 不一定更好？A：通信、权限和成本会增加。

原文核对：[Claude Managed Agents详解：把Agent的「大脑」和「双手」拆开，从原型到上线只要几天](原仓库Markdown/docs/llm/claude/managed_agents.md)。

---

<a id="a12"></a>

## 附.12 百万行代码两周迁完：Anthropic如何用Claude Code做大规模代码迁移【◻】

**一句话：** CLI 便于工具组合；Prompt 缓存只在符合服务规则的重复前缀上省计算；大型迁移案例提供任务拆分与验收思路，但数字与工具版本不宜背诵。

**例子：** 同一长项目规则被许多请求复用可能有缓存收益；每次变化的文件内容不能假定都命中缓存。百万行代码迁移案例可提炼“先做可判对错的测试，再扩大自动化范围”。

### 两个数字，先把迁移规模摆出来

第一个案例是 Bun。Jarred Sumner 用 Claude Code 把 Bun 从 Zig 迁到 Rust，**不到两周产出 100 万行代码**，合并前 CI 中原有测试全部通过。合并后暴露了 19 个回归，目前已经全部修掉。

第二个案例来自 Anthropic Labs。Mike Krieger 用一个周末，把 Python 代码库迁成 **16.5 万行 TypeScript**。过程中跑了数百个 Agent、8 道阶段闸门、3 轮对抗审查，最后逐条 diff 新旧程序的命令输出。

Anthropic 过去一个月一共迁了 10 个代码包，每个包都有数万到数十万行。

注意，这不是“Claude 一次把代码翻译对了”。**真正让规模迁移成立的，是它把迁移改造成了一条可以反复运行、机械验收、失败后继续收敛的生产线。**

![代码迁移流水线](https://file1.kamacoder.com/i/web/20260722095637_code_migration_01_compressed.jpg?v=20261005)

人不再追着每个文件改。人负责规则、裁判和异常模式，Agent 负责把队列烧完。

原文那句最值钱的话可以翻成：**别修代码，修产生代码的循环。**

### 为什么代码迁移特别适合多 Agent

大规模迁移刚好满足 Agent 最喜欢的几个条件。

**工作能并行。** 上千个文件、crate 或模块可以切成独立批次，不需要一个 Agent 从头写到尾。

**旧代码就是规格。** 需求文档可能缺，但原程序已经把行为写出来了。Agent 可以同时对照源语言、目标语言和真实输出。

**有机械裁判。** 编译器、冒烟测试、完整测试套件、命令输出 diff，都能明确告诉 Agent 对还是错。失败结果还会自动变成下一轮任务队列。

**规则可以向后传播。** 一个 Agent 发现边界问题，不是只补当前文件，而是把解法写回迁移规则，后面的 Agent 全部按新规则执行。

这也是它和普通“批量翻译代码”的根本区别：**并行只负责提速，统一规则和机械验收才负责不失控。**

### 动手前，先造一个能判新旧代码的裁判

Anthropic 把强裁判列为前置条件。没有它，你连“迁完了”是什么意思都说不清。

很多旧测试会直接调用源语言内部函数，换语言后根本跑不了。正确做法是先把测试分类，挑出可以通过外部接口表达的行为，再改造成能同时打在新旧程序上的可移植测试。

裁判还必须做一次反向验证：原程序应该通过，故意破坏的程序必须失败。**抓不住已知错误的测试，不是裁判，只是安慰剂。**

Mike 没有现成的跨语言测试，就做了 7 个真实场景的 parity harness，把 Python 和 TypeScript 输出逐项比较，任何行为差异都算 Bug。

### 六步迁移，最重要的工作都发生在大规模翻译前

**第一步，建立规则手册、依赖图和差距清单。**

规则手册先规定类型、惯用法和架构怎么映射；依赖图决定哪些文件先迁、哪些必须放在同一批；差距清单专门记录无法直接翻译的隐性知识，比如 Zig 的手动内存管理怎么落到 Rust 所有权，Python 的动态对象怎么变成 TypeScript 接口。

顺序不能反。**先定默认规则，再找默认规则盖不住的差距。**

**第二步，用小样本把规则撞一遍。**

Bun 迁移时，一个 Agent 按规则翻 3 个文件，另一个 Agent 像资深 Rust 工程师一样翻同样的文件，第三个 Agent 比较差异并补规则。只跑这轮小实验，就提前抓到两个会扩散到 1448 个文件的关键问题。

![迁移规则压力测试](https://file1.kamacoder.com/i/web/20260722095640_code_migration_02_compressed.jpg?v=20261005)

这里最反直觉的是：**试迁出来的代码全部扔掉。** 这一轮的产物不是代码，是更可靠的规则。结构不保留的重设计也一样，先让对抗 Agent 攻击设计文档，再做一次可丢弃的端到端迁移。

**第三步，再把翻译任务铺开。**

批处理脚本从磁盘重建待办队列，“目标文件存在”才算完成，所以任务天然可以暂停、恢复。高吞吐翻译交给较小模型，写规则和做审查留给更强模型。两个独立审查 Agent 意见不一致，再交给第三个 Agent 仲裁。

如果同一种错误反复出现，别手补十个文件。**给规则手册加一句，再重新生成受影响的那一批。**

**第四到第六步，依次编译、运行、对齐行为。**

编译错误由 fixer agents 分批处理；崩溃交给冒烟测试；最后把新旧程序放到同一套测试或真实场景里比行为。昂贵的全量构建由单独的 build daemon 串行执行，避免几十个 Agent 同时抢着重编。

![迁移故障反馈闭环](https://file1.kamacoder.com/i/web/20260722095642_code_migration_03_compressed.jpg?v=20261005)

这时人的注意力不该放在某一个红叉，而要看**哪些红叉总以同一种方式出现**。单点失败交给 loop 消耗，重复模式才值得人修改上游规则。

### 百万行迁完了，账也要算清楚

AI 把语言迁移从“多年豪赌”变成了“昂贵但可以重跑的工程实验”，不等于免费。

Bun 迁移消耗了 59 亿 uncached input tokens 和 6.9 亿 output tokens，按 API 价格约 **16.5 万美元**。但过去百万行迁移可能是 4 年、300 万到 400 万美元的人力项目。最坏情况也从“维护两套半成品几年”，变成“删掉分支，改好规则再跑一次”。

效果也不是只看行数。Bun 的一个 2000 次重复构建基准，内存占用从 6745 MB 降到 609 MB；Linux 和 Windows 二进制缩小 19%，部分真实负载快了 2%～5%。不过约 4% 的 Rust 代码仍在 `unsafe` 块里，说明迁移不会凭空消灭底层边界。

Python 到 TypeScript 的项目也有明确收益：原来每个平台编译约 8 分钟，整个矩阵要等 30 分钟；迁移后约 2 秒，启动快 6 倍，还删掉了一条独立部署流水线。

所以别因为“Claude 能迁”就迁。**业务收益、测试裁判、Token 预算、审查能力，四样缺一个，都可能把漂亮 Demo 变成昂贵烂尾。**

### 最后

这篇博客真正改写的，不是“AI 会不会翻译 Rust”。

它改写的是大项目的风险结构：以前人怕迁到 90% 后骑虎难下，现在可以先把裁判和规则做扎实，再让机器反复重跑。

**代码可以重生成，规则必须能解释，结果必须由机器验。**

### 参考链接

- Anthropic 官方博客（How Anthropic runs large-scale code migrations with Claude Code）：https://claude.com/blog/ai-code-migration

### 本节收尾：费曼自述与闪卡

**费曼自述：** “从工具案例里抽取稳定的工程方法，不把某次统计或产品细节记成永久规律。”

**概念闪卡：** Q：缓存命中看什么？A：服务规则及可复用前缀。Q：自动迁移前为什么先做验收？A：没有判定标准，大规模执行会放大错误。

原文核对：[百万行代码两周迁完：Anthropic如何用Claude Code做大规模代码迁移](原仓库Markdown/docs/llm/claude/ai_code_migration.md)。

---

<a id="a13"></a>

## 附.13 深入理解 Claude Code：从 CLAUDE.md 到 Hooks、Skills、Subagents【◻】

**一句话：** Skill 封装任务方法，Hook 在事件触发时执行检查，MCP 连接外部工具/资源；它们解决不同问题，不能互相替代。

**例子：** “运行测试并解释失败”可以写成复用流程；每次提交前自动格式检查适合 Hook；读取外部文档系统可通过受控工具/MCP。是否需要这些机制取决于重复任务的频率和维护成本。

### 第一阶段：先用 CLAUDE.md，让 Claude 知道怎么干活

假设你让 Claude Code 修改一个支付接口。

你每次都要重复交代：项目用 pnpm、支付金额统一用分、不能直接改数据库表、改完必须跑测试。

说一遍有用，但只对这次对话有用。任务一长、上下文一压缩，前面的要求就可能被淹没。

`CLAUDE.md` 解决的正是这个问题。它把跨任务都稳定的项目知识放进常驻上下文：

```md
### 项目规则

- 使用 pnpm，不要使用 npm
- 金额统一以分存储，禁止使用浮点数
- 修改支付模块后运行 pnpm test:payment
- 未经确认不要修改数据库 schema
```

Claude Code 每次进入项目都能看到，不需要你重新教育一遍。根目录写全局约束，子目录再补模块规则，模型走到对应目录时加载更具体的上下文。

所以 `CLAUDE.md` 的定位很清楚：**它是 always-on 的项目说明书。**

但它有两个绕不过去的边界。

第一，**看到了，不代表一定做到。**“修改后运行测试”仍然只是一条自然语言指令，最终要靠模型记住、理解并主动执行。

第二，**常驻上下文不是免费仓库。**发布手册、接口文档、排障流程全塞进去，每一轮都要带着一大包暂时用不到的信息，重点反而被稀释。

这就是下一层能力出现的原因：有些事不能只提醒，必须在正确的时间点自动发生。

### 第二阶段：再加 Hooks，把“应该做”升级成“触发就做”

还是“改完必须跑检查”这件事。

写在 `CLAUDE.md` 里的意思是：Claude，记得做。

写成 Hook 的意思是：只要编辑完成，这个检查点就会被触发。

Hooks 插在 Claude Code 的生命周期里。一次工具调用不是模型想完就直接结束，中间有一系列可以拦截和反馈的事件：

- `PreToolUse`：工具执行前，可以检查、修改或阻止这次调用；
- `PostToolUse`：工具成功后，可以格式化、检查并把结果喂回上下文；
- `PostToolUseFailure`：工具失败后记录错误或补充诊断；
- `Stop`：Claude 准备收工时，检查任务是否真的完成；
- `SubagentStart`、`SubagentStop`：子 Agent 启动和结束时追加约束或验收结果。

<!-- drawio源文件: ./drawio/claude_code_extensions_02_hooks_loop.drawio -->

![Hooks在Claude Code工具调用前后和结束验收时插入控制点并将失败证据反馈给Agent](https://file1.kamacoder.com/i/web/20260713152556_claude_code_extensions_02_hooks_loop_compressed.png?v=20261005)

这张图回答的是：Hooks 到底插在 Agent 循环哪里。它不是循环外的一份说明，而是卡在“准备行动、行动完成、准备结束”这些节点上的闸门；不满足条件，就把错误或证据送回 Claude，让它继续修。

一个最小的编辑后检查，大致长这样：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "pnpm lint"
          }
        ]
      }
    ]
  }
}
```

现在的 Hooks 也不只会跑 shell。它还可以调用 HTTP、MCP 工具，或者让一次 prompt、一个临时 Agent 做判断。但无论执行形式怎么变，它的核心都没有变：**由生命周期事件触发，而不是等 Claude 想起来。**

不过别把 Hooks 神化。

Hooks 适合“每次命中这个事件都要做”的动作，例如格式化、危险命令拦截、审计记录。它不适合承载一整套需要理解业务、根据现场调整步骤的发布手册。

**确定性动作交给 Hook，需要推理的流程不要硬塞进 Hook。**

### 第三阶段：用 Skills，把专项知识从常驻上下文里搬出去

项目继续往前走，你会遇到另一类东西：

- 发版时才需要的检查清单；
- 排查线上问题时才需要的 Runbook；
- 评审支付代码时才需要的安全规则；
- 调某个内部 API 时才需要的字段说明和脚本。

这些内容很重要，但不是每个任务都要看。

塞进 `CLAUDE.md`，上下文会越来越胖；做成 Hook，又没办法表达“先根据现象判断，再选择不同排查路径”。这时候该用 Skill。

Skill 是一套**按需加载的知识和工作流**。Claude Code 启动时通常只看到它的名称和描述，判断当前任务相关时，才加载完整内容：

```text
.claude/skills/payment-review/
├── SKILL.md
├── references/
│   ├── risk-rules.md
│   └── database-schema.md
├── scripts/
│   └── check-money-unit.sh
└── assets/
    └── review-template.md
```

这里最重要的不是文件夹长什么样，而是**上下文加载方式变了**。

`CLAUDE.md` 是“每次都带上”；Skill 是“这次用到才展开”。`SKILL.md` 还可以继续指向参考资料、脚本和模板，让 Claude 用到哪一块再读哪一块。

这就是渐进式披露。关于 Skill 怎么分类、怎么写 description、怎么积累 Gotchas，我们在 [Claude Skills 实战](原仓库Markdown/docs/llm/claude/claude_skills.md) 已经详细讲过，这里不重复。

现在可以把 Hook 和 Skill 分清了：

- Hook 由事件触发，强调**这件事必须发生**；
- Skill 由用户调用或任务意图触发，强调**这类事应该怎么做**；
- Hook 更像闸门，Skill 更像专项作业手册；
- Hook 追求稳定重复，Skill 允许 Claude 根据现场推理。

例如，`PostToolUse` Hook 可以在支付代码被修改后运行金额单位扫描；`payment-review` Skill 则告诉 Claude 发现问题以后，应该如何判断风险、补测试和给出修复方案。

### 第四阶段：拆 Subagents，把“一个 Claude 全包”变成角色分工

有了 Skills，Claude 已经能按需获得专项能力。但复杂任务还有一个更隐蔽的问题：**所有工作仍然挤在同一个上下文里。**

主 Agent 先读几十个文件定位问题，又改代码、跑测试，最后还让它审查自己的实现。上下文越来越脏，早期约束被压缩；更麻烦的是，让同一个 Agent 给自己的答案挑错，天然容易自我认可。

Subagents 解决的不是“再加一份提示词”，而是**再开一个独立的执行上下文**。

一个自定义子 Agent 可以这样定义：

```md
---
name: payment-security-reviewer
description: 支付模块改动完成后，独立检查金额、权限和幂等风险
tools: Read, Glob, Grep, Bash
model: sonnet
skills:
  - payment-review
---

只做安全评审，不修改业务代码。结论必须包含风险位置、触发条件和验证证据。
```

它拥有自己的系统提示、工具范围、模型、权限、Skills 和上下文窗口。主 Agent 只需要把任务委派出去，最后接收摘要，不必把子 Agent 读过的几十个文件和全部命令输出都塞回主线程。

<!-- drawio源文件: ./drawio/claude_code_extensions_03_subagent_isolation.drawio -->

![主Agent将测试安全和性能任务委派给独立上下文的Subagents并只接收带证据的结论](https://file1.kamacoder.com/i/web/20260713152556_claude_code_extensions_03_subagent_isolation_compressed.png?v=20261005)

这张图回答的是：独立上下文到底省掉了什么。测试、安全、性能三个 Agent 各自在干净窗口里工作，过程噪声留在各自上下文，只有带证据的结论回到主 Agent；执行者和审查者也被结构性地分开。

所以 Skill 和 Subagent 也不是一回事：

- Skill 是可以被复用的“方法”；
- Subagent 是拿着方法独立干活的“角色”；
- Skill 会把内容加载进当前上下文；
- Subagent 在另一个上下文里执行，只把结果带回来。

两者还可以组合。安全审查 Agent 预加载安全 Skill，测试 Agent 预加载测试 Skill，主 Agent 负责拆任务、汇总结论和决定是否继续修改。

这一步往后，就是我们在 [动态工作流](原仓库Markdown/docs/llm/claude/dynamic_workflows.md) 里讲的更复杂编排了：多个 Agent 并行、对抗验证、生成与筛选。但别急着冲到那一步，两个角色能讲清楚的任务，不要为了显得高级硬开十个 Agent。

### 四层能力放在一起，到底怎么选

现在回到最容易混淆的问题：一条新要求来了，应该写到哪里？

<!-- drawio源文件: ./drawio/claude_code_extensions_04_decision.drawio -->

![根据常驻知识事件触发专项方法和上下文隔离判断一条要求应该放进CLAUDE.md、Hooks、Skills还是Subagents](https://file1.kamacoder.com/i/web/20260713152556_claude_code_extensions_04_decision_compressed.png?v=20261005)

这张图回答的是：如何根据“是否每次都要知道、是否必须自动发生、是否需要专项推理、是否需要隔离上下文”逐层判断。它不是让你四选一，而是帮你找到一条要求最主要的归属层。

可以记住四个问题：

| 判断问题 | 应该放哪里 | 典型例子 |
|---|---|---|
| Claude 是否每次都应该知道？ | `CLAUDE.md` | 技术栈、目录、构建命令、禁区 |
| 是否命中事件就必须执行？ | Hooks | 格式化、阻止危险命令、记录审计 |
| 是否只在某类任务中需要一套方法？ | Skills | 发布、评审、排障、数据分析 |
| 是否需要独立上下文或独立角色？ | Subagents | 大范围探索、安全复核、并行测试 |

还是支付模块的例子：

1. `CLAUDE.md` 写清金额单位、项目命令和数据库禁区；
2. Hook 在文件修改后跑静态检查，在危险操作前做拦截；
3. Skill 保存支付评审方法、风险规则和验证脚本；
4. Subagents 分别检查安全、测试和性能，主 Agent 汇总证据。

你会发现，四层能力不是互相竞争，而是分别控制 Agent 的四个位置：

**上下文里放什么，生命周期中卡什么，任务需要时加载什么，复杂工作交给谁。**

### 最容易踩的四个坑

**第一，把所有东西都塞进 CLAUDE.md。**

结果不是 Claude 懂得更多，而是每轮都背着一仓库资料。长期稳定规则留下，专项知识移到 Skills。

**第二，把“禁止”只写成自然语言。**

“不要改 `.env`”如果是安全红线，就别只靠模型听话。能用权限和 `PreToolUse` Hook 拦截的，直接做成硬边界。

**第三，把复杂判断写成一坨 shell Hook。**

Hook 是触发器，不是所有业务逻辑的垃圾桶。需要读资料、比较方案、根据现场调整步骤，就交给 Skill 或专项 Agent。

**第四，任务一大就疯狂开 Subagents。**

子 Agent 会消耗额外 token，也有任务描述丢信息、结果汇总失真的成本。只有过程噪声很大、角色确实应该隔离，或者工作真的能并行时再拆。

### 写在最后

很多人配置 Claude Code，一上来就抄别人的全家桶：几百行 `CLAUDE.md`、几十个 Hooks、十几个 Skills，再配一队 Subagents。

**太重了，也太早了。**

同一条规则说了两次，写进 `CLAUDE.md`；某个动作必须发生，做成 Hook；某套流程反复复制，养成 Skill；某项工作需要独立探索，再交给 Subagent。

别按功能清单搭系统。**看问题卡在哪一层，就补哪一层。**

### 参考链接

- Claude Code 官方文档（扩展能力总览）：https://code.claude.com/docs/en/features-overview
- Claude Code 官方文档（Memory 与 CLAUDE.md）：https://code.claude.com/docs/en/memory
- Claude Code 官方文档（Hooks 指南）：https://code.claude.com/docs/en/hooks-guide
- Claude Code 官方文档（Skills）：https://code.claude.com/docs/en/slash-commands
- Claude Code 官方文档（Subagents）：https://code.claude.com/docs/en/sub-agents

### 本节收尾：费曼自述与闪卡

**费曼自述：** “流程说明、自动触发、外部连接是三种不同扩展点；先找到实际重复工作再加。”

**概念闪卡：** Q：Hook 与 Skill 的核心差别？A：前者按事件自动执行，后者是可选择使用的能力/流程封装。Q：MCP 直接保证数据权限吗？A：协议不替代服务端授权。

原文核对：[深入理解 Claude Code：从 CLAUDE.md 到 Hooks、Skills、Subagents](原仓库Markdown/docs/llm/claude/claude_code_extensions_evolution.md)。

---

<a id="a14"></a>

## 附.14 40万次Claude Code会话揭示：真正拉开差距的，不是Prompt【◻】

**一句话：** CLI 便于工具组合；Prompt 缓存只在符合服务规则的重复前缀上省计算；大型迁移案例提供任务拆分与验收思路，但数字与工具版本不宜背诵。

**例子：** 同一长项目规则被许多请求复用可能有缓存收益；每次变化的文件内容不能假定都命中缓存。百万行代码迁移案例可提炼“先做可判对错的测试，再扩大自动化范围”。

### 这40万次会话，到底统计了什么

研究覆盖2025年10月至2026年4月的Claude Code CLI、Claude.ai和桌面端交互，不包含IDE集成、SDK和`claude -p`这类无人值守调用。

Anthropic用隐私保护工具分析完整会话，再结合代码增删、Git提交、PR和测试结果等信号，判断用户在做什么、谁负责决策，以及任务有没有成功。

这里的“专业程度”也不是看职位。

一个高级工程师第一次写Rust，可能还是新手；一个不会Python的会计，如果能说清月末对账规则，还能抓住脚本漏掉的边界情况，在这个任务上就是专家。

**专业程度是针对当前问题的，不是简历上的头衔。**

### 最稳定的分工：人决定做什么，Agent决定怎么做

研究把决策分成两类：

- 规划决策：要做什么、选哪条路、什么算完成；
- 执行决策：改哪些文件、写什么代码、跑哪些命令。

结果很清楚：典型会话里，人做了约70%的规划决策；Claude做了约80%的执行决策。

这说明Claude Code最有效的姿势，不是人盯着每一行代码，也不是把目标都扔给Agent自己猜。

**人守住目标、约束和验收，Agent接管搜索、修改、运行与修复。**

<!-- 卡通图待生成：人负责规划，Agent负责执行 -->

这张图回答的是：人机协作为什么不是“五五开”。人像建筑师一样决定盖什么、边界在哪、怎样才算完工；Agent施工队负责选择工具和完成具体动作。

### 真正的专家，一条指令能撬动更多行动

一条Prompt发出去，Claude平均会执行约10次行动：读文件、搜代码、修改、跑命令、再根据结果继续。

但不同用户之间差距很大。

| 用户在当前任务上的水平 | 每条Prompt触发的行动 | Claude输出量 |
|---|---:|---:|
| 新手 | 约5次 | 约600词 |
| 专家 | 约12次 | 约3200词 |

别把这个结果理解成“专家的Prompt更长”。研究判断专业程度时，看的是用户能不能准确描述问题、要求Claude验证什么，以及出错后谁在纠正谁。

新手常说：“这个功能不对，帮我修一下。”

熟手会说清业务规则、失败现象、不能破坏的边界和验收证据。Agent不需要走两步就回来猜一次，自然敢连续做更多事。

**真正有效的上下文，不是字多，而是里面有判断。**

<!-- 卡通图待生成：万能Prompt对比业务地图和验收清单 -->

这张图回答的是：为什么两个人用同一个Agent，产出差距会这么大。新手只给一句模糊命令，熟手则把业务地图、边界和验收标准交给机器人。

### 从新手变成熟手，收益最大

Anthropic用了一个很严格的“验证成功”指标：不只让模型判断任务像是完成了，还要看到测试通过、匹配的Git活动或用户明确确认等硬证据。

新手会话达到验证成功的比例约15%，中级及以上达到28%—33%；如果放宽到“至少部分成功”，两边是77%和91%—92%。

更值得注意的是出错以后。

遇到明显错误、测试失败或反复重试时，新手最终验证成功的比例只有约4%，专家约15%。困难会话里，新手有19%直接放弃，其他层级只有约5%—7%。

不过，差距主要发生在“新手到中级”。从中级继续走到专家，提升反而没那么陡。

这对普通录友是个好消息：**你不需要先成为顶级专家，但至少要对问题有一套能工作的理解。**

### 不会写代码的人，已经能做出软件了吗

在真正产出代码的会话里，软件相关职业的验证成功率约34%，其他职业约29%；看“至少部分成功”，两边分别约89%和88%。

十大职业类别和软件工程师的差距都在7个百分点以内。管理类用户甚至略高，但研究也提醒，这可能和管理者更善于分派任务、表达完成有关，也可能受统计方式影响。

所以结论不是“编程已经没用了”。

更准确的说法是：**代码语法和实现正在变得更容易获得，领域知识、任务拆解和结果判断正在变得更值钱。**

律师能让Agent写合同检查工具，前提是他知道什么条款算风险；会计能让Agent写对账脚本，前提是他知道哪些差异必须报警。不会写代码不再是绝对障碍，不懂业务仍然是。

<!-- 卡通图待生成：不同领域人员带着专业规则使用Agent -->

这张图回答的是：AI降低编程门槛以后，什么能力没有贬值。法律、财务和产品人员可以把规则交给Agent实现，但最后仍要靠专业判断验收。

### Claude Code正在从“修Bug”走向“替你跑完整件事”

七个月里，修复代码的会话占比从33%降到19%；部署、配置和运行软件从14%升到21%；写作与数据分析从约10%升到20%。

研究估算的平均任务价值也提升了约27%。这个数字来自和自由职业平台任务的粗略匹配，只适合看相对变化，不能当成Claude真的替用户赚了多少钱。

趋势仍然很明显：**用户不再只让Agent补一段代码，而是开始把部署、数据、文档和完整工作流交给它。**

这和我们在[从Prompt走向Agent Loop](原仓库Markdown/docs/llm/claude/claude_code_loop.md)里讲的是同一件事。任务越完整，真正重要的越不是某一句Prompt，而是目标、工具、验证和错误恢复能不能连成闭环。

### 想让Claude Code更靠谱，先补这5件事

**第一，先把问题搞懂，再研究Prompt。** 至少说清现状、目标、不能碰的边界和失败代价。

**第二，把“什么算完成”写出来。** 测试、截图、日志、指标和人工确认，都比“你检查一下”更有用。

**第三，把专业证据交给Agent。** 业务规则、历史案例、接口契约和项目文档，才是它做判断的材料。

**第四，别替Agent遥控每一步。** 人控制规划，执行层给它足够空间；否则你只是把Claude Code用成了昂贵的代码补全。

**第五，出错时诊断，不要只换一种说法再问。** 看测试为什么失败、假设哪里错了、缺了什么证据。能从错误里把Agent拉回来，正是熟手和新手最明显的差距。

这些经验可以继续沉淀进[CLAUDE.md](原仓库Markdown/docs/llm/claude/claude_md.md)和[Skills](原仓库Markdown/docs/llm/claude/claude_skills.md)。个人脑子里的专业判断，只有变成项目规则、验收脚本和失败案例，团队里的Agent才能反复用上。

### 这份研究也不能被神化

40万次会话很多，但它仍然不是一次严格的因果实验。

职业、专业程度和成功结果主要由模型读取会话后分类；研究看不到代码最终有没有进入生产，也排除了大量无人值守和IDE里的Claude Code调用。“验证成功”比只听用户说完成更严格，但仍不等于真实商业结果。

所以不要拿这些数字证明“程序员会消失”，也不要证明“人人都能做软件”。它真正说明的是：**当实现能力越来越便宜，懂问题的人会获得更大的杠杆。**

别再只收藏万能Prompt了。

**先让自己成为那个知道该做什么、也知道怎样才算做对的人。**

### 参考链接

- Anthropic研究（Agentic coding and persistent returns to expertise）：https://www.anthropic.com/research/claude-code-expertise

### 本节收尾：费曼自述与闪卡

**费曼自述：** “从工具案例里抽取稳定的工程方法，不把某次统计或产品细节记成永久规律。”

**概念闪卡：** Q：缓存命中看什么？A：服务规则及可复用前缀。Q：自动迁移前为什么先做验收？A：没有判定标准，大规模执行会放大错误。

原文核对：[40万次Claude Code会话揭示：真正拉开差距的，不是Prompt](原仓库Markdown/docs/llm/claude/claude_code_400k_sessions.md)。

---

## 面试专题：Harness Engineering大厂面试题汇总：从Prompt到Context到Harness

### 目录

1. [Harness Engineering 是什么？从哪冒出来的？](#1-harness-engineering-是什么从哪冒出来的)
2. [从 Prompt 到 Context 到 Harness：AI 工程的三次重心转移](#2-从-prompt-到-context-到-harnessai-工程的三次重心转移)
3. [Harness 拆开看：六层核心组件](#3-harness-拆开看六层核心组件)
4. [大厂踩过的五个真实难题](#4-大厂踩过的五个真实难题)
5. [Hermes Agent vs OpenClaw：两种 Harness 实现](#5-hermes-agent-vs-openclaw两种-harness-实现)
6. [和 Prompt/Context Engineering 到底什么关系？](#6-和-promptcontext-engineering-到底什么关系)
7. [大厂真实面试追问汇总](#7-大厂真实面试追问汇总)

---

### 1. Harness Engineering 是什么？从哪冒出来的？

面试官一般这么问："你听说过 Harness Engineering 吗？"或者"Agent = Model + Harness，你怎么理解这个等式？"

#### 先搞清楚：Harness 是什么？

Harness 这个词直译叫**"马具"**，或者**"缰绳"**。

想象一下骑马：马本身有强大的力量，能跑能跳能驮东西。但如果没有缰绳和马具，这股力量就是失控的——马可能往悬崖上跑，可能甩你下来，可能跑去吃草不回来了。

马具的作用，就是让这股力量**为你所用**。

AI 系统也一样。LLM 很强，Agent 很能干，但如果没有一套东西把它们"拴住"、监测住、约束住，它们就是脱缰的野马——可能跑偏、可能幻觉、可能越权、可能悄悄变差。

**Harness Engineering 就是给 AI 系统装上缰绳的工程学科。**

#### 这个词是谁先喊出来的？

很多人跟风聊 Harness Engineering，但压根不知道它最早是谁提的。搞清楚来源，你就明白为什么它这次真的能火，而不是又一个换皮概念。

**2026 年 2 月 5 号**，Mitchell Hashimoto（HashiCorp 联合创始人，Vagrant、Terraform 的作者）发了一篇博客，叫《My AI Adoption Journey》。他把接纳 AI 的过程拆成 6 步，第 5 步的名字就叫**"Engineer the Harness"**。

他的定义特别简洁：

> 每次当你发现 Agent 犯了一个错误，就花点时间去工程化一个解决方案，让它永远不会再犯同样的错误。

你品品这个思路。绝大多数人遇到 Agent 犯错，骂两句手动改掉，祈祷下次别再犯。但 Mitchell 不是这么干的——他每次 Agent 犯错，都会停下来问自己：我能不能把这个错误永久性地修到环境里，让它下次在结构上就不可能再犯？

可能是给 AGENTS.md 加一条规则，可能是加一个 linter，可能是补一个自动化测试，也可能是搞一个 Git Hook。关键是：**这个修补必须沉淀到环境里，而不是留在人脑子里。**

这套做法是**复利**的。每次 Agent 犯错，环境就变强一点；环境变强一点，Agent 下次就更少犯错；犯错变少，你改进的速度就更快。时间一长，你的 Harness 越来越坚固，Agent 在你这个项目里越跑越稳。

博客发出来一周后，OpenAI 紧接着发了一篇官方博客，标题就叫《Harness engineering: leveraging Codex in an agent-first world》。讲的是他们内部一个小团队，从一个空仓库出发，用 5 个月时间靠 Agent 写出了 100 万行代码、合并了 1500 个 PR，全程没人手动写过一行代码。

这个词的路径很清晰：**基础设施圈的老法师先喊出来 → OpenAI 几天后发文背书 → 一周内整个 AI 圈刷屏。** 这种出身决定了它不会像很多 AI 新词一样"炒一波就凉"，它更像是在真实工程土壤里长出来的东西。

#### 一个核心等式

圈子里流传着一个特别简洁的等式：

**Agent = Model + Harness**

翻译成人话：在一个 AI Agent 系统里，除了模型本身之外，几乎所有决定它能不能稳定交付的东西，都属于 Harness。

你也可以反过来推：**Harness = Agent − Model**

这个公式把 Harness 的边界划得清清楚楚。

<!-- drawio源文件: ./drawio/harness_01_model_plus_harness.drawio -->
![Agent = Model + Harness](https://file1.kamacoder.com/i/web/20260422213747.png?v=20261005)

#### 面试核心点

别把 Harness Engineering 理解成某个具体工具或产品。面试官问的是**方法论**——你怎么设计一整套运行环境，让模型持续做对。Mitchell Hashimoto 的定义和 OpenAI 的实践，面试时要能说出来，这是概念来源。

---

### 2. 从 Prompt 到 Context 到 Harness：AI 工程的三次重心转移

面试官会问："Harness Engineering 和 Prompt Engineering、Context Engineering 到底什么区别？"

要真正讲清楚 Harness 在解决什么问题，不能一上来就讲它。因为它不是凭空冒出来的，是 AI 工程这几年一步一步**被逼出来的**。

#### Agent 本身的演进

在聊工程重心怎么转移之前，先看一眼 Agent 本身经历了什么变化——因为正是 Agent 的形态变了，才逼着工程方法跟着变。

![Agent演进：从聊天机器人到自进化Agent](https://file1.kamacoder.com/i/web/2026-04-22_21-44-10.jpg?v=20261005)

最早是**聊天机器人**——你问我答，单轮对话，模型说啥就是啥，不需要任何工程化手段。

后来接上**检索和工具**——模型能查文档、调 API 了，但问题也来了：查出来的信息怎么喂给它？工具返回的结果它能不能正确理解？这时候光靠提示词就不够了，你需要管上下文。

再后来是**自主 Agent**——模型自己规划任务、自己拆步骤、自己执行、自己检查。一跑就是几十步，中间任何一步出问题，后面全跟着错。这时候光管上下文也不够了，你需要一整套机制保证它"跑得稳"。

最前沿的是**自进化 Agent**——不只是跑得稳，还能从错误中学习、生成新技能、下次直接复用。这就把 Harness 和学习闭环绑在了一起。

**Agent 从"问答"→"干活"→"长期干活"→"越干越强"，每一步升级都暴露了前一代工程方法的短板，逼着新的工程方法诞生。**

#### 第一阶段：Prompt Engineering——让模型"听懂"

大模型本质上是一个**对上下文极度敏感的概率生成器**。你给它什么样的输入，它就沿着那个方向生成。你给它一个角色身份，它就用那个角色的思路回答；你给几个示例，它就沿那个范式补全；你强调什么约束，它就把那个约束当重点。

所以同一件事，换个说法效果能差十倍。"加个排序"可能给你一段没头没尾的代码片段，但"这是完整代码，帮我加按年龄从大到小的排序，保留所有逻辑，输出完整代码"就能给你靠谱的结果。

Prompt Engineering 解决的核心问题就一个：**模型不是不会，而是你没把话说明白。**

它在单轮对话场景里很好用。但很快，大家想做的事情变复杂了，提示词工程就撑不住了——你让大模型"分析公司财报"，它没看过你的财报，分析啥？你让它"按公司代码规范写功能"，它没看过你的规范，怎么知道该怎么写？

**提示词擅长把任务表达清楚，但不擅长凭空补出模型不知道的知识。** 它解决的是"表达"的问题，不是"信息"的问题。

#### 第二阶段：Context Engineering——让模型"知道"

为什么 Context Engineering 会火？因为大家做的产品形态变了。之前是聊天机器人，问一句答一句。后来 Agent 火了，模型要进真实环境去"干活"——多轮对话、调用工具、写代码、查数据库，要在多个步骤之间传递中间结果。

一个完整的任务，模型至少需要拿到：当前的需求文档、历史评审记录、公司相关规范、当前任务的具体目标、之前分析的中间结论。这些东西全部加起来，才叫一个完整的"上下文"。

Context Engineering 的核心思想就一句话：**模型未必知道，所以系统必须在合适的时机，把正确的信息送进去。**

但上下文窗口是有限的。更要命的是，上下文塞得太满，模型会出现**"上下文腐化"**（Context Rot）——记不住前面内容，前后矛盾，忽略最初定下的规则。像被信息淹没的人，你给他太多东西要看，反而抓不住重点。

所以 Context Engineering 要做三件事：**召回**（找最相关的信息）、**压缩**（摘要提炼省空间）、**组装**（按顺序排好，重要的放后面）。

Anthropic 的 Agent Skills 就是这个思路——一开始只给模型看"目录"，等它真的需要某个工具时再动态加载详细说明。上下文优化的本质不是"给得更多"，而是**"按需给、分层给、在正确的时机给"**。

**但到这儿还没完。**

#### 第三阶段：Harness Engineering——让模型"做对"

你把提示词写得再漂亮，把上下文管得再完美，模型在单步上的表现确实越来越好。但只要任务的链路一长，还是会出问题：

- 计划做得很好，执行时突然跑偏
- 调用工具调对了，但理解错了返回结果
- 在长任务链里悄悄偏离初衷，系统完全没察觉
- 跑着跑着忘了自己最初要干啥

提示词优化的是"意图表达"，上下文优化的是"信息供给"，但这两个都还停留在**输入侧**。当模型真正开始连续行动时，会出现一个全新的问题：**谁来监督它？谁来约束它？谁来在它跑偏时把它拉回来？**

这就是 Harness Engineering 要解决的问题。前两代工程关注的是"怎么让模型更会想"，Harness 关注的是**"怎么让模型不跑偏、跑得稳、出了错还能爬起来"**。

![Harness Engineering面试问题截图](https://file1.kamacoder.com/i/web/2026-04-22_21-44-12.png?v=20261005)

---

### 3. Harness 拆开看：六层核心组件

面试官会问："如果让你设计一个 Harness，你里面会装什么？"

去看 OpenAI、Anthropic、LangChain 这些做 Agent 的顶级团队，产品形态不同、技术栈不同，但把 Harness 掀开看内部结构，组件惊人地相似。因为"让 Agent 在真实世界稳定工作"这个命题，天然推着所有人往同一方向收敛。

一个成熟的 Harness 大致可以拆成六层，按"它在干啥"分成三组：

<!-- drawio源文件: ./drawio/harness_03_six_layers.drawio -->
![Harness六层核心组件](https://file1.kamacoder.com/i/web/20260422213752.png?v=20261005)

**输入侧**（让模型看到正确的东西）：上下文精细化管理 + 记忆与状态管理

**动作侧**（让模型做出正确的事）：工具系统 + 任务执行编排

**校验侧**（让模型知道做没做对 + 出错能爬起来）：评估观测 + 约束恢复

三组对应一个工程师在真实环境里干活的三个必要条件：**看得准 → 做得对 → 错了能兜底。**

| 层 | 解决的核心问题 |
|----|---------------|
| 上下文精细化 | 模型这一轮该看到什么？ |
| 工具系统 | 模型用什么动手？ |
| 执行编排 | 模型下一步该干啥？ |
| 记忆与状态 | 模型跨轮该记住什么？ |
| 评估与观测 | 模型做得好不好有没有尺子？ |
| 约束与恢复 | 模型出错了能不能爬起来？ |

我们一层一层看。

#### 第一层：上下文精细化

这一层管的是"空间"——**这一轮发给模型的那一坨上下文，长啥样、装了些啥、怎么排布**。它容易和第四层（记忆与状态）搞混，区别是：第一层管"这一轮看到什么"，第四层管"上一轮的事怎么流到下一轮"。

核心做三件事：

**① 把角色和目标钉死。** 大部分 Agent 跑偏，根源是身份没说清楚。模型得知道自己是谁、当前任务是啥、成功标准是什么。

**② 动态筛选，不是一次塞满。** Anthropic 把这个叫"just-in-time retrieval"——让 Agent 边干活边按需抓信息，而不是一上来把所有可能有用的东西一股脑塞进去。塞得越多，注意力越散。

**③ 结构化组织。** 固定规则放一处，动态证据放一处，中间结论放一处，三者分开。否则模型会"自我污染"——用前面错的中间结论去影响后面判断。

#### 第二层：工具系统

没有工具的大模型就是个文本预测器。接上工具之后，Agent 才真正活过来。但工具不是接得越多越好。

OpenAI 做Codex早期踩过这个坑：一开始给 Agent 接了一堆工具，想着"选择多总是好的"，结果 Agent 频繁用错工具、用错时机。后来砍掉一大半，效果反而上去了。

这一层要回答三个问题：**给它哪些工具**（只给真正需要的）、**什么时候用哪个**（该查的时候查，不该查的时候别瞎查）、**工具结果怎么喂回模型**（30条搜索结果别原样塞回去，先提炼再喂）。

MCP（Model Context Protocol）本质上就是在做工具层的标准化，让任何工具都能用同一种方式接到任何 Agent 上。

#### 第三层：执行编排

Agent 的本质，说白了就是一个 **for 循环**：思考一步 → 行动一步 → 观察结果 → 再思考下一步。经典名字叫 ReAct（Reasoning + Acting）。

但魔鬼藏在这个循环里。Agent 经常翻车的场景是：每一步它都会做，但把所有步骤串起来之后就不会了。它知道拉数据、知道写摘要，但不知道应该先拉全量再逐个分析，最后交付给你的经常是一堆半成品。把这个循环本身工程化（上下文、状态、预算、工具、终止怎么治理），就是 Loop Engineering，Loop 可以说是 Harness 的心脏，单独展开见 [从 ReAct 到 Loop Engineering](原仓库Markdown/docs/interview/llm/loop_engineering_interview.md)。

这一层的职责就是**给模型一条明确的工作轨道**，让它知道"我现在在哪一步，下一步该干啥"。

#### 第四层：记忆与状态

没有状态管理的 Agent，每一轮调用之间都是失忆的。今天跑了一遍，明天再跑完全不记得"这个任务昨天已经处理过了"，于是又处理一遍。

Anthropic 给出了一个关键做法：**Agent 的状态不应该放在上下文窗口里，而应该外化到文件系统。** 让 Agent 维护一份进度日志、一份启动脚本、一个完整的 git history，作为"长期记忆介质"。下一轮换一个全新的上下文窗口接手时，从这些文件里一读，立刻就知道"现在到哪一步了"。

记忆必须分层存：**任务状态**（写到 progress 文件里，任务完就归档）、**会话中间结果**（当轮用完就丢）、**长期记忆**（写在常驻配置里，每次调用都注入）。三类记忆生命周期完全不同，混在一起就乱了。

Claude Code 里的 CLAUDE.md、Cursor 里的 .cursorrules，就是"长期记忆"这一类的典型实现。

#### 第五层：评估与观测

这一层最容易被跳过，但跳过之后就进退两难。

太多团队做出 Agent 高高兴兴上线，跑了两周才发现实际成功率只有 50%——不是它不出结果，而是它每次都出结果，但一半时候是错的。这两周里没人发现，因为根本没有机制能告诉团队"它这次到底做得对不对"。

两件事：

**Eval 集**——手写一批典型任务，每个标注"正确答案长啥样"，每次改完 Agent 就跑一遍，对比成功率。没有 Eval 集，你对 Agent 好不好的判断永远停留在"我感觉这次变好了"的玄学阶段。

**Trace**——看到 Agent 每一步的真实足迹：做了什么决策、调了哪个工具、拿到什么返回、花了多少 token。LangSmith、Langfuse 这类 trace 系统就是干这个的。能看到 trace，调试就从"猜"变成了"看"。

#### 第六层：约束与恢复

在真实环境里，失败不是例外，是常态。这一层做三件事：

**约束**——定义"什么事 Agent 不能做"。这些约束最好硬编码到代码或 linter 规则里，而不是写在提示词里靠 Agent 自己遵守。

**校验**——在每一步输出前后做自动检查。格式对不对？频道名在不在白名单里？校验不是审美品味，是硬规则。

**恢复**——失败之后有预案。API 限流就等一会重试；发送失败就先落本地队列；token 快耗光就立即停下保存进度。每种典型失败都应该有明确恢复路径。

#### Mitchell 的"复利效应"落到哪一层？

还记得 Mitchell Hashimoto 说的"每次 Agent 犯错，把修复沉到环境里"吗？那个修复到底沉到哪？

- Agent 总是漏掉某个上下文信息 → 改第一层
- 它总是用错工具 → 改第二层
- 步骤乱 → 改第三层
- 跨天记不住进度 → 改第四层
- 没法判断做得好不好 → 搭第五层
- 一失败就崩溃 → 强化第六层

这六层不是"必须一次搭完"的任务清单，是一张**路标**——告诉你下次 Agent 犯错时，修复该落到哪里。随着时间推移，每一层被你一点一点填充、加固，Harness 就是这样一寸一寸长大的。

---

### 4. 大厂踩过的五个真实难题

面试官会问："你们做 Agent 踩过什么坑？怎么解决的？"

概念清晰是一回事，落地是另一回事。Harness 真正的难度根本不在蓝图，而在这些具体的坑里。

#### 难题一：Agent 跑久了为什么会越走越偏？

这是几乎所有做长链路 Agent 的团队都会遇到的问题。一开始 Agent 表现挺好，但跑着跑着开始"忘"——忘了最初的目标，忘了之前的决定，开始重复劳动，偏离主线。

Cognition（做 Devin 的公司）在用 Claude Sonnet 4.5 重做 Devin 时，观察到一个有趣现象，他们叫**"上下文焦虑"（Context Anxiety）**——模型自己好像也能感觉到"我快撑不住了"，不仅丢细节，还会着急收尾：突然简化方案、跳过验证、急匆匆宣布"任务完成"。更神奇的是，模型对自己"还剩多少上下文"的估计非常不准，经常以为自己快没空间了，其实还剩一大半。

很多人的第一反应是做"上下文压缩"——把历史压成摘要腾空间。这个思路对不对？对，但不够。Anthropic 挑明了一个更扎心的观察：光压缩不够，那种"已经累了"的负担感模型还是带着。

真正解开这个结的关键动作叫 **Context Reset**——直接把旧的上下文窗口整个丢掉，换一个干净的接手。

<!-- drawio源文件: ./drawio/harness_04_context_reset.drawio -->
![Context Reset：重启胜过修补](https://file1.kamacoder.com/i/web/20260422213756.png?v=20261005)状态全部外化到文件系统，新窗口从文件里读进度，立刻知道"现在到哪一步"。这特别像工程里遇到内存泄漏时的做法——不拼命优化内存，直接重启进程，从磁盘恢复状态。

**原则：重启胜过修补，状态沉到文件里。**

#### 难题二：让 Agent 自己给自己打分，为什么总偏乐观？

很多人做 Agent 时让模型干完活再自评"做得怎么样"。听起来合理，但 Agent 永远觉得自己干得不错，尤其在没标准答案的任务上，自评偏差特别明显。

Anthropic 后来想明白了一件事：**让干活的和验收的，必须是不同的人。** 他们搞出了一个三角分工：Planner（规划者）负责拆需求、Generator（生成者）负责实现、Evaluator（验收者）负责真实测试。Evaluator 不是简单看一眼代码，必须真的操作页面、看交互、检查运行结果。三个角色足够独立，才能形成有效闭环：规划 → 生成 → 验收 → 修复 → 再验收。

**原则：生产和验收必须分离，验收方必须能摸到真实世界。**

#### 难题三：Agent 反复失败，工程师到底该干啥？

当 Agent 反复失败时，绝大多数人的本能反应只有两个：再调调提示词，或者换个更强的模型。但 OpenAI 在做 Codex 项目时发现，这两招其实都是错的方向。

他们干了一件很离谱的事：在百万行代码项目里，人类工程师几乎不写代码，全部由 Agent 来写。那人在干啥？**三件事**：
* 把产品目标拆成 Agent 能力边界内的小任务；
* 当 Agent 反复失败时看"环境里缺了什么能力"然后补进去；
* 建立反馈链路让 Agent 看到自己的工作结果。

以前遇到 Agent 写代码有 bug，加一句提示词"请仔细检查代码不要有 bug"，祈祷模型听话。Codex 团队的做法是：给 Agent 接上 lint、单测、运行环境，让它自己写完自己跑，看见 bug 自己改。同样的问题，前者在求模型发挥，后者在改造环境。

**原则：Agent 反复失败时，别问模型能不能更努力，要问环境还缺什么。**

#### 难题四：规范文件越写越长，Agent 为什么反而更糊涂？

OpenAI 自己踩过的坑。早期做 Codex 时搞了一个巨大的 AGENTS.md，把所有规范、约定、最佳实践全塞进去。想法是：规则写得越全，Agent 越不会出错。结果呢？Agent 更糊涂了——上下文窗口是稀缺资源，文件越来越长，模型注意力被严重稀释。

OpenAI 后来怎么改的？把 AGENTS.md 从"百科全书"改成"目录页"——主文件只保留约 100 行核心索引，详细内容拆到子文档里。Agent 平时只看目录，真的需要某部分时才钻进去。这就是**渐进式披露**（Progressive Disclosure）——上下文优化的本质不是"给得越多越好"，而是"该给的时候给，不该给的时候藏起来"。

如果你现在在写 CLAUDE.md 或 Cursor Rules，强烈建议回头看看自己有没有"百科全书化"。如果有，赶紧拆。

**原则：规则文件宁缺毋滥，给模型看的东西少即是多。**

#### 难题五：Agent 写的代码越堆越烂，技术债怎么还？

这个特别接地气。Agent 负责写绝大多数代码后，会疯狂模仿仓库里已有的模式——好的被复制，坏的也被复制。一旦早期某段代码写歪了，Agent 把那个歪写法当"惯例"，越堆越歪。OpenAI 给它起了个扎心的名字：**AI slop（AI 代码泔水）**。

OpenAI 一开始的办法是靠人工清理——每周拿出周五一整天让工程师手工打扫。结果失败了：Agent 产出代码的速度太快，人类清理的速度跟不上，周五清一天，周一又堆满了。

最后的解法非常 Harness：**把工程师的经验写成"黄金原则"（Golden Principles）沉进仓库**，然后让一批后台 Agent 按固定节奏自动扫描仓库，找出偏离的地方，自动开修复 PR。大部分修复 PR 一分钟审完直接 auto merge。技术债从"一周一次人工清扫"变成了"每天持续自动偿还"。

OpenAI 原文里有一句话特别准：**技术债就像一笔高利息贷款，几乎永远应该每天小额还一点，而不是攒着等某一天集中还。**

**原则：技术债不是攒一堆集中还，而是每天让后台 Agent 自动偿还一点。**

#### 总结

重启胜过修补，生产验收分家，与其催模型不如改环境，规则宁缺毋滥，技术债天天还。

---

### 5. Hermes Agent vs OpenClaw：两种 Harness 实现

面试官会问："你了解 Hermes Agent 和 OpenClaw 吗？它们和 Harness Engineering 什么关系？"

#### 先定性：Harness 是方法论，Hermes 和 OpenClaw 是实现

Harness Engineering 是"方法论 / 架构思想"，Hermes Agent 和 OpenClaw 是"基于这种思想的两种具体实现"。

```
          ┌────────────────────┐
          │ Harness Engineering │  ← 方法论（抽象层）
          └─────────┬──────────┘
                    │
     ┌──────────────┼──────────────┐
     │              │              │
 Hermes Agent   Claude Code     OpenClaw
 （实现A）       （实现B）       （实现C）
```

面试时先把这个结构说出来，面试官就知道你分得清"思想"和"产品"。

![Harness Engineering与Hermes Agent、Claude Code、OpenClaw的关系](https://file1.kamacoder.com/i/web/2026-04-22_21-44-11.jpg?v=20261005)

#### OpenClaw（小龙虾）

OpenClaw 是一个开源的个人 AI 助手，你可以在自己的设备上运行。吉祥物是一只太空龙虾叫 Molty，所以圈子里叫它"小龙虾"。

它的核心定位是**消息优先、本地优先**——一个网关进程打通 25+ 消息平台（WhatsApp、Telegram、Slack、Discord、微信、QQ、飞书……），你的 AI 助手无处不在。

OpenClaw 的 Harness 特点：

| 能力 | 实现方式 |
|------|----------|
| 上下文管理 | AGENTS.md / SOUL.md / TOOLS.md 注入 |
| 工具系统 | MCP 协议 + ClawHub 技能市场 |
| 执行编排 | 单 Agent 循环 |
| 记忆与状态 | 本地文件持久化 |
| 评估与观测 | 基础日志 |
| 约束与恢复 | 沙盒后端（Docker/SSH） |

OpenClaw 是个成熟的个人助手——消息全平台覆盖、本地隐私优先、技能市场丰富。但它的 Harness 本质上是一个**工具执行系统**：你给 prompt、给 tools，它调工具、返回结果。

#### Hermes Agent（爱马仕）

Hermes Agent 是 Nous Research 出品的自进化 Agent，标语是"The agent that grows with you"（和你一起成长的 Agent）。

Hermes 做了一个非常关键的升级：**从"工具执行系统" → "自进化系统"**。

它和 OpenClaw 最大的区别，就是内置了一个**学习闭环（Learning Loop）**：

<!-- drawio源文件: ./drawio/harness_05_hermes_vs_openclaw.drawio -->
![Hermes学习闭环 vs OpenClaw线性执行](https://file1.kamacoder.com/i/web/20260422213759.png?v=20261005)

```
执行任务 → 总结经验 → 生成 skill → 存入记忆 → 下次复用
```

Hermes 的 Harness 特点：

| 能力 | 实现方式 |
|------|----------|
| 上下文管理 | AGENTS.md + 动态加载 |
| 工具系统 | MCP + 自生成技能（~/.hermes/skills/） |
| 执行编排 | 单 Agent + 子 Agent 并行委派 |
| 记忆与状态 | 持久化记忆 + Honcho 方言式用户建模 + FTS5 会话搜索 |
| 评估与观测 | 自我督促 + LLM 摘要跨会话回忆 |
| 约束与恢复 | 6 种终端后端 + 定时任务（cron） |

#### 核心对比：谁强在哪？

| 维度 | OpenClaw | Hermes Agent |
|------|----------|-------------|
| 定位 | 个人 AI 助手 | 自进化 Agent |
| 语言 | TypeScript | Python |
| 消息平台 | 25+（覆盖最广） | 6 个（CLI/Telegram/Discord/Slack/WhatsApp/Signal） |
| 技能来源 | ClawHub 市场（社区贡献） | 自主生成 + agentskills.io 标准 |
| 记忆系统 | 基础持久化 | 深度：用户建模 + 跨会话搜索 + 自我督促 |
| 学习能力 | 无（每次都是"新手"） | 有（从经验中创建技能，越用越强） |
| 研究能力 | 无 | 批量轨迹生成 + RL 训练环境 |
| 迁移友好度 | — | 内置 `hermes claw migrate` 从 OpenClaw 导入 |

**一句话总结区别**：

OpenClaw 让模型**能干活**——全平台接入，工具齐全，但每次都是"新手"。

Hermes Agent 让模型**越干越强**——它会记住做过的事、生成新技能、下次直接复用，是 Harness + Learning Loop 的结合体。

#### 面试答法

面试时先说清楚 Harness 是方法论、Hermes 和 OpenClaw 是实现。然后对比两者：OpenClaw 是"工具执行系统"，消息覆盖广；Hermes 是"自进化系统"，核心差异化是学习闭环。

---

### 6. 和 Prompt/Context Engineering 到底什么关系？

面试官会问："这三个 Engineering 是替代关系吗？我都要学吗？"

#### 不是替代，是包含

三者根本不是替代关系，而是**包含关系**：

- Prompt 是对"指令"的工程化
- Context 是对"输入环境"的工程化
- Harness 是对"整个运行系统"的工程化

边界一层比一层大。**Prompt 是 Context 的一部分，Context 是 Harness 的一部分。** 当你做 Harness 的时候，里面一定包含 Context 工程，Context 工程里又一定包含 Prompt 工程。

#### 四层递进

如果你把 Hermes Agent 的学习闭环也加进来，可以看成四层递进：

| 层次 | 解决什么 | 一句话 |
|------|---------|--------|
| Prompt Engineering | 怎么说 | 让模型听懂你想干啥 |
| Context Engineering | 给什么信息 | 让模型知道该用什么 |
| Harness Engineering | 能干什么 | 让模型持续做对 |
| Learning Loop | 会不会变强 | 让模型越干越强 |

#### 面试核心点

面试时别说"我三个都会"，要说出理解层次：先说清三者是包含关系不是替代关系，再强调 Harness 是站在更大系统视角把前面两个包进去了，最后点出真正的分水岭——**当任务还是单轮对话时 Prompt 就够，需要外部知识时 Context 关键，但进入"长链路、可执行、低容错"的真实场景，Harness 几乎是不可避免的。**

---

### 7. 大厂真实面试追问汇总

以下是各大厂在 Harness Engineering 方向的真实追问，整理汇总。

#### 概念理解类

**Q：Harness Engineering 和 MLOps 有什么区别？**

MLOps 侧重模型训练、部署、版本管理的工程化流程；Harness Engineering 侧重模型上线后的运行环境设计——工具、约束、评估、恢复。MLOps 是"怎么把模型搞上线"，Harness 是"上线后怎么让它跑得稳"。两者互补，MLOps 偏训练侧，Harness 偏运行侧。

**Q：Agent = Model + Harness 这个等式你怎么理解？**

除了模型本身之外，几乎所有决定 Agent 能不能稳定交付的东西都属于 Harness——工具、上下文文件、记忆系统、评估机制、约束规则、恢复策略。换模型提升的是天花板，搭 Harness 提升的是落地能力。在模型迭代速度放缓的今天，Harness 这部分的提升空间可能比你想象的大得多。

**Q：Context Engineering 和 RAG 是什么关系？**

RAG 是 Context Engineering 的一种具体实现技术——它解决"召回"这一步（从大量文档中找出最相关的）。Context Engineering 还包括压缩（摘要提炼）和组装（按顺序排布），RAG 只管了第一步。

#### 技术深挖类

**Q：Agent 跑久了上下文腐化怎么办？**

三步：先做上下文压缩（摘要提炼历史对话，腾出空间），如果压缩还不够（模型带着"疲惫感"），做 Context Reset（整个上下文窗口丢掉，换干净的，状态从文件系统恢复），关键是状态必须外化到文件而不是留在上下文窗口里。

**Q：AGENTS.md 越写越长效果变差怎么办？**

改成渐进式披露：主文件只保留核心索引（OpenAI 建议约 100 行），详细规则拆到子文档，Agent 按需加载。这和 Anthropic 的 Agent Skills 是同一思路——不一开始塞所有信息，而是需要时才动态注入。

**Q：Hermes Agent 的学习闭环具体怎么工作？**

Agent 执行完一个复杂任务后，自动从经验中提取模式生成一个 skill 文件（存在 ~/.hermes/skills/），下次遇到类似任务时搜索已有 skill 直接复用，同时在使用中不断改进 skill。这和 Mitchell Hashimoto 说的"复利效应"一脉相承——每次犯错都沉到环境里，环境越来越强，Agent 越来越稳。

#### 生产实战类

**Q：你们 Agent 的技术债怎么管理？**

技术债不能攒着集中还。两种做法：一是把工程师的经验写成 Golden Principles（黄金原则）沉进仓库，让 Agent 按规则写代码；二是让后台 Agent 定期自动扫描仓库找偏离，开修复 PR，人类快速审核合并。每天自动还一点，比攒到周五集中清效果好得多。

**Q：怎么给非技术人员解释 Harness Engineering 的价值？**

"LLM 就像高速公路上的自动驾驶——能跑很快，但如果不装刹车、不装安全气囊、不装仪表盘，你敢坐吗？Harness Engineering 就是给 AI 装刹车和安全气囊。没有它，AI 越强大越危险。"

---

### 写在最后

AI 工程的重心，过去两年换了三次：Prompt 解决"怎么说"，Context 解决"给什么信息"，Harness 解决"能不能持续做对"。这三个不是替代关系，是包含关系——Prompt 是 Context 的一部分，Context 是 Harness 的一部分。

Harness Engineering 之所以在 2026 年火了，不是因为又造了个新词，而是因为 AI 真正上了生产。银行用 LLM 审贷款、医院用 LLM 辅助诊断、客服用 Agent 处理投诉——**能用不等于可靠**，模型幻觉一次就是真金白银的损失。

Agent = Model + Harness。换模型提升天花板，搭 Harness 提升落地能力。Hermes Agent 比 OpenClaw 多走了一步——不只是让模型能干活，而是让模型越干越强。

如果你最近在做 Agent，别再把所有精力花在调模型、调提示词上了。回过头看看你的 Harness 长啥样——有没有规则文件、有没有校验闭环、有没有任务编排、有没有评估机制、有没有失败恢复、有没有学习闭环。这些东西，每一项都能让你的 Agent 上一个台阶。

**会搭 Agent 的人越来越多，但能让 Agent 在生产环境稳定运行的人，才是稀缺的**。

顺着这条线继续深挖：想先建立从技术架构到组织人才的整体认识，看 [生产级 Agent 全景面试详解](原仓库Markdown/docs/interview/llm/production_agent_architecture_harness_org_talent.md)；Agent 的底层原理和协议体系看 [Agent 大厂面试题汇总](原仓库Markdown/docs/interview/llm/agent_interview.md)；多 Agent 场景下 Harness 怎么编排调度看 [Multi-Agent Harness 面试详解](原仓库Markdown/docs/interview/llm/multi_agent_harness_interview.md)；怎么给 Agent 装上"仪表盘"做可观测看 [Agent Harness 可观测性面试详解](原仓库Markdown/docs/interview/llm/agent_harness_observability_interview.md)；几个主流框架的 Harness 实现对比看 [Agent 框架横评](原仓库Markdown/docs/interview/llm/agent_framework_comparison.md)。

加油

---

### 参考文献

1. Mitchell Hashimoto，《My AI Adoption Journey》 — Harness Engineering 概念的起源，第5步首次提出 "Engineer the Harness"：https://mitchellh.com/writing/my-ai-adoption-journey
2. OpenAI，《Harness engineering: leveraging Codex in an agent-first world》— OpenAI 官方背书，5个月100万行代码实践：https://openai.com/index/harness-engineering/
3. Anthropic，《Effective harnesses for long-running agents》— Anthropic 对 Harness 设计的工程实践，Context Reset 方案：https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
4. Anthropic，《Harness design for long-running application development》— 上下文焦虑、三角分工、生产验收分离：https://www.anthropic.com/engineering/harness-design-long-running-apps
5. Anthropic，《Effective context engineering for AI agents》— Context Engineering 的官方指南，渐进式披露：https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
6. Anthropic，《Building effective agents》— Agent 工作流模式、ACI 设计原则：https://www.anthropic.com/engineering/building-effective-agents
7. LangChain，《Improving Deep Agents with harness engineering》— LangChain 的 Harness 实践，Eval 集与 trace 回放：https://blog.langchain.com/improving-deep-agents-with-harness-engineering/
8. Cognition，《Rebuilding Devin for Claude Sonnet 4.5: Lessons and Challenges》— 上下文焦虑现象的首次报告：https://cognition.ai/blog/devin-sonnet-4-5-lessons-and-challenges
9. Nous Research，Hermes Agent — 自进化 Agent 开源项目（学习闭环 + 技能生成）：https://github.com/NousResearch/hermes-agent
10. OpenClaw — 个人 AI 助手开源项目（全平台消息 + ClawHub 技能市场）：https://github.com/openclaw/openclaw

**费曼自述：** 用一次真实或公开项目的任务轨迹说明工具调用、验证和失败恢复。

**概念闪卡：** Q：只展示 Agent 最终答对足够吗？A：不够，还要看权限、过程、错误恢复与成本。

原文核对：[Harness Engineering大厂面试题汇总：从Prompt到Context到Harness](原仓库Markdown/docs/interview/llm/harness_interview.md)。

## 面试专题：Agent Harness 可观测性：生产级 AI 项目必须补上的一课

### 目录

1. [为什么 Agent 能跑起来了，生产里还是不稳？](#一为什么-agent-能跑起来了生产里还是不稳)
2. [Agent Harness 可观测性到底是什么？](#二agent-harness-可观测性到底是什么)
3. [Agent 可观测性和普通日志有什么区别？](#三agent-可观测性和普通日志有什么区别)
4. [一个 Agent Trace 里到底要记录什么？](#四一个-agent-trace-里到底要记录什么)
5. [长链路任务里，怎么发现 Agent 已经跑偏了？](#五长链路任务里怎么发现-agent-已经跑偏了)
6. [工具调用出问题，怎么定位是模型错、参数错，还是工具错？](#六工具调用出问题怎么定位是模型错参数错还是工具错)
7. [Agent 成本失控，应该监控哪些指标？](#七agent-成本失控应该监控哪些指标)
8. [线上失败样本，怎么沉淀成 Harness 的长期改进？](#八线上失败样本怎么沉淀成-harness-的长期改进)
9. [如果让你设计一套生产级 Agent 可观测系统，你会怎么答？](#九如果让你设计一套生产级-agent-可观测系统你会怎么答)

### 一、为什么 Agent 能跑起来了，生产里还是不稳？

面试官可能会这么问："现在模型能力、工具调用、MCP 都成熟了，为什么 Agent 真正上生产还是容易翻车？"

这个问题很关键。

很多录友会说："因为模型还会幻觉。"

对，但不够。

Agent 生产不稳，不只是模型本身的问题，而是**长链路执行系统的问题**。

普通聊天机器人答错一句话，最多是回答质量差。

但 Agent 不一样，它会规划任务、调用工具、读写状态、修改文件、发起请求、触发工作流。它的每一步输出，都会变成下一步输入。

这就带来一个很大的差异：

**Agent 的错误会沿着执行链路传播。**

第一步目标理解偏了，后面计划全偏。

第二步工具选错了，后面拿到的证据就是错的。

第三步参数传错了，工具返回异常，Agent 可能又基于异常结果继续推理。

跑到最后，结果看起来还挺像那么回事，但中间过程已经乱了。

这也是为什么 Demo 阶段和生产阶段的关注点完全不同。

Demo 阶段，大家看的是："它能不能跑通一次？"

生产阶段，大家看的是：

- 能不能稳定跑很多次？
- 能不能在异常情况下恢复？
- 能不能控制成本？
- 能不能解释过程？
- 能不能复盘失败？
- 能不能在版本迭代后不退化？

如果没有可观测性，这些问题全都只能靠猜。

<!-- drawio源文件: ./drawio/agent_observability_01_demo_to_production.drawio -->
![Agent长链路执行中小偏差被逐步放大的过程](https://file1.kamacoder.com/i/web/20260602115939_agent_observability_01_demo_to_production_compressed.png?v=20261005)

所以面试时可以这样回答：

**Agent 生产不稳的根因，不是某一个 Prompt 没写好，而是长链路执行过程不可见。生产级 Agent 必须把计划、上下文、工具调用、状态变化、成本和评估结果全部纳入观测，否则出了错无法定位，也无法持续改进。**

这句话比单纯说"模型会幻觉"要高级得多。

因为你已经从模型视角，切到了系统工程视角。

### 二、Agent Harness 可观测性到底是什么？

面试官可能会这么问："你怎么理解 Agent Harness 可观测性？它到底要观测什么？"

先给一个定义：

**Agent Harness 可观测性，是对 Agent 执行过程中的目标、计划、上下文、工具调用、状态变化、成本、风险和结果质量进行全链路记录、度量、回放和评估的能力。**

注意，不是只记录输入输出。

Agent 真正麻烦的地方，不在输入和输出，而在中间过程。

一次复杂任务里，Agent 可能经历这些步骤：

用户给出目标。

Agent 先拆计划。

然后检索资料。

再调用工具。

再根据工具返回结果更新状态。

再重新规划。

再生成结果。

最后还要自检或交给 Evaluator 评估。

这里面任何一步都可能出问题。

所以 Agent 可观测性至少要覆盖七类对象：

**目标**：原始任务是什么，过程中有没有被改写。

**计划**：Agent 是怎么拆任务的，每一步是否服务原始目标。

**上下文**：哪些信息被塞进模型，来源是什么，是否过期或污染。

**工具**：调了什么工具，参数是什么，返回了什么，耗时多久。

**状态**：任务状态、记忆、文件、数据库记录发生了什么变化。

**成本**：每一步消耗多少 Token、时间、工具资源。

**评估**：最终结果是否正确，中间轨迹是否合理。

<!-- drawio源文件: ./drawio/agent_observability_02_scope.drawio -->
![Agent Harness可观测性沿执行链路采集目标计划上下文工具和结果](https://file1.kamacoder.com/i/web/20260602120022_agent_observability_02_scope_compressed.png?v=20261005)

这里有一个非常重要的判断：

**可观测性不是为了事后甩锅，而是为了让 Agent 的每一次执行都能变成可分析、可复现、可改进的工程数据。**

一个生产级 Agent 系统，不能只问"结果对不对"。

还要问：

- 过程合理吗？
- 证据充分吗？
- 工具调用必要吗？
- 有没有越权动作？
- 有没有重复步骤？
- 有没有成本异常？
- 有没有潜在漂移？

这些问题回答清楚了，Agent 才算从"会干活"走向"可治理"。

面试里可以这么说：

**Agent Harness 可观测性，本质上是在 Harness 里给 Agent 装眼睛。它不只看最终答案，而是把一次任务拆成目标、计划、上下文、工具、状态、成本、评估这些可观察对象，让系统能定位错误、复盘过程、控制成本，并把失败样本沉淀成后续改进。**

### 三、Agent 可观测性和普通日志有什么区别？

面试官可能会这么问："你们不是已经有日志、监控、链路追踪了吗？为什么 Agent 还要单独做可观测性？"

这个问题很容易答错。

很多人会说："我们把 Agent 输入输出打到日志里。"

这不够。

普通日志主要记录的是：

- 接口什么时候被调用
- 请求参数是什么
- 返回状态码是什么
- 服务耗时多少
- 有没有异常堆栈

这些对传统后端系统很有用。

但 Agent 的问题往往不是接口 500。

Agent 最常见的问题是：

- 它为什么选择这个工具？
- 它为什么忽略了某段关键上下文？
- 它为什么跑到另一个子任务去了？
- 它为什么把工具返回结果理解错了？
- 它为什么重复规划了 5 次？

这些问题，普通日志回答不了。

因为普通日志记录的是**发生了什么**。

而 Agent 可观测性还要解释：**它为什么这么做**。

<!-- drawio源文件: ./drawio/agent_observability_03_logs_vs_trace.drawio -->
![普通日志和Agent Trace在定位任务偏差上的区别](https://file1.kamacoder.com/i/web/20260602120106_agent_observability_03_logs_vs_trace_compressed.png?v=20261005)

举个例子。

用户让 Agent："分析最近 7 天订单异常原因。"

普通日志可能只看到：

`POST /agent/run 200`

`call search_orders success`

`call query_refund_rate success`

`LLM response success`

看起来都成功。

但最终结论错了。

为什么错？

普通日志看不出来。

Agent Trace 应该能看到：

- 原始目标是"订单异常原因"
- Agent 第一步把目标改写成了"退款率异常"
- 检索时只查了退款数据，没查支付失败数据
- 工具返回里有"支付通道超时"，但模型没有纳入结论
- 最终回答只归因到退款策略

这样你才能定位：问题不是工具失败，而是**目标改写偏了 + 证据选择偏了**。

这就是 Agent 可观测性和普通日志的差异。

普通日志适合看系统有没有挂。

Agent Trace 适合看任务有没有做对。

面试里可以这么答：

**传统日志关注服务状态，Agent 可观测性关注执行意图和决策过程。Agent 出问题时，HTTP 状态码可能全是 200，但任务已经偏了。所以生产级 Agent 需要结构化 Trace，把 Plan、Action、Observation、State Diff、Context Source 和 Eval Result 串起来，而不是只打输入输出日志。**

### 四、一个 Agent Trace 里到底要记录什么？

面试官可能会这么问："如果让你设计 Agent Trace，你会记录哪些字段？"

这个问题很实战。

不要只说"记录输入输出"。

一个真正可用的 Agent Trace，至少要分两层：**任务级 Trace** 和 **步骤级 Step**。

任务级 Trace 记录这次任务的全局信息：

- `trace_id`：一次任务的唯一 ID
- `user_goal`：用户原始目标
- `normalized_goal`：系统改写后的目标
- `agent_version`：Agent 版本
- `model_version`：模型版本
- `policy_version`：工具权限、成本预算、审批策略版本
- `start_time` / `end_time`：任务开始结束时间
- `final_status`：成功、失败、降级、人工接管
- `total_tokens` / `total_cost`：总消耗
- `final_eval`：最终评估结果

步骤级 Step 记录每一步发生了什么：

- `step_id`：第几步
- `step_type`：规划、工具调用、观察、总结、评估
- `current_goal`：当前步骤服务哪个子目标
- `context_refs`：本步使用了哪些上下文来源
- `model_input_summary`：输入摘要
- `model_output_summary`：输出摘要
- `tool_name`：调用了哪个工具
- `tool_args`：工具参数
- `tool_result_summary`：工具返回摘要
- `state_before` / `state_after`：状态变化
- `latency_ms`：耗时
- `tokens`：Token 消耗
- `risk_level`：风险等级
- `eval_result`：本步是否合理

<!-- drawio源文件: ./drawio/agent_observability_04_trace_schema.drawio -->
![Agent Trace Schema如何支持失败任务回放和评测](https://file1.kamacoder.com/i/web/20260602120106_agent_observability_04_trace_schema_compressed.png?v=20261005)

这里面有几个字段特别重要。

第一个是 `context_refs`。

因为很多 Agent 错误，不是模型突然变差，而是上下文给错了、给多了、给旧了。

你必须知道本步推理到底用了哪些文档、哪些历史消息、哪些记忆。

第二个是 `state_before` 和 `state_after`。

Agent 是会改变环境的。它可能写文件、改数据库、创建工单、发邮件。

如果没有状态差异记录，出事后你不知道它到底动了什么。

第三个是 `eval_result`。

不要等最终答案出来才评估。

长链路任务里，中间步骤就要评估。

因为越早发现偏差，修复成本越低。

面试时可以这么说：

**我会把 Agent Trace 设计成任务级和步骤级两层。任务级记录原始目标、版本、状态、总成本和最终评估；步骤级记录每一步的计划、上下文来源、模型输出、工具调用、状态差异、成本和局部评估。这样失败后既能看全局链路，也能定位到具体哪一步开始出问题。**

### 五、长链路任务里，怎么发现 Agent 已经跑偏了？

面试官可能会这么问："Agent 跑一个长任务，执行了二三十步，你怎么判断它没有偏离原始目标？"

这个问题和之前聊过的 [Agent上下文漂移](原仓库Markdown/docs/interview/llm/agent_drift_hallucination_interview.md) 强相关。

长链路任务里，Agent 跑偏不是突然发生的。

它通常是慢慢偏的。

一开始只是处理一个小问题。

后来这个小问题变成子任务。

再后来子任务抢走主线。

最后 Agent 忘了最初要干什么。

所以漂移检测不能只看最终答案，要看执行过程里的信号。

常见信号有五个：

**第一，当前动作和原始目标相关度下降。**

比如原始目标是"分析订单异常"，Agent 连续几步都在优化报表格式，这就危险了。

**第二，子任务耗时超过预算。**

支线任务可以做，但不能无限做。一个格式修复占掉整个任务 70% 的步骤，就说明主线被挤压了。

**第三，重复步骤增多。**

反复查询同一个接口，反复重写同一个计划，反复调用同一个工具，都说明 Agent 可能卡住了。

**第四，目标表述发生变化。**

最初是"分析订单异常原因"，中间变成"分析退款率"，再变成"输出退款建议"，这就是目标漂移。

**第五，关键证据没有被使用。**

工具返回里有重要异常信号，但 Agent 后续步骤完全没引用，说明注意力可能被别的信息带偏了。

<!-- drawio源文件: ./drawio/agent_observability_05_drift_detection.drawio -->
![长链路Agent任务中根据轨迹与目标主线距离检测漂移](https://file1.kamacoder.com/i/web/20260602120106_agent_observability_05_drift_detection_compressed.png?v=20261005)

工程上怎么做？

可以加三类机制。

**目标锚点**：每隔 N 步，把原始目标、当前计划、最近动作放在一起，让系统判断是否还在朝目标推进。

**检查点评估**：每完成一个子任务，就评估它对原始目标的贡献，而不是只看它有没有完成。

**漂移告警**：当相关度低、重复率高、子任务超预算时，触发重新规划、人工确认或任务终止。

注意，这里不是让模型每一步都自我反省。

那样成本会爆。

更合理的是在关键节点做轻量检测。

比如每 3 到 5 步做一次，或者在工具失败、重复调用、预算超限时触发。

面试里可以这么答：

**我不会只看最终结果判断 Agent 是否跑偏，而会在 Trace 里持续记录当前动作和原始目标的关系、子任务进度、重复步骤、目标改写和关键证据使用情况。工程上可以设置目标锚点和检查点评估，一旦出现连续低相关动作、重复工具调用或子任务超预算，就触发 Re-Planning、降级或人工确认。**

### 六、工具调用出问题，怎么定位是模型错、参数错，还是工具错？

面试官可能会这么问："Agent 调工具失败了，你怎么判断问题出在模型、参数、权限，还是工具本身？"

Agent 工具调用的问题，不能笼统归因成"模型不稳定"。

生产里要分清楚故障类型。

否则你根本不知道该修 Prompt、修 Schema、修权限，还是修工具服务。

常见工具调用故障可以分成五类。

**第一类，工具选择错误。**

用户问订单数据，Agent 却调用了知识库搜索。

这是模型在工具选择上出了问题，通常要优化工具描述、路由策略或工具候选集。

**第二类，参数生成错误。**

工具选对了，但参数错了。

比如日期格式不对、字段名不存在、必填参数缺失。

这类问题要靠 JSON Schema、参数校验、错误反馈和少量示例修。

**第三类，权限边界错误。**

Agent 调用了不该调用的写操作。

比如用户只是询问退款规则，Agent 却尝试执行退款。

这类问题不能靠 Prompt 兜，要靠 Tool Registry、权限策略和高风险动作审批。

**第四类，工具服务异常。**

工具本身超时、接口 500、依赖服务挂了。

这不是模型问题，要走重试、熔断、降级和告警。

**第五类，结果理解错误。**

工具返回是对的，但 Agent 看错了。

比如接口返回"支付失败率升高"，Agent 却归因成"退款率升高"。

这类问题要记录工具返回摘要、证据引用、输出校验和评估结果。

<!-- drawio源文件: ./drawio/agent_observability_06_tool_attribution.drawio -->
![Agent工具调用故障从工具选择到结果理解的逐段归因路径](https://file1.kamacoder.com/i/web/20260602120154_agent_observability_06_tool_attribution_compressed.png?v=20261005)

所以工具调用可观测性要记录完整链路：

工具调用前，记录为什么选这个工具。

工具调用中，记录工具名、参数、权限、耗时、重试。

工具调用后，记录返回结果、模型如何解释结果、结果有没有被最终答案引用。

这里有一个很实用的排查顺序：

先看工具是否存在。

再看权限是否允许。

再看参数是否通过 Schema。

再看工具服务是否成功。

再看模型是否正确理解返回结果。

这个顺序能把大部分问题拆开。

面试里可以这么答：

**工具调用失败不能一概说是模型幻觉。我会把链路拆成工具选择、参数生成、权限校验、工具执行、结果理解五段，每段都有结构化 Trace。选择错了修工具描述和路由，参数错了修 Schema 和校验，权限错了加审批边界，工具异常走熔断降级，结果理解错了做证据引用和输出评估。**

### 七、Agent 成本失控，应该监控哪些指标？

面试官可能会这么问："Agent 一跑长任务 Token 就爆，你会怎么做成本可观测和控制？"

这个问题很现实。

很多 Agent Demo 看起来很猛，一到生产就贵得离谱。

原因也很简单：

Agent 不是一次模型调用。

它是多次规划、多次工具调用、多次上下文拼接、多次自检、多次重试。

一次用户请求，背后可能烧掉几十次 LLM 调用。

所以 Agent 成本不能只看总 Token，要拆到步骤级。

至少要监控七类指标：

**第一，总 Token 和单步 Token。**

总 Token 只能告诉你贵不贵，单步 Token 才能告诉你贵在哪一步。

**第二，上下文长度。**

很多成本不是推理难，而是上下文塞太多。尤其是把历史记录、检索文档、工具返回全部原样塞进去，成本很快爆。

**第三，工具重试次数。**

工具失败一次不可怕，反复失败才可怕。每次失败都可能触发新的模型分析和重试。

**第四，重复规划次数。**

Agent 不断 Re-Planning，说明它对任务状态不确定，或者前面步骤没有推进。

**第五，无效步骤占比。**

有些步骤执行了，但对目标没有贡献。这个指标越高，说明 Agent 在空转。

**第六，模型路由成本。**

是否所有步骤都用了大模型？简单分类、格式转换、规则判断能不能用小模型或规则？

**第七，端到端延迟。**

生产系统里，时间也是成本。Agent 跑 3 分钟才给结果，很多业务场景根本不能接受。

<!-- drawio源文件: ./drawio/agent_observability_07_cost_breakdown.drawio -->
![Agent成本失控来自循环重试和上下文膨胀的过程](https://file1.kamacoder.com/i/web/20260602120154_agent_observability_07_cost_breakdown_compressed.png?v=20261005)

成本控制不是简单砍 Token。

乱砍 Token，可能把质量也砍没了。

更合理的做法是做预算分层：

简单任务给低预算。

复杂任务给中预算。

高价值任务才允许高预算。

超过预算时，不是直接失败，而是触发策略：

- 压缩上下文
- 换小模型
- 减少候选工具
- 停止低价值子任务
- 要求用户补充信息
- 转人工或降级输出

面试里可以这么答：

**Agent 成本可观测不能只看总 Token，要拆到每个 Step。重点看单步 Token、上下文长度、工具重试、重复规划、无效步骤、模型路由和端到端延迟。控制上要做预算分层，超过阈值后触发上下文压缩、小模型路由、停止低价值子任务、降级输出或人工接管，而不是简单让 Agent 继续烧。**

### 八、线上失败样本，怎么沉淀成 Harness 的长期改进？

面试官可能会这么问："Agent 线上失败一次之后，你怎么保证下次不会再犯同样的错？"

这个问题直接考 Harness 思维。

很多团队处理线上失败，是这样的：

出错了。

查一下日志。

改一下 Prompt。

上线。

然后祈祷下次别再出事。

这不叫工程化。

真正的 Harness 改进，要把失败样本变成资产。

一次线上失败，至少要沉淀出四类东西：

**第一，失败归因。**

到底是目标理解错、上下文缺失、工具选择错、参数错、权限错、模型误读，还是评估漏掉？

没有归因，就没有改进方向。

**第二，评测样本。**

把这次失败任务变成固定测试用例。

以后 Prompt 改了、模型换了、工具描述改了，都要跑一遍。

**第三，Harness 规则。**

如果是结构性错误，就要补到 Harness 里。

比如高风险写操作必须审批，连续三次工具失败必须停止，目标相关度低于阈值必须重新规划。

**第四，监控指标。**

如果这次失败有早期信号，就要把信号变成告警。

比如重复调用、成本异常、关键证据未引用、工具返回错误率升高。

<!-- drawio源文件: ./drawio/agent_observability_08_failure_loop.drawio -->
![线上Agent失败样本沉淀为评测规则和监控资产的闭环](https://file1.kamacoder.com/i/web/20260602120155_agent_observability_08_failure_loop_compressed.png?v=20261005)

这就是 Harness Engineering 的核心精神：

**不要只修这一次错误，要把错误修进环境里。**

Agent 犯错不可怕。

可怕的是每次犯错都靠人肉记忆，下次换个版本又犯。

生产级 Agent 的进步，不是靠写一个完美 Prompt，而是靠持续积累失败样本、评测集、规则、监控和回归流程。

面试里可以这么答：

**线上失败后，我会先从 Trace 做失败归因，把问题定位到目标理解、上下文、工具、权限、状态、成本或评估某一层。然后把失败任务沉淀成评测样本，把结构性问题沉淀成 Harness 规则，把早期信号沉淀成监控指标。以后每次模型、Prompt、工具或 Harness 改动，都用这些样本做回归，避免同类错误反复出现。**

### 九、如果让你设计一套生产级 Agent 可观测系统，你会怎么答？

面试官可能会这么问："如果让你从 0 设计一套生产级 Agent 可观测系统，你会怎么设计？"

这类问题不要上来就讲技术组件。

先讲目标，再讲架构。

目标很简单：

**让每一次 Agent 执行都可追踪、可解释、可复盘、可评估、可改进。**

架构可以拆成六层。

**第一层，Trace 采集层。**

在 Harness 的关键节点打点：任务开始、计划生成、上下文组装、工具调用、状态更新、评估、任务结束。

这里要注意，采集点应该在 Harness 里，而不是散落在各个 Agent Prompt 里。

否则后面一定乱。

**第二层，事件存储层。**

把 Agent 运行事件结构化存起来。

原始输入输出可以脱敏存储，结构化字段用于查询分析。

比如 trace_id、step_id、tool_name、tokens、latency、status、risk_level、eval_score。

**第三层，指标计算层。**

从 Trace 里计算关键指标：

- 任务成功率
- 平均步骤数
- 平均 Token
- 工具失败率
- 重试率
- 漂移告警率
- 人工接管率
- 评测通过率

**第四层，可视化与告警层。**

给研发看 Trace 回放。

给业务看任务成功率和成本。

给运维看延迟、失败率、异常工具。

高风险动作、成本超限、连续失败要触发告警。

**第五层，回放与评测层。**

线上失败任务可以一键回放。

回放时固定输入、上下文、工具返回，判断新版本有没有修好旧问题。

**第六层，改进闭环层。**

把失败归因沉淀到 Prompt、工具描述、权限策略、上下文策略、评测集和 Harness 规则里。

<!-- drawio源文件: ./drawio/agent_observability_09_system_architecture.drawio -->
![生产级Agent可观测系统从Trace采集到Harness改进的架构闭环](https://file1.kamacoder.com/i/web/20260602120155_agent_observability_09_system_architecture_compressed.png?v=20261005)

这个系统不是一天建成的。

刚开始可以很轻。

先把 Trace 跑通。

再补工具调用和成本指标。

再做失败样本回放。

最后做评测闭环和自动告警。

但方向一定要对：

**不要把 Agent 可观测性做成日志大屏，要做成 Agent 改进系统。**

面试里可以这么答：

**我会把生产级 Agent 可观测系统拆成 Trace 采集、事件存储、指标计算、可视化告警、回放评测和改进闭环六层。Harness 在关键节点统一打点，记录计划、上下文、工具、状态、成本和评估；失败任务进入回放和评测集；结构性问题沉淀成规则、权限、预算和上下文策略。最终目标不是多一个监控大屏，而是让 Agent 的每次失败都能推动系统变稳。**

最后说一句。

Agent 真正进生产后，最怕的不是它犯错。

**最怕的是它犯错了，你不知道它为什么错。**

没有可观测性，Harness 就没有眼睛。

**费曼自述：** 用一次真实或公开项目的任务轨迹说明工具调用、验证和失败恢复。

**概念闪卡：** Q：只展示 Agent 最终答对足够吗？A：不够，还要看权限、过程、错误恢复与成本。

原文核对：[Agent Harness 可观测性：生产级 AI 项目必须补上的一课](原仓库Markdown/docs/interview/llm/agent_harness_observability_interview.md)。

## 附录完成标准

能画出任务闭环；说清 Prompt、工具、MCP、Skill、Hook 和 Harness 各负责什么；自己写出有权限检查、预算、停止条件和日志的最小循环。产品专有命令用时查官方文档，不作为背诵任务。

