# 02-Prompt与模型调用（重构实战版 v3）

本章聚焦大模型应用最核心的工程接口层：从 API 重试与终态判别、角色与注入隔离、推理策略路由，到结构化输出、Function Calling、流式 SSE、上下文原子裁剪、有效成本监控与回归评测。

代码说明：本章采用工业级工程校验写法（ch2_code/step_01..09.py），全部通过 python3 断言自测。

## 目录导航

[2.1 接入模型：请求、响应与失败状态【🔥 必会】](http://./2.1-%E6%8E%A5%E5%85%A5%E6%A8%A1%E5%9E%8B%EF%BC%9A%E8%AF%B7%E6%B1%82%E3%80%81%E5%93%8D%E5%BA%94%E4%B8%8E%E5%A4%B1%E8%B4%A5%E7%8A%B6%E6%80%81.md)

[2.2 Prompt 结构：角色、变量与不可信资料【🔥 必会】](http://./2.2-Prompt%E7%BB%93%E6%9E%84%EF%BC%9A%E8%A7%92%E8%89%B2%E3%80%81%E5%8F%98%E9%87%8F%E4%B8%8E%E4%B8%8D%E5%8F%AF%E4%BF%A1%E8%B5%84%E6%96%99.md)

[2.3 Few-shot、推理与反思：何时有帮助【⭐ 理解】](http://./2.3-Few-shot%E3%80%81%E6%8E%A8%E7%90%86%E4%B8%8E%E5%8F%8D%E6%80%9D%EF%BC%9A%E4%BD%95%E6%97%B6%E6%9C%89%E5%B8%AE%E5%8A%A9.md)

[2.4 结构化输出：JSON、Schema 与业务校验【🔥 必会】](http://./2.4-%E7%BB%93%E6%9E%84%E5%8C%96%E8%BE%93%E5%87%BA%EF%BC%9AJSON%E3%80%81Schema%E4%B8%8E%E4%B8%9A%E5%8A%A1%E6%A0%A1%E9%AA%8C.md)

[2.5 Function Calling：参数、结果与调用 ID【🔥 必会】](./2.5-Function Calling：参数、结果与调用ID.md)

[2.6 同步、异步与流式：完成状态和异常【🔥 必会】](http://./2.6-%E5%90%8C%E6%AD%A5%E3%80%81%E5%BC%82%E6%AD%A5%E4%B8%8E%E6%B5%81%E5%BC%8F%EF%BC%9A%E5%AE%8C%E6%88%90%E7%8A%B6%E6%80%81%E5%92%8C%E5%BC%82%E5%B8%B8.md)

[2.7 上下文窗口：预算与完整轮次【🔥 必会】](http://./2.7-%E4%B8%8A%E4%B8%8B%E6%96%87%E7%AA%97%E5%8F%A3%EF%BC%9A%E9%A2%84%E7%AE%97%E4%B8%8E%E5%AE%8C%E6%95%B4%E8%BD%AE%E6%AC%A1.md)

[2.8 Token、成本与延迟：重试也要计入【🔥 必会】](http://./2.8-Token%E3%80%81%E6%88%90%E6%9C%AC%E4%B8%8E%E5%BB%B6%E8%BF%9F%EF%BC%9A%E9%87%8D%E8%AF%95%E4%B9%9F%E8%A6%81%E8%AE%A1%E5%85%A5.md)

[2.9 模型版本与Prompt 迭代：先核对再比较【⭐ 理解】](http://./2.9-%E6%A8%A1%E5%9E%8B%E7%89%88%E6%9C%AC%E4%B8%8EPrompt%E8%BF%AD%E4%BB%A3%EF%BC%9A%E5%85%88%E6%A0%B8%E5%AF%B9%E5%86%8D%E6%AF%94%E8%BE%83.md)

