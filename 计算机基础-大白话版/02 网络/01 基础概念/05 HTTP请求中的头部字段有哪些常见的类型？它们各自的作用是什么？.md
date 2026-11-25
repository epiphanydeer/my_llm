# 05 HTTP请求中的头部字段有哪些常见的类型？它们各自的作用是什么？

## 1. 🎯 一句话听懂 + 生活大白话例子

- **一句话听懂**：HTTP 头部字段（Headers）就是贴在快递包裹外面的“物流面单”，用来写清楚寄件人是谁、收件地址在哪、箱子里装的是什么格式、能不能缓存，而真正的业务数据都在箱子里面（Body）。
- **生活大白话例子**：
    - 你网购了一箱生鲜，快递盒外面的贴纸分四类：
        1. **买家填写的面单（请求头）**：写着“收件人手机型号、我要开发票、这是我的会员卡号”（`User-Agent`、`Accept`、`Authorization`）。
        2. **卖家贴的回执（响应头）**：写着“发货仓库编号、下次直接来这个新地址买、给你一张会员卡留着用”（`Server`、`Location`、`Set-Cookie`）。
        3. **物流通用规则（通用头）**：写着“这箱菜能保鲜 1 小时、快递员送完先别走等我下一单”（`Cache-Control`、`Connection`）。
        4. **箱内物品说明书（实体头）**：写着“里面装的是 JSON 格式水果、总重 5 公斤、用了真空压缩包装”（`Content-Type`、`Content-Length`、`Content-Encoding`）。

## 2. 🧠 底层机制与核心对比

### 核心子概念（大白话版）

- **请求头（Request Headers）**：客户端单方面发给服务器的“自我介绍与要求”，比如 `Host`（我要找这台机器上的哪个域名）、`User-Agent`（我是 Chrome 还是手机 App）、`Authorization`（这是我的通行令牌）、`Cookie`（之前你发给我的小纸条）。
- **响应头（Response Headers）**：服务器回给客户端的“附加指令与说明”，比如 `Set-Cookie`（给你存个小纸条）、`Location`（配合 3xx 状态码让你跳去新网址）、`Server`（服务器软件版本，生产环境常隐藏防黑客探测）。
- **通用头（General Headers）**：请求和响应两边都能用的“传输控制开关”，管的是缓存策略（`Cache-Control`）、TCP 连接要不要留着复用（`Connection: keep-alive`）和发送时间（`Date`）。
- **实体头（Entity Headers）**：专门描述消息体（Body）本身属性的“货物说明书”，比如 `Content-Type`（里面是 HTML、JSON 还是图片）、`Content-Length`（货物一共多少字节）、`Content-Encoding`（是不是用 gzip 压缩过）。

**为什么实体头不合并进通用头？** 因为职责分离：通用头管的是**“快递怎么送”**（连不连、缓不缓存），实体头只管**“箱子里装的是啥”**（格式、大小、压缩方式）。

### 四大头部字段分类对比表

| **字段类型** | **管什么事（大白话）** | **谁发出的** | **高频代表字段与作用** |
| --- | --- | --- | --- |
| **请求头（Request）** | 买家自我介绍 + 提要求 | 仅客户端发 | `Host`（目标域名）、`User-Agent`（设备信息）、`Accept`（想要啥格式）、`Authorization`（Token 凭证）、`Cookie` |
| **响应头（Response）** | 卖家回执 + 下达指令 | 仅服务端回 | `Set-Cookie`（种 Cookie）、`Location`（跳转地址）、`Server`（服务端标识）、`WWW-Authenticate`（401 认证提示） |
| **通用头（General）** | 快递运输与缓存规则 | 双方都能带 | `Cache-Control`（缓存有效期）、`Connection`（长连接复用）、`Date`（报文时间） |
| **实体头（Entity）** | 箱子里的货物说明书 | 有 Body 的一方带 | `Content-Type`（数据类型）、`Content-Length`（字节大小）、`Content-Encoding`（gzip 压缩）、`Last-Modified` |

### 原理解析图解

- **HTTP 常见字段分类总结**：

![HTTP 常见字段分类总结](../../%E5%9B%BE%E7%89%87/642faac70d6d42ef43df.jpg)

## 3. 🗣️ 面试官视角：怎么串成体系回答

回答头部字段千万别零散地报菜名，按 **“四大分类 + 每类举 2~3 个最关键字段及作用”** 来答：

“HTTP 头部字段按职责分为四大类： 第一类是**请求头**，由客户端发出，用来说明目标和身份，比如 HTTP/1.1 必带的 `Host`（解决单 IP 多域名）、`User-Agent`（设备标识）、`Accept`（期望格式）以及携带 Token 的 `Authorization`。 第二类是**响应头**，由服务端返回，比如用来种凭证的 `Set-Cookie`、配合 3xx 跳转的 `Location`。 第三类是**通用头**，请求和响应都能用，负责连接和缓存控制，比如管理长连接的 `Connection: keep-alive` 和控制强缓存的 `Cache-Control`。 第四类是**实体头**，专门描述 Body 货物的属性，比如声明格式的 `Content-Type`、声明大小的 `Content-Length`（动态流式传输时改用 `Transfer-Encoding: chunked`）和声明压缩方式的 `Content-Encoding`。”

## 4. ❓ 高频面试追问与参考答法

- **Q1：为什么 HTTP/1.1 规定请求头里必须有** `Host` **字段？没有会怎样？**
    - **参考答法**：因为云服务器时代，**一个公网 IP 上往往托管了十几个不同的网站域名（虚拟主机）**。数据包通过 IP 找到这台服务器后，如果请求头里不带 `Host: a.com`，服务器就不知道你到底想访问这台机器上的哪一个网站，所以 HTTP/1.1 强制要求必传 `Host`，否则直接返回 400 Bad Request。
- **Q2：如果服务器一边生成数据一边往外吐，还不知道总大小，那** `Content-Length` **怎么填？**
    - **参考答法**：不知道总大小时就**不填** `Content-Length`，而是改用 `Transfer-Encoding: chunked`（分块传输编码），把数据切成一小块一小块往外发，每块前面标上这一小块的大小，最后发一个长度为 0 的块表示“发完了”。

## 5. 🤖 和大模型/AI 开发有什么关系

- **鉴权与格式声明**：调大模型 API 时，最核心的两个请求头就是 `Authorization: Bearer sk-xxx`（带 API Key 身份凭证）和 `Content-Type: application/json`（告诉模型服务我发过去的是 JSON）。
- **流式打字机效果的底层秘密**：大模型吐字（Streaming）时，服务器还不知道整段回答最终有多少个字，所以响应头里不会有 `Content-Length`，而是使用 `Content-Type: text/event-stream` 配合 `Transfer-Encoding: chunked`（或 HTTP/2 帧），让前端收到一个词就立刻往屏幕上渲染一个词。
- **反爬与多模态压缩**：做 RAG 网页抓取时，爬虫经常需要伪装 `User-Agent`；而传输大段文本或 Prompt 时开启 `Accept-Encoding: gzip` 能大幅压缩网络流量、降低延迟。

## 6. 📝 节末费曼自检

1. 实体头（如 `Content-Type`）明明请求和响应里都会出现，为什么不把它归类为“通用头”？
2. 如果你用大模型 API 做流式输出，服务器提前不知道回答有多长，它是怎么告诉浏览器“数据还没发完，别断开”的？
3. 一台服务器只绑定了一个 IP 地址，却同时跑着博客和商城两个网站，它靠哪个请求头区分用户访问的是谁？

[卡码笔记原文](https://notes.kamacoder.com/base/http-headers.html)
