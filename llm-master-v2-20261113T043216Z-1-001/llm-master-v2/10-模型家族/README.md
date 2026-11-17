# 10-模型家族（重构实战版 v3）

本章系统梳理大模型选型与架构演进：从业务硬约束选型、Encoder/Decoder 注意力可见性、Dense vs MoE 路由与显存瓶颈，到 Embedding/Reranker 分工、VLM 视觉 Token 预算、部署 TCO 核算与面试级失败切片归因。

代码说明：本章采用工业级工程校验写法（ch10_code/step_01..07.py），全部通过 python3 断言自测。

## 目录导航

[10.1 模型选型：先确认任务与能力边界【🔥 必会】](http://./10.1-%E6%A8%A1%E5%9E%8B%E9%80%89%E5%9E%8B%EF%BC%9A%E5%85%88%E7%A1%AE%E8%AE%A4%E4%BB%BB%E5%8A%A1%E4%B8%8E%E8%83%BD%E5%8A%9B%E8%BE%B9%E7%95%8C.md)

[10.2 Encoder、Decoder 与 Encoder-Decoder 的可见性【🔥 必会】](http://./10.2-Encoder%E3%80%81Decoder%E4%B8%8EEncoder-Decoder%E7%9A%84%E5%8F%AF%E8%A7%81%E6%80%A7.md)

[10.3 Dense 与 MoE：路由、负载和部署【🔥 必会】](http://./10.3-Dense%E4%B8%8EMoE%EF%BC%9A%E8%B7%AF%E7%94%B1%E3%80%81%E8%B4%9F%E8%BD%BD%E5%92%8C%E9%83%A8%E7%BD%B2.md)

[10.4 生成模型、Embedding 与 Reranker【🔥 必会】](http://./10.4-%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B%E3%80%81Embedding%E4%B8%8EReranker.md)

[10.5 VLM 选型：视觉编码、对齐与生成【🔥 必会】](http://./10.5-VLM%E9%80%89%E5%9E%8B%EF%BC%9A%E8%A7%86%E8%A7%89%E7%BC%96%E7%A0%81%E3%80%81%E5%AF%B9%E9%BD%90%E4%B8%8E%E7%94%9F%E6%88%90.md)

[10.6 API、托管与自部署：核对约束【⭐ 理解】](http://./10.6-API%E3%80%81%E6%89%98%E7%AE%A1%E4%B8%8E%E8%87%AA%E9%83%A8%E7%BD%B2%EF%BC%9A%E6%A0%B8%E5%AF%B9%E7%BA%A6%E6%9D%9F.md)

[10.7 模型选型面试：同任务与失败切片【🔥 必会】](http://./10.7-%E6%A8%A1%E5%9E%8B%E9%80%89%E5%9E%8B%E9%9D%A2%E8%AF%95%EF%BC%9A%E5%90%8C%E4%BB%BB%E5%8A%A1%E4%B8%8E%E5%A4%B1%E8%B4%A5%E5%88%87%E7%89%87.md)

