"""3.10 Step 10: 在 Step 7/9 基础上新增 Agentic RAG 动态检索循环（含 CRAG 评估与三层死循环熔断）"""
import hashlib


def agentic_rag_loop(
    query: str,
    search_fn,
    max_steps: int = 3,
) -> dict[str, object]:
    """
    【Step 10 新增】动态检索状态机：
    1. 意图路由（闲聊/改写类免检索）；
    2. CRAG 证据评估（Insufficient 时改写 Query 重试）；
    3. 三层熔断（max_steps 硬上限 + Query SHA-256 去重防震荡 + 降级兜底）。
    """
    # [必记设计决策-意图路由] 非知识类改写/闲聊请求直接走 DIRECT_LLM，免去无意义的检索延迟与噪声
    if "改得更口语化" in query:
        return {"status": "DIRECT_LLM", "steps": 0, "evidences": [], "trace": ["route:skip_rag"]}

    seen_hashes: set[str] = set()
    curr_query = query
    collected: list[str] = []
    trace: list[str] = []

    # [必记设计决策-三层熔断] 1) max_steps 硬上限；2) Query SHA-256 去重防改写震荡；3) 耗尽降级 DEGRADED_FALLBACK
    for step in range(1, max_steps + 1):
        q_hash = hashlib.sha256(curr_query.strip().encode("utf-8")).hexdigest()
        # [必背手撕-循环去重] 改写后的 Query 若与历史哈希重复，立即 break 熔断，防止死循环烧钱
        if q_hash in seen_hashes:
            trace.append(f"step{step}:cycle_broken")
            break
        seen_hashes.add(q_hash)

        hits = search_fn(curr_query)
        collected.extend(hits)
        trace.append(f"step{step}:query='{curr_query}'->hits={len(hits)}")

        # 模拟 CRAG 证据充分性评估：需同时覆盖两个对比维度才算充分
        if any("登高" in h for h in collected) and any("密闭" in h for h in collected):
            return {"status": "SUFFICIENT", "steps": step, "evidences": list(dict.fromkeys(collected)), "trace": trace}

        # 证据不足时触发 Query 改写（严格绑定原始意图，防 Context Drift）
        # [必记设计决策-防意图漂移] Query 改写必须锚定原始 query，防止多轮改写偏离用户初始问题 (Context Drift)
        curr_query = f"{query} 补充检索:密闭空间" if step == 1 else curr_query

    return {"status": "DEGRADED_FALLBACK", "steps": len(trace), "evidences": list(dict.fromkeys(collected)), "trace": trace}


if __name__ == "__main__":
    def mock_search(q: str) -> list[str]:
        if "补充检索:密闭空间" in q:
            return ["密闭空间作业监护人须持有限空间证。"]
        return ["登高作业监护人须持高处作业证。"]

    r1 = agentic_rag_loop("对比登高与密闭空间监护人资质", mock_search, max_steps=3)
    assert r1["status"] == "SUFFICIENT" and r1["steps"] == 2

    r2 = agentic_rag_loop("对比登高与密闭空间监护人资质", lambda q: ["无关片段"], max_steps=3)
    assert r2["status"] == "DEGRADED_FALLBACK" and "cycle_broken" in r2["trace"][-1]
    assert agentic_rag_loop("把这段话改得更口语化", mock_search)["status"] == "DIRECT_LLM"
    print("Step 10 self-test passed!")
