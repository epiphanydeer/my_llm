"""3.6 Step 6: 在 Step 5 基础上新增全链路可观测 Trace 与六层因果穿透排障状态机"""
from dataclasses import dataclass
from enum import Enum


# [必记设计决策-六层漏斗] RAG 故障归因必须按因果单向穿透：源头缺失->解析丢失->初筛漏召->精排挤出->未引->幻觉
class FailureStage(str, Enum):
    SOURCE_DEFICIT = "SOURCE_DEFICIT"
    PARSER_LOSS = "PARSER_LOSS"
    RETRIEVAL_MISS = "RETRIEVAL_MISS"
    RERANK_DROPPED = "RERANK_DROPPED"
    CITATION_UNGROUNDED = "CITATION_UNGROUNDED"
    LLM_HALLUCINATION = "LLM_HALLUCINATION"
    HEALTHY = "HEALTHY"


@dataclass(frozen=True)
# [无需死背-脚手架] 全链路埋点快照结构，重点记住需落盘 retrieved_top_k_ids / final_context_ids / cited_ids
class PipelineTrace:
    """【Step 6 新增】单次请求全链路中间快照（Trace ID 穿透落盘）。"""
    trace_id: str
    query: str
    gold_chunk_id: str
    source_has_fact: bool
    parsed_chunks_has_fact: bool
    retrieved_top_k_ids: tuple[str, ...]
    final_context_ids: tuple[str, ...]
    cited_ids: tuple[str, ...]
    is_factually_faithful: bool


# [必背手撕-因果穿透归因] 严格遵守“上游未过关，绝不归咎下游”的短路判断顺序
def diagnose_rag_pipeline(trace: PipelineTrace) -> tuple[FailureStage, str]:
    """沿因果链单向向下穿透，锁定第一个破坏链路的根因阶段。"""
    if not trace.source_has_fact:
        return FailureStage.SOURCE_DEFICIT, "源头文档缺失该事实，应触发拒答并补充知识库。"
    if not trace.parsed_chunks_has_fact:
        return FailureStage.PARSER_LOSS, "原文存在但解析/切片丢失（检查 PDF 表格或增大 overlap）。"
    if trace.gold_chunk_id not in trace.retrieved_top_k_ids:
        return FailureStage.RETRIEVAL_MISS, "初筛未命中（检查前置权限误杀、切换混合检索或改写 Query）。"
    if trace.gold_chunk_id not in trace.final_context_ids:
        return FailureStage.RERANK_DROPPED, "初筛命中但被精排或窗口预算截断（调优 Reranker / 增大 Top-N）。"
    if trace.gold_chunk_id not in trace.cited_ids:
        return FailureStage.CITATION_UNGROUNDED, "上下文正确但模型未标注引用（强化 Prompt 锚点与后置校验）。"
    if not trace.is_factually_faithful:
        return FailureStage.LLM_HALLUCINATION, "证据已引用但模型发生内在幻觉（调低 temperature=0 并精简噪声）。"
    return FailureStage.HEALTHY, "全链路证据流转正常。"


if __name__ == "__main__":
    t1 = PipelineTrace("T1", "三级动火谁审批", "C_FIRE_03", True, False, (), (), (), False)
    assert diagnose_rag_pipeline(t1)[0] == FailureStage.PARSER_LOSS

    t2 = PipelineTrace("T2", "三级动火谁审批", "C_FIRE_03", True, True, ("C_FIRE_03", "C_02"), ("C_02",), (), False)
    assert diagnose_rag_pipeline(t2)[0] == FailureStage.RERANK_DROPPED

    t3 = PipelineTrace("T3", "三级动火谁审批", "C_FIRE_03", True, True, ("C_FIRE_03",), ("C_FIRE_03",), ("C_FIRE_03",), False)
    assert diagnose_rag_pipeline(t3)[0] == FailureStage.LLM_HALLUCINATION
    print("Step 6 self-test passed!")
