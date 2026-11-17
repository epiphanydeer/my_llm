"""1.2 应用落地链路：输入、模型、证据与交付（工业流水线校验写法）"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass
class DeliveryRecord:
    request_id: str
    sanitized_input: str
    model_output: Dict[str, Any]
    evidence_refs: List[str]
    status: str  # "delivered" | "fallback_human_review"


class IndustrialAppPipeline:
    """带输入清洗、证据绑定与低置信度兜底的应用交付流水线。"""

    def __init__(self, llm_fn: Callable[[str], Dict[str, Any]], min_confidence: float = 0.75) -> None:
        self.llm_fn = llm_fn
        self.min_confidence = min_confidence

    @staticmethod
    def sanitize_input(raw_text: str) -> str:
        cleaned = raw_text.strip()
        if not cleaned:
            raise ValueError("输入不能为空")
        return cleaned[:4000]

    def run(self, request_id: str, raw_text: str, retrieved_docs: List[str]) -> DeliveryRecord:
        query = self.sanitize_input(raw_text)
        if not retrieved_docs:
            return DeliveryRecord(request_id, query, {"answer": "缺少可验证依据，转人工复核"}, [], "fallback_human_review")
        resp = self.llm_fn(query)
        conf = float(resp.get("confidence", 0.0))
        status = "delivered" if conf >= self.min_confidence else "fallback_human_review"
        return DeliveryRecord(request_id, query, resp, retrieved_docs, status)


if __name__ == "__main__":
    mock_llm = lambda q: {"answer": f"已基于规程核查：{q[:12]}", "confidence": 0.91}
    pipe = IndustrialAppPipeline(mock_llm, min_confidence=0.8)
    ok_rec = pipe.run("req-001", "  皮带机检修前必须断电挂牌吗？ ", ["《安全规程》第4.2条"])
    assert ok_rec.status == "delivered" and len(ok_rec.evidence_refs) == 1
    no_ev = pipe.run("req-002", "未检索到规程的问题", [])
    assert no_ev.status == "fallback_human_review"
    print("[1.2 PASS]", ok_rec.status, no_ev.status)
