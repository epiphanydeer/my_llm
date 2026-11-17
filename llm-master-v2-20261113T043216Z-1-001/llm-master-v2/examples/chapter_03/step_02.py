"""3.2 Step 2: 在 Step 1 基础上封装 MiniRAGPipeline，打通离线切片、Prompt 组装与后置引用核验"""
from dataclasses import dataclass
from datetime import date
import re


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    content: str
    version: str
    effective_date: date
    expire_date: date
    roles: tuple[str, ...] = ("employee",)


@dataclass(frozen=True)
class Chunk:
    """【Step 2 新增】切片数据结构：携带原文归属与权限元数据。"""
    chunk_id: str
    doc_id: str
    version: str
    content: str
    effective_date: date
    expire_date: date
    roles: tuple[str, ...]


class MiniRAGPipeline:
    """【Step 2 新增】离线建库 + 在线检索 + Prompt 装配 + 引用守卫闭环。"""
    def __init__(self, chunk_size: int = 40) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须为正整数")
        self.chunk_size = chunk_size
        self.chunks: list[Chunk] = []

    def ingest(self, docs: list[Document]) -> None:
        """离线阶段：清洗并切分为 Chunk（继承文档权限与时效）。"""
        if not docs:
            raise ValueError("文档列表不能为空")
        self.chunks.clear()
        for doc in docs:
            text = re.sub(r"\s+", " ", doc.content).strip()
            for idx, start in enumerate(range(0, len(text), self.chunk_size)):
                piece = text[start : start + self.chunk_size]
                self.chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}_c{idx:02d}",
                    doc_id=doc.doc_id,
                    version=doc.version,
                    content=piece,
                    effective_date=doc.effective_date,
                    expire_date=doc.expire_date,
                    roles=doc.roles,
                ))
        assert len(self.chunks) > 0, "入库切片总数必须大于 0"

    def retrieve(self, query: str, user_role: str, today: date, top_k: int = 2) -> list[Chunk]:
        """在线检索：沿用 Step 1 前置过滤 + 双字滑窗打分。"""
        if not query.strip():
            raise ValueError("查询不能为空")
        q = query.strip().lower()
        q_bigrams = {q[i : i + 2] for i in range(len(q) - 1) if q[i : i + 2].strip()} or {q}
        scored: list[tuple[int, Chunk]] = []
        for ch in self.chunks:
            if user_role not in ch.roles or not (ch.effective_date <= today <= ch.expire_date):
                continue
            hits = sum(1 for bg in q_bigrams if bg in ch.content.lower())
            if hits > 0:
                scored.append((hits, ch))
        scored.sort(key=lambda x: (-x[0], x[1].chunk_id))
        return [c for _, c in scored[:top_k]]

    @staticmethod
    def build_prompt(query: str, contexts: list[Chunk]) -> str:
        """组装防先验 Prompt，显式注入来源锚点标签。"""
        evidence_block = "\n".join(f"[来源: {c.chunk_id}] {c.content}" for c in contexts)
        return (
            "系统指令：仅依据以下参考资料作答，严禁使用先验记忆推测，并在句末标注 [来源: chunk_id]。\n"
            f"参考资料：\n{evidence_block}\n用户问题：{query}"
        )

    @staticmethod
    def verify_citations(answer: str, allowed_chunks: list[Chunk]) -> bool:
        """后置引用核验：拦截模型捏造未注入上下文的虚假引用 ID。"""
        cited = set(re.findall(r"\[来源:\s*([^\]]+)\]", answer))
        allowed = {c.chunk_id for c in allowed_chunks}
        return bool(cited) and cited.issubset(allowed)

    def answer(self, query: str, user_role: str = "employee", today: date = date(2026, 11, 13)) -> dict[str, object]:
        hits = self.retrieve(query, user_role, today, top_k=2)
        if not hits:
            return {"status": "REFUSED", "answer": "依据不足，无法回答。", "prompt": "", "citation_valid": False}
        prompt = self.build_prompt(query, hits)
        mock_llm_output = f"根据规章：{hits[0].content} [来源: {hits[0].chunk_id}]"
        is_valid = self.verify_citations(mock_llm_output, hits)
        assert is_valid, "受控生成必须通过引用真实性校验"
        return {"status": "SUCCESS", "answer": mock_llm_output, "prompt": prompt, "citation_valid": is_valid}


if __name__ == "__main__":
    docs = [
        Document("DOC_SAFE", "安全规范", "高处作业人员必须佩戴全身式五点安全带，并高挂低用。", "v2", date(2026, 1, 1), date(2027, 12, 31)),
    ]
    pipe = MiniRAGPipeline(chunk_size=30)
    pipe.ingest(docs)
    out = pipe.answer("高处作业怎么戴安全带")
    assert out["status"] == "SUCCESS" and out["citation_valid"] is True
    assert pipe.verify_citations("罚款一万 [来源: FAKE_999]", pipe.chunks) is False, "必须拦截虚假引用"
    print("Step 2 self-test passed!")
