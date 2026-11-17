"""3.1 Step 1: 权限前置过滤与无证据确定性拒答的最小 RAG 原型（纯标准库）"""
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Document:
    """受控企业知识文档（含版本、时效与权限元数据）。"""
    doc_id: str
    title: str
    content: str
    version: str
    effective_date: date
    expire_date: date
    roles: tuple[str, ...] = ("employee",)


def safety_guard_rag(
    docs: list[Document],
    query: str,
    user_role: str = "employee",
    today: date = date(2026, 11, 13),
    top_k: int = 2,
    min_overlap: int = 2,
) -> dict[str, object]:
    """先执行权限与时效前置过滤（Pre-filtering），再按双字滑窗匹配，无证据时确定性拒答。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("查询问题不能为空")
    if top_k <= 0 or min_overlap <= 0:
        raise ValueError("top_k 与 min_overlap 必须为正整数")

    q = query.strip().lower()
    q_bigrams = {q[i : i + 2] for i in range(len(q) - 1) if q[i : i + 2].strip()} or {q}

    candidates: list[tuple[int, Document]] = []
    for doc in docs:
        # 1. 前置过滤（Pre-filtering）：无权限或过期文档绝不进入检索候选池
        if user_role not in doc.roles:
            continue
        if not (doc.effective_date <= today <= doc.expire_date):
            continue
        c_lower = doc.content.lower()
        hits = sum(1 for bg in q_bigrams if bg in c_lower)
        if hits >= min_overlap:
            candidates.append((hits, doc))

    candidates.sort(key=lambda x: (-x[0], x[1].doc_id))
    selected = [doc for _, doc in candidates[:top_k]]
    assert len(selected) <= top_k, "返回结果数不能超过 top_k"

    if not selected:
        return {"status": "REFUSED", "answer": "未找到您权限范围内且当前有效的相关依据，无法回答。", "citations": []}

    citations = [f"{d.doc_id}:{d.version}" for d in selected]
    context_summary = " | ".join(d.content for d in selected)
    return {"status": "SUCCESS", "answer": f"依据有效制度：{context_summary}", "citations": citations}


if __name__ == "__main__":
    kb = [
        Document("DOC_TRAVEL_V1", "差旅制度V1", "出差乘坐飞机不可报销。", "v1", date(2024, 1, 1), date(2025, 12, 31), ("employee",)),
        Document("DOC_TRAVEL_V2", "差旅制度V2", "出差乘坐飞机经济舱须提前审批后方可报销。", "v2", date(2026, 1, 1), date(2027, 12, 31), ("employee",)),
        Document("DOC_EXEC_PAY", "高管薪酬", "高管年度奖金与股票期权方案。", "v1", date(2026, 1, 1), date(2027, 12, 31), ("hr_director",)),
    ]
    res_ok = safety_guard_rag(kb, "出差坐飞机能报销吗", user_role="employee")
    assert res_ok["status"] == "SUCCESS" and res_ok["citations"] == ["DOC_TRAVEL_V2:v2"]

    res_deny = safety_guard_rag(kb, "高管年度奖金怎么发", user_role="employee")
    assert res_deny["status"] == "REFUSED", "普通员工绝不能越权检索高管薪酬"
    print("Step 1 self-test passed!")
