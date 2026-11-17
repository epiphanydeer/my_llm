"""10.4 生成模型、Embedding 与 Reranker：两阶段召回+精排流水线"""
import numpy as np
from typing import List, Tuple


def bi_encoder_recall(query_vec: np.ndarray, doc_vecs: np.ndarray, top_k: int) -> List[int]:
    """向量预计算后仅做余弦相似度点积，复杂度 O(1) 实时编码。"""
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    d_norm = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-12)
    scores = d_norm @ q_norm
    return np.argsort(scores)[::-1][:top_k].tolist()


def cross_encoder_rerank(query: str, candidate_indices: List[int], docs: List[str]) -> List[Tuple[int, float]]:
    """模拟 Cross-Encoder 拼接 [query, doc] 做细粒度词面+语义交互精排。"""
    q_terms = set(query.split())
    scored = []
    for idx in candidate_indices:
        overlap = len(q_terms & set(docs[idx].split()))
        scored.append((idx, round(0.5 + 0.25 * overlap, 3)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


if __name__ == "__main__":
    docs = ["破碎机 轴承 温度 报警 处理", "皮带机 跑偏 停机 规程", "破碎机 衬板 磨损 更换"]
    q_vec = np.array([0.9, 0.2, 0.1])
    d_vecs = np.array([[0.85, 0.25, 0.1], [0.1, 0.9, 0.2], [0.88, 0.15, 0.15]])
    recalled = bi_encoder_recall(q_vec, d_vecs, top_k=2)
    reranked = cross_encoder_rerank("破碎机 温度 报警", recalled, docs)
    assert reranked[0][0] == 0
    print("[10.4 PASS] recalled:", recalled, "reranked:", reranked)
