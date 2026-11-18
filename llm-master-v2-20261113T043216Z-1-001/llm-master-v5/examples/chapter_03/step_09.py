"""3.9 Step 9: 在 Step 7/8 基础上挂载离线评测套件（Hit/Recall/MRR/nDCG + 防空拒答漏洞的忠实度）"""
import math
from statistics import mean


# [必背手撕-检索五大指标] Hit@K / Recall@K / Precision@K / MRR@K / nDCG@K 是 RAG 评测必背手撕函数
def evaluate_retrieval(ranked_ids: list[str], gold_rels: dict[str, int], k: int = 3) -> dict[str, float]:
    """计算单条查询的 Hit@K, Recall@K, Precision@K, MRR@K, nDCG@K。"""
    if k <= 0 or not gold_rels:
        raise ValueError("k 必须为正整数且 gold_rels 不能为空")
    top_k = ranked_ids[:k]
    hits = [cid for cid in top_k if gold_rels.get(cid, 0) > 0]
    total_pos = sum(1 for v in gold_rels.values() if v > 0)

    hit_k = 1.0 if hits else 0.0
    recall_k = len(hits) / total_pos
    precision_k = len(hits) / k
    mrr_k = next((1.0 / idx for idx, cid in enumerate(top_k, 1) if gold_rels.get(cid, 0) > 0), 0.0)

    # [必背手撕-nDCG折损] 分母为 log2(rank + 1)（1-indexed），再用排序后的理想前 k 个算 IDCG 归一化
    dcg = sum(gold_rels.get(cid, 0) / math.log2(idx + 1) for idx, cid in enumerate(top_k, 1))
    ideal_rels = sorted([v for v in gold_rels.values() if v > 0], reverse=True)[:k]
    idcg = sum(rel / math.log2(idx + 1) for idx, rel in enumerate(ideal_rels, 1))
    ndcg_k = dcg / idcg if idcg > 0 else 0.0

    assert 0.0 <= recall_k <= 1.0 and 0.0 <= ndcg_k <= 1.0
    return {"hit": hit_k, "recall": recall_k, "precision": precision_k, "mrr": mrr_k, "ndcg": ndcg_k}


# [必记设计决策-防刷分漏洞] 必须校验 has_gold_in_context：上下文明明有答案却过度拒答，忠实度直接判 0 分
def evaluate_faithfulness(claims_supported: list[bool], is_refused: bool, has_gold_in_context: bool) -> float:
    """声明级忠实度核算：防备“上下文明明有答案却过度拒答刷 100% 忠实度”的评测漏洞。"""
    if is_refused:
        return 0.0 if has_gold_in_context else 1.0
    if not claims_supported:
        return 0.0
    return mean(1.0 if ok else 0.0 for ok in claims_supported)


if __name__ == "__main__":
    m = evaluate_retrieval(["C", "A", "D"], {"A": 3, "B": 2}, k=3)
    assert m["hit"] == 1.0 and m["recall"] == 0.5 and m["mrr"] == 0.5
    assert math.isclose(m["ndcg"], (3 / math.log2(3)) / (3 / math.log2(2) + 2 / math.log2(3)), abs_tol=1e-6)
    assert evaluate_faithfulness([], is_refused=True, has_gold_in_context=True) == 0.0, "有证据却拒答不能给满分"
    assert evaluate_faithfulness([True, True, False], is_refused=False, has_gold_in_context=True) == 2 / 3
    print("Step 9 self-test passed!")
