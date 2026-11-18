"""3.7 Step 7: 在 Step 4~6 基础上新增 BM25 稀疏检索 + 加权 RRF 融合 + 同文档多样性去重"""
from collections import Counter
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RankedItem:
    chunk_id: str
    doc_id: str
    rank: int  # 从 1 开始


# [无需死背-脚手架] MiniBM25Retriever 整类初始化与 Counter 统计不用死背，只需记住 BM25 的两个核心调节参数：
# k1（词频饱和度，防高频词刷分）和 b（文档长度归一化惩罚）
class MiniBM25Retriever:
    """【Step 7 新增】纯标准库 BM25 稀疏检索器，弥补向量对标准号/型号的漏召回。"""
    def __init__(self, chunks: dict[str, str], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tokenized = {cid: list(text.lower()) for cid, text in chunks.items()}
        self.avg_dl = sum(len(t) for t in self.tokenized.values()) / max(1, len(self.tokenized))
        self.df: Counter[str] = Counter()
        for tokens in self.tokenized.values():
            self.df.update(set(tokens))
        self.n_docs = len(self.tokenized)

    def search(self, query: str, top_k: int = 10) -> list[str]:
        q_tokens = list(query.strip().lower())
        scores: list[tuple[float, str]] = []
        for cid, doc_tokens in self.tokenized.items():
            tf_map = Counter(doc_tokens)
            dl = len(doc_tokens)
            score = 0.0
            for qt in q_tokens:
                if qt not in tf_map:
                    continue
                df_t = self.df[qt]
                idf = math.log((self.n_docs - df_t + 0.5) / (df_t + 0.5) + 1.0)
                tf = tf_map[qt]
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl))
            if score > 0:
                scores.append((score, cid))
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [cid for _, cid in scores[:top_k]]


# [必记设计决策-参数与边界] 1) k=60 平滑名次差异，防单路第 1 名噪声霸榜；2) max_per_doc 限制同一文档垄断 Top-N
def weighted_rrf(
    routes: list[tuple[str, list[str], float]],
    chunk_to_doc: dict[str, str],
    k: int = 60,
    top_n: int = 5,
    max_per_doc: int = 2,
) -> list[tuple[str, float]]:
    """【Step 7 新增】加权倒数排名融合 + 同文档多样性去重（防止长文档霸榜）。"""
    if k <= 0 or top_n <= 0:
        raise ValueError("k 与 top_n 必须为正整数")

    # [必背手撕-核心 6 行] 面试高频手撕：多路检索结果去重校验 + RRF 倒数排名无量纲累加
    scores: dict[str, float] = {}
    for route_name, ranked_ids, weight in routes:
        if len(ranked_ids) != len(set(ranked_ids)):
            raise ValueError(f"检索路 {route_name} 存在重复 ID")
        for rank, cid in enumerate(ranked_ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank)

    ordered = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

    # [必背手撕-核心 7 行] 面试加分项：按 doc_id 计数执行同文档多样性截断（防长文档切片霸榜）
    final: list[tuple[str, float]] = []
    doc_counts: Counter[str] = Counter()
    for cid, sc in ordered:
        did = chunk_to_doc.get(cid, cid)
        if doc_counts[did] >= max_per_doc:
            continue
        doc_counts[did] += 1
        final.append((cid, sc))
        if len(final) >= top_n:
            break
    return final


if __name__ == "__main__":
    # [必记设计决策-测试用例] 构造“精确标准号(GB/T 3608) + 同文档多切片(DOCA_c1/c2/c3)”验证混合召回与多样性截断
    corpus = {
        "DOCA_c1": "根据GB/T 3608规定，三级高处作业基准面高度为15至30米。",
        "DOCA_c2": "根据GB/T 3608规定，一级高处作业基准面高度为2至5米。",
        "DOCA_c3": "根据GB/T 3608规定，二级高处作业基准面高度为5至15米。",
        "DOCB_c1": "高空作业人员必须佩戴安全带。",
    }
    doc_map = {cid: cid.split("_")[0] for cid in corpus}
    bm25 = MiniBM25Retriever(corpus)
    bm25_res = bm25.search("GB/T 3608 三级高处作业", top_k=4)
    assert bm25_res[0] == "DOCA_c1", f"BM25 应精准命中标准号，实际: {bm25_res}"

    vec_res = ["DOCB_c1", "DOCA_c1", "DOCA_c2", "DOCA_c3"]
    fused = weighted_rrf([("bm25", bm25_res, 0.5), ("dense", vec_res, 0.5)], doc_map, k=60, top_n=3, max_per_doc=2)
    assert fused[0][0] == "DOCA_c1", "两路均排前列的 DOCA_c1 应在 RRF 融合后夺冠"
    assert sum(1 for cid, _ in fused if doc_map[cid] == "DOCA") <= 2, "同文档切片不得超过 max_per_doc=2"
    print("Step 7 self-test passed!")
