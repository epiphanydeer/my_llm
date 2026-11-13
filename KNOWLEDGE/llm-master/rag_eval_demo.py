"""最小 RAG 检索与重排评测；--self-test 无需安装模型依赖。"""
import math
import os
import statistics
import sys
import time

# 演示文本和标注是虚构的，不代表真实法规或实际业务测试集。
DOCS = [
    ("A", "高处作业应佩戴安全带并检查挂点。"),
    ("B", "动火作业前必须审批并配备灭火器。"),
    ("C", "电气设备检修前需要断电并验电。"),
    ("D", "高处作业前应检查脚手架的稳定性。"),
    ("E", "进入施工现场需要正确佩戴安全帽。"),
]
CASES = [
    ("高空作业需要什么防护和检查？", {"A", "D"}),
    ("焊接动火需要什么手续？", {"B"}),
    ("检修电气设备之前怎么做？", {"C"}),
]
PREFIX = "为这个句子生成表示以用于检索相关文章："


def metrics(gold, ranking, k):
    if not gold or k <= 0:
        raise ValueError("金标准不能为空且 k 必须大于 0")
    selected = ranking[:k]
    found = gold.intersection(selected)
    first = next((i for i, doc_id in enumerate(selected, 1) if doc_id in gold), None)
    return {"hit": float(bool(found)), "recall": len(found) / len(gold),
            "mrr": 1 / first if first else 0.0}


def percentile(values, p):
    ordered = sorted(values)
    return ordered[math.ceil(p * len(ordered)) - 1]


def self_test():
    a = metrics({"A", "B"}, ["B", "C"], 2)
    assert a == {"hit": 1.0, "recall": 0.5, "mrr": 1.0}
    assert metrics({"A"}, ["B", "A"], 2)["mrr"] == 0.5
    assert metrics({"A"}, ["B"], 1)["hit"] == 0.0
    assert percentile([1, 2, 10, 20], 0.95) == 20
    print("指标自检通过")


def main():
    import numpy as np
    import torch
    import hnswlib
    from sentence_transformers import CrossEncoder, SentenceTransformer

    # 模型加载、首次下载及离线建库不计入线上请求延迟。
    embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    reranker = CrossEncoder("BAAI/bge-reranker-base")

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def timed(fn):
        sync()
        start = time.perf_counter()
        value = fn()
        sync()
        return value, (time.perf_counter() - start) * 1000

    vectors, build_embedding_ms = timed(lambda: embedder.encode(
        [text for _, text in DOCS], normalize_embeddings=True).astype(np.float32))
    index = hnswlib.Index(space="cosine", dim=vectors.shape[1])
    index.init_index(max_elements=len(DOCS), M=16, ef_construction=200)
    _, build_index_ms = timed(lambda: index.add_items(vectors, np.arange(len(DOCS))))
    index.set_ef(50)
    candidate_k, final_k = min(3, len(DOCS)), 2

    def query(question):
        vector, embed_ms = timed(lambda: embedder.encode(
            [PREFIX + question], normalize_embeddings=True).astype(np.float32))
        result, search_ms = timed(lambda: index.knn_query(vector, k=candidate_k))
        ids = [int(i) for i in result[0][0]]
        scores, rerank_ms = timed(lambda: reranker.predict(
            [(question, DOCS[i][1]) for i in ids]))
        reordered = sorted(range(len(ids)), key=lambda j: float(scores[j]), reverse=True)
        baseline = [DOCS[i][0] for i in ids]
        reranked = [DOCS[ids[j]][0] for j in reordered]
        return baseline, reranked, (embed_ms, search_ms, rerank_ms)

    query(CASES[0][0])  # 预热不计入统计
    rows = []
    for question, gold in CASES:
        baseline, reranked, times = query(question)
        rows.append((metrics(gold, baseline, candidate_k),
                     metrics(gold, baseline, final_k),
                     metrics(gold, reranked, final_k), times))
        print(f"问题: {question}\n  召回: {baseline}; 重排: {reranked}; 标准: {sorted(gold)}")

    print(f"离线建库：编码 {build_embedding_ms:.1f} ms，建索引 {build_index_ms:.1f} ms")
    for label, column in ((f"候选 Recall@{candidate_k}", 0),
                          (f"重排前 Top-{final_k}", 1),
                          (f"重排后 Top-{final_k}", 2)):
        print(label, {key: round(statistics.mean(row[column][key] for row in rows), 3)
                      for key in ("hit", "recall", "mrr")})
    times = [row[3] for row in rows]
    for label, values in (("查询向量化", [t[0] for t in times]),
                          ("HNSW", [t[1] for t in times]),
                          ("重排", [t[2] for t in times]),
                          ("检索总计", [sum(t) for t in times])):
        print(f"{label}: P50={percentile(values, .50):.1f} ms, P95={percentile(values, .95):.1f} ms")
    # 假设资源完全按请求独占计费：仅为估算，不含空闲、建库、生成、存储等。
    price = os.getenv("RAG_HOURLY_PRICE")
    if price is not None:
        hourly = float(price)
        if hourly < 0:
            raise ValueError("每小时价格不能为负")
        mean_ms = statistics.mean(sum(t) for t in times)
        print(f"估算每千次检索算力费用: {mean_ms * 1000 / 3600000 * hourly:.4f}（价格货币单位）")
    else:
        print("未设置 RAG_HOURLY_PRICE，不估算算力费用")
    print("样本仅供演示；未测生成模型、网络、并发、完整账单或真实业务效果。")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    else:
        main()
