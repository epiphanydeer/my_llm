"""3.4 Step 4: 在 Step 3 基础上新增 DenseVectorRetriever（L2 归一化 + 矩阵余弦 + 非对称前缀）"""
import numpy as np


# [必记设计决策-非对称前缀] BGE 等非对称模型仅 Query 加检索前缀，Document 入库绝不加前缀
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


# [必背手撕-数值稳定L2] 面试高频手撕：先查 NaN/Inf，再除以 norm + eps 防止全零向量除零崩溃
def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """对二维向量矩阵 (N, D) 执行数值稳定的 L2 行归一化，防零向量除零崩溃。"""
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("输入必须是形状为 (N, D) 的非空二维矩阵")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("向量矩阵包含 NaN 或 Inf")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + eps
    normalized = vectors / norms
    assert normalized.shape == vectors.shape
    return normalized


class DenseVectorRetriever:
    """【Step 4 新增】支持非对称前缀与几何等价校验的稠密向量检索器。"""
    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.chunk_ids: list[str] = []
        self.doc_matrix: np.ndarray | None = None

    # [无需死背-脚手架] 教学用确定性字符哈希编码器，生产中直接调用 BGE / OpenAI Embedding
    def _mock_encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """确定性教学编码器：模拟语义映射（生产中替换为 BGE/OpenAI Embedding）。"""
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            # Query 必须加非对称前缀，Document 绝不加前缀
            proc = f"{BGE_QUERY_PREFIX}{text}" if is_query else text
            for ch in proc:
                if ch in BGE_QUERY_PREFIX and is_query:
                    continue
                vecs[i, ord(ch) % self.dim] += 1.0
        return l2_normalize(vecs)

    def index_chunks(self, chunk_ids: list[str], texts: list[str]) -> None:
        if len(chunk_ids) != len(texts) or not chunk_ids:
            raise ValueError("chunk_ids 与 texts 必须非空且等长")
        self.chunk_ids = list(chunk_ids)
        self.doc_matrix = self._mock_encode(texts, is_query=False)
        assert self.doc_matrix.shape == (len(chunk_ids), self.dim)

    def search(self, query: str, top_k: int = 2) -> list[tuple[str, float]]:
        if self.doc_matrix is None:
            raise RuntimeError("请先调用 index_chunks 构建索引")
        q_vec = self._mock_encode([query], is_query=True)  # (1, D)
        # 归一化后，点积严格等价于余弦相似度，且与 L2 平方距离单调负相关
        # [必背手撕-矩阵余弦] 单位向量归一化后，点积等价于余弦相似度，且 L2^2 = 2 - 2*cos
        dot_scores = (self.doc_matrix @ q_vec.T).squeeze(axis=1)  # (N,)
        l2_sq_dists = np.sum((self.doc_matrix - q_vec) ** 2, axis=1)
        assert np.allclose(l2_sq_dists, 2.0 - 2.0 * dot_scores, atol=1e-5), "L2平方距离必须严格等于 2 - 2*cos"

        top_indices = np.argsort(-dot_scores)[:top_k]
        return [(self.chunk_ids[int(idx)], float(dot_scores[int(idx)])) for idx in top_indices]


if __name__ == "__main__":
    retriever = DenseVectorRetriever(dim=32)
    retriever.index_chunks(
        ["C1", "C2", "C3"],
        ["高处作业必须佩戴全身式五点安全带", "动火作业须配备灭火器并设专人监护", "食堂午餐十二点开始供应"],
    )
    hits = retriever.search("高处作业安全带佩戴要求", top_k=2)
    assert hits[0][0] == "C1", f"应优先召回 C1，实际为 {hits}"
    zero_safe = l2_normalize(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
    assert np.all(np.isfinite(zero_safe)) and np.allclose(zero_safe, 0.0)
    print("Step 4 self-test passed!")
