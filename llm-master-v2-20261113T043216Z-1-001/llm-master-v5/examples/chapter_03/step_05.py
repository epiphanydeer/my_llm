"""3.5 Step 5: 在 Step 4 基础上新增 Flat 精确搜索 vs 近似图搜索（HNSW 原理模拟与对齐评测）"""
import time
import numpy as np
from step_04 import l2_normalize


# [必记设计决策-HNSW权衡] 用 M 控制建图邻接数(内存/建图耗时)，用 ef_search 控制在线波束宽度(召回/延迟)
class SimpleNavigableGraphIndex:
    """【Step 5 新增】轻量教学版可导航小世界图索引（演示 M 与 efSearch 原理）。"""
    def __init__(self, vectors: np.ndarray, m_neighbors: int = 16, seed: int = 42) -> None:
        self.vectors = l2_normalize(vectors)
        self.n = self.vectors.shape[0]
        self.m = min(m_neighbors, max(1, self.n - 1))
        # 离线建图：为每个节点连接最相似的 M 个近邻（简化单层 NSW 图）
        sim_matrix = self.vectors @ self.vectors.T
        np.fill_diagonal(sim_matrix, -np.inf)
        self.adj: list[list[int]] = [
            np.argsort(-sim_matrix[i])[: self.m].tolist() for i in range(self.n)
        ]
        self.entry_point = int(np.random.default_rng(seed).integers(0, self.n))

    def search(self, query_vec: np.ndarray, top_k: int = 5, ef_search: int = 32) -> list[int]:
        """在线贪心波束搜索：仅访问图上局部节点，避免 O(N) 全量扫描。"""
        q = l2_normalize(query_vec.reshape(1, -1)).squeeze(0)
        visited = {self.entry_point}
        candidates = [(float(self.vectors[self.entry_point] @ q), self.entry_point)]
        best = list(candidates)
        steps = 0
        # [必背手撕-贪心波束搜索] 维护 visited 集合与候选队列，按相似度扩展局部邻居，避免 O(N) 全量扫描
        while candidates and steps < ef_search:
            candidates.sort(key=lambda x: -x[0])
            _, curr = candidates.pop(0)
            steps += 1
            for nb in self.adj[curr]:
                if nb not in visited:
                    visited.add(nb)
                    s = float(self.vectors[nb] @ q)
                    candidates.append((s, nb))
                    best.append((s, nb))
        best.sort(key=lambda x: -x[0])
        return [nid for _, nid in best[:top_k]]


# [无需死背-脚手架] 随机矩阵生成与 time.perf_counter 压测基准函数无需死背
def benchmark_flat_vs_ann(n_vecs: int = 400, dim: int = 32, top_k: int = 5, ef_search: int = 48) -> float:
    rng = np.random.default_rng(42)
    db = l2_normalize(rng.standard_normal((n_vecs, dim)).astype(np.float32))
    q = l2_normalize(rng.standard_normal((1, dim)).astype(np.float32))

    t0 = time.perf_counter()
    flat_scores = (db @ q.T).squeeze(1)
    flat_top = np.argsort(-flat_scores)[:top_k].tolist()
    flat_ms = (time.perf_counter() - t0) * 1000

    ann_index = SimpleNavigableGraphIndex(db, m_neighbors=16)
    t1 = time.perf_counter()
    ann_top = ann_index.search(q, top_k=top_k, ef_search=ef_search)
    ann_ms = (time.perf_counter() - t1) * 1000

    # [必记设计决策-召回基准] ANN 上线前必须与 Flat 精确搜索对齐 Agreement@K，用数据验证 ef_search 甜点
    agreement = len(set(flat_top) & set(ann_top)) / top_k
    assert 0.0 <= agreement <= 1.0
    print(f"Flat={flat_ms:.3f}ms, ANN={ann_ms:.3f}ms, Agreement@{top_k}={agreement:.2%}")
    return agreement


if __name__ == "__main__":
    agr_low = benchmark_flat_vs_ann(ef_search=10)
    agr_high = benchmark_flat_vs_ann(ef_search=80)
    assert agr_high >= agr_low, "增大 ef_search 应提高或保持 ANN 召回重合度"
    assert agr_high >= 0.6, f"高 ef_search 下重合度应 >= 60%，实际 {agr_high:.2%}"
    print("Step 5 self-test passed!")
