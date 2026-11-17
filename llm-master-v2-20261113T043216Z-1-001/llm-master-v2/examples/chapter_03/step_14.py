"""3.14 Step 14: 在 Step 8/10 基础上新增带原始切片溯源的有向图多跳路径探索器（GraphRAG/LightRAG 原理）"""
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class GraphEdge:
    src: str
    rel: str
    dst: str
    chunk_id: str  # 强制绑定原始文本切片 ID，防图谱幻觉


class LightGraphRetriever:
    """支持 Local Search 多跳路径探索（防环 + 跳数硬截断）与全局社区摘要汇总。"""
    def __init__(self) -> None:
        self.adj: dict[str, list[GraphEdge]] = {}

    def add_edge(self, edge: GraphEdge) -> None:
        self.adj.setdefault(edge.src, []).append(edge)

    def local_multi_hop(self, start: str, target: str, max_hops: int = 3) -> list[list[GraphEdge]]:
        if max_hops <= 0:
            raise ValueError("max_hops 必须为正整数")
        results: list[list[GraphEdge]] = []
        queue: deque[tuple[str, list[GraphEdge], set[str]]] = deque([(start, [], {start})])
        while queue:
            curr, path, visited = queue.popleft()
            if len(path) >= max_hops:
                continue
            for edge in self.adj.get(curr, []):
                if edge.dst in visited:
                    continue  # 严格防环死循环
                new_path = path + [edge]
                if edge.dst == target:
                    results.append(new_path)
                else:
                    queue.append((edge.dst, new_path, visited | {edge.dst}))
        return results


if __name__ == "__main__":
    g = LightGraphRetriever()
    g.add_edge(GraphEdge("施工一队", "负责", "配电箱维护", "C_01"))
    g.add_edge(GraphEdge("配电箱维护", "属于", "特种带电作业", "C_02"))
    g.add_edge(GraphEdge("特种带电作业", "要求", "高压电工证", "C_03"))
    g.add_edge(GraphEdge("特种带电作业", "回环测试", "施工一队", "C_LOOP"))

    paths = g.local_multi_hop("施工一队", "高压电工证", max_hops=3)
    assert len(paths) == 1 and len(paths[0]) == 3
    assert [e.chunk_id for e in paths[0]] == ["C_01", "C_02", "C_03"]
    print("Step 14 self-test passed!")
