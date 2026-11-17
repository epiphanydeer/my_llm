"""Step 9: DAG Task Scheduler (Plan-and-Execute 与拓扑并行调度 + 断点续跑).

实现基于 Kahn 算法的 DAG 任务调度器：
1. 环检测 (Cycle Detection)：存在循环依赖时拒绝执行并报错。
2. 并行分批 (Parallel Wave Batching)：入度为 0 的无依赖任务自动编入同一并行波次。
3. 断点续跑 (Checkpoint Resume)：支持持久化已完成节点状态，失败恢复时跳过已完成波次。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Set


@dataclass
class TaskNode:
    task_id: str
    deps: List[str]
    action: Callable[[Dict[str, str]], str]


@dataclass
class DAGTaskScheduler:
    """支持并行分批与断点续跑的 DAG 调度器."""

    nodes: Dict[str, TaskNode]
    completed_outputs: Dict[str, str] = field(default_factory=dict)

    def plan_waves(self) -> List[List[str]]:
        """使用 Kahn 算法生成并行执行波次，并检测循环依赖."""
        in_degree: Dict[str, int] = {tid: 0 for tid in self.nodes}
        children: Dict[str, List[str]] = {tid: [] for tid in self.nodes}

        for tid, node in self.nodes.items():
            for dep in node.deps:
                if dep not in self.nodes:
                    raise ValueError(f"任务 {tid} 依赖不存在的节点 {dep}")
                in_degree[tid] += 1
                children[dep].append(tid)

        queue: deque[str] = deque(sorted(tid for tid, deg in in_degree.items() if deg == 0))
        waves: List[List[str]] = []
        visited_count = 0

        while queue:
            current_wave = list(queue)
            queue.clear()
            waves.append(current_wave)
            next_ready: List[str] = []
            for tid in current_wave:
                visited_count += 1
                for child in children[tid]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_ready.append(child)
            queue.extend(sorted(next_ready))

        if visited_count != len(self.nodes):
            raise ValueError("检测到 DAG 循环依赖 (Cycle Detected)，无法调度")
        return waves

    def run(self, fail_once_on: Set[str] | None = None) -> Dict[str, str]:
        """按波次执行 DAG，自动跳过 checkpoint 中已完成的节点."""
        waves = self.plan_waves()
        fail_once_on = fail_once_on or set()

        for wave in waves:
            for tid in wave:
                if tid in self.completed_outputs:
                    continue
                if tid in fail_once_on:
                    fail_once_on.remove(tid)
                    raise RuntimeError(f"节点 {tid} 模拟执行中断")
                dep_results = {d: self.completed_outputs[d] for d in self.nodes[tid].deps}
                self.completed_outputs[tid] = self.nodes[tid].action(dep_results)
        return dict(self.completed_outputs)


if __name__ == "__main__":
    exec_counts = {"fetch_logs": 0, "fetch_metrics": 0, "diagnose": 0}

    def make_action(name: str) -> Callable[[Dict[str, str]], str]:
        def _fn(deps: Dict[str, str]) -> str:
            exec_counts[name] += 1
            return f"{name}_ok(deps={list(deps.keys())})"
        return _fn

    scheduler = DAGTaskScheduler(
        nodes={
            "fetch_logs": TaskNode("fetch_logs", [], make_action("fetch_logs")),
            "fetch_metrics": TaskNode("fetch_metrics", [], make_action("fetch_metrics")),
            "diagnose": TaskNode("diagnose", ["fetch_logs", "fetch_metrics"], make_action("diagnose")),
        }
    )
    waves = scheduler.plan_waves()
    assert waves == [["fetch_logs", "fetch_metrics"], ["diagnose"]], f"并行波次错误: {waves}"

    fail_set = {"diagnose"}
    try:
        scheduler.run(fail_once_on=fail_set)
        assert False, "应当抛出中断异常"
    except RuntimeError:
        pass

    final_out = scheduler.run(fail_once_on=fail_set)
    assert "diagnose" in final_out
    assert exec_counts["fetch_logs"] == 1 and exec_counts["fetch_metrics"] == 1
    assert exec_counts["diagnose"] == 1

    cyclic = DAGTaskScheduler(
        nodes={
            "A": TaskNode("A", ["B"], lambda _: "A"),
            "B": TaskNode("B", ["A"], lambda _: "B"),
        }
    )
    try:
        cyclic.plan_waves()
        assert False, "环依赖必须报错"
    except ValueError as e:
        assert "Cycle" in str(e)
    print("Step 9 self-test passed!")
