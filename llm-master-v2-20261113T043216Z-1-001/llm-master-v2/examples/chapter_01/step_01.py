"""1.1 应用算法岗位：能力地图与项目证据链校验器（工业工程校验写法）"""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EvidenceItem:
    """简历/面试项目中的单条可验证证据。"""
    baseline_metric: str
    intervention: str
    final_metric: str
    reproducible_artifact: str

    def is_complete(self) -> bool:
        return all([
            bool(self.baseline_metric.strip()),
            bool(self.intervention.strip()),
            bool(self.final_metric.strip()),
            bool(self.reproducible_artifact.strip()),
        ])


@dataclass
class RoleReadinessEvaluator:
    """面向应用算法 / 模型微调 / 工程落地岗位的证据链校验器。"""
    role_name: str
    required_pillars: List[str]
    evidences: Dict[str, EvidenceItem] = field(default_factory=dict)

    def add_evidence(self, pillar: str, item: EvidenceItem) -> None:
        self.evidences[pillar] = item

    def evaluate(self) -> Dict[str, object]:
        missing = [p for p in self.required_pillars if p not in self.evidences]
        incomplete = [
            p for p, ev in self.evidences.items() if not ev.is_complete()
        ]
        passed = (len(missing) == 0) and (len(incomplete) == 0)
        return {
            "role": self.role_name,
            "passed": passed,
            "missing_pillars": missing,
            "incomplete_pillars": incomplete,
            "coverage": round((len(self.required_pillars) - len(missing)) / len(self.required_pillars), 2),
        }


if __name__ == "__main__":
    evaluator = RoleReadinessEvaluator(
        role_name="大模型应用算法工程师",
        required_pillars=["模型调用与结构化输出", "RAG/Agent检索编排", "评测与生产排障"],
    )
    evaluator.add_evidence(
        "模型调用与结构化输出",
        EvidenceItem("JSON解析失败率14%", "引入Schema约束+错误回传自修复", "解析失败率降至0.4%", "eval_report_v2.json"),
    )
    evaluator.add_evidence(
        "RAG/Agent检索编排",
        EvidenceItem("Recall@5=0.62", "切分重叠调优+BGE-Rerank精排", "Recall@5=0.89", "rag_benchmark.csv"),
    )
    evaluator.add_evidence(
        "评测与生产排障",
        EvidenceItem("P99延迟6.8s", "前缀缓存+异步并发流式", "P99延迟2.1s", "load_test_locust.html"),
    )
    res = evaluator.evaluate()
    assert res["passed"] is True and res["coverage"] == 1.0
    print("[1.1 PASS]", res)
