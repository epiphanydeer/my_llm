"""1.1 应用算法岗位：能力地图与项目证据链校验器（工业工程校验写法）"""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EvidenceItem:  # [无需死背-脚手架] 数据类容器定义
    """简历/面试项目中的单条可验证证据。"""
    baseline_metric: str
    intervention: str
    final_metric: str
    reproducible_artifact: str

    def is_complete(self) -> bool:
        return all([  # [必记设计决策-四维证据闭环] 基线+干预+结果+可复现产物缺一不可
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
        missing = [p for p in self.required_pillars if p not in self.evidences]  # [必背手撕-支柱覆盖率与缺失项校验]
        incomplete = [
            p for p, ev in self.evidences.items() if not ev.is_complete()
        ]
        passed = (len(missing) == 0) and (len(incomplete) == 0)  # [必记设计决策-双条件门禁] 既不能缺支柱也不能有残缺证据
        return {
            "role": self.role_name,
            "passed": passed,
            "missing_pillars": missing,
            "incomplete_pillars": incomplete,
            "coverage": round((len(self.required_pillars) - len(missing)) / len(self.required_pillars), 2),
        }


if __name__ == "__main__":  # [无需死背-脚手架] 本地断言测试入口
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
