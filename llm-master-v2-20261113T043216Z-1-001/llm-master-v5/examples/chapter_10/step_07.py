"""10.7 模型选型面试：同任务评测与失败切片归因分析器"""
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class EvalSample:  # [无需死背-脚手架] 评测样本切片记录类
    sample_id: str
    slice_tag: str  # e.g. "long_context" | "json_schema" | "small_object_ocr"
    passed: bool


def analyze_failure_slices(samples: List[EvalSample]) -> Dict[str, object]:
    total = len(samples)
    passed = sum(s.passed for s in samples)
    fail_by_slice = Counter(s.slice_tag for s in samples if not s.passed)  # [必背手撕-失败切片频次归因] 按场景标签聚合失败样本
    return {
        "overall_accuracy": round(passed / total, 3),
        "top_failure_slice": fail_by_slice.most_common(1)[0] if fail_by_slice else None,  # [必记设计决策-定位短板切片] 面试选型不仅报总准确率，更要指出头号失败切片
        "failure_breakdown": dict(fail_by_slice),
    }


if __name__ == "__main__":  # [无需死背-脚手架] 四样本切片统计测试
    eval_set = [
        EvalSample("s1", "long_context", True),
        EvalSample("s2", "small_object_ocr", False),
        EvalSample("s3", "small_object_ocr", False),
        EvalSample("s4", "json_schema", True),
    ]
    report = analyze_failure_slices(eval_set)
    assert report["overall_accuracy"] == 0.5 and report["top_failure_slice"][0] == "small_object_ocr"
    print("[10.7 PASS]", report)
