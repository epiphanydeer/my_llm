"""10.7 模型选型面试：同任务评测与失败切片归因分析器"""
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class EvalSample:
    sample_id: str
    slice_tag: str  # e.g. "long_context" | "json_schema" | "small_object_ocr"
    passed: bool


def analyze_failure_slices(samples: List[EvalSample]) -> Dict[str, object]:
    total = len(samples)
    passed = sum(s.passed for s in samples)
    fail_by_slice = Counter(s.slice_tag for s in samples if not s.passed)
    return {
        "overall_accuracy": round(passed / total, 3),
        "top_failure_slice": fail_by_slice.most_common(1)[0] if fail_by_slice else None,
        "failure_breakdown": dict(fail_by_slice),
    }


if __name__ == "__main__":
    eval_set = [
        EvalSample("s1", "long_context", True),
        EvalSample("s2", "small_object_ocr", False),
        EvalSample("s3", "small_object_ocr", False),
        EvalSample("s4", "json_schema", True),
    ]
    report = analyze_failure_slices(eval_set)
    assert report["overall_accuracy"] == 0.5 and report["top_failure_slice"][0] == "small_object_ocr"
    print("[10.7 PASS]", report)
