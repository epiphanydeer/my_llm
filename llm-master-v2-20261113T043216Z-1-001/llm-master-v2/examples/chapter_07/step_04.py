"""7.4 Step 4: 多模态训练数据组织与事件级防泄漏切分 (Event-Level Split).

在 Step 3 输入校验基础上新增数据集清单校验：
1. 强制每条样本包含规范字段且允许合规的 `unknown` 拒答标签。
2. 按 `event_id` 隔离划分 train/validation/test，严禁同一事件的连拍帧跨集合泄漏。
"""

from __future__ import annotations

from typing import Dict, List, Set


VALID_SPLITS = frozenset({"train", "validation", "test"})
VALID_LABELS = frozenset({"hazard", "safe", "unknown"})


def validate_event_split_dataset(records: List[Dict[str, str]]) -> Dict[str, object]:
    """校验字段完整性、样本 ID 唯一性，并阻断跨分区事件级数据泄漏."""
    if not records:
        raise ValueError("数据集不能为空")

    seen_ids: Set[str] = set()
    event_to_split: Dict[str, str] = {}
    split_counts: Dict[str, int] = {s: 0 for s in VALID_SPLITS}
    unknown_count = 0

    for idx, row in enumerate(records, start=1):
        for key in ("id", "event_id", "split", "label"):
            if not str(row.get(key, "")).strip():
                raise ValueError(f"第 {idx} 条样本缺少必填字段: {key}")

        sid = row["id"].strip()
        ev = row["event_id"].strip()
        sp = row["split"].strip()
        lb = row["label"].strip()

        if sid in seen_ids:
            raise ValueError(f"发现重复样本 ID: {sid}")
        seen_ids.add(sid)

        if sp not in VALID_SPLITS:
            raise ValueError(f"非法分区 {sp}，必须属于 {VALID_SPLITS}")
        if lb not in VALID_LABELS:
            raise ValueError(f"非法标签 {lb}，必须属于 {VALID_LABELS}")

        if ev in event_to_split and event_to_split[ev] != sp:
            raise ValueError(
                f"DATA_LEAKAGE: 事件 {ev} 同时出现在 {event_to_split[ev]} 与 {sp}！"
            )
        event_to_split[ev] = sp
        split_counts[sp] += 1
        if lb == "unknown":
            unknown_count += 1

    train_events = {r["event_id"] for r in records if r["split"] == "train"}
    test_events = {r["event_id"] for r in records if r["split"] == "test"}
    assert len(train_events & test_events) == 0, "训练集与测试集存在事件交集！"

    return {
        "total": len(records),
        "unique_events": len(event_to_split),
        "split_counts": split_counts,
        "unknown_count": unknown_count,
    }


if __name__ == "__main__":
    clean_rows = [
        {"id": "s1", "event_id": "cam01_1000", "split": "train", "label": "hazard"},
        {"id": "s2", "event_id": "cam01_1000", "split": "train", "label": "unknown"},
        {"id": "s3", "event_id": "cam02_1400", "split": "test", "label": "safe"},
    ]
    summary = validate_event_split_dataset(clean_rows)
    assert summary["unique_events"] == 2 and summary["unknown_count"] == 1
    print(f"MM Step 4 self-test passed! summary={summary}")
