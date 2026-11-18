"""7.1 Step 1: 矩形框正面积校验、IoU 计算与一对一贪心匹配 (TP/FP/FN).

业务场景：工业巡检安全帽/反光衣检测评测，绝不用 VQA 文本准确率代替空间定位评测。
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np


def validate_boxes(boxes: np.ndarray, name: str) -> None:
    """校验边界框形状为 (N, 4)、有限浮点坐标且严格满足 x2>x1, y2>y1."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"{name} 必须是 (N, 4) 形状，当前为 {boxes.shape}")
    if not np.issubdtype(boxes.dtype, np.floating) or not np.isfinite(boxes).all():
        raise ValueError(f"{name} 必须为有限浮点坐标")
    if boxes.shape[0] > 0 and not bool((boxes[:, 2:] > boxes[:, :2]).all()):  # [必记设计决策-正面积边界框强校验防脏坐标]
        raise ValueError(f"{name} 必须满足 x2 > x1 且 y2 > y1")


def compute_box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """计算两组 (N, 4) 与 (M, 4) 边界框的两两 IoU 矩阵，返回 (N, M)."""
    n, m = boxes_a.shape[0], boxes_b.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])  # [必背手撕-向量化交集左上角坐标]
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)  # [必背手撕-向量化交集面积计算]
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter  # [必背手撕-并集面积等于两框之和减交集]
    return inter / np.maximum(union, 1e-12)


def match_one_class(
    predicted: np.ndarray,
    scores: np.ndarray,
    gold: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Optional[float] | int]:
    """按置信度降序执行一对一贪心匹配，统计 TP/FP/FN 与 Precision/Recall."""
    validate_boxes(predicted, "预测框")
    validate_boxes(gold, "金标框")
    if scores.ndim != 1 or scores.shape[0] != predicted.shape[0]:
        raise ValueError("scores 必须是一维且长度等于预测框数量")
    if not np.isfinite(scores).all() or not ((scores >= 0.0) & (scores <= 1.0)).all():
        raise ValueError("scores 必须在 [0, 1] 区间内")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("IoU 阈值必须在 [0, 1] 区间内")

    overlaps = compute_box_iou(predicted, gold)
    assert overlaps.shape == (predicted.shape[0], gold.shape[0])

    used_gold: set[int] = set()
    hits = 0
    for pred_idx in np.argsort(-scores).tolist():  # [必背手撕-按置信度降序执行一对一贪心匹配]
        candidates = [j for j in range(gold.shape[0]) if j not in used_gold]
        if not candidates:
            continue
        best_j = max(candidates, key=lambda j: float(overlaps[pred_idx, j]))
        if float(overlaps[pred_idx, best_j]) >= threshold:
            used_gold.add(best_j)  # [必记设计决策-每个金标框仅允许匹配一次防重复刷分]
            hits += 1

    fp = int(predicted.shape[0] - hits)
    fn = int(gold.shape[0] - hits)
    assert 0 <= hits <= min(predicted.shape[0], gold.shape[0])
    return {
        "tp": hits,
        "fp": fp,
        "fn": fn,
        "precision": hits / predicted.shape[0] if predicted.shape[0] > 0 else None,
        "recall": hits / gold.shape[0] if gold.shape[0] > 0 else None,
    }


if __name__ == "__main__":  # [无需死背-脚手架]
    preds = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]], dtype=np.float64)
    scores = np.array([0.9, 0.8], dtype=np.float64)
    gold = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float64)
    res = match_one_class(preds, scores, gold, threshold=0.5)
    assert res == {"tp": 1, "fp": 1, "fn": 0, "precision": 0.5, "recall": 1.0}
    print(f"MM Step 1 self-test passed! res={res}")
