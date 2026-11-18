"""3.13 Step 13: 在 Step 2/3 基础上新增基于 SHA-256 与权限版本对比的增量索引调度器（含防误删熔断）"""
from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class IndexRecord:
    doc_id: str
    content_hash: str
    policy_rev: str


# [必记设计决策-归一化哈希] 计算正文哈希前先归一化空白符，避免无意义换行/空格变动触发全量重算 Embedding
def sha256_norm(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def plan_incremental_sync(
    old_records: list[IndexRecord],
    new_docs: list[tuple[str, str, str]],  # (doc_id, normalized_text, policy_rev)
    is_snapshot_complete: bool,
    max_delete_ratio: float = 0.20,
) -> list[tuple[str, str]]:
    """
    增量调度决策：
    - 正文哈希未变仅权限变 -> UPDATE_METADATA_ONLY（零 Embedding 算力开销）
    - 正文哈希变 -> UPDATE_VECTOR
    - 快照不完整或删除比例超限 -> PENDING_CHECK（熔断挂起防误删全库）
    """
    old_map = {r.doc_id: r for r in old_records}
    new_ids = {d[0] for d in new_docs}
    plan: list[tuple[str, str]] = []

    for doc_id, text, pol in new_docs:
        h = sha256_norm(text)
        if doc_id not in old_map:
            plan.append((doc_id, "INSERT"))
        # [必记设计决策-算力分流] 正文变了才 UPDATE_VECTOR；仅权限变则 UPDATE_METADATA_ONLY，零 Embedding 成本
        elif old_map[doc_id].content_hash != h:
            plan.append((doc_id, "UPDATE_VECTOR"))
        elif old_map[doc_id].policy_rev != pol:
            plan.append((doc_id, "UPDATE_METADATA_ONLY"))
        else:
            plan.append((doc_id, "NOOP"))

    missing_ids = [did for did in old_map if did not in new_ids]
    delete_ratio = len(missing_ids) / len(old_map) if old_map else 0.0
    # [必记设计决策-防误删熔断] 上游快照不完整或缺失比例超阈值(如20%)时，必须熔断挂起(PENDING_CHECK)防误删全库
    action_for_missing = "TOMBSTONE" if (is_snapshot_complete and delete_ratio <= max_delete_ratio) else "PENDING_CHECK"
    for did in missing_ids:
        plan.append((did, action_for_missing))
    return plan


if __name__ == "__main__":
    old = [
        IndexRecord("D1", sha256_norm("条款一"), "pol_v1"),
        IndexRecord("D2", sha256_norm("条款二"), "pol_v1"),
        IndexRecord("D3", sha256_norm("条款三"), "pol_v1"),
    ]
    new = [("D1", "条款一", "pol_v2"), ("D2", "条款二-修订", "pol_v1"), ("D4", "条款四", "pol_v1")]
    p = dict(plan_incremental_sync(old, new, is_snapshot_complete=True, max_delete_ratio=0.20))
    assert p["D1"] == "UPDATE_METADATA_ONLY" and p["D2"] == "UPDATE_VECTOR" and p["D4"] == "INSERT"
    assert p["D3"] == "PENDING_CHECK", "删除比例超阈值必须熔断挂起"
    print("Step 13 self-test passed!")
