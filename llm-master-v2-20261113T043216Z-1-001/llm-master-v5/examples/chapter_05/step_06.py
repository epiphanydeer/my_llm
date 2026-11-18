"""5.6 Step 6: SFT 数据清洗、指纹去重与 PII 隐私合规脱敏."""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Set


PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")  # [无需死背-脚手架]
ID_CARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")


def mask_pii(text: str) -> str:
    text = PHONE_RE.sub("[PHONE_MASKED]", text)  # [必记设计决策-入库前确定性PII脱敏防记忆背诵]
    text = ID_CARD_RE.sub("[ID_MASKED]", text)
    return text


def sanitize_sft_corpus(samples: List[Dict[str, str]], min_chars: int = 3) -> Dict[str, object]:
    seen_hashes: Set[str] = set()
    cleaned: List[Dict[str, str]] = []
    dropped_short = 0
    dropped_dup = 0

    for row in samples:
        q = mask_pii(row.get("instruction", "").strip())
        a = mask_pii(row.get("output", "").strip())
        if len(q) < min_chars or len(a) < min_chars:  # [必记设计决策-极短噪声样本过滤]
            dropped_short += 1
            continue
        fp = hashlib.sha256(f"{q.lower()}||{a.lower()}".encode("utf-8")).hexdigest()  # [必背手撕-归一化文本指纹精确去重]
        if fp in seen_hashes:
            dropped_dup += 1
            continue
        seen_hashes.add(fp)
        cleaned.append({"instruction": q, "output": a})

    return {
        "cleaned": cleaned,
        "kept": len(cleaned),
        "dropped_short": dropped_short,
        "dropped_dup": dropped_dup,
    }


if __name__ == "__main__":  # [无需死背-脚手架]
    raw = [
        {"instruction": "查工单", "output": "请联系 13812345678 处理"},
        {"instruction": "查工单", "output": "请联系 13812345678 处理"},
        {"instruction": "a", "output": "b"},
    ]
    out = sanitize_sft_corpus(raw, min_chars=3)
    assert out["kept"] == 1 and out["dropped_dup"] == 1 and out["dropped_short"] == 1
    assert "[PHONE_MASKED]" in out["cleaned"][0]["output"]
    print("Ch5 Step 6 self-test passed!")
