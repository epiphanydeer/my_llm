"""7.8 Step 8: 多模态 RAG 视觉观察与制度法规双路解耦核验器.

彻底阻断“以法索图（看了严厉法规就脑补模糊画面违规）”幻觉：
1. 视觉盲审链（不看法规）只输出 hazard / safe / unknown。
2. 法规文本链独立校验 rule_id 存在性与 as_of_date 生效有效期。
3. 只有“视觉实锤 + 法规现行有效”才放行，否则一律确定性降级为 MANUAL_REVIEW。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List


@dataclass(frozen=True)
class RegulatoryRule:
    rule_id: str
    title: str
    version: str
    start_date: str
    end_date: str


@dataclass(frozen=True)
class VisualObservation:
    case_id: str
    observation: str  # "hazard" | "safe" | "unknown"
    rule_id: str


class MultimodalRAGVerifier:
    """双路解耦核验引擎：物理隔离视觉事实与法规检索."""

    def __init__(self, rules: List[RegulatoryRule]) -> None:
        if not rules:
            raise ValueError("制度库不能为空")
        self._rules: Dict[str, RegulatoryRule] = {}
        self._ranges: Dict[str, tuple[date, date]] = {}
        for r in rules:
            s, e = date.fromisoformat(r.start_date), date.fromisoformat(r.end_date)
            if s > e:
                raise ValueError(f"制度 {r.rule_id} 生效日期晚于失效日期")
            self._rules[r.rule_id] = r
            self._ranges[r.rule_id] = (s, e)

    def verify(self, obs: VisualObservation, as_of: date) -> Dict[str, str]:
        if obs.observation not in {"hazard", "safe", "unknown"}:
            raise ValueError(f"非法视觉观察状态: {obs.observation}")
        if obs.observation == "unknown":  # [必记设计决策-视觉证据不足直接降级人工复核防以法索图]
            return {"status": "MANUAL_REVIEW", "reason": "画面证据不足(unknown)，严禁凭法规脑补"}
        if obs.observation == "safe":
            return {"status": "PASS_SAFE", "reason": "视觉确认合规"}

        if obs.rule_id not in self._rules:  # [必记设计决策-缺失有效法规ID阻断自动化裁决]
            return {"status": "MANUAL_REVIEW", "reason": f"未检索到有效制度 {obs.rule_id}"}

        s_date, e_date = self._ranges[obs.rule_id]
        if not (s_date <= as_of <= e_date):  # [必背手撕-制度有效期区间硬门禁校验]
            return {
                "status": "MANUAL_REVIEW",
                "reason": f"制度 {obs.rule_id}({self._rules[obs.rule_id].version}) 不在有效期 [{s_date}, {e_date}]",
            }

        return {
            "status": "CONFIRMED_HAZARD",
            "reason": f"视觉实锤违规，且制度 {obs.rule_id}({self._rules[obs.rule_id].version}) 现行有效",
        }


if __name__ == "__main__":  # [无需死背-脚手架]
    verifier = MultimodalRAGVerifier(
        [
            RegulatoryRule("R-NEW", "高处作业规范", "v2026", "2026-01-01", "2027-12-31"),
            RegulatoryRule("R-OLD", "旧版安全规程", "v2023", "2023-01-01", "2025-12-31"),
        ]
    )
    today = date(2026, 11, 13)
    assert verifier.verify(VisualObservation("C1", "unknown", "R-NEW"), today)["status"] == "MANUAL_REVIEW"
    assert verifier.verify(VisualObservation("C2", "hazard", "R-OLD"), today)["status"] == "MANUAL_REVIEW"
    assert verifier.verify(VisualObservation("C3", "hazard", "R-NEW"), today)["status"] == "CONFIRMED_HAZARD"
    print("MM Step 8 self-test passed!")
