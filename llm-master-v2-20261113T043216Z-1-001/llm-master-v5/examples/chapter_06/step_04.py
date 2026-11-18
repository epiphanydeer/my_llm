"""6.4 Step 4: 流式并发压测与 TTFT / TPOT / P95 / Goodput 统计套件."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass(frozen=True)
class StreamMetrics:
    req_id: str
    queue_ms: float
    prefill_ms: float
    decode_token_ms: List[float]

    @property
    def ttft_ms(self) -> float:
        first_tok = self.decode_token_ms[0] if self.decode_token_ms else 0.0
        return self.queue_ms + self.prefill_ms + first_tok  # [必背手撕-TTFT包含排队+Prefill+首Token]

    @property
    def tpot_ms(self) -> float:
        if len(self.decode_token_ms) <= 1:
            return 0.0
        return sum(self.decode_token_ms[1:]) / (len(self.decode_token_ms) - 1)  # [必背手撕-TPOT仅统计首字之后的平均生成间隔]

    @property
    def e2e_ms(self) -> float:
        return self.queue_ms + self.prefill_ms + sum(self.decode_token_ms)


def evaluate_sla(
    logs: List[StreamMetrics],
    ttft_slo_ms: float = 500.0,
    tpot_slo_ms: float = 50.0,
) -> Dict[str, float | int]:
    if not logs:
        raise ValueError("压测记录不能为空")

    ttfts = [x.ttft_ms for x in logs]
    tpots = [x.tpot_ms for x in logs]
    e2es = [x.e2e_ms for x in logs]
    good_reqs = sum(1 for x in logs if x.ttft_ms <= ttft_slo_ms and x.tpot_ms <= tpot_slo_ms)  # [必记设计决策-Goodput需同时满足TTFT与TPOT双SLO]

    return {
        "total_reqs": len(logs),
        "ttft_p95_ms": round(float(np.percentile(ttfts, 95)), 2),  # [必记设计决策-各阶段P95不可直接相加须按请求级统计]
        "tpot_p95_ms": round(float(np.percentile(tpots, 95)), 2),
        "e2e_p95_ms": round(float(np.percentile(e2es, 95)), 2),
        "goodput_rate": round(good_reqs / len(logs), 4),
    }


if __name__ == "__main__":  # [无需死背-脚手架]
    reqs = [
        StreamMetrics("r1", queue_ms=400.0, prefill_ms=50.0, decode_token_ms=[20.0, 20.0]),
        StreamMetrics("r2", queue_ms=20.0, prefill_ms=300.0, decode_token_ms=[20.0, 40.0, 40.0, 40.0]),
        StreamMetrics("r3", queue_ms=30.0, prefill_ms=60.0, decode_token_ms=[20.0, 25.0, 25.0]),
    ]
    sla = evaluate_sla(reqs, ttft_slo_ms=500.0, tpot_slo_ms=50.0)
    assert sla["goodput_rate"] == 1.0
    print(f"Ch6 Step 4 self-test passed! sla={sla}")
