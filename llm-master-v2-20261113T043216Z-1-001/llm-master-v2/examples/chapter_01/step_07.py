"""1.7 API 成本：输入、缓存与输出分项核算器"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelPricingPerMillion:
    input_usd: float
    cached_input_usd: float
    output_usd: float


def compute_call_cost_usd(
    uncached_input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    pricing: ModelPricingPerMillion,
) -> Dict[str, float]:
    """按百万 Token 单价拆分核算未命中输入、缓存命中输入与输出费用。"""
    in_cost = uncached_input_tokens * pricing.input_usd / 1_000_000
    cache_cost = cached_input_tokens * pricing.cached_input_usd / 1_000_000
    out_cost = output_tokens * pricing.output_usd / 1_000_000
    return {
        "input_cost": round(in_cost, 6),
        "cache_cost": round(cache_cost, 6),
        "output_cost": round(out_cost, 6),
        "total_cost": round(in_cost + cache_cost + out_cost, 6),
    }


if __name__ == "__main__":
    pricing = ModelPricingPerMillion(input_usd=2.0, cached_input_usd=0.5, output_usd=8.0)
    no_cache = compute_call_cost_usd(10_000, 0, 1_000, pricing)
    with_cache = compute_call_cost_usd(2_000, 8_000, 1_000, pricing)
    assert with_cache["total_cost"] < no_cache["total_cost"]
    print("[1.7 PASS] no_cache:", no_cache["total_cost"], "with_cache:", with_cache["total_cost"])
