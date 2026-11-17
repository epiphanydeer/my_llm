"""10.6 API、托管与自部署：不同并发下的月度 TCO 核算器"""
from typing import Dict


def compare_monthly_tco(
    monthly_requests: int,
    avg_in_tokens: int,
    avg_out_tokens: int,
    api_in_per_1m: float,
    api_out_per_1m: float,
    self_host_gpu_monthly_usd: float,
) -> Dict[str, object]:
    api_cost = (
        monthly_requests * avg_in_tokens / 1_000_000 * api_in_per_1m
        + monthly_requests * avg_out_tokens / 1_000_000 * api_out_per_1m
    )
    winner = "self_host" if self_host_gpu_monthly_usd < api_cost else "api"
    return {
        "api_monthly_usd": round(api_cost, 2),
        "self_host_monthly_usd": round(self_host_gpu_monthly_usd, 2),
        "cheaper_option": winner,
    }


if __name__ == "__main__":
    low_qps = compare_monthly_tco(50_000, 1500, 300, 1.5, 6.0, 2200.0)
    high_qps = compare_monthly_tco(2_000_000, 1500, 300, 1.5, 6.0, 2200.0)
    assert low_qps["cheaper_option"] == "api" and high_qps["cheaper_option"] == "self_host"
    print("[10.6 PASS] low_qps:", low_qps, "high_qps:", high_qps)
