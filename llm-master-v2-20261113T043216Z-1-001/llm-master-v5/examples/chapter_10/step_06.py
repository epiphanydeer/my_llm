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
    api_cost = (  # [必背手撕-API月度可变成本] 按月请求量与输入/输出单价线性累加
        monthly_requests * avg_in_tokens / 1_000_000 * api_in_per_1m
        + monthly_requests * avg_out_tokens / 1_000_000 * api_out_per_1m
    )
    winner = "self_host" if self_host_gpu_monthly_usd < api_cost else "api"  # [必记设计决策-TCO盈亏平衡选型] 低QPS用API按量付费，高QPS自部署摊薄固定GPU成本
    return {
        "api_monthly_usd": round(api_cost, 2),
        "self_host_monthly_usd": round(self_host_gpu_monthly_usd, 2),
        "cheaper_option": winner,
    }


if __name__ == "__main__":  # [无需死背-脚手架] 低/高并发场景对比断言
    low_qps = compare_monthly_tco(50_000, 1500, 300, 1.5, 6.0, 2200.0)
    high_qps = compare_monthly_tco(2_000_000, 1500, 300, 1.5, 6.0, 2200.0)
    assert low_qps["cheaper_option"] == "api" and high_qps["cheaper_option"] == "self_host"
    print("[10.6 PASS] low_qps:", low_qps, "high_qps:", high_qps)
