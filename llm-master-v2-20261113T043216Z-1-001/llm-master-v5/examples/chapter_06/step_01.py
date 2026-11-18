"""6.1 Step 1: 显存容量测算 (静态权重 + 动态 KV Cache) 与云 API vs 自部署 TCO 盈亏平衡点."""

from __future__ import annotations

from typing import Dict


def estimate_vram_and_concurrency(
    params_billion: float,
    weight_bytes: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    kv_bytes: int,
    seq_len: int,
    gpu_vram_gb: float,
    runtime_reserve_gb: float = 4.0,
) -> Dict[str, float | int]:
    """根据标准注意力 KV Cache 公式测算单卡显存分布与最大理论并发数."""
    if params_billion <= 0 or seq_len <= 0 or gpu_vram_gb <= runtime_reserve_gb:
        raise ValueError("模型参数量、序列长度与可用显存必须合法")

    weight_gb = (params_billion * 1e9 * weight_bytes) / (1024**3)  # [必背手撕-静态权重显存测算公式]
    per_token_bytes = 2 * num_layers * num_kv_heads * head_dim * kv_bytes  # [必背手撕-GQA/MQA单Token KV Cache字节公式]
    per_req_kv_gb = (per_token_bytes * seq_len) / (1024**3)  # [必背手撕-单并发长序列KV显存占用]

    free_for_kv_gb = gpu_vram_gb - weight_gb - runtime_reserve_gb  # [必记设计决策-扣除运行时预留显存防OOM]
    max_concurrency = max(0, int(free_for_kv_gb // per_req_kv_gb))  # [必记设计决策-KV空间决定理论并发上限]

    return {
        "weight_gb": round(weight_gb, 2),
        "per_token_kib": round(per_token_bytes / 1024, 2),
        "per_req_kv_gb": round(per_req_kv_gb, 2),
        "free_for_kv_gb": round(max(0.0, free_for_kv_gb), 2),
        "max_concurrency": max_concurrency,
    }


def tco_break_even(
    monthly_gpu_cost_usd: float,
    api_cost_per_1m_tokens_usd: float,
) -> Dict[str, float]:
    """计算自建 GPU 相比按量付费云 API 的月度 Token 盈亏平衡点 (单位: 百万 Token)."""
    if monthly_gpu_cost_usd <= 0 or api_cost_per_1m_tokens_usd <= 0:
        raise ValueError("成本参数必须为正数")
    break_even_m_tokens = monthly_gpu_cost_usd / api_cost_per_1m_tokens_usd  # [必背手撕-自建与云API月度盈亏平衡点]
    return {"break_even_million_tokens_per_month": round(break_even_m_tokens, 2)}


if __name__ == "__main__":  # [无需死背-脚手架]
    res = estimate_vram_and_concurrency(
        params_billion=8.0, weight_bytes=2, num_layers=32, num_kv_heads=8,
        head_dim=128, kv_bytes=2, seq_len=32768, gpu_vram_gb=80.0,
    )
    assert res["per_token_kib"] == 128.0 and res["per_req_kv_gb"] == 4.0 and res["max_concurrency"] >= 10
    print(f"Ch6 Step 1 self-test passed! res={res}")
