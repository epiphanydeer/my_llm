"""10.3 Dense 与 MoE：Top-K 门控路由与负载均衡辅助损失估算器"""
import numpy as np


def moe_topk_route_and_aux_loss(router_logits: np.ndarray, top_k: int = 2) -> dict:
    """
    router_logits: [num_tokens, num_experts]
    计算 Top-K 路由分配比例与 Switch/Mixtral 风格的负载均衡 Aux Loss。
    """
    num_tokens, num_experts = router_logits.shape
    shifted = router_logits - np.max(router_logits, axis=-1, keepdims=True)  # [必背手撕-二维轴向稳定Softmax] 按最后一维归一化路由概率
    probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)

    topk_idx = np.argsort(probs, axis=-1)[:, -top_k:]  # [必背手撕-TopK专家路由选择]
    dispatch_counts = np.zeros(num_experts, dtype=float)
    for row in topk_idx:
        for e in row:
            dispatch_counts[e] += 1.0

    f_i = dispatch_counts / (num_tokens * top_k)  # 每个专家实际分到的 Token 比例  # [必记设计决策-实际负载比例f_i] 统计各专家真实分到的 token 占比
    p_i = np.mean(probs, axis=0)                  # 每个专家的平均路由概率
    aux_loss = float(num_experts * np.sum(f_i * p_i))  # [必背手撕-MoE负载均衡AuxLoss] N * sum(f_i * p_i) 惩罚路由坍缩
    return {
        "expert_load_fraction": np.round(f_i, 3).tolist(),
        "aux_loss": round(aux_loss, 4),
    }


if __name__ == "__main__":  # [无需死背-脚手架] 均衡与坍缩路由对比测试
    balanced_logits = np.array([[2.0, 0.1, 0.1, 0.1], [0.1, 2.0, 0.1, 0.1], [0.1, 0.1, 2.0, 0.1], [0.1, 0.1, 0.1, 2.0]])
    collapsed_logits = np.tile([5.0, 4.0, -2.0, -2.0], (4, 1))
    r_bal = moe_topk_route_and_aux_loss(balanced_logits, top_k=1)
    r_col = moe_topk_route_and_aux_loss(collapsed_logits, top_k=1)
    assert r_bal["aux_loss"] < r_col["aux_loss"]
    print("[10.3 PASS] balanced:", r_bal, "collapsed:", r_col)
