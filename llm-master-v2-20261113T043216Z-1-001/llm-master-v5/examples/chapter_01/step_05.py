"""1.5 蒸馏：教师、学生与软目标 KL 散度损失计算"""
import numpy as np


def softmax_with_temp(logits: np.ndarray, temp: float) -> np.ndarray:
    scaled = logits / temp  # [必背手撕-高温软化分布] 除以温度 T 放大次优类别信息
    scaled = scaled - np.max(scaled)
    e = np.exp(scaled)
    return e / np.sum(e)


def distillation_loss(  # [无需死背-脚手架] 损失函数签名与默认超参
    teacher_logits: np.ndarray,
    student_logits: np.ndarray,
    hard_label_idx: int,
    temperature: float = 2.0,
    alpha: float = 0.7,
) -> float:
    """Hinton 知识蒸馏总损失：alpha * T^2 * KL(p_T || q_T) + (1 - alpha) * CE(y, q_1)。"""
    p_t = softmax_with_temp(teacher_logits, temperature)
    q_t = softmax_with_temp(student_logits, temperature)
    eps = 1e-12
    kl_soft = float(np.sum(p_t * np.log((p_t + eps) / (q_t + eps))))  # [必背手撕-软目标KL散度] 教师与学生高温概率分布对齐
    q_1 = softmax_with_temp(student_logits, 1.0)
    ce_hard = float(-np.log(q_1[hard_label_idx] + eps))
    return round(alpha * (temperature ** 2) * kl_soft + (1.0 - alpha) * ce_hard, 6)  # [必记设计决策-T平方梯度补偿] 乘以 T^2 保持软硬损失梯度量级一致


if __name__ == "__main__":  # [无需死背-脚手架] 蒸馏损失单调性验证
    t_logits = np.array([5.0, 2.5, 0.5])
    s_bad = np.array([1.0, 3.0, 2.0])
    s_good = np.array([4.8, 2.4, 0.6])
    loss_bad = distillation_loss(t_logits, s_bad, hard_label_idx=0)
    loss_good = distillation_loss(t_logits, s_good, hard_label_idx=0)
    assert loss_good < loss_bad
    print("[1.5 PASS] loss_bad:", loss_bad, "loss_good:", loss_good)
