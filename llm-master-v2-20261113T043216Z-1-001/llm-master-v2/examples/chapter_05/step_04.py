"""5.4 Step 4: 微调项目防泄漏与训练排障监控器."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class SFTTrainingGuard:
    patience: int = 2
    max_grad_norm: float = 5.0
    _best_eval_loss: float = float("inf")
    _worse_epochs: int = 0
    _alerts: List[str] = field(default_factory=list)

    @staticmethod
    def check_template_leakage(train_templates: Set[str], eval_templates: Set[str]) -> None:
        overlap = train_templates & eval_templates
        if overlap:
            raise ValueError(f"TEMPLATE_LEAKAGE: 发现跨集合模板指纹重叠 {sorted(overlap)}")

    def step_check(self, train_loss: float, eval_loss: float, grad_norm: float) -> Dict[str, object]:
        if not (train_loss == train_loss and eval_loss == eval_loss):
            return {"action": "ABORT_NAN", "reason": "Loss 出现 NaN，请检查 -100 掩码或降低学习率"}
        if grad_norm > self.max_grad_norm:
            self._alerts.append(f"GRAD_SPIKE:{grad_norm}")
        if eval_loss < self._best_eval_loss:
            self._best_eval_loss = eval_loss
            self._worse_epochs = 0
            return {"action": "SAVE_CHECKPOINT", "worse_epochs": 0}
        self._worse_epochs += 1
        if self._worse_epochs >= self.patience:
            return {"action": "EARLY_STOP", "reason": "Eval Loss 连续上升，防止过拟合背诵"}
        return {"action": "CONTINUE", "worse_epochs": self._worse_epochs}


if __name__ == "__main__":
    guard = SFTTrainingGuard(patience=2)
    assert guard.step_check(1.2, 1.1, 0.8)["action"] == "SAVE_CHECKPOINT"
    assert guard.step_check(0.8, 1.2, 1.0)["action"] == "CONTINUE"
    assert guard.step_check(0.5, 1.3, 1.1)["action"] == "EARLY_STOP"
    print("Ch5 Step 4 self-test passed!")
