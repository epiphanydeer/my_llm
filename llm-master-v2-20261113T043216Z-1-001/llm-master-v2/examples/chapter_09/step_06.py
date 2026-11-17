"""9.6 Step 6: TinyTransformerLM 端到端拼接、交叉熵 Loss 验算与自回归生成循环."""

from __future__ import annotations

from typing import List
import numpy as np
from step_04 import RMSNorm
from step_05 import TransformerBlock


class TinyTransformerLM:
    """将 Embedding + 位置编码 + 多层 Pre-LN Block + Final Norm + LM Head 拼装成完整语言模型."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_seq_len: int = 64,
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        rng = np.random.default_rng(seed)
        self.tok_emb = rng.normal(0.0, 0.02, (vocab_size, d_model))
        self.pos_emb = rng.normal(0.0, 0.02, (max_seq_len, d_model))
        self.blocks = [TransformerBlock(d_model, num_heads, d_ff, seed=seed + i * 10) for i in range(num_layers)]
        self.final_norm = RMSNorm(d_model)
        self.lm_head = self.tok_emb.T  # Weight Tying

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        if token_ids.ndim != 2:
            raise ValueError(f"token_ids 必须为二维 (B, T)，当前 {token_ids.shape}")
        b, t = token_ids.shape
        if t > self.max_seq_len:
            raise ValueError(f"序列长度 {t} 超过 max_seq_len={self.max_seq_len}")

        x = self.tok_emb[token_ids] + self.pos_emb[:t][None, :, :]
        for blk in self.blocks:
            x = blk.forward(x, causal=True)
        x = self.final_norm.forward(x)
        logits = x @ self.lm_head
        assert logits.shape == (b, t, self.vocab_size)
        return logits

    def generate(self, prompt_ids: List[int], max_new_tokens: int = 4) -> List[int]:
        """自回归贪心解码：每步只取最后一个时间步 logits[:, -1, :] 采样下一个 token."""
        ids = list(prompt_ids)
        for _ in range(max_new_tokens):
            ctx = np.array([ids[-self.max_seq_len :]], dtype=np.int64)
            logits = self.forward(ctx)
            next_id = int(np.argmax(logits[0, -1, :]))
            ids.append(next_id)
        return ids


if __name__ == "__main__":
    lm = TinyTransformerLM(vocab_size=20, d_model=16, num_heads=4, d_ff=48, num_layers=2, max_seq_len=32)
    batch = np.array([[1, 4, 7, 2], [3, 5, 8, 9]], dtype=np.int64)
    assert lm.forward(batch).shape == (2, 4, 20)
    gen = lm.generate([1, 4, 7], max_new_tokens=5)
    assert len(gen) == 8
    print(f"Ch9 Step 6 self-test passed! generated={gen}")
