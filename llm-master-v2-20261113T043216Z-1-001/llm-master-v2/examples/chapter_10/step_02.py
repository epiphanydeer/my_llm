"""10.2 Encoder、Decoder 与 Encoder-Decoder 的注意力可见性掩码对比"""
import numpy as np


def build_visibility_masks(seq_len: int) -> dict:
    """生成三种经典架构的 0/1 注意力可见性矩阵（1=可见，0=屏蔽）。"""
    encoder_mask = np.ones((seq_len, seq_len), dtype=int)
    decoder_causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=int))
    prefix_lm_mask = decoder_causal_mask.copy()
    prefix_len = seq_len // 2
    prefix_lm_mask[:prefix_len, :prefix_len] = 1  # 前缀双向可见，后缀因果自回归
    return {
        "encoder_bidirectional": encoder_mask,
        "decoder_causal": decoder_causal_mask,
        "prefix_lm": prefix_lm_mask,
    }


if __name__ == "__main__":
    masks = build_visibility_masks(4)
    assert masks["encoder_bidirectional"].sum() == 16
    assert masks["decoder_causal"][0, 1] == 0 and masks["decoder_causal"][2, 1] == 1
    assert masks["prefix_lm"][0, 1] == 1 and masks["prefix_lm"][2, 3] == 0
    print("[10.2 PASS] masks verified")
