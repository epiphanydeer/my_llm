"""6.2 Step 2: 手写 KV Cache 增量解码内核与 PagedAttention 块表模拟."""

from __future__ import annotations

import math
from typing import Dict, List, Optional
import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def full_causal_attention(x: np.ndarray, wq: np.ndarray, wk: np.ndarray, wv: np.ndarray) -> np.ndarray:
    """无 KV Cache 朴素全量重算 (T, D) -> (T, D)."""
    t, d = x.shape
    q, k, v = x @ wq, x @ wk, x @ wv
    scores = (q @ k.T) / math.sqrt(d)
    mask = np.triu(np.full((t, t), -1e9), k=1)
    return softmax(scores + mask) @ v


class IncrementalKVCacheAttention:
    """单头手写 KV Cache 增量注意力内核."""

    def __init__(self, d_model: int, wq: np.ndarray, wk: np.ndarray, wv: np.ndarray) -> None:
        self.d_model = d_model
        self.wq, self.wk, self.wv = wq, wk, wv
        self.k_cache: Optional[np.ndarray] = None
        self.v_cache: Optional[np.ndarray] = None

    def prefill(self, x_prompt: np.ndarray) -> np.ndarray:
        out = full_causal_attention(x_prompt, self.wq, self.wk, self.wv)
        self.k_cache = x_prompt @ self.wk
        self.v_cache = x_prompt @ self.wv
        return out

    def decode_step(self, x_new: np.ndarray) -> np.ndarray:
        assert x_new.shape == (1, self.d_model)
        assert self.k_cache is not None and self.v_cache is not None

        q_t = x_new @ self.wq
        k_t = x_new @ self.wk
        v_t = x_new @ self.wv

        self.k_cache = np.concatenate([self.k_cache, k_t], axis=0)
        self.v_cache = np.concatenate([self.v_cache, v_t], axis=0)

        scores = (q_t @ self.k_cache.T) / math.sqrt(self.d_model)
        return softmax(scores) @ self.v_cache


class PagedBlockAllocator:
    """模拟 PagedAttention 物理块池与逻辑块表按需映射."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.block_size = block_size
        self.free_blocks: List[int] = list(range(num_blocks))
        self.block_tables: Dict[str, List[int]] = {}

    def allocate_tokens(self, req_id: str, num_tokens: int) -> List[int]:
        needed_blocks = math.ceil(num_tokens / self.block_size)
        table = self.block_tables.setdefault(req_id, [])
        while len(table) < needed_blocks:
            if not self.free_blocks:
                raise MemoryError("KV_POOL_OOM: 物理 KV 块耗尽，触发排队或抢占")
            table.append(self.free_blocks.pop(0))
        return table

    def free_request(self, req_id: str) -> None:
        for blk in self.block_tables.pop(req_id, []):
            self.free_blocks.append(blk)


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    d = 16
    wq, wk, wv = rng.normal(0, 0.2, (d, d)), rng.normal(0, 0.2, (d, d)), rng.normal(0, 0.2, (d, d))
    seq = rng.normal(0, 1.0, (6, d))

    ref_out = full_causal_attention(seq, wq, wk, wv)
    kv_attn = IncrementalKVCacheAttention(d, wq, wk, wv)
    inc_out = np.concatenate([kv_attn.prefill(seq[:4]), kv_attn.decode_step(seq[4:5]), kv_attn.decode_step(seq[5:6])], axis=0)
    assert np.allclose(ref_out, inc_out, atol=1e-6)
    print("Ch6 Step 2 self-test passed!")
