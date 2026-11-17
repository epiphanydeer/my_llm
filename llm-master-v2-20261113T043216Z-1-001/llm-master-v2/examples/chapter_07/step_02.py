"""7.2 Step 2: ViT 图像 Patch 切块展平与 Multimodal Projector 维度对齐投影.

在 Step 1 空间坐标校验基础上，理解 VLM 底层如何把二维图像 (H, W, C) 变成 LLM 可读的视觉 Token 序列。
"""

from __future__ import annotations

import numpy as np


class MicroVisionProjector:
    """模拟 ViT Patchify + Linear Projector 对齐至 LLM hidden_dim."""

    def __init__(self, patch_size: int = 16, channels: int = 3, llm_hidden_dim: int = 64) -> None:
        if patch_size <= 0 or channels <= 0 or llm_hidden_dim <= 0:
            raise ValueError("patch_size、channels 与 llm_hidden_dim 必须为正整数")
        self.patch_size = patch_size
        self.channels = channels
        self.llm_hidden_dim = llm_hidden_dim
        self.patch_dim = patch_size * patch_size * channels
        rng = np.random.default_rng(42)
        self.proj_weight = rng.normal(0.0, 0.02, size=(self.patch_dim, llm_hidden_dim)).astype(np.float32)

    def patchify(self, image: np.ndarray) -> np.ndarray:
        """将 (H, W, C) 图像切分为 (num_patches, P*P*C) 的 Patch 序列."""
        if image.ndim != 3 or image.shape[2] != self.channels:
            raise ValueError(f"图像必须为 (H, W, {self.channels})，当前为 {image.shape}")
        h, w, c = image.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(f"图像尺寸 ({h}, {w}) 必须能被 patch_size={self.patch_size} 整除")

        gh, gw = h // self.patch_size, w // self.patch_size
        patches = (
            image.reshape(gh, self.patch_size, gw, self.patch_size, c)
            .transpose(0, 2, 1, 3, 4)
            .reshape(gh * gw, self.patch_dim)
        )
        assert patches.shape == (gh * gw, self.patch_dim)
        return patches

    def forward(self, image: np.ndarray) -> np.ndarray:
        """输出对齐后的视觉 Token 序列: (num_patches, llm_hidden_dim)."""
        patches = self.patchify(image)
        visual_tokens = patches @ self.proj_weight
        assert visual_tokens.shape == (patches.shape[0], self.llm_hidden_dim)
        assert np.isfinite(visual_tokens).all(), "视觉 Token 出现 NaN/Inf"
        return visual_tokens


if __name__ == "__main__":
    proj = MicroVisionProjector(patch_size=16, channels=3, llm_hidden_dim=64)
    img = np.ones((32, 48, 3), dtype=np.float32)
    tokens = proj.forward(img)
    assert tokens.shape == (6, 64), f"期望 (6, 64)，实际 {tokens.shape}"
    print(f"MM Step 2 self-test passed! visual_tokens.shape={tokens.shape}")
