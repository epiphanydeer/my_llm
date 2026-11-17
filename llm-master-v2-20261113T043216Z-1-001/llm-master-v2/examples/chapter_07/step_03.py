"""7.3 Step 3: Processor、字节级坏图拦截与 <image> 占位符强一致校验.

在 Step 2 基础上新增输入流水线守卫：
1. 拦截空字节或非 PNG/JPEG 魔数的坏图输入。
2. 渲染多模态 Chat Template，并强制断言模板中的 `<image>` 占位符数量与实际视觉 Token 序列数一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
from step_02 import MicroVisionProjector


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


def validate_image_bytes(raw_bytes: bytes) -> str:
    """字节级坏图拦截：校验非空与标准图像魔数 (PNG/JPEG)."""
    if not raw_bytes or len(raw_bytes) < 8:
        raise ValueError("BAD_IMAGE: 图像字节流为空或过短")
    if raw_bytes.startswith(PNG_MAGIC):
        return "PNG"
    if raw_bytes.startswith(JPEG_MAGIC):
        return "JPEG"
    raise ValueError("BAD_IMAGE: 不支持的图像格式或文件头已损坏")


@dataclass
class SafeMultimodalProcessor:
    """模拟 HF AutoProcessor：统一校验图像、渲染模板并对齐视觉占位符."""

    projector: MicroVisionProjector
    image_token: str = "<image>"

    def prepare_inputs(
        self,
        raw_bytes_list: List[bytes],
        images_rgb: List[np.ndarray],
        prompt_template: str,
    ) -> dict[str, object]:
        if len(raw_bytes_list) != len(images_rgb):
            raise ValueError("图像字节列表与解码数组数量不一致")

        formats = [validate_image_bytes(b) for b in raw_bytes_list]
        placeholder_count = prompt_template.count(self.image_token)
        if placeholder_count != len(images_rgb):
            raise ValueError(
                f"TEMPLATE_MISMATCH: 模板中 {self.image_token} 数量 ({placeholder_count}) "
                f"与传入图片数 ({len(images_rgb)}) 不一致！"
            )

        visual_token_batches = [self.projector.forward(img) for img in images_rgb]
        total_visual_tokens = sum(vt.shape[0] for vt in visual_token_batches)
        return {
            "formats": formats,
            "prompt": prompt_template,
            "num_images": len(images_rgb),
            "total_visual_tokens": total_visual_tokens,
        }


if __name__ == "__main__":
    proc = SafeMultimodalProcessor(MicroVisionProjector(patch_size=16, channels=3, llm_hidden_dim=64))
    fake_png = PNG_MAGIC + b"\x00" * 32
    img = np.zeros((32, 32, 3), dtype=np.float32)

    out = proc.prepare_inputs([fake_png], [img], "请检查图中安全带：<image>")
    assert out["formats"] == ["PNG"] and out["total_visual_tokens"] == 4
    print("MM Step 3 self-test passed!")
