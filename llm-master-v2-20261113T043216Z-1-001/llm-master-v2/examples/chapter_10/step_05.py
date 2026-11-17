"""10.5 VLM 选型：动态切片与视觉 Token 预算估算器"""
import math
from typing import Dict


def estimate_vlm_visual_tokens(
    width: int,
    height: int,
    patch_size: int = 14,
    merge_stride: int = 2,
    max_tiles: int = 6,
    tile_size: int = 448,
) -> Dict[str, int]:
    """估算动态分辨率（AnyRes/Qwen-VL 风格）的视觉 Token 占用。"""
    patches_per_tile = (tile_size // patch_size) ** 2
    tokens_per_tile = patches_per_tile // (merge_stride ** 2)
    tiles_w = math.ceil(width / tile_size)
    tiles_h = math.ceil(height / tile_size)
    used_tiles = min(max_tiles, max(1, tiles_w * tiles_h))
    # 常见实现会额外附带 1 张全局缩略图
    total_visual_tokens = (used_tiles + 1) * tokens_per_tile
    return {
        "used_tiles": used_tiles,
        "tokens_per_tile": tokens_per_tile,
        "total_visual_tokens": total_visual_tokens,
    }


if __name__ == "__main__":
    low_res = estimate_vlm_visual_tokens(448, 448)
    high_res = estimate_vlm_visual_tokens(1920, 1080)
    assert high_res["total_visual_tokens"] > low_res["total_visual_tokens"]
    print("[10.5 PASS] 448p:", low_res, "1080p:", high_res)
