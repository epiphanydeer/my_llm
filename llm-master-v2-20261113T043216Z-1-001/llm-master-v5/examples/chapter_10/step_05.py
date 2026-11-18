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
    patches_per_tile = (tile_size // patch_size) ** 2  # [必背手撕-ViT分块Patch数计算]
    tokens_per_tile = patches_per_tile // (merge_stride ** 2)  # [必记设计决策-相邻Token合并降采样] 通过 2x2 merge 压缩4倍视觉序列长度
    tiles_w = math.ceil(width / tile_size)
    tiles_h = math.ceil(height / tile_size)
    used_tiles = min(max_tiles, max(1, tiles_w * tiles_h))  # [必记设计决策-动态切片上限封顶] 限制 max_tiles 防高分辨率爆显存
    # 常见实现会额外附带 1 张全局缩略图
    total_visual_tokens = (used_tiles + 1) * tokens_per_tile  # [必背手撕-视觉Token总预算] 切片数加1张全局缩略图
    return {
        "used_tiles": used_tiles,
        "tokens_per_tile": tokens_per_tile,
        "total_visual_tokens": total_visual_tokens,
    }


if __name__ == "__main__":  # [无需死背-脚手架] 448p与1080p对比测试
    low_res = estimate_vlm_visual_tokens(448, 448)
    high_res = estimate_vlm_visual_tokens(1920, 1080)
    assert high_res["total_visual_tokens"] > low_res["total_visual_tokens"]
    print("[10.5 PASS] 448p:", low_res, "1080p:", high_res)
