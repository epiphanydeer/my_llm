"""7.7 Step 7: 视频时序自适应抽帧、PTS 时间戳对齐与 BGR->RGB 转换.

解决视频理解三大工程坑点：
1. 变帧率 (VFR) 场景下不能只按帧号 * 1/fps 算时间，必须校验 PTS 时间戳严格单调递增。
2. OpenCV 默认读入为 BGR 通道序，送入 VLM 前必须显式反转为 RGB。
3. 严格限制最大送入帧数 budget，防止长视频炸穿显存与账单。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass(frozen=True)
class RawVideoFrame:
    pts_sec: float
    bgr_image: np.ndarray


def sample_video_frames(
    frames: List[RawVideoFrame],
    min_interval_sec: float = 1.0,
    max_frames: int = 8,
) -> List[dict[str, object]]:
    """按 PTS 时间间隔自适应抽帧，校验单调递增并执行 BGR -> RGB 转换."""
    if not frames:
        raise ValueError("视频帧序列不能为空")
    if min_interval_sec <= 0 or max_frames <= 0:
        raise ValueError("min_interval_sec 与 max_frames 必须为正数")

    prev_pts = -1.0
    sampled: List[dict[str, object]] = []
    last_kept_pts = float("-inf")

    for f in frames:
        if f.pts_sec <= prev_pts:
            raise ValueError(f"PTS_NOT_MONOTONIC: 当前 {f.pts_sec} <= 上一帧 {prev_pts}")
        prev_pts = f.pts_sec

        if f.bgr_image.ndim != 3 or f.bgr_image.shape[2] != 3:
            raise ValueError(f"帧图像必须为 (H, W, 3)，当前 {f.bgr_image.shape}")

        if f.pts_sec - last_kept_pts >= min_interval_sec:
            rgb_img = f.bgr_image[..., ::-1].copy()
            sampled.append({"pts_sec": round(f.pts_sec, 3), "rgb": rgb_img})
            last_kept_pts = f.pts_sec
            if len(sampled) >= max_frames:
                break

    assert len(sampled) <= max_frames
    assert all(sampled[i]["pts_sec"] < sampled[i + 1]["pts_sec"] for i in range(len(sampled) - 1))
    return sampled


if __name__ == "__main__":
    bgr = np.zeros((16, 16, 3), dtype=np.uint8)
    bgr[..., 0] = 255  # OpenCV Blue channel
    stream = [RawVideoFrame(pts_sec=t * 0.5, bgr_image=bgr) for t in range(10)]
    out = sample_video_frames(stream, min_interval_sec=1.0, max_frames=3)
    assert len(out) == 3 and out[0]["rgb"][0, 0, 2] == 255
    print("MM Step 7 self-test passed!")
