"""6.3 Step 3: 对称量化与非对称量化算子及反量化保真度审计 (MSE / 余弦相似度)."""

from __future__ import annotations

from typing import Dict
import numpy as np


def symmetric_quantize_int8(w: np.ndarray) -> tuple[np.ndarray, float]:
    """对称 INT8 量化 (zero_point = 0, q in [-127, 127])."""
    max_abs = float(np.max(np.abs(w)))
    scale = max(max_abs / 127.0, 1e-12)
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, scale


def asymmetric_quantize_uint8(act: np.ndarray) -> tuple[np.ndarray, float, int]:
    """非对称 UINT8 量化 (适配 ReLU/GELU 偏斜激活值, q in [0, 255])."""
    w_min, w_max = float(np.min(act)), float(np.max(act))
    scale = max((w_max - w_min) / 255.0, 1e-12)
    zp = int(np.clip(round(-w_min / scale), 0, 255))
    q = np.clip(np.round(act / scale) + zp, 0, 255).astype(np.uint8)
    return q, scale, zp


def audit_fidelity(orig: np.ndarray, recon: np.ndarray) -> Dict[str, float]:
    mse = float(np.mean((orig - recon) ** 2))
    denom = float(np.linalg.norm(orig) * np.linalg.norm(recon))
    cos_sim = float(np.sum(orig * recon) / max(denom, 1e-12))
    return {"mse": round(mse, 8), "cosine_sim": round(cos_sim, 6)}


if __name__ == "__main__":
    rng = np.random.default_rng(10)
    weights = rng.normal(0.0, 0.5, size=(128, 128)).astype(np.float32)
    q_w, s_w = symmetric_quantize_int8(weights)
    assert audit_fidelity(weights, q_w.astype(np.float32) * s_w)["cosine_sim"] > 0.999

    acts = np.maximum(0.0, rng.normal(1.5, 0.8, size=(64, 64))).astype(np.float32)
    q_a, s_a, zp_a = asymmetric_quantize_uint8(acts)
    assert audit_fidelity(acts, (q_a.astype(np.float32) - zp_a) * s_a)["cosine_sim"] > 0.999
    print("Ch6 Step 3 self-test passed!")
