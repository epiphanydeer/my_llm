"""1.3 从训练到推理：Token、BPE 压缩率对比与长度估算器"""
from typing import Dict


def estimate_tokens_and_ratio(text: str, chars_per_token_en: float = 4.0, chars_per_token_zh: float = 1.6) -> Dict[str, float]:
    """按工程经验口径估算中英混合文本 Token 数与压缩率（Bytes / Token）。"""
    if not text:
        return {"estimated_tokens": 0, "utf8_bytes": 0, "bytes_per_token": 0.0}
    zh_chars = sum(1 for ch in text if "一" <= ch <= "鿿")
    other_chars = len(text) - zh_chars
    est_tokens = max(1, round(zh_chars / chars_per_token_zh + other_chars / chars_per_token_en))
    utf8_bytes = len(text.encode("utf-8"))
    return {
        "zh_chars": zh_chars,
        "other_chars": other_chars,
        "estimated_tokens": est_tokens,
        "utf8_bytes": utf8_bytes,
        "bytes_per_token": round(utf8_bytes / est_tokens, 2),
        "chars_per_token": round(len(text) / est_tokens, 2),
    }


if __name__ == "__main__":
    en_res = estimate_tokens_and_ratio("Transformer attention mechanisms scale quadratically with sequence length.")
    zh_res = estimate_tokens_and_ratio("大模型应用在长上下文场景下需要重点关注显存与首字延迟。")
    assert en_res["chars_per_token"] > zh_res["chars_per_token"]
    print("[1.3 PASS] EN:", en_res, "ZH:", zh_res)
