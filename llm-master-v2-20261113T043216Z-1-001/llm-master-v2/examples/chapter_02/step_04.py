"""2.4 结构化输出：JSON Schema 校验 + 业务规则二次校验与自修复"""
import json
from typing import Any, Callable, Dict, Tuple


def validate_hazard_json(raw_text: str) -> Tuple[bool, Dict[str, Any], str]:
    """同时校验 JSON 语法、字段类型以及跨字段业务规则。"""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return False, {}, f"JSON语法错误: {e}"

    required = {"hazard_level", "location", "rectify_hours"}
    if not required.issubset(data.keys()):
        return False, data, f"缺少必填字段: {required - set(data.keys())}"

    if data["hazard_level"] not in {"high", "medium", "low"}:
        return False, data, "hazard_level 必须为 high/medium/low"

    # 业务规则校验：高风险隐患整改时限不得超过 4 小时（Schema 往往查不出这种跨字段逻辑）
    if data["hazard_level"] == "high" and int(data["rectify_hours"]) > 4:
        return False, data, "业务规则冲突：high 级别隐患 rectify_hours 必须 <= 4"

    return True, data, "OK"


def extract_with_repair(llm_fn: Callable[[str], str], prompt: str, max_repair: int = 2) -> Dict[str, Any]:
    cur_prompt = prompt
    for _ in range(max_repair + 1):
        raw = llm_fn(cur_prompt)
        ok, parsed, err = validate_hazard_json(raw)
        if ok:
            return parsed
        cur_prompt = f"{prompt}\n上次输出校验失败，原因：{err}，请修正并仅输出合法 JSON。"
    raise ValueError("超出最大修复次数仍未通过校验")


if __name__ == "__main__":
    responses = iter([
        '{"hazard_level": "high", "location": "破碎站", "rectify_hours": 24}',  # Schema对但业务错
        '{"hazard_level": "high", "location": "破碎站", "rectify_hours": 2}',
    ])
    result = extract_with_repair(lambda _: next(responses), "提取隐患")
    assert result["rectify_hours"] == 2
    print("[2.4 PASS]", result)
