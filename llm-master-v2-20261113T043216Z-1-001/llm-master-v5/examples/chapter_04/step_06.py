"""Step 6: Resilient Executor (失败恢复：指数退避重试 + Schema 自愈 + 幂等去重).

在 Step 3/4 工具执行层基础上新增 ResilientExecutor：
1. 瞬态故障（如 429/503/Timeout）执行指数退避 + 随机抖动重试。
2. 参数校验失败 (SchemaError) 时构造结构化纠偏提示，触发有限次自愈重调。
3. 写操作强制要求幂等键 (idempotency_key)，防止网络超时重试导致重复建单/重复退款。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set


class TransientToolError(RuntimeError):
    """可重试的瞬态错误（如超时、限流 429）."""


class SchemaValidationError(ValueError):
    """参数 Schema 校验错误（需反馈给 LLM 自愈）."""


@dataclass
class ExecutionResult:
    """工具执行的标准化返回."""

    ok: bool
    output: Any = None
    attempts: int = 0
    deduplicated: bool = False
    error_hint: Optional[str] = None


@dataclass
class ResilientExecutor:
    """带退避重试、Schema 自愈和幂等去重的工业级工具执行器."""

    max_retries: int = 2
    base_delay_sec: float = 0.01  # 测试中使用短延迟，生产通常 0.5~1.0s
    _executed_idempotency_keys: Set[str] = field(default_factory=set)
    _idempotency_cache: Dict[str, Any] = field(default_factory=dict)

    def execute(
        self,
        tool_name: str,
        fn: Callable[[Dict[str, Any]], Any],
        args: Dict[str, Any],
        is_write: bool = False,
        idempotency_key: Optional[str] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> ExecutionResult:
        """执行工具调用，处理幂等性、瞬态重试与结构化错误反馈."""
        # [必记设计决策-写操作幂等门禁] 所有写操作强制要求非空 idempotency_key，命中缓存直接返回历史结果并标 deduplicated=True
        if is_write:
            if not idempotency_key or not idempotency_key.strip():
                return ExecutionResult(
                    ok=False,
                    attempts=0,
                    error_hint=f"写操作 {tool_name} 必须提供非空 idempotency_key",
                )
            if idempotency_key in self._executed_idempotency_keys:
                return ExecutionResult(
                    ok=True,
                    output=self._idempotency_cache[idempotency_key],
                    attempts=0,
                    deduplicated=True,
                )

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 2):
            try:
                res = fn(args)
                if is_write and idempotency_key:
                    self._executed_idempotency_keys.add(idempotency_key)
                    self._idempotency_cache[idempotency_key] = res
                return ExecutionResult(ok=True, output=res, attempts=attempt)
            # [必记设计决策-错误分层] 参数 Schema 错误重试无意义，必须立即中断重试并返回结构化纠偏提示给 LLM 自愈
            except SchemaValidationError as exc:
                # Schema 错误重试无意义，必须立即返回纠偏提示让模型修正参数
                return ExecutionResult(
                    ok=False,
                    attempts=attempt,
                    error_hint=f"[SCHEMA_ERROR] 工具 {tool_name} 参数不合法: {exc}。请修正参数后重试。",
                )
            # [必背手撕-指数退避加抖动] 仅对 429/503/超时等瞬态错误做 base * 2^(attempt-1) + jitter 退避重试
            except TransientToolError as exc:
                last_error = exc
                if attempt <= self.max_retries:
                    # [必记设计决策-防惊群效应] 必须叠加随机 jitter 抖动，防止多实例同时重试打垮下游
                    jitter = random.uniform(0.0, self.base_delay_sec)
                    delay = self.base_delay_sec * (2 ** (attempt - 1)) + jitter
                    sleep_fn(delay)
                    continue
                break

        return ExecutionResult(
            ok=False,
            attempts=self.max_retries + 1,
            error_hint=f"[TRANSIENT_EXHAUSTED] 工具 {tool_name} 重试耗尽: {last_error}",
        )


if __name__ == "__main__":
    executor = ResilientExecutor(max_retries=2, base_delay_sec=0.005)
    call_counter = {"flaky": 0, "write": 0}

    def flaky_query(args: Dict[str, Any]) -> Dict[str, Any]:
        call_counter["flaky"] += 1
        if call_counter["flaky"] < 3:
            raise TransientToolError("503 Service Unavailable")
        return {"ticket_id": args["ticket_id"], "status": "OPEN"}

    def strict_create_refund(args: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(args.get("amount"), (int, float)) or args["amount"] <= 0:
            raise SchemaValidationError("amount 必须为正数")
        call_counter["write"] += 1
        return {"refund_id": "RF-900", "amount": args["amount"]}

    r1 = executor.execute("query_ticket", flaky_query, {"ticket_id": "TK-101"})
    assert r1.ok and r1.attempts == 3, f"应在第 3 次重试成功: {r1}"

    r2 = executor.execute(
        "create_refund",
        strict_create_refund,
        {"amount": -50},
        is_write=True,
        idempotency_key="idem-001",
    )
    assert not r2.ok and r2.attempts == 1 and "SCHEMA_ERROR" in (r2.error_hint or "")

    r3 = executor.execute(
        "create_refund",
        strict_create_refund,
        {"amount": 50},
        is_write=True,
        idempotency_key="idem-001",
    )
    r4 = executor.execute(
        "create_refund",
        strict_create_refund,
        {"amount": 50},
        is_write=True,
        idempotency_key="idem-001",
    )
    assert r3.ok and not r3.deduplicated
    assert r4.ok and r4.deduplicated and call_counter["write"] == 1
    print("Step 6 self-test passed!")
