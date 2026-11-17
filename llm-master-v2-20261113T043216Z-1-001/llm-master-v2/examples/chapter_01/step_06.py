"""1.6 Token 用量：上下文、工具与缓存口径统计器"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class TurnUsage:
    system_tokens: int
    tools_schema_tokens: int
    history_tokens: int
    user_tokens: int
    cached_tokens: int
    completion_tokens: int

    @property
    def prompt_tokens(self) -> int:
        return self.system_tokens + self.tools_schema_tokens + self.history_tokens + self.user_tokens

    @property
    def uncached_input_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.cached_tokens)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def summary(self) -> Dict[str, float]:
        hit_rate = self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0
        return {
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "uncached_input_tokens": self.uncached_input_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cache_hit_rate": round(hit_rate, 4),
        }


if __name__ == "__main__":
    turn2 = TurnUsage(
        system_tokens=600,
        tools_schema_tokens=900,
        history_tokens=1200,
        user_tokens=100,
        cached_tokens=1500,  # System + Tools Schema 命中前缀缓存
        completion_tokens=320,
    )
    stats = turn2.summary()
    assert stats["prompt_tokens"] == 2800 and stats["uncached_input_tokens"] == 1300
    assert stats["total_tokens"] == 3120
    print("[1.6 PASS]", stats)
