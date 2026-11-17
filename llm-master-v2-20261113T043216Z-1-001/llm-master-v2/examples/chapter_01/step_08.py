"""1.8 学习路线：复述、手写、排障与间隔复习调度器"""
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List


@dataclass
class KnowledgeNode:
    topic: str
    feynman_passed: bool
    code_tested: bool
    troubleshoot_logged: bool
    last_review: date
    interval_days: int = 1

    def next_review_date(self) -> date:
        if not (self.feynman_passed and self.code_tested):
            return self.last_review + timedelta(days=1)
        next_iv = min(30, self.interval_days * 2)
        return self.last_review + timedelta(days=next_iv)


def due_topics(nodes: List[KnowledgeNode], today: date) -> List[str]:
    return [n.topic for n in nodes if n.next_review_date() <= today]


if __name__ == "__main__":
    today = date(2026, 11, 13)
    nodes = [
        KnowledgeNode("1.4 采样参数", True, True, True, date(2026, 11, 10), interval_days=1),
        KnowledgeNode("1.6 Token口径", True, True, True, date(2026, 11, 10), interval_days=4),
    ]
    due = due_topics(nodes, today)
    assert due == ["1.4 采样参数"]
    print("[1.8 PASS] due today:", due)
