"""1.8 学习路线：复述、手写、排障与间隔复习调度器"""
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List


@dataclass
class KnowledgeNode:  # [无需死背-脚手架] 知识节点状态数据结构
    topic: str
    feynman_passed: bool
    code_tested: bool
    troubleshoot_logged: bool
    last_review: date
    interval_days: int = 1

    def next_review_date(self) -> date:
        if not (self.feynman_passed and self.code_tested):  # [必记设计决策-掌握度双门禁] 复述与手写跑通任一未过则次日重测
            return self.last_review + timedelta(days=1)
        next_iv = min(30, self.interval_days * 2)  # [必背手撕-指数间隔封顶] 通过后间隔翻倍并封顶30天
        return self.last_review + timedelta(days=next_iv)


def due_topics(nodes: List[KnowledgeNode], today: date) -> List[str]:
    return [n.topic for n in nodes if n.next_review_date() <= today]  # [必背手撕-到期任务筛选]


if __name__ == "__main__":  # [无需死背-脚手架] 日期样例与断言
    today = date(2026, 11, 13)
    nodes = [
        KnowledgeNode("1.4 采样参数", True, True, True, date(2026, 11, 10), interval_days=1),
        KnowledgeNode("1.6 Token口径", True, True, True, date(2026, 11, 10), interval_days=4),
    ]
    due = due_topics(nodes, today)
    assert due == ["1.4 采样参数"]
    print("[1.8 PASS] due today:", due)
