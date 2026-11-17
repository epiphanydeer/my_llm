"""1.9 模型、应用与工具运行环境：沙箱与权限边界模拟器"""
from dataclasses import dataclass
from typing import Set


@dataclass
class ToolRuntimeGuard:
    allowed_roots: Set[str]
    require_approval_tools: Set[str]

    def authorize(self, tool_name: str, target_path: str, user_approved: bool = False) -> str:
        if not any(target_path.startswith(root) for root in self.allowed_roots):
            return "DENY_OUT_OF_SANDBOX"
        if tool_name in self.require_approval_tools and not user_approved:
            return "REQUIRE_USER_APPROVAL"
        return "ALLOW"


if __name__ == "__main__":
    guard = ToolRuntimeGuard(allowed_roots={"/workspace/project"}, require_approval_tools={"shell_exec", "git_push"})
    assert guard.authorize("read_file", "/workspace/project/app.py") == "ALLOW"
    assert guard.authorize("read_file", "/etc/passwd") == "DENY_OUT_OF_SANDBOX"
    assert guard.authorize("shell_exec", "/workspace/project/run.sh", user_approved=False) == "REQUIRE_USER_APPROVAL"
    assert guard.authorize("shell_exec", "/workspace/project/run.sh", user_approved=True) == "ALLOW"
    print("[1.9 PASS] sandbox boundary verified")
