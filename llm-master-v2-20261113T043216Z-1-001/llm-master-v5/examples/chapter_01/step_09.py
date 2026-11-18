"""1.9 模型、应用与工具运行环境：沙箱与权限边界模拟器"""
from dataclasses import dataclass
from typing import Set


@dataclass
class ToolRuntimeGuard:  # [无需死背-脚手架] 权限守卫配置类
    allowed_roots: Set[str]
    require_approval_tools: Set[str]

    def authorize(self, tool_name: str, target_path: str, user_approved: bool = False) -> str:
        if not any(target_path.startswith(root) for root in self.allowed_roots):  # [必背手撕-沙箱路径白名单校验] 越界路径直接拒绝
            return "DENY_OUT_OF_SANDBOX"
        if tool_name in self.require_approval_tools and not user_approved:  # [必记设计决策-高危工具二次确认] 读写越界外，执行/推送需人工审批
            return "REQUIRE_USER_APPROVAL"
        return "ALLOW"


if __name__ == "__main__":  # [无需死背-脚手架] 四种权限路径断言
    guard = ToolRuntimeGuard(allowed_roots={"/workspace/project"}, require_approval_tools={"shell_exec", "git_push"})
    assert guard.authorize("read_file", "/workspace/project/app.py") == "ALLOW"
    assert guard.authorize("read_file", "/etc/passwd") == "DENY_OUT_OF_SANDBOX"
    assert guard.authorize("shell_exec", "/workspace/project/run.sh", user_approved=False) == "REQUIRE_USER_APPROVAL"
    assert guard.authorize("shell_exec", "/workspace/project/run.sh", user_approved=True) == "ALLOW"
    print("[1.9 PASS] sandbox boundary verified")
