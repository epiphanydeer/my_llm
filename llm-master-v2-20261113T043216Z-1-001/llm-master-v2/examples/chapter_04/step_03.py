"""4.3 Step 3: 在 Step 2 基础上新增 ToolRegistry 与权限沙箱（Schema 校验 + 读写分级 + 防越权）"""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    mode: str  # "READ" | "WRITE"
    required_fields: tuple[str, ...]
    handler: Callable[[dict[str, object]], str]


class ToolSandboxRegistry:
    """工具注册与安全守门员：模型只能提请求，系统负责校验 Schema 与权限边界。"""
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def execute(self, tool_name: str, args: dict[str, object], role: str, confirmed: bool = False) -> dict[str, object]:
        if tool_name not in self._tools:
            return {"ok": False, "error": f"TOOL_NOT_FOUND: {tool_name}"}
        spec = self._tools[tool_name]
        # 1. 读写权限前置拦截
        if spec.mode == "WRITE" and role not in ("admin", "oncall"):
            return {"ok": False, "error": "PERMISSION_DENIED: 无权执行写操作"}
        if spec.mode == "WRITE" and not confirmed:
            return {"ok": False, "error": "CONFIRMATION_REQUIRED: 高风险写操作需人工确认"}
        # 2. 必填 Schema 校验
        missing = [f for f in spec.required_fields if f not in args or not str(args[f]).strip()]
        if missing:
            return {"ok": False, "error": f"SCHEMA_ERROR: 缺少必填参数 {missing}"}
        # 3. 路径穿越沙箱拦截
        if "path" in args and (".." in str(args["path"]) or str(args["path"]).startswith("/")):
            return {"ok": False, "error": "SANDBOX_VIOLATION: 禁止访问工作区外路径"}
        return {"ok": True, "result": spec.handler(args)}


if __name__ == "__main__":
    reg = ToolSandboxRegistry()
    reg.register(ToolSpec("read_log", "READ", ("path",), lambda a: f"logs:{a['path']}"))
    reg.register(ToolSpec("restart_svc", "WRITE", ("service",), lambda a: f"restarted:{a['service']}"))

    assert reg.execute("read_log", {"path": "../etc/passwd"}, "viewer")["error"].startswith("SANDBOX_VIOLATION")
    assert reg.execute("restart_svc", {"service": "pay"}, "viewer")["error"].startswith("PERMISSION_DENIED")
    assert reg.execute("restart_svc", {"service": "pay"}, "oncall", confirmed=False)["error"].startswith("CONFIRMATION_REQUIRED")
    assert reg.execute("restart_svc", {"service": "pay"}, "oncall", confirmed=True)["ok"] is True
    print("Agent Step 3 self-test passed!")
