"""Step 8: Mini MCP Server (基于 JSON-RPC 2.0 的标准化工具协议网关).

将 Step 3 的 ToolSandboxRegistry 封装为符合 MCP 思路的 JSON-RPC 2.0 协议服务端：
1. 支持标准 `tools/list` 能力发现（返回 name / description / inputSchema）。
2. 支持标准 `tools/call` 远程调用，统一透传鉴权上下文 (auth_context)。
3. 严格区分协议层错误 (-32601/-32602) 与工具业务执行结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass
class MCPToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    required_role: str
    handler: Callable[[Dict[str, Any], Dict[str, Any]], Any]


class MiniMCPServer:
    """极简 JSON-RPC 2.0 MCP Server."""

    def __init__(self) -> None:
        self._tools: Dict[str, MCPToolSpec] = {}

    def register_tool(self, spec: MCPToolSpec) -> None:
        self._tools[spec.name] = spec

    def handle_request(self, req: Dict[str, Any], auth_context: Dict[str, Any]) -> Dict[str, Any]:
        """处理单条 JSON-RPC 2.0 请求."""
        req_id = req.get("id")
        # [无需死背-脚手架] JSON-RPC 2.0 报文字段校验与标准错误码(-32600/-32601/-32602)可随时查表
        if req.get("jsonrpc") != "2.0" or "method" not in req:
            return self._error(req_id, -32600, "Invalid JSON-RPC 2.0 Request")

        method = req["method"]
        params = req.get("params") or {}

        # [必记设计决策-能力自发现] MCP 通过标准 tools/list 暴露 name / description / inputSchema，实现客户端与工具解耦
        if method == "tools/list":
            tools_payload: List[Dict[str, Any]] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.input_schema,
                }
                for t in self._tools.values()
            ]
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools_payload}}

        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments") or {}
            if tool_name not in self._tools:
                return self._error(req_id, -32601, f"Tool not found: {tool_name}")

            spec = self._tools[tool_name]
            # [必记设计决策-鉴权上下文隔离] 权限角色必须从服务端透传的 auth_context 读取，绝不放在模型可控的 params 里
            user_role = auth_context.get("role", "guest")
            if spec.required_role == "admin" and user_role != "admin":
                return self._error(req_id, -32003, f"Permission denied for role={user_role}")

            required_fields = spec.input_schema.get("required", [])
            missing = [f for f in required_fields if f not in arguments]
            if missing:
                return self._error(req_id, -32602, f"Missing required params: {missing}")

            # [必记设计决策-协议与业务分层] 区分协议层错误(error 对象)与工具执行结果(content + isError)
            output = spec.handler(arguments, auth_context)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "json", "json": output}], "isError": False},
            }

        return self._error(req_id, -32601, f"Method not found: {method}")

    @staticmethod
    def _error(req_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


if __name__ == "__main__":
    server = MiniMCPServer()
    server.register_tool(
        MCPToolSpec(
            name="query_ticket",
            description="查询企业工单详情",
            input_schema={
                "type": "object",
                "properties": {"ticket_id": {"type": "string"}},
                "required": ["ticket_id"],
            },
            required_role="support",
            handler=lambda args, ctx: {"ticket_id": args["ticket_id"], "operator": ctx["user_id"]},
        )
    )
    server.register_tool(
        MCPToolSpec(
            name="restart_service",
            description="重启生产微服务（仅管理员）",
            input_schema={
                "type": "object",
                "properties": {"service": {"type": "string"}},
                "required": ["service"],
            },
            required_role="admin",
            handler=lambda args, ctx: {"service": args["service"], "status": "RESTARTED"},
        )
    )

    list_resp = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, {"role": "support"})
    assert len(list_resp["result"]["tools"]) == 2

    call_ok = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "query_ticket", "arguments": {"ticket_id": "TK-101"}},
        },
        {"user_id": "u_007", "role": "support"},
    )
    assert call_ok["result"]["content"][0]["json"]["operator"] == "u_007"

    call_denied = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "restart_service", "arguments": {"service": "auth-api"}},
        },
        {"user_id": "u_007", "role": "support"},
    )
    assert call_denied["error"]["code"] == -32003
    print("Step 8 self-test passed!")
