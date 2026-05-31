"""OpenAI-tool shim for the local backend (ARCHITECTURE.md §9.3).

The `darwin-mcp` server is attached natively by both the Claude backend and OpenHands, so the
**same** tool surface drives every backend (§9.4). This shim is the *fallback* for a harness
that speaks **only** OpenAI function-calling: it enumerates the MCP tool schemas, exposes them
as OpenAI `tools=[...]` function specs, and on a function-call response forwards the call to the
MCP server and returns the result as the tool message. It is deliberately thin and
transport-free — the actual call is delegated to an injected `invoke(name, arguments)` callable
— so the translation logic is unit-testable without a live MCP server or model.

MCP tools are duck-typed: anything with `.name`, `.description`, and `.inputSchema` (a JSON
Schema dict) works — which is exactly what FastMCP's `list_tools()` returns.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable

# A tool invoker: given a tool name + parsed arguments, run it and return a JSON-able result.
ToolInvoker = Callable[[str, dict[str, Any]], Any]


def mcp_tool_to_openai(name: str, description: str, input_schema: dict | None) -> dict:
    """Translate one MCP tool definition into an OpenAI function spec."""
    parameters = input_schema or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or "",
            "parameters": parameters,
        },
    }


def mcp_tools_to_openai(tools: Iterable[Any]) -> list[dict]:
    """Translate MCP tool definitions (duck-typed: .name/.description/.inputSchema)."""
    return [
        mcp_tool_to_openai(t.name, getattr(t, "description", ""), getattr(t, "inputSchema", None))
        for t in tools
    ]


def parse_tool_call(tool_call: Any) -> tuple[str, str, dict[str, Any]]:
    """Extract (call_id, name, arguments) from an OpenAI tool-call (dict or SDK object).

    `arguments` arrives as a JSON string in the wire format; an empty/blank string parses to
    `{}`. Already-parsed dict arguments pass through.
    """
    func = _get(tool_call, "function", {})
    name = _get(func, "name", "")
    call_id = _get(tool_call, "id", "")
    raw_args = _get(func, "arguments", "{}")
    if isinstance(raw_args, dict):
        args = raw_args
    elif isinstance(raw_args, str):
        args = json.loads(raw_args) if raw_args.strip() else {}
    else:
        args = {}
    return call_id, name, args


def tool_result_message(call_id: str, name: str, result: Any) -> dict:
    """Wrap a tool result as an OpenAI `role:"tool"` message (content is JSON text)."""
    if isinstance(result, str):
        content = result
    else:
        content = json.dumps(result, default=str)
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}


class OpenAIToolShim:
    """Exposes MCP tools as OpenAI function specs and dispatches calls back to MCP."""

    def __init__(self, specs: list[dict], invoke: ToolInvoker):
        self._specs = specs
        self._invoke = invoke

    @classmethod
    def from_mcp(cls, tools: Iterable[Any], invoke: ToolInvoker) -> "OpenAIToolShim":
        """Build from MCP tool definitions + an invoker that forwards to the MCP server."""
        return cls(mcp_tools_to_openai(tools), invoke)

    def specs(self) -> list[dict]:
        """The `tools=[...]` value to pass to an OpenAI-compatible chat completion."""
        return list(self._specs)

    def tool_names(self) -> set[str]:
        return {s["function"]["name"] for s in self._specs}

    def handle(self, tool_call: Any) -> dict:
        """Dispatch one OpenAI tool-call to MCP and return the tool-result message.

        An unknown tool or a raising invoker yields a structured error in the tool message
        (rather than crashing the harness loop) so the model can read it and adjust.
        """
        call_id, name, args = parse_tool_call(tool_call)
        if name not in self.tool_names():
            return tool_result_message(call_id, name, {"error": f"unknown tool {name!r}"})
        try:
            result = self._invoke(name, args)
        except Exception as exc:  # surface to the model, don't kill the loop
            return tool_result_message(call_id, name, {"error": str(exc)})
        return tool_result_message(call_id, name, result)


def _get(obj: Any, key: str, default: Any) -> Any:
    """Read `key` from a dict or an attribute from an SDK object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
