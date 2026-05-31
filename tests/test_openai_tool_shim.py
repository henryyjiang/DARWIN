"""OpenAI-tool shim: MCP <-> OpenAI translation + dispatch (ARCHITECTURE.md §9.3)."""

from dataclasses import dataclass

from darwin.mutation_agent.openai_tool_shim import (
    OpenAIToolShim,
    mcp_tool_to_openai,
    mcp_tools_to_openai,
    parse_tool_call,
    tool_result_message,
)


@dataclass
class FakeMCPTool:
    name: str
    description: str
    inputSchema: dict | None


def test_mcp_tool_to_openai_shape():
    spec = mcp_tool_to_openai(
        "smoke_run", "run the smoke test", {"type": "object", "properties": {"summary": {"type": "string"}}}
    )
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "smoke_run"
    assert spec["function"]["parameters"]["properties"]["summary"]["type"] == "string"


def test_mcp_tool_missing_schema_defaults_to_empty_object():
    spec = mcp_tool_to_openai("finalize", "end early", None)
    assert spec["function"]["parameters"] == {"type": "object", "properties": {}}


def test_translate_tool_list():
    tools = [
        FakeMCPTool("memory_recent", "recent memory", {"type": "object", "properties": {}}),
        FakeMCPTool("finalize", "end", None),
    ]
    specs = mcp_tools_to_openai(tools)
    assert {s["function"]["name"] for s in specs} == {"memory_recent", "finalize"}


def test_parse_tool_call_json_string_arguments():
    call = {"id": "c1", "function": {"name": "cost_report", "arguments": '{"amount_usd": 2.0, "reason": "x"}'}}
    cid, name, args = parse_tool_call(call)
    assert cid == "c1"
    assert name == "cost_report"
    assert args == {"amount_usd": 2.0, "reason": "x"}


def test_parse_tool_call_empty_and_dict_arguments():
    _, _, empty = parse_tool_call({"id": "c", "function": {"name": "finalize", "arguments": ""}})
    assert empty == {}
    _, _, passthrough = parse_tool_call({"id": "c", "function": {"name": "f", "arguments": {"a": 1}}})
    assert passthrough == {"a": 1}


def test_tool_result_message_serializes_non_string():
    msg = tool_result_message("c1", "smoke_run", {"passed": True})
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "c1"
    assert msg["content"] == '{"passed": true}'
    # strings pass through unchanged
    assert tool_result_message("c2", "x", "hello")["content"] == "hello"


def test_shim_dispatches_to_invoker():
    tools = [FakeMCPTool("smoke_run", "smoke", {"type": "object", "properties": {}})]
    calls = []

    def invoke(name, args):
        calls.append((name, args))
        return {"passed": True, "committed": True}

    shim = OpenAIToolShim.from_mcp(tools, invoke)
    assert shim.tool_names() == {"smoke_run"}
    msg = shim.handle({"id": "c1", "function": {"name": "smoke_run", "arguments": '{"summary":"s"}'}})
    assert calls == [("smoke_run", {"summary": "s"})]
    assert '"passed": true' in msg["content"]


def test_shim_unknown_tool_returns_error_message():
    shim = OpenAIToolShim.from_mcp([FakeMCPTool("finalize", "", None)], lambda n, a: None)
    msg = shim.handle({"id": "c1", "function": {"name": "nope", "arguments": "{}"}})
    assert "unknown tool" in msg["content"]


def test_shim_invoker_error_is_caught():
    def boom(name, args):
        raise RuntimeError("mcp down")

    shim = OpenAIToolShim.from_mcp([FakeMCPTool("finalize", "", None)], boom)
    msg = shim.handle({"id": "c1", "function": {"name": "finalize", "arguments": "{}"}})
    assert "mcp down" in msg["content"]
