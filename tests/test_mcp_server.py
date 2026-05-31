"""Integration test: the FastMCP server builds and registers the memory tools (§9.3)."""

import asyncio
import json

from darwin.memory import MemoryStore
from darwin.mcp.server import create_server

EXPECTED_TOOLS = {
    "memory_get_global",
    "memory_recent",
    "memory_search",
    "memory_write_iteration",
}


def test_server_registers_expected_tools(tmp_path):
    server = create_server(MemoryStore(tmp_path))
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert EXPECTED_TOOLS <= names


def test_write_tool_schema_excludes_controller_fields(tmp_path):
    server = create_server(MemoryStore(tmp_path))
    tools = asyncio.run(server.list_tools())
    write_tool = next(t for t in tools if t.name == "memory_write_iteration")
    props = set(write_tool.inputSchema.get("properties", {}))
    assert "thesis" in props  # agent-owned field present
    for forbidden in ("final_fitness", "mutation_failed", "finetune_failed"):
        assert forbidden not in props


def test_end_to_end_write_then_recent_via_call_tool(tmp_path):
    """Drive a write and a read through the actual MCP `call_tool` dispatch path."""
    server = create_server(MemoryStore(tmp_path))

    async def scenario():
        write_args = dict(
            model="model7",
            iteration=0,
            generation=1,
            parent_survivor="model3",
            mutator="model7",
            backend="claude",
            base_fitness=0.5,
            cost_usd=1.0,
            thesis="raise lora rank",
            changes="rank 16 -> 32",
            smoke_results="green",
            outcome="promising",
        )
        _, write_result = await server.call_tool("memory_write_iteration", write_args)
        assert write_result["ok"] is True

        content, _ = await server.call_tool("memory_recent", {"model": "model7", "k": 5})
        # FastMCP emits one content block per list element.
        return [json.loads(c.text) for c in content]

    recent = asyncio.run(scenario())
    assert len(recent) == 1
    assert recent[0]["thesis"] == "raise lora rank"
