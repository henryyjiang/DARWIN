"""Tests for the agent-facing MCP tools smoke.run / finalize (ARCHITECTURE.md §9.3)."""

import asyncio
import sys
from pathlib import Path

import pytest

from darwin.memory import MemoryStore
from darwin.mcp import AgentToolset
from darwin.mcp.server import create_server
from darwin.mutation_agent import GitCheckpointer, SmokeTest
from darwin.mutation_agent.backend import MutationContext


def make_ctx(tmp_path: Path) -> MutationContext:
    genome = tmp_path / "genome"
    genome.mkdir(parents=True, exist_ok=True)
    (genome / "recipe.py").write_text("OK = True\n", encoding="utf-8")
    (genome / "smoke_test.py").write_text(
        "import recipe, sys\nsys.exit(0 if recipe.OK else 1)\n", encoding="utf-8"
    )
    ctx = MutationContext(
        offspring_id="7", genome_dir=genome, model="model7", parent_survivor="model3",
        mutator="model2", generation=5, iteration=0, backend_name="claude",
        base_fitness=0.5, directive="d", checkpointer=GitCheckpointer(genome),
        smoke=SmokeTest(command=[sys.executable, "smoke_test.py"]),
        store=MemoryStore(tmp_path / "store"),
    )
    ctx.checkpointer.init_offspring("7")
    return ctx


def test_smoke_run_commits_on_green(tmp_path):
    ctx = make_ctx(tmp_path)
    res = AgentToolset(ctx).smoke_run("checkpoint")
    assert res["passed"] is True
    assert res["committed"] is True
    assert res["commit"] == ctx.checkpointer.last_green()


def test_smoke_run_no_commit_on_red(tmp_path):
    ctx = make_ctx(tmp_path)
    (ctx.genome_dir / "recipe.py").write_text("OK = False\n", encoding="utf-8")
    res = AgentToolset(ctx).smoke_run("checkpoint")
    assert res["passed"] is False
    assert res["committed"] is False
    assert not ctx.checkpointer.has_last_green()


def test_finalize_sets_flag(tmp_path):
    tools = AgentToolset(make_ctx(tmp_path))
    assert tools.finalized is False
    assert tools.finalize() == {"ok": True, "finalized": True}
    assert tools.finalized is True


def test_server_registers_agent_tools_with_context(tmp_path):
    ctx = make_ctx(tmp_path)
    server = create_server(ctx.store, agent_context=ctx)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"smoke_run", "finalize"} <= names
    assert {"memory_get_global", "memory_write_iteration"} <= names  # memory still present


def test_server_omits_agent_tools_without_context(tmp_path):
    names = {t.name for t in asyncio.run(create_server(MemoryStore(tmp_path)).list_tools())}
    assert "smoke_run" not in names
    assert "finalize" not in names


def test_smoke_run_through_call_tool_dispatch(tmp_path):
    ctx = make_ctx(tmp_path)
    server = create_server(ctx.store, agent_context=ctx)
    _, result = asyncio.run(server.call_tool("smoke_run", {"summary": "via mcp"}))
    assert result["passed"] is True and result["committed"] is True
    assert ctx.checkpointer.has_last_green()
