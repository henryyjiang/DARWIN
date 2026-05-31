"""Tests for the MCP cost.* tools (ARCHITECTURE.md §9.3)."""

import asyncio
import sys
from pathlib import Path

from darwin.config import CostConfig
from darwin.cost import BudgetGuard, CostLedger
from darwin.memory import MemoryStore
from darwin.mcp import CostToolset
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


def test_report_attributes_to_bound_generation_and_model(tmp_path):
    led = CostLedger(tmp_path / "cost.jsonl")
    tools = CostToolset(led, generation=5, model="model7")
    res = tools.report(2.5, "claude mutation tokens")
    assert res["ok"] is True
    assert res["generation_total"] == 2.5

    entries = led.entries()
    assert len(entries) == 1
    assert entries[0].generation == 5
    assert entries[0].model == "model7"
    assert entries[0].kind == "api"


def test_report_rejects_negative(tmp_path):
    led = CostLedger(tmp_path / "cost.jsonl")
    res = CostToolset(led, generation=0).report(-1.0, "bad")
    assert res["ok"] is False
    assert "error" in res
    assert led.entries() == []


def test_get_budget_without_guard_is_uncapped(tmp_path):
    led = CostLedger(tmp_path / "cost.jsonl")
    led.record(generation=5, kind="api", amount_usd=3.0, reason="x")
    res = CostToolset(led, generation=5).get_budget()
    assert res["gen_budget_usd"] is None
    assert res["generation_spend"] == 3.0
    assert res["exhausted"] is False


def test_get_budget_with_guard_reports_cap(tmp_path):
    led = CostLedger(tmp_path / "cost.jsonl")
    guard = BudgetGuard(led, CostConfig(gen_budget_usd=10.0))
    led.record(generation=5, kind="finetune", amount_usd=11.0, reason="x")
    res = CostToolset(led, generation=5, budget_guard=guard).get_budget()
    assert res["gen_budget_usd"] == 10.0
    assert res["remaining"] == -1.0
    assert res["exhausted"] is True


def test_server_registers_cost_tools_when_ledger_present(tmp_path):
    ctx = make_ctx(tmp_path)
    led = CostLedger(tmp_path / "cost.jsonl")
    server = create_server(ctx.store, agent_context=ctx, ledger=led)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"cost_report", "cost_get_budget"} <= names


def test_server_omits_cost_tools_without_ledger(tmp_path):
    ctx = make_ctx(tmp_path)
    server = create_server(ctx.store, agent_context=ctx)  # no ledger
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert "cost_report" not in names
    assert "cost_get_budget" not in names


def test_cost_report_through_call_tool_dispatch(tmp_path):
    ctx = make_ctx(tmp_path)
    led = CostLedger(tmp_path / "cost.jsonl")
    server = create_server(ctx.store, agent_context=ctx, ledger=led)
    _, result = asyncio.run(
        server.call_tool("cost_report", {"amount_usd": 1.25, "reason": "tokens"})
    )
    assert result["ok"] is True
    assert led.total(5) == 1.25  # attributed to ctx.generation == 5
