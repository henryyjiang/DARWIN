"""FastMCP wiring for `darwin-mcp` (ARCHITECTURE.md §9.3).

Thin layer that registers `MemoryToolset` methods as MCP tools. Both agent backends attach
this same server (Claude via `ClaudeAgentOptions.mcp_servers`; OpenHands natively), which is
what makes the mutation directive and tool semantics identical across backends (§9.4).

Tool names use the `memory_*` convention (FastMCP tool identifiers can't contain a dot); the
directive refers to them as the `memory.*` group.

Run as a stdio MCP server:
    uv run --with mcp python -m darwin.mcp.server --root .
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from darwin.cost import BudgetGuard, CostLedger
from darwin.memory import MemoryStore
from darwin.mcp.tools import MemoryToolset, AgentToolset, CostToolset


def create_server(
    store: MemoryStore,
    agent_context: Any | None = None,
    name: str = "darwin-mcp",
    ledger: CostLedger | None = None,
    budget_guard: BudgetGuard | None = None,
) -> FastMCP:
    """Build a FastMCP server exposing the memory tool group over `store`.

    When `agent_context` (a `MutationContext`) is supplied — i.e. the server is attached to a
    live mutation window — the agent's `smoke.run` and `finalize` tools (§9.3) are also
    registered, bound to that offspring's checkpointer. When a `ledger` is also supplied, the
    `cost.report` / `cost.get_budget` tools are registered, bound to that offspring's
    generation/model (and the optional `budget_guard` for cap status).
    """
    server = FastMCP(name)
    tools = MemoryToolset(store)

    @server.tool(name="memory_get_global", description="Read the shared global memory store (objectives, what's working, todo, cost ledger). Read-only.")
    def memory_get_global() -> dict[str, str]:
        return tools.get_global()

    @server.tool(name="memory_recent", description="Return the k most recent iteration memory files for a model, newest first.")
    def memory_recent(model: str, k: int = 5) -> list[dict[str, Any]]:
        return tools.recent(model, k)

    @server.tool(name="memory_search", description="Keyword search over a model's own iteration memory history.")
    def memory_search(model: str, query: str) -> list[dict[str, Any]]:
        return tools.search(model, query)

    @server.tool(name="memory_write_iteration", description="Write this iteration's per-model memory file (schema-validated). Provide thesis, changes, smoke_results, outcome and the iteration metadata. Do not pass benchmark results; the controller fills those in.")
    def memory_write_iteration(
        model: str,
        iteration: int,
        generation: int,
        parent_survivor: str,
        mutator: str,
        backend: str,
        base_fitness: float,
        cost_usd: float,
        thesis: str,
        changes: str,
        smoke_results: str,
        outcome: str,
        datasets_used: list[str] | None = None,
        papers_cited: list[str] | None = None,
    ) -> dict[str, Any]:
        return tools.write_iteration(
            model=model,
            iteration=iteration,
            generation=generation,
            parent_survivor=parent_survivor,
            mutator=mutator,
            backend=backend,
            base_fitness=base_fitness,
            cost_usd=cost_usd,
            thesis=thesis,
            changes=changes,
            smoke_results=smoke_results,
            outcome=outcome,
            datasets_used=datasets_used,
            papers_cited=papers_cited,
        )

    if agent_context is not None:
        agent_tools = AgentToolset(agent_context)
        server.darwin_agent_tools = agent_tools  # exposed so the controller can read .finalized

        @server.tool(name="smoke_run", description="Run the read-only smoke test on the current genome. A pass auto-commits a green checkpoint. Returns pass/fail, exit code, and the tail of the log.")
        def smoke_run(summary: str = "smoke checkpoint") -> dict[str, Any]:
            return agent_tools.smoke_run(summary)

        @server.tool(name="finalize", description="Declare convergence to end your mutation window early once your code is green and your memory file is written.")
        def finalize() -> dict[str, Any]:
            return agent_tools.finalize()

        if ledger is not None:
            cost_tools = CostToolset(
                ledger,
                generation=agent_context.generation,
                model=agent_context.model,
                budget_guard=budget_guard,
            )

            @server.tool(name="cost_report", description="Log spend (USD) you incurred this window (e.g. API calls). Attributed to your generation and model. Provide amount_usd and a short reason.")
            def cost_report(amount_usd: float, reason: str, kind: str = "api") -> dict[str, Any]:
                return cost_tools.report(amount_usd, reason, kind)

            @server.tool(name="cost_get_budget", description="Read this generation's budget status: cap, spend so far, remaining, and whether the budget is exhausted. Cost is a first-class constraint — keep your strategy affordable.")
            def cost_get_budget() -> dict[str, Any]:
                return cost_tools.get_budget()

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the darwin-mcp stdio server.")
    parser.add_argument(
        "--root",
        default=".",
        help="Repo root containing models/ and memory/global/ (default: cwd).",
    )
    args = parser.parse_args()
    server = create_server(MemoryStore(Path(args.root)))
    server.run()


if __name__ == "__main__":
    main()
