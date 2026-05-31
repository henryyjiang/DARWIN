"""Memory tool logic for the DARWIN MCP server (ARCHITECTURE.md §9.3, memory group).

This is the backend-agnostic logic layer: it wraps a `MemoryStore` and returns JSON-friendly
values, with no dependency on the MCP transport so it can be unit-tested directly. The
FastMCP server (`server.py`) is a thin layer that registers these methods as tools.

Tools implemented here:
- `memory.get_global()`        -> read the shared global memory store
- `memory.recent(model, k)`    -> the k most recent iterations (newest first)
- `memory.search(model, query)`-> keyword search over a model's own history
- `memory.write_iteration(...)`-> schema-validated per-model memory write

Invariants enforced at this boundary (§7.2 / §7.3):
- agents write per-model memory **only** through the schema-validated `write_iteration`;
- the controller-owned post-benchmark fields (`final_fitness`, `mutation_failed`,
  `finetune_failed`) are **not** accepted from agents here — only the controller patches them;
- there is **no** global-memory write tool — population/mutation agents only read it.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from darwin.cost import BudgetGuard, CostLedger
from darwin.memory import IterationMemory, MemoryStore, MemoryValidationError

if TYPE_CHECKING:
    from darwin.mutation_agent.backend import MutationContext


class MemoryToolset:
    """Memory tools bound to a single MemoryStore."""

    def __init__(self, store: MemoryStore):
        self.store = store

    # ------------------------------------------------------------------ reads
    def get_global(self) -> dict[str, str]:
        """Return the global memory store as a section->content mapping (`memory.get_global`)."""
        return asdict(self.store.get_global())

    def recent(self, model: str, k: int = 5) -> list[dict[str, Any]]:
        """Return the k most recent iterations for a model, newest first (`memory.recent`)."""
        return [m.to_dict() for m in self.store.recent(model, k)]

    def search(self, model: str, query: str) -> list[dict[str, Any]]:
        """Keyword search over a model's own memory history (`memory.search`)."""
        return [m.to_dict() for m in self.store.search(model, query)]

    # ------------------------------------------------------------------ write
    def write_iteration(
        self,
        *,
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
        """Schema-validated per-model memory write (`memory.write_iteration`, §7.2).

        Only agent-owned fields are accepted; controller-owned post-benchmark fields are
        deliberately absent from the signature. On a schema violation this returns a
        structured error (rather than raising) so the agent can read the problem and retry.
        """
        mem = IterationMemory(
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
            datasets_used=list(datasets_used or []),
            papers_cited=list(papers_cited or []),
        )
        try:
            path = self.store.write_iteration(mem)
        except MemoryValidationError as exc:
            return {"ok": False, "error": str(exc)}
        return {
            "ok": True,
            "model": model,
            "iteration": iteration,
            "path": str(path),
        }


class AgentToolset:
    """Mutation-window tools bound to one offspring's context (ARCHITECTURE.md §9.3).

    These are the agent's *only* eval-like / control tools:
    - `smoke.run`  — run the read-only smoke test (§4.4.1); a pass auto-commits a green
      checkpoint (§4.4). There is deliberately **no** agent-callable scored-benchmark tool —
      scored benchmarking is controller-only and post-finetune (§6.2), so agents cannot probe
      the held-out eval.
    - `finalize`   — the agent self-declares convergence to end its window early (§4.3); the
      controller observes the flag.
    """

    def __init__(self, ctx: "MutationContext"):
        self._ctx = ctx
        self._finalized = False

    def smoke_run(self, summary: str = "smoke checkpoint") -> dict[str, Any]:
        result = self._ctx.run_smoke()
        commit = None
        if result.passed:
            commit = self._ctx.checkpointer.commit_green(summary)
        return {
            "passed": result.passed,
            "exit_code": result.exit_code,
            "committed": commit is not None,
            "commit": commit,
            "log": result.log[-4000:],
        }

    def finalize(self) -> dict[str, Any]:
        self._finalized = True
        return {"ok": True, "finalized": True}

    @property
    def finalized(self) -> bool:
        return self._finalized


class CostToolset:
    """Cost-ledger tools bound to one offspring's generation/model (ARCHITECTURE.md §9.3).

    - `cost.report(amount, reason)` — the agent logs API/compute spend it incurred this
      window; entries are attributed to the bound generation + model.
    - `cost.get_budget()` — the agent reads the generation's budget status so it can self-
      regulate (cost is a first-class constraint surfaced in memory, §1.3 / §5.4).

    Bound context (generation/model) is fixed by the controller, not chosen by the agent, so
    an agent cannot mis-attribute spend to another generation/model. The optional
    `BudgetGuard` answers `get_budget`; without it, budget is reported as uncapped.
    """

    def __init__(
        self,
        ledger: CostLedger,
        *,
        generation: int,
        model: str | None = None,
        budget_guard: BudgetGuard | None = None,
    ):
        self.ledger = ledger
        self.generation = generation
        self.model = model
        self.budget_guard = budget_guard

    def report(self, amount_usd: float, reason: str, kind: str = "api") -> dict[str, Any]:
        try:
            entry = self.ledger.record(
                generation=self.generation,
                kind=kind,  # type: ignore[arg-type]  (validated in record)
                amount_usd=amount_usd,
                reason=reason,
                model=self.model,
            )
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return {
            "ok": True,
            "amount_usd": entry.amount_usd,
            "generation_total": self.ledger.total(self.generation),
        }

    def get_budget(self) -> dict[str, Any]:
        if self.budget_guard is None:
            return {
                "generation": self.generation,
                "gen_budget_usd": None,
                "generation_spend": self.ledger.total(self.generation),
                "total_spend": self.ledger.total(),
                "remaining": None,
                "exhausted": False,
            }
        return asdict(self.budget_guard.status(self.generation))
