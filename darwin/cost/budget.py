"""Budget enforcement over the cost ledger (ARCHITECTURE.md §5.4).

Two enforcement levels are distinct mechanisms (§5.4): the *soft* `lambda_cost` fitness
penalty lives in the fitness reduction (§6.3); this module is the *hard* controller cap. On
hitting `gen_budget_usd` mid-generation the controller **lets in-flight finetunes finish**
(killing them wastes spend already incurred) but **launches no new ones** — so the guard's job
is to answer "may I launch another job?" given what this generation has already spent.

It also answers the agent-facing `cost.get_budget()` MCP query (§9.3) via `status()`.
"""

from __future__ import annotations

from dataclasses import dataclass

from darwin.config import CostConfig
from darwin.cost.ledger import CostLedger


@dataclass
class BudgetStatus:
    """A generation's spend against its cap — the `cost.get_budget()` payload (§9.3)."""

    generation: int
    gen_budget_usd: float | None  # None => no per-generation cap configured
    generation_spend: float
    total_spend: float
    remaining: float | None  # None when uncapped; may go negative if a cap was overshot
    exhausted: bool  # True => stop launching new jobs this generation (§5.4)


class BudgetGuard:
    """Hard per-generation budget cap over a `CostLedger` (§5.4)."""

    def __init__(self, ledger: CostLedger, cost_config: CostConfig):
        self.ledger = ledger
        self.config = cost_config

    def status(self, generation: int) -> BudgetStatus:
        cap = self.config.gen_budget_usd
        spend = self.ledger.total(generation)
        remaining = None if cap is None else cap - spend
        return BudgetStatus(
            generation=generation,
            gen_budget_usd=cap,
            generation_spend=spend,
            total_spend=self.ledger.total(),
            remaining=remaining,
            exhausted=cap is not None and spend >= cap,
        )

    def can_launch(self, generation: int, estimated_usd: float = 0.0) -> bool:
        """Whether a new job (optionally with an estimated cost) may be launched (§5.4).

        Uncapped => always yes. Capped => yes only while the generation's *already-incurred*
        spend plus the estimate stays within the cap. In-flight jobs are not pre-charged here;
        the controller checks `can_launch` before each launch, never killing a running job.
        """
        cap = self.config.gen_budget_usd
        if cap is None:
            return True
        return self.ledger.total(generation) + estimated_usd <= cap
