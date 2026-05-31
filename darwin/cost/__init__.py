"""DARWIN v2 cost subsystem (ARCHITECTURE.md §5.4 / §7.4).

Every finetune (GPU-hours x rate) and every API-backed agent session writes to a single
append-only **cost ledger**. The ledger's running total + per-generation breakdown feed two
mechanisms:

- **Soft** steering: the per-offspring `cost_usd` lands in fitness via `lambda_cost` (§6.3).
- **Hard** enforcement: the `BudgetGuard` lets the controller stop launching new finetunes
  once a generation's `gen_budget_usd` cap is hit, while letting in-flight jobs finish (§5.4).

It also backs the MCP `cost.*` tools (§9.3) and is rendered into the global `cost_ledger.md`
section by the global-memory pass (§7.4). This package is pure filesystem/data — no infra.
"""

from darwin.cost.ledger import CostEntry, CostLedger, COST_KINDS
from darwin.cost.budget import BudgetGuard, BudgetStatus

__all__ = [
    "CostEntry",
    "CostLedger",
    "COST_KINDS",
    "BudgetGuard",
    "BudgetStatus",
]
