"""Append-only cost ledger (ARCHITECTURE.md §5.4 / §7.4).

Records every dollar DARWIN spends — finetune GPU-hours x rate (§5.3), Claude-backed agent
sessions and the global-memory pass (§7.4), benchmarking — as one JSONL line per entry so the
record is crash-safe and auditable. Reads compute running totals and per-generation
breakdowns; `render_markdown` produces the raw table the global-memory pass narrates into
`cost_ledger.md`.

JSONL (not a single rewritten JSON blob) so concurrent per-offspring writes during a
generation only ever *append* — no read-modify-write race across parallel containers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Literal

# What a charge is for. `finetune` carries gpu_hours; `api`/`agent` are Claude API spend
# (mutation when backend=claude, the global-memory pass); `benchmark` is eval compute.
CostKind = Literal["finetune", "api", "agent", "benchmark", "other"]
COST_KINDS: tuple[CostKind, ...] = ("finetune", "api", "agent", "benchmark", "other")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CostEntry:
    """One charge against the budget."""

    generation: int
    kind: CostKind
    amount_usd: float
    reason: str
    model: str | None = None  # the offspring/model the charge is attributed to, if any
    gpu_hours: float | None = None  # set for finetune entries (amount = gpu_hours x rate)
    timestamp: str = ""  # ISO-8601 UTC; filled at record time if blank

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> "CostEntry":
        data = json.loads(line)
        return cls(**data)


class CostLedger:
    """JSONL-backed running record of DARWIN's spend, rooted at a single file."""

    def __init__(self, path: Path | str, *, now: Callable[[], str] = _utcnow_iso):
        self.path = Path(path)
        self._now = now

    # ------------------------------------------------------------------ writes
    def record(
        self,
        *,
        generation: int,
        kind: CostKind,
        amount_usd: float,
        reason: str,
        model: str | None = None,
        gpu_hours: float | None = None,
    ) -> CostEntry:
        """Append one charge to the ledger and return it."""
        if amount_usd < 0:
            raise ValueError("amount_usd must be non-negative")
        if kind not in COST_KINDS:
            raise ValueError(f"unknown cost kind {kind!r}; expected one of {COST_KINDS}")
        entry = CostEntry(
            generation=generation,
            kind=kind,
            amount_usd=float(amount_usd),
            reason=reason,
            model=model,
            gpu_hours=gpu_hours,
            timestamp=self._now(),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(entry.to_json() + "\n")
        return entry

    def record_gpu(
        self,
        *,
        generation: int,
        model: str,
        gpu_hours: float,
        rate_usd_per_h: float,
        reason: str = "finetune",
        kind: CostKind = "finetune",
    ) -> CostEntry:
        """Convenience: record a GPU charge as `gpu_hours x rate` (§5.3 cost contract)."""
        return self.record(
            generation=generation,
            kind=kind,
            amount_usd=gpu_hours * rate_usd_per_h,
            reason=reason,
            model=model,
            gpu_hours=gpu_hours,
        )

    # ------------------------------------------------------------------ reads
    def entries(self, generation: int | None = None) -> list[CostEntry]:
        """All entries (oldest first), optionally filtered to one generation."""
        if not self.path.exists():
            return []
        out: list[CostEntry] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entry = CostEntry.from_json(line)
            if generation is None or entry.generation == generation:
                out.append(entry)
        return out

    def total(self, generation: int | None = None) -> float:
        return sum(e.amount_usd for e in self.entries(generation))

    def generations(self) -> list[int]:
        """Sorted distinct generation numbers present in the ledger."""
        return sorted({e.generation for e in self.entries()})

    def totals_by_kind(self, generation: int | None = None) -> dict[str, float]:
        out: dict[str, float] = {}
        for e in self.entries(generation):
            out[e.kind] = out.get(e.kind, 0.0) + e.amount_usd
        return out

    # ------------------------------------------------------------------ rendering
    def render_markdown(self) -> str:
        """Per-generation spend table + grand total — raw input for `cost_ledger.md` (§7.4)."""
        gens = self.generations()
        header = (
            "| generation | finetune | api | agent | benchmark | other | total |\n"
            "|---|---|---|---|---|---|---|"
        )
        if not gens:
            return header + "\n| _no spend recorded_ |"
        rows = []
        for g in gens:
            by_kind = self.totals_by_kind(g)
            cells = [f"{by_kind.get(k, 0.0):.4g}" for k in COST_KINDS]
            rows.append(f"| {g} | " + " | ".join(cells) + f" | {self.total(g):.4g} |")
        rows.append(f"| **all** | | | | | | **{self.total():.4g}** |")
        return header + "\n" + "\n".join(rows)
