"""Anti-gaming flags & report (ARCHITECTURE.md §6.4).

The §6.4 defenses are *producers* of penalties: each heuristic (contamination scan,
genome-diff hack inspection, plausibility/generalization-gap) emits zero or more
`AntiGamingFlag`s, collected into an `AntiGamingReport`. The report's `count` (sum of flag
severities) is the integer the §6.3 fitness reduction multiplies by `lambda_penalty` — so
gaming is *selected against*, not merely logged ("penalties feed fitness", §6.4).

Pure data; the heuristics that build these live alongside (`contamination.py`,
`genome_review.py`, `plausibility.py`) and are composed by `scan.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AntiGamingFlag:
    """One anti-gaming hit (§6.4).

    `kind` groups the producing heuristic ("contamination" / "genome_hack" /
    "generalization_gap"); `detail` is a short human-readable reason for the audit log;
    `severity` weights how hard it hits fitness (default 1 == one `lambda_penalty` unit).
    """

    kind: str
    detail: str
    severity: int = 1


@dataclass
class AntiGamingReport:
    """All flags raised for one offspring this generation (§6.4)."""

    flags: list[AntiGamingFlag] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Total penalty weight fed to fitness as `antigaming_flags` (§6.3)."""
        return sum(f.severity for f in self.flags)

    @property
    def kinds(self) -> set[str]:
        return {f.kind for f in self.flags}

    @property
    def clean(self) -> bool:
        return not self.flags

    def extend(self, flags: list[AntiGamingFlag]) -> None:
        self.flags.extend(flags)

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "flags": [
                {"kind": f.kind, "detail": f.detail, "severity": f.severity}
                for f in self.flags
            ],
        }
