"""Wall-clock deadline management for a mutation window (ARCHITECTURE.md §4.3).

The 2-4 hr window is enforced by the controller, not trusted to the model. Three layers:
- **Soft deadline (T-minus soft_lead):** inject a wrap-up nudge — stop new experiments, make
  the current code green, commit, write the memory file.
- **Hard deadline (T-0):** send `FINALIZE`; the agent must commit + write memory.
- **Kill (T+grace):** if the agent hasn't returned, force-stop and recover the last green
  commit (handled by the orchestrator / checkpointer).

The clock is injectable so the phase logic is testable without real time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class Phase(str, Enum):
    RUNNING = "running"  # free to run new experiments
    SOFT = "soft"  # T-minus soft_lead reached: wrap up
    HARD = "hard"  # T-0 reached: finalize now
    KILL = "kill"  # T+grace exceeded: force-stop


@dataclass
class DeadlineManager:
    """Maps elapsed wall-clock onto the four deadline phases (§4.3)."""

    window_s: float
    soft_lead_s: float  # soft deadline fires at window_s - soft_lead_s
    kill_grace_s: float  # kill fires at window_s + kill_grace_s
    clock: Callable[[], float] = time.monotonic
    start: float = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.start is None:
            self.start = self.clock()

    @classmethod
    def from_config(cls, mutation_cfg, clock: Callable[[], float] = time.monotonic):
        """Build from a MutationConfig (hours/minutes → seconds)."""
        return cls(
            window_s=mutation_cfg.mutation_window_h * 3600.0,
            soft_lead_s=mutation_cfg.soft_deadline_min * 60.0,
            kill_grace_s=mutation_cfg.kill_grace_min * 60.0,
            clock=clock,
        )

    def elapsed(self) -> float:
        return self.clock() - self.start

    def remaining(self) -> float:
        """Seconds until the hard deadline (negative once past it)."""
        return self.window_s - self.elapsed()

    def phase(self) -> Phase:
        elapsed = self.elapsed()
        if elapsed >= self.window_s + self.kill_grace_s:
            return Phase.KILL
        if elapsed >= self.window_s:
            return Phase.HARD
        if elapsed >= self.window_s - self.soft_lead_s:
            return Phase.SOFT
        return Phase.RUNNING
