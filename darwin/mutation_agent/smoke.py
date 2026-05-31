"""Smoke-test runner (ARCHITECTURE.md §4.4.1).

"Green" must mean *the recipe will train*, not merely *the code imports*. The full smoke test
is a fast, tiny end-to-end finetune dry-run (import+config validation, a data-pipeline check,
one real train step with a finite loss and a non-zero gradient, and a materialized adapter),
run deterministically with a fixed seed.

That finetune-specific harness needs the training stack and lands with the finetune pipeline
(Phase 3). What lives here now is the **generic, controller-owned runner**: it invokes the
genome's declared smoke entrypoint in the genome dir and treats exit code 0 as green. The
runner is mounted read-only into the container so the agent can't weaken it to force false
greens (§4.4.1 / §8.2). Determinism (fixed seed, fixed tiny data) is the entrypoint's job.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SmokeResult:
    """Outcome of one smoke-test run."""

    passed: bool
    exit_code: int
    log: str
    duration_s: float


@dataclass
class SmokeTest:
    """Runs the genome's smoke entrypoint as a subprocess; exit 0 == green."""

    command: list[str]
    timeout_s: float = 120.0  # §4.4.1 targets < ~2 min
    env: dict[str, str] | None = None

    def run(self, cwd: Path | str) -> SmokeResult:
        start = time.monotonic()
        try:
            proc = subprocess.run(
                self.command,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                env=self.env,
            )
            exit_code = proc.returncode
            log = (proc.stdout or "") + (proc.stderr or "")
        except subprocess.TimeoutExpired as exc:
            exit_code = -1
            log = f"smoke test timed out after {self.timeout_s}s\n{exc.output or ''}"
        duration = time.monotonic() - start
        return SmokeResult(
            passed=exit_code == 0,
            exit_code=exit_code,
            log=log,
            duration_s=duration,
        )
