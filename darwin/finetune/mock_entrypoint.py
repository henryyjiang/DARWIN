"""Mock finetune entrypoint — the test-profile substitute for the GPU finetune (TEST_RUN_PLAN §3.1).

The default `darwin-finetune` entrypoint (`entrypoint.py`) loads a 32B model and trains a LoRA
adapter on a GPU ($, hours). For the budget-free full-run test we swap in *this* module: it runs
in a **real container** through the same `ContainerFinetuneBackend` / `SubprocessFinetuneBackend`
contract and the same `DARWIN_*` env (`finetune_env`), but instead of training it writes a tiny
deterministic "adapter" — a **genome fingerprint** — so the eval stage downstream has a real,
genome-dependent signal to score (mirroring the real data flow: the genome's effect reaches eval
only through the adapter, §6.2).

Why a fingerprint adapter: the eval container sees only `base + adapter`, never the genome. So the
mock finetune must encode the genome's identity (its source sha) and how much it has been mutated
(the count of the §3.3 improvement marker) *into the adapter*, which the mock eval then reads back
to produce genome-dependent scores that trend up as a lineage accumulates green mutations.

Design mirrors the real entrypoint: the fingerprint + payload assembly are **pure functions**
(`fingerprint_genome`, `count_markers`, `build_adapter_payload`) unit-tested with no Docker; `main`
reads the env, optionally sleeps so wall-clock GPU-hours are non-zero (the ledger/budget has
something to record), honors `DARWIN_MOCK_FAIL` to exercise the §5.3 failure taxonomy, and writes
the adapter JSON to `DARWIN_ADAPTER_OUT`.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from darwin.finetune.entrypoint import FinetuneRunConfig

# The token the §3.3 mock mutation backend appends to the genome on each green edit. Counting its
# occurrences across the genome source gives "how many accumulated improvements this lineage has",
# which the eval step turns into a score gain — so a surviving lineage trends up over generations.
IMPROVEMENT_MARKER = "darwin-improve"

# Files that are not part of the "genome source" for fingerprinting (VCS / caches / the adapter).
_IGNORE_DIRS = {".git", "__pycache__", ".pytest_cache"}


def _iter_source_files(genome_dir: Path):
    """Yield the genome's source files (sorted, VCS/cache excluded) for a stable fingerprint."""
    files = [
        p
        for p in genome_dir.rglob("*")
        if p.is_file() and not any(part in _IGNORE_DIRS for part in p.relative_to(genome_dir).parts)
    ]
    return sorted(files, key=lambda p: p.relative_to(genome_dir).as_posix())


def fingerprint_genome(genome_dir: Path | str) -> str:
    """A deterministic sha256 over the sorted genome source files (path + bytes).

    Stable across machines and runs for the same tree; changes the moment the mutator edits any
    file — which is exactly the "the genome changed" signal the eval step keys its base score on.
    """
    genome_dir = Path(genome_dir)
    h = hashlib.sha256()
    if genome_dir.exists():
        for p in _iter_source_files(genome_dir):
            h.update(p.relative_to(genome_dir).as_posix().encode("utf-8"))
            h.update(b"\0")
            h.update(p.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def count_markers(genome_dir: Path | str, marker: str = IMPROVEMENT_MARKER) -> int:
    """Count occurrences of the improvement marker across the genome source (pure)."""
    genome_dir = Path(genome_dir)
    if not genome_dir.exists():
        return 0
    total = 0
    for p in _iter_source_files(genome_dir):
        try:
            total += p.read_text(encoding="utf-8").count(marker)
        except (UnicodeDecodeError, OSError):
            continue  # binary / unreadable file: contributes no markers
    return total


def build_adapter_payload(cfg: FinetuneRunConfig, genome_sha: str, markers: int) -> dict:
    """Assemble the deterministic "adapter" the eval step reads back (pure).

    This is the entire mock adapter: the genome's identity (`genome_sha`), its accumulated
    improvement count (`markers`), and the resolved LoRA knobs/method (so a genome that changes its
    hyperparameters also changes the adapter, hence the score)."""
    return {
        "genome_sha": genome_sha,
        "markers": markers,
        "lora_rank": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "method": cfg.method,
        "mock": True,
    }


@dataclass
class MockFinetuneConfig:
    """The mock-specific env knobs (the real finetune knobs come from `FinetuneRunConfig`)."""

    genome_dir: str
    sleep_s: float = 1.0  # brief sleep so wall-clock GPU-hours are non-zero (ledger/budget, §5.4)
    fail: str = ""  # "" | "oom" | "nonzero": inject the §5.3 failure taxonomy through the container

    @classmethod
    def from_env(cls, env: dict[str, str]) -> "MockFinetuneConfig":
        return cls(
            genome_dir=env.get("DARWIN_GENOME_DIR") or os.getcwd(),
            sleep_s=_env_float(env, "DARWIN_MOCK_SLEEP_S", 1.0),
            fail=env.get("DARWIN_MOCK_FAIL", "").strip().lower(),
        )


def _env_float(env, key, default):
    try:
        return float(env[key])
    except (KeyError, ValueError):
        return default


def main(env: dict[str, str] | None = None) -> int:
    """Write the genome-fingerprint adapter to `DARWIN_ADAPTER_OUT` (the mock finetune)."""
    env = dict(os.environ if env is None else env)
    cfg = FinetuneRunConfig.from_env(env)
    mock = MockFinetuneConfig.from_env(env)

    if mock.sleep_s > 0:
        time.sleep(mock.sleep_s)

    # §5.3 failure injection — exercised end-to-end through the real container path.
    if mock.fail == "oom":
        print("[mock-finetune] CUDA out of memory (injected DARWIN_MOCK_FAIL=oom)")
        return 1
    if mock.fail in ("nonzero", "1", "fail"):
        print(f"[mock-finetune] injected failure DARWIN_MOCK_FAIL={mock.fail}")
        return 2

    genome_sha = fingerprint_genome(mock.genome_dir)
    markers = count_markers(mock.genome_dir)
    payload = build_adapter_payload(cfg, genome_sha, markers)

    out = Path(cfg.adapter_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(
        f"[mock-finetune] genome_sha={genome_sha[:12]} markers={markers} "
        f"rank={cfg.lora_rank} method={cfg.method} -> {out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
