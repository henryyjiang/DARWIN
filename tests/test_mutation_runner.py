"""End-to-end mutation-window orchestration with a fake backend (ARCHITECTURE.md §4.2-§4.4).

No Claude API / Docker: a scripted backend drives the lifecycle against a real Git repo +
trivial genome, proving the always-green finalization and the zero-green fallback.
"""

import sys
from pathlib import Path

import pytest

from darwin.memory import MemoryStore
from darwin.mutation_agent import (
    DeadlineManager,
    GitCheckpointer,
    SmokeTest,
    run_mutation_window,
)
from darwin.mutation_agent.backend import MutationContext


def make_genome(path: Path, ok: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "recipe.py").write_text(f"OK = {ok}\n", encoding="utf-8")
    (path / "smoke_test.py").write_text(
        "import recipe, sys\nsys.exit(0 if recipe.OK else 1)\n", encoding="utf-8"
    )


def make_ctx(tmp_path: Path, model="model7", iteration=0) -> MutationContext:
    genome = tmp_path / "genome"
    make_genome(genome)
    return MutationContext(
        offspring_id="7",
        genome_dir=genome,
        model=model,
        parent_survivor="model3",
        mutator="model2",
        generation=5,
        iteration=iteration,
        backend_name="claude",
        base_fitness=0.5,
        directive="do the thing",
        checkpointer=GitCheckpointer(genome),
        smoke=SmokeTest(command=[sys.executable, "smoke_test.py"]),
        store=MemoryStore(tmp_path / "store"),
    )


def never_ending_deadline() -> DeadlineManager:
    return DeadlineManager(window_s=100, soft_lead_s=20, kill_grace_s=10,
                           clock=lambda: 0.0, start=0.0)


class GreenBackend:
    """Makes one good edit, checkpoints it green, writes its memory."""

    def run(self, ctx: MutationContext, deadline: DeadlineManager) -> None:
        (ctx.genome_dir / "recipe.py").write_text("OK = True\n# improved\n", encoding="utf-8")
        assert ctx.checkpoint("improved recipe") is True
        ctx.write_memory(thesis="t", changes="c", smoke_results="green", outcome="o", cost_usd=1.0)


class BrokenBackend:
    """Breaks the genome (never goes green) but still reflects in memory."""

    def run(self, ctx: MutationContext, deadline: DeadlineManager) -> None:
        (ctx.genome_dir / "recipe.py").write_text("OK = False\n", encoding="utf-8")
        assert ctx.checkpoint("attempt") is False  # smoke fails → no commit
        ctx.write_memory(thesis="t", changes="broke it", smoke_results="red", outcome="o", cost_usd=1.0)


def test_green_window(tmp_path):
    ctx = make_ctx(tmp_path)
    result = run_mutation_window(ctx, GreenBackend(), never_ending_deadline())

    assert result.produced_green is True
    assert result.mutation_failed is False
    assert result.memory_written is True
    assert ctx.checkpointer.has_last_green()
    assert result.final_commit == ctx.checkpointer.last_green()
    # final genome is green and the edit survived
    assert ctx.smoke.run(ctx.genome_dir).passed
    assert "# improved" in (ctx.genome_dir / "recipe.py").read_text(encoding="utf-8")
    assert ctx.store.read_iteration("model7", 0).changes == "c"


def test_zero_green_window_falls_back_to_clone(tmp_path):
    ctx = make_ctx(tmp_path)
    result = run_mutation_window(ctx, BrokenBackend(), never_ending_deadline())

    assert result.produced_green is False
    assert result.mutation_failed is True   # fell back to clone of S (§4.3)
    assert result.memory_written is True    # reflective memory still written
    assert not ctx.checkpointer.has_last_green()
    assert result.final_commit == ctx.checkpointer.base_commit
    # genome restored to the green clone
    assert ctx.smoke.run(ctx.genome_dir).passed
    assert (ctx.genome_dir / "recipe.py").read_text(encoding="utf-8") == "OK = True\n"
