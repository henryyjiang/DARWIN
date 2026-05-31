"""Run status dashboard (ARCHITECTURE.md §9.5)."""

from darwin.cost import CostLedger
from darwin.controller.state import (
    GenerationState,
    GenerationStateStore,
    OffspringState,
)
from darwin.observability import (
    render_generation_markdown,
    render_run_markdown,
    summarize_generation,
    summarize_run,
)


def _off(name, status, fitness, **kw):
    return OffspringState(
        name=name, parent_survivor="s0", mutator="s1", backend="claude", iteration=0,
        finetune_status=status, fitness=fitness, **kw,
    )


def _gen_state(generation=0, completed=True, phase="checkpoint"):
    return GenerationState(
        generation=generation,
        phase=phase,
        offspring=[
            _off("o0", "ok", 1.4, cost_usd=2.0),
            _off("o1", "finetune_failed", float("-inf"), cost_usd=1.0),
            _off("o2", "deferred", None),
            _off("o3", "ok", 0.9, antigaming_flags=1, cost_usd=2.0),
        ],
        survivors_after_cull=["s0", "s1"],
        completed=completed,
    )


def test_summarize_generation_counts(tmp_path):
    ledger = CostLedger(tmp_path / "cost.jsonl")
    ledger.record(generation=0, kind="finetune", amount_usd=5.0, reason="ft")
    summary = summarize_generation(_gen_state(), ledger)
    assert summary.generation == 0
    assert summary.completed is True
    assert summary.deferred_count == 1
    assert summary.failed_count == 1
    assert summary.flagged_count == 1
    assert summary.best_fitness == 1.4
    assert summary.spend_usd == 5.0
    assert summary.survivors == ["s0", "s1"]


def test_summarize_generation_without_ledger():
    summary = summarize_generation(_gen_state(), None)
    assert summary.spend_usd == 0.0
    assert summary.best_fitness == 1.4


def test_render_generation_markdown_has_rows():
    md = render_generation_markdown(summarize_generation(_gen_state(), None))
    assert "Generation 0 — complete" in md
    assert "| o0 |" in md and "| o2 |" in md
    assert "deferred" in md
    assert "floor" in md  # finetune_failed offspring rendered as floor
    assert "—" in md  # deferred offspring's None fitness


def test_in_progress_generation_label():
    md = render_generation_markdown(
        summarize_generation(_gen_state(completed=False, phase="offspring_done"), None)
    )
    assert "in-progress (offspring_done)" in md


def test_summarize_run_walks_all_generations(tmp_path):
    store = GenerationStateStore(tmp_path / "runs")
    store.save(_gen_state(generation=0))
    store.save(_gen_state(generation=1))
    ledger = CostLedger(tmp_path / "cost.jsonl")
    ledger.record(generation=0, kind="finetune", amount_usd=4.0, reason="ft")
    ledger.record(generation=1, kind="finetune", amount_usd=6.0, reason="ft")

    run = summarize_run(store, ledger)
    assert len(run.generations) == 2
    assert run.latest_generation == 1
    assert run.total_spend_usd == 10.0

    md = render_run_markdown(run)
    assert "generations recorded: 2" in md
    assert "total spend: $10" in md


def test_summarize_empty_run(tmp_path):
    store = GenerationStateStore(tmp_path / "runs")
    run = summarize_run(store, None)
    assert run.generations == []
    assert run.latest_generation is None
    assert "no generations recorded yet" in render_run_markdown(run)
