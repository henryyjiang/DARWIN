"""Cost ledger + budget guard (ARCHITECTURE.md §5.4 / §7.4)."""

import pytest

from darwin.config import CostConfig
from darwin.cost import BudgetGuard, CostEntry, CostLedger


def make_ledger(tmp_path):
    # deterministic timestamps so assertions don't depend on wall-clock
    counter = {"n": 0}

    def fake_now():
        counter["n"] += 1
        return f"2026-05-30T00:00:{counter['n']:02d}+00:00"

    return CostLedger(tmp_path / "runs" / "cost_ledger.jsonl", now=fake_now)


def test_record_appends_and_totals(tmp_path):
    led = make_ledger(tmp_path)
    led.record(generation=0, kind="api", amount_usd=2.0, reason="global pass")
    led.record(generation=0, kind="agent", amount_usd=3.0, reason="mutation", model="model7")
    led.record(generation=1, kind="benchmark", amount_usd=1.5, reason="eval")

    assert led.total() == pytest.approx(6.5)
    assert led.total(generation=0) == pytest.approx(5.0)
    assert led.total(generation=1) == pytest.approx(1.5)
    assert led.generations() == [0, 1]


def test_record_gpu_converts_hours_times_rate(tmp_path):
    led = make_ledger(tmp_path)
    entry = led.record_gpu(
        generation=2, model="model3", gpu_hours=1.5, rate_usd_per_h=1.10
    )
    assert entry.kind == "finetune"
    assert entry.gpu_hours == 1.5
    assert entry.amount_usd == pytest.approx(1.65)
    assert led.total(generation=2) == pytest.approx(1.65)


def test_persistence_roundtrip_via_fresh_instance(tmp_path):
    led = make_ledger(tmp_path)
    led.record(generation=0, kind="api", amount_usd=2.0, reason="x")
    # a fresh instance reading the same file sees the entry (crash-safe append-only)
    reopened = CostLedger(led.path)
    entries = reopened.entries()
    assert len(entries) == 1
    assert isinstance(entries[0], CostEntry)
    assert entries[0].amount_usd == 2.0
    assert entries[0].timestamp  # filled at record time


def test_negative_and_bad_kind_rejected(tmp_path):
    led = make_ledger(tmp_path)
    with pytest.raises(ValueError):
        led.record(generation=0, kind="api", amount_usd=-1.0, reason="x")
    with pytest.raises(ValueError):
        led.record(generation=0, kind="bogus", amount_usd=1.0, reason="x")


def test_totals_by_kind_and_render(tmp_path):
    led = make_ledger(tmp_path)
    led.record(generation=0, kind="api", amount_usd=2.0, reason="x")
    led.record(generation=0, kind="finetune", amount_usd=4.0, reason="ft", gpu_hours=2.0)
    by_kind = led.totals_by_kind(0)
    assert by_kind["api"] == 2.0
    assert by_kind["finetune"] == 4.0

    md = led.render_markdown()
    assert "generation" in md and "total" in md
    assert "**all**" in md
    # empty ledger renders a placeholder, not a crash
    assert "no spend recorded" in CostLedger(tmp_path / "empty.jsonl").render_markdown()


# ------------------------------------------------------------------ budget guard


def test_budget_guard_uncapped_always_launches(tmp_path):
    led = make_ledger(tmp_path)
    guard = BudgetGuard(led, CostConfig())  # gen_budget_usd None => uncapped
    led.record(generation=0, kind="finetune", amount_usd=1000.0, reason="big")
    status = guard.status(0)
    assert status.gen_budget_usd is None
    assert status.remaining is None
    assert status.exhausted is False
    assert guard.can_launch(0, estimated_usd=500.0) is True


def test_budget_guard_caps_launches_when_exhausted(tmp_path):
    led = make_ledger(tmp_path)
    guard = BudgetGuard(led, CostConfig(gen_budget_usd=10.0))

    assert guard.can_launch(0, estimated_usd=4.0) is True
    led.record(generation=0, kind="finetune", amount_usd=8.0, reason="a")

    status = guard.status(0)
    assert status.remaining == pytest.approx(2.0)
    assert status.exhausted is False
    # an 8-spend + 4-estimate would exceed the 10 cap -> refuse to launch
    assert guard.can_launch(0, estimated_usd=4.0) is False
    # a small job still fits
    assert guard.can_launch(0, estimated_usd=1.0) is True

    led.record(generation=0, kind="finetune", amount_usd=3.0, reason="b")
    status = guard.status(0)
    assert status.exhausted is True  # spend (11) >= cap (10)
    assert guard.can_launch(0) is False
    # caps are per-generation: a fresh generation is unconstrained again
    assert guard.can_launch(1, estimated_usd=9.0) is True
