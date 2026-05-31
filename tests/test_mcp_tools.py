"""Tests for the MCP memory toolset logic layer (ARCHITECTURE.md §9.3)."""

import pytest

from darwin.memory import IterationMemory, MemoryStore, GlobalMemory
from darwin.mcp import MemoryToolset


def write_args(model="model7", iteration=0, **overrides) -> dict:
    base = dict(
        model=model,
        iteration=iteration,
        generation=1,
        parent_survivor="model3",
        mutator=model,
        backend="claude",
        base_fitness=0.5,
        cost_usd=1.0,
        thesis=f"thesis {iteration}",
        changes=f"changes {iteration}",
        smoke_results="green",
        outcome=f"outcome {iteration}",
    )
    base.update(overrides)
    return base


@pytest.fixture
def tools(tmp_path):
    return MemoryToolset(MemoryStore(tmp_path))


def test_write_iteration_success(tools):
    result = tools.write_iteration(**write_args(iteration=3))
    assert result["ok"] is True
    assert result["model"] == "model7"
    assert result["iteration"] == 3
    # readable back through the store
    assert tools.store.read_iteration("model7", 3).thesis == "thesis 3"


def test_write_iteration_validation_error_is_structured(tools):
    result = tools.write_iteration(**write_args(thesis=""))  # empty body section
    assert result["ok"] is False
    assert "Thesis" in result["error"]
    # nothing persisted
    assert tools.store.iteration_numbers("model7") == []


def test_write_iteration_rejects_bad_backend(tools):
    result = tools.write_iteration(**write_args(backend="openai"))
    assert result["ok"] is False
    assert "backend" in result["error"]


def test_recent_returns_dicts_newest_first(tools):
    for n in range(4):
        tools.write_iteration(**write_args(iteration=n))
    recent = tools.recent("model7", k=2)
    assert [m["iteration"] for m in recent] == [3, 2]
    assert isinstance(recent[0], dict)
    assert recent[0]["thesis"] == "thesis 3"


def test_search_ranks_results(tools):
    tools.write_iteration(**write_args(iteration=0, thesis="explore lora rank"))
    tools.write_iteration(**write_args(iteration=1, thesis="lora lora rank rank"))
    tools.write_iteration(**write_args(iteration=2, thesis="unrelated"))
    results = tools.search("model7", "lora rank")
    assert [m["iteration"] for m in results] == [1, 0]


def test_get_global_returns_sections(tools):
    tools.store.write_global(
        GlobalMemory(objectives="o", whats_working="w", todo="t", cost_ledger="c")
    )
    gm = tools.get_global()
    assert gm == {"objectives": "o", "whats_working": "w", "todo": "t", "cost_ledger": "c"}


def test_write_iteration_signature_excludes_controller_fields(tools):
    # The agent-facing write tool must not accept controller-owned post-benchmark fields.
    import inspect

    params = inspect.signature(tools.write_iteration).parameters
    for forbidden in ("final_fitness", "mutation_failed", "finetune_failed"):
        assert forbidden not in params
