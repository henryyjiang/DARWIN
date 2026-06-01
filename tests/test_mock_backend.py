"""Mock mutation backend (TEST_RUN_PLAN §3.3).

Drives the real `run_mutation_window` over a real temp git genome with the offline
`MockMutationBackend` (no Claude/Docker) and asserts: a green checkpoint was produced, the genome
gained one improvement marker, and a per-model memory file was written.
"""

import sys
from pathlib import Path

from darwin.finetune.mock_entrypoint import IMPROVEMENT_MARKER, count_markers
from darwin.memory import MemoryStore
from darwin.mutation_agent import (
    DeadlineManager,
    GitCheckpointer,
    MutationContext,
    SmokeTest,
    run_mutation_window,
)
from darwin.mutation_agent.mock_backend import MockMutationBackend, apply_marker_edit


def _genome(tmp_path: Path) -> Path:
    g = tmp_path / "genome"
    g.mkdir(parents=True)
    (g / "recipe.py").write_text("OK = True\n", encoding="utf-8")
    (g / "smoke_test.py").write_text(
        "import recipe, sys; sys.exit(0 if recipe.OK else 1)\n", encoding="utf-8"
    )
    return g


def test_apply_marker_edit_appends_to_recipe(tmp_path):
    g = _genome(tmp_path)
    target = apply_marker_edit(g, 5)
    assert target.name == "recipe.py"
    assert f"# {IMPROVEMENT_MARKER} 5" in target.read_text(encoding="utf-8")
    assert count_markers(g) == 1


def test_apply_marker_edit_fallback_file_when_no_recipe(tmp_path):
    g = tmp_path / "g"
    g.mkdir()
    target = apply_marker_edit(g, 0)
    assert target.name == "improvements.py"
    assert count_markers(g) == 1


def test_mock_window_green_commit_and_memory(tmp_path):
    g = _genome(tmp_path)
    store = MemoryStore(tmp_path / "store")
    ctx = MutationContext(
        offspring_id="o0", genome_dir=g, model="o0", parent_survivor="s0", mutator="s1",
        generation=0, iteration=0, backend_name="mock", base_fitness=0.5, directive="",
        checkpointer=GitCheckpointer(g), smoke=SmokeTest(command=[sys.executable, "smoke_test.py"]),
        store=store,
    )
    deadline = DeadlineManager(window_s=100, soft_lead_s=20, kill_grace_s=10,
                               clock=lambda: 0.0, start=0.0)
    result = run_mutation_window(ctx, MockMutationBackend(), deadline)

    assert result.produced_green is True
    assert result.mutation_failed is False
    assert result.memory_written is True
    # the marker edit survived to the green final genome
    assert count_markers(g) == 1
    # the per-model memory file records the deterministic change
    mem = store.read_iteration("o0", 0)
    assert IMPROVEMENT_MARKER in mem.changes
    assert mem.backend == "mock"
