"""Tests for the smoke runner and Git checkpointer (ARCHITECTURE.md §4.4)."""

import subprocess
import sys
from pathlib import Path

import pytest

from darwin.mutation_agent import GitCheckpointer, SmokeTest


def make_genome(path: Path, ok: bool = True) -> None:
    (path / "recipe.py").write_text(f"OK = {ok}\n", encoding="utf-8")
    (path / "smoke_test.py").write_text(
        "import recipe, sys\nsys.exit(0 if recipe.OK else 1)\n", encoding="utf-8"
    )


def smoke() -> SmokeTest:
    # Use the running interpreter (bare "python" hits a stub on this Windows host).
    return SmokeTest(command=[sys.executable, "smoke_test.py"])


def _branch(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "branch", "--show-current"],
        capture_output=True, text=True,
    ).stdout.strip()


def _subject(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "log", "-1", "--pretty=%s"],
        capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def genome(tmp_path):
    make_genome(tmp_path)
    return tmp_path


# --- smoke runner ---

def test_smoke_passes_on_good_genome(genome):
    result = smoke().run(genome)
    assert result.passed and result.exit_code == 0


def test_smoke_fails_on_broken_genome(genome):
    make_genome(genome, ok=False)
    result = smoke().run(genome)
    assert not result.passed and result.exit_code == 1


# --- checkpointer ---

def test_init_offspring_creates_branch_and_base(genome):
    cp = GitCheckpointer(genome)
    base = cp.init_offspring("7", parent_survivor="model3")
    assert base == cp.head()
    assert _branch(genome) == "offspring/7"
    assert not cp.has_last_green()  # no green produced yet


def test_commit_green_advances_last_green_tag(genome):
    cp = GitCheckpointer(genome)
    cp.init_offspring("7")
    (genome / "recipe.py").write_text("OK = True\n# raise rank\n", encoding="utf-8")
    sha = cp.commit_green("raise lora rank")
    assert cp.has_last_green()
    assert cp.last_green() == sha == cp.head()
    assert _subject(genome).startswith("darwin-green:")


def test_finalize_resets_to_last_green(genome):
    cp = GitCheckpointer(genome)
    cp.init_offspring("7")
    (genome / "recipe.py").write_text("OK = True\n# good change\n", encoding="utf-8")
    green = cp.commit_green("good change")
    # a later broken, uncommitted edit
    (genome / "recipe.py").write_text("OK = False\n", encoding="utf-8")

    final, fell_back = cp.finalize_genome()
    assert final == green and fell_back is False
    assert (genome / "recipe.py").read_text(encoding="utf-8") == "OK = True\n# good change\n"


def test_zero_green_fallback_to_clone(genome):
    cp = GitCheckpointer(genome)
    base = cp.init_offspring("7")
    # the agent broke the genome and never produced a green commit
    (genome / "recipe.py").write_text("OK = False\n", encoding="utf-8")

    final, fell_back = cp.finalize_genome()
    assert fell_back is True and final == base
    # back to the unchanged clone of S (green by construction)
    assert (genome / "recipe.py").read_text(encoding="utf-8") == "OK = True\n"
    assert smoke().run(genome).passed
