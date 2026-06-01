"""GPU-free end-to-end smoke test of the local-model (OpenHands) mutation path.

Drives a *real* OpenHands V1-SDK mutation window (`backend="local"`) against a small,
tool-capable model served by **Ollama** over its OpenAI-compatible endpoint — no GPU and no vLLM
needed, since the harness only needs a `base_url`/`api_key`/`model` (`build_llm_kwargs`). It is the
live counterpart of the unit tests: it actually exercises OpenHands → `darwin-mcp` tools → genome
file edits → `smoke.run` → `memory.write_iteration` against a throwaway git genome.

Run it inside the `containers/smoke-local` compose stack (which provides Ollama + the
`openhands-sdk`/`openhands-tools` deps); see that directory's README. Everything is configurable by
env; the defaults match the compose file.

This is a debugging/validation tool, not part of the test suite.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# The trivial "genome": a recipe module + a read-only smoke test that passes when the recipe
# imports cleanly. The local model is asked (via the "small" directive) to make one green edit.
RECIPE_PY = "OK = True\n"
SMOKE_TEST_PY = "import recipe, sys\nsys.exit(0 if recipe.OK else 1)\n"


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def setup_genome(genome_dir: Path) -> None:
    """Create a throwaway git genome with an initial green commit."""
    genome_dir.mkdir(parents=True, exist_ok=True)
    (genome_dir / "recipe.py").write_text(RECIPE_PY, encoding="utf-8")
    (genome_dir / "smoke_test.py").write_text(SMOKE_TEST_PY, encoding="utf-8")
    if not (genome_dir / ".git").exists():
        _git(["init", "-q"], genome_dir)
        _git(["config", "user.email", "smoke@darwin.local"], genome_dir)
        _git(["config", "user.name", "darwin-smoke"], genome_dir)
    _git(["add", "-A"], genome_dir)
    # Allow an empty commit on re-runs where files are unchanged.
    subprocess.run(
        ["git", "commit", "-q", "--allow-empty", "-m", "smoke: initial genome"],
        cwd=genome_dir, check=True, capture_output=True,
    )


def main() -> int:
    work = Path(os.environ.get("SMOKE_WORK_DIR", tempfile.gettempdir() + "/darwin-smoke"))
    genome_dir = Path(os.environ.get("DARWIN_GENOME_DIR", str(work / "genome")))
    store_root = Path(os.environ.get("DARWIN_STORE_ROOT", str(work / "store")))
    result_out = Path(os.environ.get("DARWIN_RESULT_OUT", str(work / "result.json")))
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434/v1")
    model = os.environ.get("SMOKE_MODEL", "qwen2.5-coder:7b")

    setup_genome(genome_dir)
    store_root.mkdir(parents=True, exist_ok=True)

    # The serve endpoint points at Ollama (OpenAI-compatible); host/port are split from base_url.
    host, _, port = base_url.split("//", 1)[1].split("/", 1)[0].partition(":")
    env_overrides = {
        "DARWIN_BACKEND": "local",
        "DARWIN_OFFSPRING_ID": os.environ.get("DARWIN_OFFSPRING_ID", "smoke1"),
        "DARWIN_MODEL": os.environ.get("DARWIN_MODEL", "qwen-smoke"),
        "DARWIN_PARENT_SURVIVOR": os.environ.get("DARWIN_PARENT_SURVIVOR", "qwen-smoke"),
        "DARWIN_MUTATOR": os.environ.get("DARWIN_MUTATOR", "qwen-smoke"),
        "DARWIN_GENERATION": os.environ.get("DARWIN_GENERATION", "0"),
        "DARWIN_ITERATION": os.environ.get("DARWIN_ITERATION", "0"),
        "DARWIN_GENOME_DIR": str(genome_dir),
        "DARWIN_STORE_ROOT": str(store_root),
        "DARWIN_RESULT_OUT": str(result_out),
        "DARWIN_SMOKE_CMD": json.dumps([sys.executable, "smoke_test.py"]),
        "DARWIN_DIRECTIVE_STYLE": os.environ.get("DARWIN_DIRECTIVE_STYLE", "small"),
        # Short window so a CPU-served model finishes the smoke quickly.
        "DARWIN_WINDOW_H": os.environ.get("DARWIN_WINDOW_H", "0.17"),
        "DARWIN_SOFT_DEADLINE_MIN": os.environ.get("DARWIN_SOFT_DEADLINE_MIN", "3"),
        "DARWIN_KILL_GRACE_MIN": os.environ.get("DARWIN_KILL_GRACE_MIN", "2"),
        # Point the local serve config at the Ollama endpoint.
        "DARWIN_SERVE_HOST": host,
        "DARWIN_SERVE_PORT": port or "80",
        "DARWIN_SERVE_MODEL_NAME": model,
        "DARWIN_SERVE_API_KEY": os.environ.get("DARWIN_SERVE_API_KEY", "ollama"),
    }
    os.environ.update(env_overrides)

    print("=== darwin local-mutator smoke ===")
    print(f"  genome : {genome_dir}")
    print(f"  store  : {store_root}")
    print(f"  model  : {model} @ {base_url}")
    print(f"  window : {env_overrides['DARWIN_WINDOW_H']}h "
          f"(soft -{env_overrides['DARWIN_SOFT_DEADLINE_MIN']}m, "
          f"kill +{env_overrides['DARWIN_KILL_GRACE_MIN']}m)")
    print("===================================", flush=True)

    # Import after env is set; main() reads os.environ via _default_backend_factory.
    from darwin.mutation_agent.entrypoint import main as entrypoint_main

    rc = entrypoint_main()

    if result_out.exists():
        result = json.loads(result_out.read_text(encoding="utf-8"))
        print("\n=== result ===")
        print(json.dumps(result, indent=2))
        green = result.get("produced_green")
        print(f"\nproduced_green={green} mutation_failed={result.get('mutation_failed')} "
              f"memory_written={result.get('memory_written')}")
        # The genome head + any iteration memory the agent wrote are left in place for inspection.
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
