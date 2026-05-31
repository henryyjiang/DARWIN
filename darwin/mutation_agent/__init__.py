"""Mutation Agent (ARCHITECTURE.md §4).

One offspring's mutation = one autonomous agent session inside one Docker container, driven by
a wall-clock budget and checkpointed in Git so the final genome is always a green commit.

Backend-agnostic core (this package):
- `smoke.SmokeTest`        — the §4.4.1 smoke-test runner ("green" = the recipe trains).
- `checkpoint.GitCheckpointer` — offspring branch, green commits, last-green tag, revert (§4.4).
- `deadline.DeadlineManager`   — soft / hard / kill wall-clock phases (§4.3).
- `directive`              — the structured mutation directive + deadline nudges (§4.1/§4.8).
- `backend.MutationContext`/`MutationBackend`/`MutationResult` — the §4.2 lifecycle contract.
- `runner.run_mutation_window` — orchestrates a window; guarantees an always-green final genome.

Backends behind the one interface:
- `claude_backend.ClaudeMutationBackend` — Claude Agent SDK headless session (§4.5).
- `local_backend.LocalMutationBackend` — the population model as mutator via vLLM + an agentic
  harness (§4.6); the live session is deferred behind an injectable harness runner.
- `make_mutation_backend_factory` — the default `claude`/`local` router for the controller seam.
"""

from darwin.mutation_agent.smoke import SmokeTest, SmokeResult
from darwin.mutation_agent.checkpoint import GitCheckpointer
from darwin.mutation_agent.deadline import DeadlineManager, Phase
from darwin.mutation_agent.backend import (
    MutationBackend,
    MutationContext,
    MutationResult,
)
from darwin.mutation_agent.runner import run_mutation_window
from darwin.mutation_agent.memory_synthesis import (
    ClaudeMemorySynthesizer,
    MemorySynthesizer,
    SynthesisContext,
    git_log,
    read_transcript_excerpt,
)
from darwin.mutation_agent.vllm_serving import (
    VLLMServeConfig,
    VLLMServer,
    VLLMServerError,
    build_serve_command,
)
from darwin.mutation_agent.local_backend import (
    HarnessConfig,
    LocalMutationBackend,
    build_harness_config,
    make_mutation_backend_factory,
)

__all__ = [
    "SmokeTest",
    "SmokeResult",
    "GitCheckpointer",
    "DeadlineManager",
    "Phase",
    "MutationBackend",
    "MutationContext",
    "MutationResult",
    "run_mutation_window",
    "MemorySynthesizer",
    "ClaudeMemorySynthesizer",
    "SynthesisContext",
    "git_log",
    "read_transcript_excerpt",
    "VLLMServeConfig",
    "VLLMServer",
    "VLLMServerError",
    "build_serve_command",
    "HarnessConfig",
    "LocalMutationBackend",
    "build_harness_config",
    "make_mutation_backend_factory",
]
