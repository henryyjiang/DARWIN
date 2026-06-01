"""Backend B — local-model harness (ARCHITECTURE.md §4.6).

When the mutator M is the population model itself, M's LoRA-merged weights are served by vLLM
(`vllm_serving.py`) behind an OpenAI-compatible endpoint, and an **agentic harness** (OpenHands
by default; the §9.3 shim as fallback) turns that raw chat model into a tool-using coder driving
the **same** `darwin-mcp` tools + directive as the Claude backend. That tool/directive parity is
what makes §3.2's "M's identity drives the mutation" hold meaningfully and keeps results
comparable across backends.

This implements the §4.2 `MutationBackend.run(ctx, deadline)` contract, so it plugs into the
controller's `mutation_backend_factory` seam exactly where `backend="local"` routes (the
controller already does that routing, §4.7). The pure, testable surface is `build_harness_config`;
the live OpenHands session needs a GPU + the harness package (optional `local` extra) and is
delegated to an injectable `harness_runner` (deferred default), mirroring the deferral of the
Claude SDK session and the Lambda finetune. The always-green finalization + the kill-and-recover
guarantee are backend-agnostic and handled by `run_mutation_window` / the checkpointer (§9.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from darwin.mutation_agent.backend import MutationContext
from darwin.mutation_agent.deadline import DeadlineManager
from darwin.mutation_agent.directive import DIRECTIVE_SYSTEM_PROMPT
from darwin.mutation_agent.vllm_serving import VLLMServeConfig

# A larger turn budget than the Claude backend, within the same wall-clock window: a 32B coder
# is materially weaker at long-horizon agentic work, so it gets more, smaller steps (§4.6).
DEFAULT_MAX_ITERATIONS = 250


@dataclass
class HarnessConfig:
    """What the agentic harness needs to drive one mutation window (§4.6)."""

    workspace: str  # the offspring genome repo (cwd)
    base_url: str  # vLLM OpenAI-compatible endpoint
    api_key: str
    model: str  # the served model name (M's weights)
    system_prompt: str  # the shared directive system prompt (tool/directive parity, §4.6)
    task: str  # the predetermined mutation directive
    mcp_servers: dict[str, Any]  # darwin-mcp attach (OpenHands native; shim as fallback, §9.3)
    max_iterations: int


# A harness runner drives the live session; injected so the backend is testable with a fake.
HarnessRunner = Callable[[HarnessConfig, MutationContext, DeadlineManager], None]


def build_harness_config(
    ctx: MutationContext,
    serve: VLLMServeConfig,
    *,
    mcp_servers: dict[str, Any] | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> HarnessConfig:
    """Build the harness config for an offspring's window (pure → testable)."""
    return HarnessConfig(
        workspace=str(ctx.genome_dir),
        base_url=serve.base_url,
        api_key=serve.api_key,
        model=serve.served_model_name,
        system_prompt=DIRECTIVE_SYSTEM_PROMPT,
        task=ctx.directive,
        mcp_servers=dict(mcp_servers or {}),
        max_iterations=max_iterations,
    )


class LocalMutationBackend:
    """Drives the population model as the mutator via an agentic harness (§4.6)."""

    def __init__(
        self,
        serve_config: VLLMServeConfig,
        mcp_servers: dict[str, Any] | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        harness_runner: HarnessRunner | None = None,
    ):
        self.serve_config = serve_config
        self.mcp_servers = mcp_servers or {}
        self.max_iterations = max_iterations
        self.harness_runner = harness_runner

    def run(self, ctx: MutationContext, deadline: DeadlineManager) -> None:
        config = build_harness_config(
            ctx,
            self.serve_config,
            mcp_servers=self.mcp_servers,
            max_iterations=self.max_iterations,
        )
        runner = self.harness_runner or _run_openhands
        runner(config, ctx, deadline)


def _run_openhands(
    config: HarnessConfig, ctx: MutationContext, deadline: DeadlineManager
) -> None:
    """Live OpenHands session against the vLLM endpoint (§4.6) — scaffold.

    The live path: configure OpenHands with `base_url`/`api_key`/`model`, attach `darwin-mcp`
    (native MCP support), set the workspace to the genome repo and `max_iterations`, run the
    `task`; inject the soft-deadline nudge via OpenHands' message/event API and enforce the hard
    deadline + kill by stopping the harness process (the orchestrator then recovers `last-green`,
    §9.3). Needs a GPU + the harness package (optional `local` extra), so it is deferred — inject
    a `harness_runner` to exercise the contract without it.
    """
    raise NotImplementedError(
        "LocalMutationBackend's default OpenHands runner is scaffolded; the live session needs "
        "a GPU-served vLLM endpoint + the OpenHands harness (optional `local` extra). Inject a "
        "harness_runner to drive the window without it."
    )


def make_mutation_backend_factory(
    *,
    serve_config: VLLMServeConfig | None = None,
    mcp_servers: dict[str, Any] | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    harness_runner: HarnessRunner | None = None,
    claude_transcript_dir: Path | str | None = None,
) -> Callable[[str, MutationContext], Any]:
    """A default `mutation_backend_factory` for `LocalGenerationOps` (§4.7 routing).

    Maps the per-offspring backend name to a concrete backend: `mock` → `MockMutationBackend` (the
    offline test-profile mutator, §3.3); `local` → `LocalMutationBackend` (requires a
    `serve_config`); anything else (`claude`, and the degenerate <2-survivor fallback) →
    `ClaudeMutationBackend`. The live backends attach the same `darwin-mcp` so the directive and
    tools are identical across them (§9.4).
    """
    # Lazy import to avoid a hard dependency cycle / keep import light.
    from darwin.mutation_agent.claude_backend import ClaudeMutationBackend

    def factory(backend_name: str, ctx: MutationContext) -> Any:
        if backend_name == "mock":
            from darwin.mutation_agent.mock_backend import MockMutationBackend

            return MockMutationBackend()
        if backend_name == "local":
            if serve_config is None:
                raise ValueError("local backend requires a VLLMServeConfig (serve_config)")
            return LocalMutationBackend(
                serve_config=serve_config,
                mcp_servers=mcp_servers,
                max_iterations=max_iterations,
                harness_runner=harness_runner,
            )
        transcript = None
        if claude_transcript_dir is not None:
            transcript = Path(claude_transcript_dir) / f"{ctx.offspring_id}.jsonl"
        return ClaudeMutationBackend(mcp_servers=mcp_servers or {}, transcript_path=transcript)

    return factory
