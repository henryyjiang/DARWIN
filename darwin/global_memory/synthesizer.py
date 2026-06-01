"""Global-memory synthesizers (ARCHITECTURE.md §7.4 step 3).

A `Synthesizer` turns a `GenerationDigest` + the current `GlobalMemory` into the rewritten
`GlobalMemory`. The default `ClaudeSynthesizer` runs the focused Claude reasoning pass; the
interface lets a local model drive it under `strict-local` (§4.7) and lets tests inject a
deterministic fake.

The Anthropic call uses adaptive thinking + `effort` (the modern replacement for the §7.4
"reason for ~an hour" extended-thinking budget), structured outputs to guarantee the four
sections come back as valid JSON, prompt caching on the stable system prompt, and streaming
(the reasoning pass can run long). The prompt-building and response-parsing are factored out
as pure functions so they're unit-testable without the network.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from darwin.global_memory.digest import GenerationDigest
from darwin.memory import GlobalMemory, GLOBAL_SECTIONS

MODEL = "claude-opus-4-8"
DEFAULT_EFFORT = "high"
MAX_TOKENS = 32000

SYSTEM_PROMPT = """\
You are the DARWIN global-memory pass: the system's long-horizon "what works / what to do \
next" brain (ARCHITECTURE.md §7.4). DARWIN evolves a population of code-mutating models that \
improve their own finetuning/architecture code; after each generation you read every model's \
per-model memory ("lab notebook", including the papers it cited and the data it used) plus the \
fitness and cost tables, and you REWRITE the four shared global-memory sections. This is the \
system's advanced-reasoning step: reason hard about overarching guiding principles and reason \
quantitatively about which concepts are most worth testing next.

You are the ONLY writer of global memory. Population/mutation models never write it — they \
only read it to orient at the start of their mutation window. Keep the shared signal \
coherent, grounded, and non-contradictory.

Standing research direction to uphold and refine (not discard):
- The primary capability lever is GROWING the model's parameters from an already-trained \
checkpoint — depth expansion (block/layer stacking) and MoE upcycling (dense->sparse experts) \
— and, beyond those seeds, INVENTING improved scaling methods. Promote what the population has \
shown to work and propose concrete next variants.
- Performance is the target: also weigh paper-derived techniques and data-mix optimization.
- COST and TRAIN TIME are first-class constraints. Param-scaling on large token budgets (up to \
250B tokens, multi-GPU) is expensive — reason in fitness-per-dollar and per-GPU-hour and steer \
toward the cheapest method that yields the gain.

Rewrite all four sections in full (they replace the previous versions):
- objectives: the current high-level direction + the overarching GUIDING PRINCIPLES the \
population should follow toward SOTA (synthesize/refine them from what the evidence now supports).
- whats_working: patterns CORRELATED WITH FITNESS GAINS across the population this and prior \
generations. Be concrete; cite which models/changes/methods drove gains and at what cost.
- todo: a prioritized list of CONCRETE CONCEPTS TO TEST next. Reason over the papers the models \
cited (and the math/claims in them) plus what's working to derive specific, testable hypotheses \
— name the method, the expected mechanism of improvement, and a rough cost/benefit. Prefer \
high-expected-value, affordable experiments; include scaling-method variants to try.
- cost_ledger: update the running spend table and per-generation cost guidance; flag strategies \
whose cost isn't justified by their gains.

Ground every claim in the provided per-model memories, citations, and tables. Do not invent \
results or fabricate citations. Return the four sections via the required structured-output \
format."""

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {section: {"type": "string"} for section in GLOBAL_SECTIONS},
    "required": list(GLOBAL_SECTIONS),
    "additionalProperties": False,
}


class Synthesizer(Protocol):
    """Rewrites global memory from a generation digest + the current global memory."""

    def synthesize(
        self, digest: GenerationDigest, current: GlobalMemory
    ) -> GlobalMemory: ...


def build_user_prompt(digest: GenerationDigest, current: GlobalMemory) -> str:
    """Render the (volatile) user message: current global memory + this gen's inputs."""
    return f"""\
# Generation {digest.generation} — global-memory pass

## Current global memory (to be rewritten)

### objectives
{current.objectives or "(empty)"}

### whats_working
{current.whats_working or "(empty)"}

### todo
{current.todo or "(empty)"}

### cost_ledger
{current.cost_ledger or "(empty)"}

## This generation's fitness & cost table
{digest.fitness_table()}

## This generation's per-model memory files
{digest.render_memories()}

Now rewrite all four global-memory sections."""


def parse_global_memory(data: dict[str, Any]) -> GlobalMemory:
    """Build a GlobalMemory from a structured-output object (inverse of the schema)."""
    return GlobalMemory(**{section: data.get(section, "") for section in GLOBAL_SECTIONS})


class MockSynthesizer:
    """Deterministic, offline global-memory synthesizer for the test profile (§7.4 / TEST_RUN_PLAN §3.4).

    The real §7.4 pass is an hours-long Claude reasoning step; this one is instant and offline so
    the generational loop's global-memory stage runs in the budget-free test. It derives the new
    global memory from the digest: it carries `objectives` forward (so a real direction can be
    seeded once and persist), and appends a one-line `gen N: best=<model> fitness=<x>` summary to
    `whats_working` — enough for the dashboard/acceptance checks to show the pass updating memory
    each generation. `todo`/`cost_ledger` are carried through unchanged.
    """

    DEFAULT_OBJECTIVES = (
        "Increase fitness across the benchmark suite via green genome mutations (test profile)."
    )

    def synthesize(
        self, digest: GenerationDigest, current: GlobalMemory
    ) -> GlobalMemory:
        best_line = self._best_line(digest)
        whats_working = (current.whats_working + "\n" if current.whats_working else "") + best_line
        return GlobalMemory(
            objectives=current.objectives or self.DEFAULT_OBJECTIVES,
            whats_working=whats_working.strip(),
            todo=current.todo,
            cost_ledger=current.cost_ledger,
        )

    @staticmethod
    def _best_line(digest: GenerationDigest) -> str:
        scored = [m for m in digest.memories if m.final_fitness is not None]
        if not scored:
            return f"gen {digest.generation}: no scored models"
        best = max(scored, key=lambda m: m.final_fitness)
        return f"gen {digest.generation}: best={best.model} fitness={best.final_fitness:.4g}"


class CappedSynthesizer:
    """Wrap a `Synthesizer` with a wall-clock cap; on timeout/error keep `current` (§3.4).

    For the opt-in real-Claude global pass in the test: the §7.4 reasoning pass can run long, and a
    budget-free smoke test must never stall on it. This runs the wrapped synthesizer in a worker
    thread and waits `timeout_s`; if it doesn't finish (or raises), the loop proceeds with the
    *unchanged* current global memory rather than blocking or crashing. The abandoned call is left
    to finish in its daemon thread (dev-only; acceptable for the throwaway test box)."""

    def __init__(self, inner: "Synthesizer", timeout_s: float = 300.0):
        self.inner = inner
        self.timeout_s = timeout_s

    def synthesize(
        self, digest: GenerationDigest, current: GlobalMemory
    ) -> GlobalMemory:
        import concurrent.futures

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self.inner.synthesize, digest, current)
        try:
            return future.result(timeout=self.timeout_s)
        except concurrent.futures.TimeoutError:
            print(f"[global-memory] synthesizer exceeded {self.timeout_s}s cap; keeping current memory")
            return current
        except Exception as exc:  # pragma: no cover - defensive: never stall the loop on a pass error
            print(f"[global-memory] synthesizer failed ({exc!r}); keeping current memory")
            return current
        finally:
            # don't block shutdown on the (possibly still-running) call; let the daemon thread die
            executor.shutdown(wait=False)


class ClaudeSynthesizer:
    """Global-memory synthesizer backed by the Anthropic API (the §7.4 default)."""

    def __init__(
        self,
        client: Any | None = None,
        model: str = MODEL,
        effort: str = DEFAULT_EFFORT,
    ):
        self._client = client
        self.model = model
        self.effort = effort

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic  # lazy: pure helpers/tests don't require the SDK

            self._client = anthropic.Anthropic()
        return self._client

    def _request_kwargs(
        self, digest: GenerationDigest, current: GlobalMemory, *, adaptive_thinking: bool
    ) -> dict[str, Any]:
        """Build the streaming-request kwargs (pure). `adaptive_thinking` is dropped on the retry
        path for models that don't support it (e.g. Haiku)."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA},
            },
            "system": [
                {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": build_user_prompt(digest, current)}],
        }
        if adaptive_thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        return kwargs

    def _stream_final(self, client: Any, kwargs: dict[str, Any]) -> Any:
        with client.messages.stream(**kwargs) as stream:
            return stream.get_final_message()

    def synthesize(
        self, digest: GenerationDigest, current: GlobalMemory
    ) -> GlobalMemory:
        client = self._get_client()
        try:
            message = self._stream_final(
                client, self._request_kwargs(digest, current, adaptive_thinking=True)
            )
        except Exception as exc:  # noqa: BLE001 - inspect the message, then narrow or re-raise
            if "thinking" in str(exc).lower():
                # model doesn't support adaptive thinking (e.g. Haiku) → retry without it
                message = self._stream_final(
                    client, self._request_kwargs(digest, current, adaptive_thinking=False)
                )
            else:
                raise

        text = next(b.text for b in message.content if b.type == "text")
        return parse_global_memory(json.loads(text))
