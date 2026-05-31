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
next" brain (ARCHITECTURE.md §7.4). DARWIN evolves a population of code-mutating models; \
after each generation you read every model's per-model memory ("lab notebook") plus the \
fitness and cost tables, and you REWRITE the four shared global-memory sections.

You are the ONLY writer of global memory. Population/mutation models never write it — they \
only read it to orient at the start of their mutation window. Keep the shared signal \
coherent, grounded, and non-contradictory.

Rewrite all four sections in full (they replace the previous versions):
- objectives: current high-level direction toward state-of-the-art targets.
- whats_working: patterns CORRELATED WITH FITNESS GAINS across the population this and prior \
generations. Be concrete; cite which models/changes drove gains. Cost is a first-class \
constraint — steer away from expensive strategies when fitness gains don't justify them.
- todo: open problems and unexplored ideas for next generation's mutators to pick up.
- cost_ledger: update the running spend table and per-generation cost guidance.

Ground every claim in the provided per-model memories and tables. Do not invent results. \
Return the four sections via the required structured-output format."""

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

    def synthesize(
        self, digest: GenerationDigest, current: GlobalMemory
    ) -> GlobalMemory:
        client = self._get_client()
        with client.messages.stream(
            model=self.model,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA},
            },
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {"role": "user", "content": build_user_prompt(digest, current)}
            ],
        ) as stream:
            message = stream.get_final_message()

        text = next(b.text for b in message.content if b.type == "text")
        return parse_global_memory(json.loads(text))
